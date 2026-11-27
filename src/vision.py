# -*- coding: utf-8 -*-
"""
Módulo 3: Clasificación de Enfermedades en Hojas de Arroz (Visión por Computadora).

Modelo supervisado entrenado con fotografías REALES de campo (ver src/vision_dataset.py):
    Hoja Sana | Tizón Bacteriano (Xanthomonas oryzae) | Mancha Marrón (Bipolaris oryzae)

Pipeline:
    imagen -> redimensionado 128x128 -> vector de 123 descriptores
             (color RGB/HSV, histograma conjunto H-S, fracciones de píxeles
              verdes/amarillos/marrones/blanquecinos, textura LBP, orientación de gradientes)
           -> HistGradientBoostingClassifier -> probabilidades por clase

Evaluación: partición 80/20 estratificada POR GRUPOS de casi-duplicados
(StratifiedGroupKFold), para que fotos casi idénticas no queden en train y test a la vez.
"""

import os
import sys

import joblib
import numpy as np
from PIL import Image

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'vision_hojas_arroz.pkl')
SAMPLE_DIR = os.path.join(BASE_DIR, 'data', 'sample_leaves')

FEATURE_VERSION = 1
IMG_SIZE = 128
CLASES_ID = ['hoja_sana', 'tizon_bacteriano', 'mancha_marron']
CLASES_NOMBRE = {
    'hoja_sana': 'Hoja Sana',
    'tizon_bacteriano': 'Tizón Bacteriano (Xanthomonas)',
    'mancha_marron': 'Mancha Marrón (Bipolaris)',
}
CLASES_CORTO = {'hoja_sana': 'Hoja Sana', 'tizon_bacteriano': 'Tizón Bacteriano', 'mancha_marron': 'Mancha Marrón'}
UMBRAL_CONFIANZA = 60.0  # por debajo de este % se advierte que el diagnóstico es incierto


# ------------------------------------------------------------------------------
# Extracción de características
# ------------------------------------------------------------------------------
def _rgb_to_hsv(a):
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx, mn = a.max(-1), a.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    m = d > 1e-6
    rm = m & (mx == r)
    gm = m & (mx == g) & ~rm
    bm = m & ~rm & ~gm
    h[rm] = ((g - b)[rm] / d[rm]) % 6
    h[gm] = (b - r)[gm] / d[gm] + 2
    h[bm] = (r - g)[bm] / d[bm] + 4
    h /= 6.0
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0)
    return np.stack([h, s, mx], -1)


def _lbp_uniforme(gray):
    """Histograma Local Binary Pattern uniforme (8 vecinos, radio 1) -> 10 bins."""
    c = gray[1:-1, 1:-1]
    offs = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]
    bits = np.stack([gray[i:i + c.shape[0], j:j + c.shape[1]] >= c for i, j in offs], -1).astype(np.uint8)
    trans = np.abs(np.diff(np.concatenate([bits, bits[..., :1]], -1), axis=-1)).sum(-1)
    code = np.where(trans <= 2, bits.sum(-1), 9)
    h = np.bincount(code.ravel(), minlength=10).astype(np.float64)
    return h / h.sum()


def extraer_caracteristicas(img):
    """Devuelve el vector de 123 descriptores de una imagen PIL."""
    a = np.asarray(img.convert('RGB').resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
    hsv = _rgb_to_hsv(a)
    out = []
    # Momentos de color (media, desviación, asimetría) en RGB y HSV -> 18
    for arr in (a, hsv):
        for k in range(3):
            x = arr[..., k].ravel()
            mu, sd = x.mean(), x.std()
            out += [mu, sd, ((x - mu) ** 3).mean() / (sd ** 3 + 1e-6)]
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # Histograma conjunto Tono x Saturación (18 x 4) -> 72
    hs, _, _ = np.histogram2d(H.ravel(), S.ravel(), bins=[18, 4], range=[[0, 1], [0, 1]])
    out += list((hs / hs.sum()).ravel())
    # Histograma de brillo -> 8
    vh, _ = np.histogram(V, bins=8, range=(0, 1))
    out += list(vh / vh.sum())
    # Fracción de píxeles por tipo de tejido -> 4
    hue = H * 360
    out += [
        np.mean((hue > 70) & (hue < 170) & (S > 0.25)),   # verde (clorofila)
        np.mean((hue >= 40) & (hue <= 70) & (S > 0.25)),  # amarillo (clorosis / halo)
        np.mean((hue < 40) & (S > 0.25) & (V < 0.75)),    # marrón (necrosis / manchas)
        np.mean((S < 0.18) & (V > 0.65)),                 # blanquecino (tejido seco del tizón)
    ]
    # Textura LBP -> 10
    gray = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    out += list(_lbp_uniforme(gray))
    # Orientación y magnitud de gradientes -> 9 + 2
    gy, gx = np.gradient(gray)
    mag = np.hypot(gx, gy)
    ang = np.arctan2(gy, gx) % np.pi
    oh, _ = np.histogram(ang, bins=9, range=(0, np.pi), weights=mag)
    out += list(oh / (oh.sum() + 1e-6))
    out += [mag.mean(), mag.std()]
    return np.array(out, dtype=np.float32)


def construir_matriz(meta, data_dir):
    X = np.stack([extraer_caracteristicas(Image.open(os.path.join(data_dir, f))) for f in meta['archivo']])
    y = meta['clase'].map({c: i for i, c in enumerate(CLASES_ID)}).to_numpy()
    return X, y


def particion_train_test(meta, seed=42):
    """80/20 estratificado por grupos de casi-duplicados. Devuelve (idx_train, idx_test)."""
    from sklearn.model_selection import StratifiedGroupKFold
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    return next(cv.split(meta, meta['clase'], groups=meta['grupo']))


def nuevo_modelo():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=42)


# ------------------------------------------------------------------------------
# Clasificador usado por la aplicación
# ------------------------------------------------------------------------------
class PlantDiseaseClassifier:
    def __init__(self):
        self.classes = [CLASES_NOMBRE[c] for c in CLASES_ID]
        self.model = None
        self.metrics = None
        self.treatment_recommendations = {
            'Hoja Sana': {
                'diagnostico': 'El tejido foliar se encuentra vigoroso, con fotosíntesis activa y sin necrosis visible.',
                'accion': 'Continuar con monitoreo preventivo. Aplicar bioestimulantes y mantener drenes limpios para evitar anegamiento.',
                'severidad': 'Baja (Seguro)',
                'color': 'green'
            },
            'Tizón Bacteriano (Xanthomonas)': {
                'diagnostico': 'Lesiones alargadas amarillentas a blanquecinas que avanzan desde la punta y los bordes de la hoja. Favorecida por temperaturas > 30°C y lluvias torrenciales.',
                'accion': 'Urgente: Reducir nivel de fertilización nitrogenada. Desinfectar compuertas de riego y aplicar bactericidas a base de cobre aprobados por SENASA.',
                'severidad': 'Alta (Crítico ante El Niño)',
                'color': 'red'
            },
            'Mancha Marrón (Bipolaris)': {
                'diagnostico': 'Manchas ovales de color marrón oscuro con halo amarillento. Común en suelos con asfixia radicular y exceso de sales post-inundación.',
                'accion': 'Drenar parcelas encharcadas de inmediato. Aplicar sulfato de potasio foliar y fungicidas sistémicos (Triazoles o Estrobilurinas).',
                'severidad': 'Media-Alta (Alerta Naranja)',
                'color': 'orange'
            }
        }

    # ---------------- entrenamiento ----------------
    def train_model(self, exportar_muestras=True):
        """Entrena con el dataset real, evalúa en hold-out y exporta models/vision_hojas_arroz.pkl."""
        import sklearn
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
        try:
            from src.vision_dataset import cargar_metadata, OUT_DIR
        except ImportError:  # ejecutado como `python src/vision.py`
            from vision_dataset import cargar_metadata, OUT_DIR

        meta = cargar_metadata()
        X, y = construir_matriz(meta, OUT_DIR)
        idx_tr, idx_te = particion_train_test(meta)

        model = nuevo_modelo()
        model.fit(X[idx_tr], y[idx_tr])
        y_pred = model.predict(X[idx_te])

        nombres = [CLASES_CORTO[c] for c in CLASES_ID]
        self.metrics = {
            'accuracy': round(float(accuracy_score(y[idx_te], y_pred)), 4),
            'f1_macro': round(float(f1_score(y[idx_te], y_pred, average='macro')), 4),
            'n_train': int(len(idx_tr)),
            'n_test': int(len(idx_te)),
            'matriz_confusion': confusion_matrix(y[idx_te], y_pred).tolist(),
            'reporte': classification_report(y[idx_te], y_pred, target_names=nombres, output_dict=True),
        }
        self.model = model
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump({
            'model': model,
            'clases': CLASES_ID,
            'feature_version': FEATURE_VERSION,
            'img_size': IMG_SIZE,
            'metrics': self.metrics,
            'sklearn_version': sklearn.__version__,
        }, MODEL_PATH)

        if exportar_muestras:
            self._exportar_muestras(meta, idx_te, y, y_pred, OUT_DIR)

        print(f"[OK] Modelo de visión entrenado. Accuracy hold-out = {self.metrics['accuracy']:.3f} | "
              f"F1 macro = {self.metrics['f1_macro']:.3f} (n_test = {len(idx_te)})")
        return self.metrics

    @staticmethod
    def _exportar_muestras(meta, idx_te, y, y_pred, data_dir, por_clase=2):
        """Copia a data/sample_leaves/ fotos del conjunto de PRUEBA (nunca vistas en entrenamiento)."""
        os.makedirs(SAMPLE_DIR, exist_ok=True)
        for f in os.listdir(SAMPLE_DIR):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                os.remove(os.path.join(SAMPLE_DIR, f))
        rng = np.random.default_rng(7)
        for k, clase in enumerate(CLASES_ID):
            # muestras bien clasificadas, para ilustrar cada clase en la demo
            cand = [i for i, p in zip(idx_te, y_pred) if y[i] == k and p == k]
            for n, i in enumerate(rng.choice(cand, size=min(por_clase, len(cand)), replace=False), 1):
                Image.open(os.path.join(data_dir, meta['archivo'].iloc[i])).save(
                    os.path.join(SAMPLE_DIR, f'{clase}_{n}.jpg'), quality=92)

    # ---------------- inferencia ----------------
    def load_model(self):
        """Carga el modelo exportado; si falta o es incompatible, lo reentrena."""
        try:
            pack = joblib.load(MODEL_PATH)
            if pack.get('feature_version') != FEATURE_VERSION:
                raise ValueError('versión de características distinta')
            self.model = pack['model']
            self.metrics = pack.get('metrics')
        except Exception as e:
            print(f"[AVISO] No se pudo cargar {MODEL_PATH} ({e}). Reentrenando...")
            self.train_model(exportar_muestras=False)

    def predict_image(self, img_input):
        if self.model is None:
            self.load_model()
        image = Image.open(img_input) if isinstance(img_input, str) else img_input
        x = extraer_caracteristicas(image.convert('RGB')).reshape(1, -1)
        probs = self.model.predict_proba(x)[0]

        best_idx = int(np.argmax(probs))
        predicted_class = self.classes[best_idx]
        confianza = float(probs[best_idx]) * 100.0
        details = self.treatment_recommendations[predicted_class]

        return {
            'clase_detectada': predicted_class,
            'confianza_pct': round(confianza, 1),
            'probabilidades': {CLASES_CORTO[c]: round(float(p) * 100, 1) for c, p in zip(CLASES_ID, probs)},
            'diagnostico': details['diagnostico'],
            'accion_recomendada': details['accion'],
            'severidad': details['severidad'],
            'color': details['color'],
            'confianza_baja': confianza < UMBRAL_CONFIANZA,
        }


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    clf = PlantDiseaseClassifier()
    m = clf.train_model()
    print("Matriz de confusión (filas = real, columnas = predicho):", m['matriz_confusion'])
    for f in sorted(os.listdir(SAMPLE_DIR)):
        r = clf.predict_image(os.path.join(SAMPLE_DIR, f))
        print(f"{f:22} -> {r['clase_detectada']} ({r['confianza_pct']}%)")
