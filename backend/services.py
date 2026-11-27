# -*- coding: utf-8 -*-
"""
Capa de servicios: instancia UNA vez los 5 modelos de src/ (cargados desde models/)
y expone funciones puras que la API convierte en JSON.
"""

import os
import threading

import pandas as pd

from src.prediction import CropYieldPredictor
from src.timeseries import AgroTimeSeriesForecaster
from src.vision import PlantDiseaseClassifier, SAMPLE_DIR
from src.clustering import VulnerabilityClusterer
from src.summarizer import EnfenReportSummarizer

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TABLA_MAESTRA = os.path.join(DATA_DIR, 'tabla_maestra_agro_nino.csv')
COMUNICADOS_DIR = os.path.join(DATA_DIR, 'comunicados_enfen')

ESCENARIOS = {
    'Extraordinario': {'nombre': 'Extraordinario (ICEN +2.6 °C)', 'icen': 2.6},
    'Fuerte': {'nombre': 'Fuerte (ICEN +2.1 °C)', 'icen': 2.1},
    'Moderado': {'nombre': 'Moderado (ICEN +1.4 °C)', 'icen': 1.4},
    'Normal': {'nombre': 'Normal / Neutro (ICEN +0.2 °C)', 'icen': 0.2},
}

DOCUMENTOS = {
    'enfen-04-2026': ('Comunicado Oficial ENFEN N° 04-2026 (Alerta El Niño Extraordinario)',
                      'comunicado_enfen_n04_2026.txt'),
    'senamhi-182-2026': ('Aviso Meteorológico SENAMHI N° 182-2026 (Lluvias Extremas Costa Norte)',
                         'aviso_senamhi_lluvias_norte.txt'),
}

ACCIONES_GRUPO = {
    'Vulnerabilidad Extrema': 'Prioridad inmediata para traslado de motobombas pesadas de 12 pulgadas y descolmatación de drenes troncales.',
    'Vulnerabilidad Alta': 'Monitoreo continuo de deslizamientos en quebradas y vigilancia de las zonas altas de la cuenca.',
    'Vulnerabilidad Moderada': 'Menor anegamiento de tierras de cultivo; habilitación como centros de acopio temporal.',
}

COLOR_A_NIVEL = {'green': 'ok', 'orange': 'alerta', 'red': 'critico'}
RENDIMIENTO_BASE = {'Limon': 12.5, 'Arroz': 8.8}


class Modelos:
    """Contenedor de los modelos cargados en memoria (se crea al iniciar la API)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.predictor = CropYieldPredictor()
        self.predictor.load_models()
        self.forecaster = AgroTimeSeriesForecaster()
        for c in ('Limon', 'Arroz'):
            self.forecaster.load_model(c)
        self.vision = PlantDiseaseClassifier()
        self.vision.load_model()
        self.clusterer = VulnerabilityClusterer()
        self.silueta = self.clusterer.load_model().get('silhouette')
        self.summarizer = EnfenReportSummarizer()
        self.tabla = pd.read_csv(TABLA_MAESTRA)

    # ---------------- Panel ----------------
    def panel(self, cultivo):
        df = self.tabla[self.tabla['cultivo'] == cultivo].sort_values('fecha')
        ult = df.iloc[-1]
        normales = df[df['evento_nino'] == 'Normal']
        lluvia_clim = normales[normales['mes'] == ult['mes']]['lluvia_mensual_mm'].mean()
        precio_normal = normales['precio_chacra_soles_kg'].mean()
        base = RENDIMIENTO_BASE[cultivo]
        perdida = max(0.0, (base - ult['rendimiento_tha']) / base * 100)

        def tono(valor, alerta, critico):
            return 'critico' if valor >= critico else 'alerta' if valor >= alerta else 'ok'

        kpis = [
            {'etiqueta': 'Índice ICEN (último mes)', 'valor': f"{ult['icen']:+.2f} °C",
             'detalle': f"Evento: {ult['evento_nino']}", 'tono': tono(ult['icen'], 1.0, 2.0)},
            {'etiqueta': 'Lluvia mensual en Piura', 'valor': f"{ult['lluvia_mensual_mm']:.0f} mm",
             'detalle': (f"Climatología del mes: {lluvia_clim:.0f} mm" if pd.notna(lluvia_clim) else 'Sin climatología'),
             'tono': tono(ult['lluvia_mensual_mm'], 100, 250)},
            {'etiqueta': f'Pérdida de rendimiento ({cultivo})', 'valor': f"-{perdida:.1f} %",
             'detalle': f"{ult['rendimiento_tha']:.2f} t/ha vs {base} t/ha normal", 'tono': tono(perdida, 15, 35)},
            {'etiqueta': f'Precio en chacra ({cultivo})', 'valor': f"S/ {ult['precio_chacra_soles_kg']:.2f}/kg",
             'detalle': f"{(ult['precio_chacra_soles_kg'] / precio_normal - 1) * 100:+.0f} % vs años normales",
             'tono': tono(ult['precio_chacra_soles_kg'] / precio_normal, 1.3, 2.0)},
        ]
        serie = df[['fecha', 'rendimiento_tha', 'icen', 'precio_chacra_soles_kg', 'lluvia_mensual_mm']].copy()
        serie['fecha'] = serie['fecha'].str[:7]
        return {
            'cultivo': cultivo,
            'ultimo_mes': str(ult['fecha'])[:7],
            'evento_nino': str(ult['evento_nino']),
            'kpis': kpis,
            'serie': serie.to_dict(orient='records'),
        }

    # ---------------- Módulo 1 ----------------
    def predecir_rendimiento(self, req):
        r = self.predictor.predict(req.cultivo, req.icen, req.anomalia_tsm, req.lluvia_mm, req.temp_max, req.mes)
        r['nivel'] = COLOR_A_NIVEL[r.pop('color')]
        r['rendimiento_estimado_tha'] = float(r['rendimiento_estimado_tha'])
        r['importancia_variables'] = {k: float(v) for k, v in r['importancia_variables'].items()}
        return r

    # ---------------- Módulo 2 ----------------
    def pronosticar_precios(self, req):
        esc = ESCENARIOS[req.escenario]
        with self._lock:  # forecast_future puede reentrenar si el .pkl es inválido
            hist, fut = self.forecaster.forecast_future(req.cultivo, req.meses, esc['nombre'])
        pack = self.forecaster.models.get(req.cultivo, {})
        return {
            'cultivo': req.cultivo,
            'escenario': {'id': req.escenario, **esc},
            'historico': [{'fecha': f, 'precio_soles_kg': float(p)}
                          for f, p in zip(hist['fecha'], hist['precio_chacra_soles_kg'])],
            'pronostico': [{'fecha': f, 'precio_soles_kg': float(p)}
                           for f, p in zip(fut['fecha'], fut['precio_pronosticado_soles_kg'])],
            'metricas': pack.get('metrics'),
        }

    # ---------------- Módulo 3 ----------------
    def muestras_vision(self):
        return sorted(f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png')))

    def info_vision(self):
        m = self.vision.metrics or {}
        return {
            'modelo': 'HistGradientBoostingClassifier sobre 123 descriptores de color y textura',
            'n_train': m.get('n_train', 0), 'n_test': m.get('n_test', 0),
            'accuracy': m.get('accuracy', 0.0), 'f1_macro': m.get('f1_macro', 0.0),
            'clases': ['Hoja Sana', 'Tizón Bacteriano', 'Mancha Marrón'],
            'matriz_confusion': m.get('matriz_confusion', []),
            'muestras': self.muestras_vision(),
            'alcance': ('Entrenado solo con fotografías de hojas de arroz (en campo o sobre fondo blanco). '
                        'No es válido para limón ni para dibujos. El tizón bacteriano solo aparece en fotos de campo, '
                        'así que una hoja con tizón fotografiada sobre fondo blanco puede clasificarse mal.'),
        }

    def predecir_imagen(self, image):
        r = self.vision.predict_image(image)
        r['nivel'] = COLOR_A_NIVEL[r.pop('color')]
        r['confianza_baja'] = bool(r['confianza_baja'])
        return r

    # ---------------- Módulo 4 ----------------
    def clustering(self):
        df = self.clusterer.get_clustered_data()
        cols = ['ubigeo', 'provincia', 'valle_principal', 'cultivo_predominante', 'lluvia_max_historica_mm',
                'pct_area_inundable', 'perdida_cosecha_nino_pct', 'deficit_drenaje_pct', 'cluster_id',
                'nivel_vulnerabilidad', 'color_alerta', 'pca_x', 'pca_y']
        provincias = df[cols].to_dict(orient='records')
        grupos = []
        for prefijo, accion in ACCIONES_GRUPO.items():
            sub = df[df['nivel_vulnerabilidad'].str.startswith(prefijo)]
            if len(sub):
                grupos.append({'nivel': sub['nivel_vulnerabilidad'].iloc[0], 'color': sub['color_alerta'].iloc[0],
                               'provincias': sub['provincia'].tolist(), 'accion': accion})
        return {'silueta': self.silueta,'variables': list(self.clusterer.features), 'provincias': provincias, 'grupos': grupos}

    # ---------------- Módulo 5 ----------------
    def documentos(self):
        return [{'id': k, 'titulo': v[0]} for k, v in DOCUMENTOS.items()]

    def documento(self, doc_id):
        titulo, archivo = DOCUMENTOS[doc_id]
        with open(os.path.join(COMUNICADOS_DIR, archivo), encoding='utf-8') as f:
            return {'id': doc_id, 'titulo': titulo, 'texto': f.read()}

    def resumir(self, req):
        r = self.summarizer.summarize(req.texto, num_sentences=req.num_oraciones)
        r['ratio_compresion_pct'] = float(r['ratio_compresion_pct'])
        r['palabras_clave'] = [str(p) for p in r['palabras_clave']]
        return r
