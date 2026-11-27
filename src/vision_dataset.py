# -*- coding: utf-8 -*-
"""
Preparación del dataset de imágenes REALES de hojas de arroz para el Módulo 3.

Fuente: Hugging Face `girish787/riceLeafDataset` (fotografías de campo 300x300 px,
recopiladas de los datasets públicos de Kaggle tipo "Paddy Doctor" / "Rice Leaf Diseases").
Se usa la partición `test` de ese repositorio (91 MB) como corpus completo y, a partir
de ella, se generan nuestras propias particiones de entrenamiento y prueba.

Clases utilizadas (las 3 del módulo):
    healthy                -> hoja_sana
    bacterial_leaf_blight  -> tizon_bacteriano   (Xanthomonas oryzae pv. oryzae)
    Brown Spot             -> mancha_marron      (Bipolaris oryzae)

Limpieza aplicada:
  1. Duplicados exactos (mismo dHash): se conserva una sola copia; si las copias tienen
     etiquetas distintas, se descartan todas (etiqueta ambigua).
  2. Casi-duplicados (distancia de Hamming <= 6 en dHash de 64 bits): se agrupan con
     union-find y el grupo se guarda en metadata.csv. El entrenamiento usa estos grupos
     para que un casi-duplicado nunca quede a la vez en entrenamiento y en prueba
     (evita fuga de información y métricas infladas).

Uso:
    python -m src.vision_dataset
Salida:
    data/raw/hojas_arroz/<clase>/<id>.jpg
    data/raw/hojas_arroz/metadata.csv   (archivo, clase, grupo)
"""

import io
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
OUT_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'hojas_arroz')
PARQUET_URL = ('https://huggingface.co/datasets/girish787/riceLeafDataset/resolve/main/'
               'data/test-00000-of-00001.parquet')
PARQUET_PATH = os.path.join(OUT_DIR, '_fuente_test.parquet')

# id de etiqueta en el dataset de origen -> nombre de clase en AgroSmart
LABEL_MAP = {13: 'hoja_sana', 8: 'tizon_bacteriano', 0: 'mancha_marron'}
HAMMING_CASI_DUPLICADO = 6


def dhash(img, n=8):
    """Hash perceptual por diferencias (64 bits) para detectar imágenes repetidas."""
    g = np.asarray(img.convert('L').resize((n + 1, n), Image.BILINEAR), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


def descargar_parquet():
    if os.path.exists(PARQUET_PATH):
        return PARQUET_PATH
    import requests
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Descargando dataset desde {PARQUET_URL} ...")
    tmp = PARQUET_PATH + '.part'
    with requests.get(PARQUET_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    os.replace(tmp, PARQUET_PATH)
    return PARQUET_PATH


def _agrupar(hashes):
    """Union-find sobre pares con distancia de Hamming <= umbral."""
    dist = (hashes[:, None, :] != hashes[None, :, :]).sum(-1)
    parent = list(range(len(hashes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    ii, jj = np.where(np.triu(dist <= HAMMING_CASI_DUPLICADO, 1))
    for i, j in zip(ii, jj):
        parent[find(i)] = find(j)
    raices = [find(i) for i in range(len(hashes))]
    # renumerar grupos 0..G-1
    mapa = {r: k for k, r in enumerate(dict.fromkeys(raices))}
    return np.array([mapa[r] for r in raices])


def preparar_dataset():
    import pyarrow.parquet as pq

    df = pq.read_table(descargar_parquet()).to_pandas()
    df = df[df['label'].isin(LABEL_MAP)].reset_index(drop=True)

    imagenes, clases, hashes = [], [], []
    for row in df.itertuples():
        img = Image.open(io.BytesIO(row.image['bytes'])).convert('RGB')
        imagenes.append(img)
        clases.append(LABEL_MAP[row.label])
        hashes.append(dhash(img))
    hashes = np.array(hashes)

    # 1) Duplicados exactos
    claves = [h.tobytes() for h in hashes]
    por_clave = {}
    for i, k in enumerate(claves):
        por_clave.setdefault(k, []).append(i)
    conservar, ambiguos, repetidos = [], 0, 0
    for idxs in por_clave.values():
        if len({clases[i] for i in idxs}) > 1:
            ambiguos += len(idxs)
            continue
        conservar.append(idxs[0])
        repetidos += len(idxs) - 1
    conservar.sort()

    # 2) Grupos de casi-duplicados
    grupos = _agrupar(hashes[conservar])

    os.makedirs(OUT_DIR, exist_ok=True)
    filas = []
    for n, (i, g) in enumerate(zip(conservar, grupos)):
        clase = clases[i]
        rel = os.path.join(clase, f'{clase}_{n:04d}.jpg')
        os.makedirs(os.path.join(OUT_DIR, clase), exist_ok=True)
        imagenes[i].save(os.path.join(OUT_DIR, rel), quality=92)
        filas.append({'archivo': rel.replace('\\', '/'), 'clase': clase, 'grupo': int(g)})

    meta = pd.DataFrame(filas)
    meta.to_csv(os.path.join(OUT_DIR, 'metadata.csv'), index=False, encoding='utf-8')

    print(f"Imágenes de origen (3 clases): {len(df)}")
    print(f"Duplicados exactos eliminados: {repetidos} | descartadas por etiqueta ambigua: {ambiguos}")
    print(f"Imágenes finales: {len(meta)} | grupos de casi-duplicados: {meta['grupo'].nunique()}")
    print(meta['clase'].value_counts().to_string())
    return meta


def cargar_metadata(preparar_si_falta=True):
    path = os.path.join(OUT_DIR, 'metadata.csv')
    if not os.path.exists(path):
        if not preparar_si_falta:
            raise FileNotFoundError(path)
        preparar_dataset()
    return pd.read_csv(path)


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    preparar_dataset()
