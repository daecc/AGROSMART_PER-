# -*- coding: utf-8 -*-
"""
API REST de AgroSmart Perú (FastAPI).

Ejecutar en desarrollo (desde la raíz del proyecto):
    uvicorn backend.main:app --reload --port 8000

Documentación interactiva: http://localhost:8000/docs
Si existe frontend/dist (npm run build), también sirve la interfaz React en http://localhost:8000/
"""

import io
import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from backend import schemas as S
from backend.services import BASE_DIR, DOCUMENTOS, ESCENARIOS, Modelos
from src.vision import SAMPLE_DIR

MAX_IMAGEN_BYTES = 10 * 1024 * 1024
FRONTEND_DIST = os.path.join(BASE_DIR, 'frontend', 'dist')

modelos: Modelos = None


@asynccontextmanager
async def lifespan(_app):
    global modelos
    modelos = Modelos()  # carga los 5 modelos una sola vez
    yield


app = FastAPI(
    title='AgroSmart Perú API',
    description='Sistema inteligente para mitigar el impacto de El Niño Costero en la agricultura de Piura.',
    version='1.0.0',
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get('AGROSMART_CORS', 'http://localhost:5173,http://127.0.0.1:5173').split(','),
    allow_methods=['GET', 'POST'],
    allow_headers=['*'],
)


# ------------------------------------------------------------------ general
@app.get('/api/health', tags=['general'])
def health():
    return {'estado': 'ok', 'modelos_cargados': modelos is not None}


@app.get('/api/panel', response_model=S.PanelResponse, tags=['panel'])
def panel(cultivo: S.Cultivo = Query('Limon')):
    return modelos.panel(cultivo)


# ------------------------------------------------------------------ módulo 1
@app.post('/api/rendimiento/prediccion', response_model=S.RendimientoResponse, tags=['1. Rendimiento'])
def predecir_rendimiento(req: S.RendimientoRequest):
    return modelos.predecir_rendimiento(req)


# ------------------------------------------------------------------ módulo 2
@app.get('/api/precios/escenarios', response_model=List[S.EscenarioInfo], tags=['2. Precios'])
def escenarios():
    return [{'id': k, **v} for k, v in ESCENARIOS.items()]


@app.post('/api/precios/pronostico', response_model=S.PronosticoResponse, tags=['2. Precios'])
def pronosticar(req: S.PronosticoRequest):
    return modelos.pronosticar_precios(req)


# ------------------------------------------------------------------ módulo 3
@app.get('/api/vision/info', response_model=S.VisionInfo, tags=['3. Visión'])
def vision_info():
    return modelos.info_vision()


@app.get('/api/vision/muestras/{nombre}', tags=['3. Visión'])
def vision_muestra(nombre: str):
    if nombre not in modelos.muestras_vision():  # evita acceder a rutas arbitrarias
        raise HTTPException(404, 'Muestra no encontrada')
    return FileResponse(os.path.join(SAMPLE_DIR, nombre))


@app.post('/api/vision/muestras/{nombre}/diagnostico', response_model=S.VisionResponse, tags=['3. Visión'])
def vision_diagnostico_muestra(nombre: str):
    if nombre not in modelos.muestras_vision():
        raise HTTPException(404, 'Muestra no encontrada')
    return modelos.predecir_imagen(Image.open(os.path.join(SAMPLE_DIR, nombre)))


@app.post('/api/vision/diagnostico', response_model=S.VisionResponse, tags=['3. Visión'])
async def vision_diagnostico(imagen: UploadFile = File(...)):
    datos = await imagen.read(MAX_IMAGEN_BYTES + 1)
    if len(datos) > MAX_IMAGEN_BYTES:
        raise HTTPException(413, 'La imagen supera los 10 MB')
    try:
        img = Image.open(io.BytesIO(datos))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(415, 'El archivo no es una imagen válida (use JPG o PNG)')
    return modelos.predecir_imagen(img)


# ------------------------------------------------------------------ módulo 4
@app.get('/api/clustering', response_model=S.ClusteringResponse, tags=['4. Clustering'])
def clustering():
    return modelos.clustering()


# ------------------------------------------------------------------ módulo 5
@app.get('/api/resumen/documentos', response_model=List[S.Documento], tags=['5. Resumen'])
def documentos():
    return modelos.documentos()


@app.get('/api/resumen/documentos/{doc_id}', response_model=S.DocumentoTexto, tags=['5. Resumen'])
def documento(doc_id: str):
    if doc_id not in DOCUMENTOS:
        raise HTTPException(404, 'Documento no encontrado')
    return modelos.documento(doc_id)


@app.post('/api/resumen', response_model=S.ResumenResponse, tags=['5. Resumen'])
def resumir(req: S.ResumenRequest):
    return modelos.resumir(req)


# ------------------------------------------------------------------ frontend compilado
if os.path.isdir(FRONTEND_DIST):
    app.mount('/assets', StaticFiles(directory=os.path.join(FRONTEND_DIST, 'assets')), name='assets')

    @app.get('/{ruta:path}', include_in_schema=False)
    def spa(ruta: str):
        archivo = os.path.join(FRONTEND_DIST, ruta)
        if ruta and os.path.isfile(archivo) and os.path.abspath(archivo).startswith(FRONTEND_DIST):
            return FileResponse(archivo)
        return FileResponse(os.path.join(FRONTEND_DIST, 'index.html'))
