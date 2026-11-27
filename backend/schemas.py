# -*- coding: utf-8 -*-
"""Esquemas Pydantic de entrada/salida de la API de AgroSmart Perú."""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

Cultivo = Literal['Limon', 'Arroz']
Escenario = Literal['Extraordinario', 'Fuerte', 'Moderado', 'Normal']


# ---------- Panel general ----------
class Kpi(BaseModel):
    etiqueta: str
    valor: str
    detalle: str
    tono: Literal['ok', 'alerta', 'critico', 'neutro']


class PuntoHistorico(BaseModel):
    fecha: str
    rendimiento_tha: float
    icen: float
    precio_chacra_soles_kg: float
    lluvia_mensual_mm: float


class PanelResponse(BaseModel):
    cultivo: Cultivo
    ultimo_mes: str
    evento_nino: str
    kpis: List[Kpi]
    serie: List[PuntoHistorico]


# ---------- Módulo 1: rendimiento ----------
class RendimientoRequest(BaseModel):
    cultivo: Cultivo = 'Limon'
    mes: int = Field(3, ge=1, le=12)
    icen: float = Field(2.2, ge=-2.0, le=5.0)
    anomalia_tsm: float = Field(2.5, ge=-3.0, le=6.0)
    lluvia_mm: float = Field(250.0, ge=0.0, le=1500.0)
    temp_max: float = Field(34.5, ge=20.0, le=45.0)


class RendimientoResponse(BaseModel):
    cultivo: Cultivo
    rendimiento_estimado_tha: float
    rendimiento_historico_base: float
    perdida_estimada_pct: float
    alerta_riesgo: str
    nivel: Literal['ok', 'alerta', 'critico']
    importancia_variables: Dict[str, float]


# ---------- Módulo 2: precios ----------
class PronosticoRequest(BaseModel):
    cultivo: Cultivo = 'Limon'
    meses: int = Field(6, ge=1, le=24)
    escenario: Escenario = 'Fuerte'


class PuntoPrecio(BaseModel):
    fecha: str
    precio_soles_kg: float


class EscenarioInfo(BaseModel):
    id: Escenario
    nombre: str
    icen: float


class PronosticoResponse(BaseModel):
    cultivo: Cultivo
    escenario: EscenarioInfo
    historico: List[PuntoPrecio]
    pronostico: List[PuntoPrecio]
    metricas: Optional[Dict[str, float]] = None


# ---------- Módulo 3: visión ----------
class VisionResponse(BaseModel):
    clase_detectada: str
    confianza_pct: float
    confianza_baja: bool
    probabilidades: Dict[str, float]
    diagnostico: str
    accion_recomendada: str
    severidad: str
    nivel: Literal['ok', 'alerta', 'critico']


class VisionInfo(BaseModel):
    modelo: str
    n_train: int
    n_test: int
    accuracy: float
    f1_macro: float
    clases: List[str]
    matriz_confusion: List[List[int]]
    muestras: List[str]
    alcance: str


# ---------- Módulo 4: clustering ----------
class Provincia(BaseModel):
    ubigeo: int
    provincia: str
    valle_principal: str
    cultivo_predominante: str
    lluvia_max_historica_mm: float
    pct_area_inundable: float
    perdida_cosecha_nino_pct: float
    deficit_drenaje_pct: float
    cluster_id: int
    nivel_vulnerabilidad: str
    color_alerta: str
    pca_x: float
    pca_y: float


class GrupoRiesgo(BaseModel):
    nivel: str
    color: str
    provincias: List[str]
    accion: str


class ClusteringResponse(BaseModel):
    silueta: Optional[float]
    variables: List[str]
    provincias: List[Provincia]
    grupos: List[GrupoRiesgo]


# ---------- Módulo 5: resumen ----------
class Documento(BaseModel):
    id: str
    titulo: str


class DocumentoTexto(Documento):
    texto: str


class ResumenRequest(BaseModel):
    texto: str = Field(..., min_length=50, max_length=200_000)
    num_oraciones: int = Field(4, ge=1, le=10)


class ResumenResponse(BaseModel):
    resumen: List[str]
    total_oraciones_original: int
    oraciones_resumidas: int
    ratio_compresion_pct: float
    palabras_clave: List[str]
