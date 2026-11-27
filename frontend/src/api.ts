// Cliente tipado de la API FastAPI (backend/main.py)

export type Cultivo = 'Limon' | 'Arroz'
export type Escenario = 'Extraordinario' | 'Fuerte' | 'Moderado' | 'Normal'
export type Nivel = 'ok' | 'alerta' | 'critico'

export interface Kpi { etiqueta: string; valor: string; detalle: string; tono: Nivel | 'neutro' }
export interface PuntoHistorico {
  fecha: string; rendimiento_tha: number; icen: number; precio_chacra_soles_kg: number; lluvia_mensual_mm: number
}
export interface Panel { cultivo: Cultivo; ultimo_mes: string; evento_nino: string; kpis: Kpi[]; serie: PuntoHistorico[] }

export interface RendimientoReq {
  cultivo: Cultivo; mes: number; icen: number; anomalia_tsm: number; lluvia_mm: number; temp_max: number
}
export interface RendimientoRes {
  cultivo: Cultivo; rendimiento_estimado_tha: number; rendimiento_historico_base: number
  perdida_estimada_pct: number; alerta_riesgo: string; nivel: Nivel; importancia_variables: Record<string, number>
}

export interface EscenarioInfo { id: Escenario; nombre: string; icen: number }
export interface PuntoPrecio { fecha: string; precio_soles_kg: number }
export interface PronosticoRes {
  cultivo: Cultivo; escenario: EscenarioInfo; historico: PuntoPrecio[]; pronostico: PuntoPrecio[]
  metricas: { mape_pct: number; rmse_soles: number } | null
}

export interface VisionInfo {
  modelo: string; n_train: number; n_test: number; accuracy: number; f1_macro: number
  clases: string[]; matriz_confusion: number[][]; muestras: string[]; alcance: string
}
export interface VisionRes {
  clase_detectada: string; confianza_pct: number; confianza_baja: boolean; probabilidades: Record<string, number>
  diagnostico: string; accion_recomendada: string; severidad: string; nivel: Nivel
}

export interface Provincia {
  ubigeo: number; provincia: string; valle_principal: string; cultivo_predominante: string
  lluvia_max_historica_mm: number; pct_area_inundable: number; perdida_cosecha_nino_pct: number
  deficit_drenaje_pct: number; cluster_id: number; nivel_vulnerabilidad: string; color_alerta: string
  pca_x: number; pca_y: number
}
export interface Grupo { nivel: string; color: string; provincias: string[]; accion: string }
export interface ClusteringRes { silueta: number | null; variables: string[]; provincias: Provincia[]; grupos: Grupo[] }

export interface Documento { id: string; titulo: string }
export interface ResumenRes {
  resumen: string[]; total_oraciones_original: number; oraciones_resumidas: number
  ratio_compresion_pct: number; palabras_clave: string[]
}

export class ApiError extends Error {}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`/api${path}`, init)
  } catch {
    throw new ApiError('No se pudo conectar con el servidor. ¿Está corriendo el backend (uvicorn backend.main:app)?')
  }
  if (!res.ok) {
    let detalle = `Error ${res.status}`
    try {
      const body = await res.json()
      if (typeof body.detail === 'string') detalle = body.detail
      else if (Array.isArray(body.detail)) detalle = body.detail.map((d: { msg: string }) => d.msg).join('; ')
    } catch { /* respuesta sin JSON */ }
    throw new ApiError(detalle)
  }
  return res.json() as Promise<T>
}

const postJson = <T,>(path: string, body: unknown) =>
  request<T>(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })

export const api = {
  panel: (cultivo: Cultivo) => request<Panel>(`/panel?cultivo=${cultivo}`),
  predecirRendimiento: (req: RendimientoReq) => postJson<RendimientoRes>('/rendimiento/prediccion', req),
  escenarios: () => request<EscenarioInfo[]>('/precios/escenarios'),
  pronosticar: (cultivo: Cultivo, meses: number, escenario: Escenario) =>
    postJson<PronosticoRes>('/precios/pronostico', { cultivo, meses, escenario }),
  visionInfo: () => request<VisionInfo>('/vision/info'),
  urlMuestra: (nombre: string) => `/api/vision/muestras/${encodeURIComponent(nombre)}`,
  diagnosticarMuestra: (nombre: string) =>
    request<VisionRes>(`/vision/muestras/${encodeURIComponent(nombre)}/diagnostico`, { method: 'POST' }),
  diagnosticarImagen: (archivo: File) => {
    const form = new FormData()
    form.append('imagen', archivo)
    return request<VisionRes>('/vision/diagnostico', { method: 'POST', body: form })
  },
  clustering: () => request<ClusteringRes>('/clustering'),
  documentos: () => request<Documento[]>('/resumen/documentos'),
  documento: (id: string) => request<Documento & { texto: string }>(`/resumen/documentos/${id}`),
  resumir: (texto: string, num_oraciones: number) => postJson<ResumenRes>('/resumen', { texto, num_oraciones }),
}
