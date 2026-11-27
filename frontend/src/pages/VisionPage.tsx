import { useEffect, useRef, useState } from 'react'
import { Camera, Leaf, Upload } from 'lucide-react'
import { api } from '../api'
import { Alert, Card, Empty, Loading, PageHead, COLORES, fmt, useAction, useApi } from '../components/ui'

const COLOR_CLASE: Record<string, string> = {
  'Hoja Sana': COLORES.ok, 'Tizón Bacteriano': COLORES.critico, 'Mancha Marrón': COLORES.alerta,
}
const etiqueta = (archivo: string) => archivo.replace(/\.\w+$/, '').replace(/_(\d+)$/, ' $1').replace(/_/g, ' ')

export default function VisionPage() {
  const info = useApi(api.visionInfo, [])
  const [modo, setModo] = useState<'muestra' | 'propia'>('muestra')
  const [muestra, setMuestra] = useState<string | null>(null)
  const [archivo, setArchivo] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [drag, setDrag] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const diag = useAction((m: 'muestra' | 'propia', valor: string | File) =>
    m === 'muestra' ? api.diagnosticarMuestra(valor as string) : api.diagnosticarImagen(valor as File))
  const res = diag.data

  // seleccionar la primera muestra al cargar
  useEffect(() => {
    if (info.data && !muestra && info.data.muestras.length) elegirMuestra(info.data.muestras[0])
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [info.data])

  useEffect(() => () => { if (preview?.startsWith('blob:')) URL.revokeObjectURL(preview) }, [preview])

  function elegirMuestra(nombre: string) {
    setMuestra(nombre)
    setPreview(api.urlMuestra(nombre))
    diag.run('muestra', nombre)
  }

  function elegirArchivo(f: File | undefined) {
    if (!f) return
    setArchivo(f)
    setPreview(URL.createObjectURL(f))
    diag.run('propia', f)
  }

  return (
    <>
      <PageHead eyebrow="Módulo 3 · Visión por computadora" title="Diagnóstico fitosanitario de hojas de arroz">
        Clasifica la foto en hoja sana, tizón bacteriano (<em>Xanthomonas oryzae</em>) o mancha marrón (<em>Bipolaris oryzae</em>)
        y entrega la recomendación de manejo.
      </PageHead>

      {info.error && <Alert tone="critico">{info.error}</Alert>}
      {info.loading && <Loading />}

      {info.data && (
        <div className="stack">
          <Alert tone="info">
            <strong>Modelo:</strong> {info.data.modelo}, entrenado con {info.data.n_train} fotos reales.
            Evaluado en {info.data.n_test} fotos no vistas: <strong>accuracy {fmt(info.data.accuracy * 100, 1)} %</strong>,
            F1 macro {fmt(info.data.f1_macro * 100, 1)} %.
          </Alert>
          <Alert tone="alerta"><strong>Alcance:</strong> {info.data.alcance}</Alert>

          <div className="grid grid-side">
            <Card title="Imagen a diagnosticar" icon={<Camera size={18} />}>
              <div className="segmented" style={{ marginBottom: 14 }}>
                <button className={modo === 'muestra' ? 'on' : ''} onClick={() => { setModo('muestra'); if (muestra) elegirMuestra(muestra) }}>Fotos de muestra</button>
                <button className={modo === 'propia' ? 'on' : ''} onClick={() => { setModo('propia'); setPreview(archivo ? URL.createObjectURL(archivo) : null); diag.setData(null) }}>Subir mi foto</button>
              </div>

              {modo === 'muestra' ? (
                <>
                  <p className="hint muted" style={{ marginTop: 0 }}>Fotos reales del conjunto de prueba (el modelo no las vio al entrenar).</p>
                  <div className="thumbs">
                    {info.data.muestras.map((m) => (
                      <button key={m} className={`thumb ${m === muestra ? 'on' : ''}`} onClick={() => elegirMuestra(m)} title={etiqueta(m)}>
                        <img src={api.urlMuestra(m)} alt={etiqueta(m)} />
                      </button>
                    ))}
                  </div>
                </>
              ) : (
                <label
                  className={`dropzone ${drag ? 'drag' : ''}`}
                  onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
                  onDragLeave={() => setDrag(false)}
                  onDrop={(e) => { e.preventDefault(); setDrag(false); elegirArchivo(e.dataTransfer.files[0]) }}
                >
                  <Upload size={26} />
                  <div><strong>Arrastra una foto</strong> o haz clic para elegirla</div>
                  <div className="hint">JPG o PNG, máximo 10 MB</div>
                  <input ref={inputRef} type="file" accept="image/jpeg,image/png" hidden onChange={(e) => elegirArchivo(e.target.files?.[0])} />
                </label>
              )}

              {preview && <img className="preview" src={preview} alt="Imagen seleccionada" style={{ marginTop: 14 }} />}
            </Card>

            <div className="stack">
              {diag.error && <Alert tone="critico">{diag.error}</Alert>}
              {diag.loading && <Card><Loading text="Analizando la imagen…" /></Card>}
              {!diag.loading && !res && !diag.error && (
                <Card><Empty icon={<Leaf size={40} />}>Elige una foto de muestra o sube la tuya.</Empty></Card>
              )}
              {!diag.loading && res && (
                <>
                  <Card>
                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'flex-start' }}>
                      <div>
                        <div className="muted" style={{ fontSize: 13 }}>Diagnóstico</div>
                        <div style={{ fontSize: 24, fontWeight: 800, color: COLORES[res.nivel] }}>{res.clase_detectada}</div>
                        <span className={`badge ${res.nivel}`}>Severidad: {res.severidad}</span>
                      </div>
                      <div style={{ textAlign: 'right' }}>
                        <div className="muted" style={{ fontSize: 13 }}>Confianza</div>
                        <div className="big-number" style={{ fontSize: 34 }}>{fmt(res.confianza_pct, 1)}<small> %</small></div>
                      </div>
                    </div>
                    {res.confianza_baja && (
                      <div style={{ marginTop: 12 }}><Alert tone="alerta">Confianza baja: confirme el diagnóstico con un técnico de SENASA.</Alert></div>
                    )}
                    <p style={{ marginBottom: 6 }}><strong>Descripción:</strong> {res.diagnostico}</p>
                    <Alert tone={res.nivel}><strong>Acción recomendada (SENASA):</strong> {res.accion_recomendada}</Alert>
                  </Card>

                  <Card title="Probabilidad por clase">
                    {Object.entries(res.probabilidades).map(([clase, p]) => (
                      <div className="prob-row" key={clase}>
                        <span>{clase}</span>
                        <div className="prob-bar"><div style={{ width: `${p}%`, background: COLOR_CLASE[clase] ?? COLORES.gris }} /></div>
                        <span style={{ textAlign: 'right', fontWeight: 600 }}>{fmt(p, 1)} %</span>
                      </div>
                    ))}
                  </Card>
                </>
              )}

              <Card title="Matriz de confusión (conjunto de prueba)" sub="Filas: clase real · Columnas: clase predicha.">
                <MatrizConfusion clases={info.data.clases} matriz={info.data.matriz_confusion} />
              </Card>
            </div>
          </div>
        </div>
      )}
    </>
  )
}

function MatrizConfusion({ clases, matriz }: { clases: string[]; matriz: number[][] }) {
  const max = Math.max(1, ...matriz.flat())
  return (
    <div className="cm" style={{ gridTemplateColumns: `130px repeat(${clases.length}, 1fr)` }}>
      <div />
      {clases.map((c) => <div key={c} className="muted" style={{ textAlign: 'center', fontWeight: 600 }}>{c}</div>)}
      {matriz.map((fila, i) => (
        <FilaMatriz key={clases[i]} clase={clases[i]} fila={fila} i={i} max={max} />
      ))}
    </div>
  )
}

function FilaMatriz({ clase, fila, i, max }: { clase: string; fila: number[]; i: number; max: number }) {
  return (
    <>
      <div style={{ fontWeight: 600, alignSelf: 'center' }}>{clase}</div>
      {fila.map((v, j) => {
        const a = v / max
        const bg = i === j ? `rgba(46,125,50,${0.12 + a * 0.75})` : v ? `rgba(198,40,40,${0.1 + a * 2})` : '#f4f6f3'
        return <div key={j} className="cm-cell" style={{ background: bg, color: i === j && a > 0.5 ? '#fff' : 'inherit' }}>{v}</div>
      })}
    </>
  )
}
