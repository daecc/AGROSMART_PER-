import { useEffect, useState } from 'react'
import { FileText, Sparkles } from 'lucide-react'
import { api } from '../api'
import { Alert, Card, Empty, PageHead, Slider, fmt, useAction, useApi } from '../components/ui'

export default function ResumenPage() {
  const docs = useApi(api.documentos, [])
  const [docId, setDocId] = useState<string>('')
  const [texto, setTexto] = useState('')
  const [num, setNum] = useState(4)
  const resumen = useAction(api.resumir)
  const res = resumen.data

  useEffect(() => {
    if (docs.data?.length && !docId) setDocId(docs.data[0].id)
  }, [docs.data, docId])

  useEffect(() => {
    if (!docId) return
    let vivo = true
    api.documento(docId).then((d) => { if (vivo) { setTexto(d.texto); resumen.setData(null) } }).catch(() => {})
    return () => { vivo = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [docId])

  return (
    <>
      <PageHead eyebrow="Módulo 5 · Procesamiento de lenguaje natural" title="Resumen de comunicados ENFEN / SENAMHI">
        TextRank sobre la matriz de similitud coseno de vectores TF-IDF: extrae las oraciones más representativas de un boletín técnico.
      </PageHead>

      {docs.error && <Alert tone="critico">{docs.error}</Alert>}

      <div className="grid grid-2">
        <Card title="Documento" icon={<FileText size={18} />}>
          <div className="field">
            <label>Documento oficial precargado</label>
            <select value={docId} onChange={(e) => setDocId(e.target.value)}>
              {(docs.data ?? []).map((d) => <option key={d.id} value={d.id}>{d.titulo}</option>)}
            </select>
          </div>
          <div className="field">
            <label>Texto del boletín <span className="muted" style={{ fontWeight: 400 }}>{texto.length.toLocaleString('es-PE')} caracteres</span></label>
            <textarea value={texto} onChange={(e) => setTexto(e.target.value)} rows={14} />
            <div className="hint">Puedes pegar aquí cualquier comunicado extenso.</div>
          </div>
          <Slider label="Número de conclusiones clave" value={num} min={2} max={8} step={1} onChange={setNum} />
          <button className="btn" disabled={resumen.loading || texto.trim().length < 50} onClick={() => resumen.run(texto, num)}>
            {resumen.loading ? <div className="spinner" /> : <Sparkles size={16} />} Generar ficha ejecutiva
          </button>
        </Card>

        <div className="stack">
          {resumen.error && <Alert tone="critico">{resumen.error}</Alert>}
          {!res && !resumen.error && (
            <Card><Empty icon={<Sparkles size={40} />}>Presiona <strong>Generar ficha ejecutiva</strong> para extraer las conclusiones del texto.</Empty></Card>
          )}
          {res && (
            <>
              <div className="grid grid-3">
                <div className="card kpi ok"><div className="label">Compresión</div><div className="value">{fmt(res.ratio_compresion_pct, 1)} %</div><div className="detail">del texto original</div></div>
                <div className="card kpi neutro"><div className="label">Oraciones evaluadas</div><div className="value">{res.total_oraciones_original}</div><div className="detail">en el documento</div></div>
                <div className="card kpi neutro"><div className="label">Seleccionadas</div><div className="value">{res.oraciones_resumidas}</div><div className="detail">conclusiones clave</div></div>
              </div>
              <Card title="Ficha ejecutiva para agricultores y técnicos">
                <ol className="summary-list">
                  {res.resumen.map((s, i) => <li key={i}><b>{i + 1}</b><span>{s}</span></li>)}
                </ol>
              </Card>
              <Card title="Palabras clave detectadas">
                <div className="chips">{res.palabras_clave.map((p) => <span key={p} className="chip">{p}</span>)}</div>
              </Card>
            </>
          )}
        </div>
      </div>
    </>
  )
}
