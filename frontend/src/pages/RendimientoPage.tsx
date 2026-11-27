import { useState } from 'react'
import { Gauge, Play, Sprout } from 'lucide-react'
import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api, type RendimientoReq } from '../api'
import { Alert, Card, CultivoToggle, Empty, PageHead, Slider, COLORES, fmt, useAction } from '../components/ui'

const NOMBRES_VAR: Record<string, string> = {
  icen: 'ICEN', anomalia_tsm: 'Anomalía TSM', lluvia_mensual_mm: 'Lluvia mensual', temp_max_prom: 'Temp. máxima', mes: 'Mes',
}
const MESES = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']

export default function RendimientoPage() {
  const [req, setReq] = useState<RendimientoReq>({
    cultivo: 'Limon', mes: 3, icen: 2.2, anomalia_tsm: 2.5, lluvia_mm: 250, temp_max: 34.5,
  })
  const set = <K extends keyof RendimientoReq>(k: K, v: RendimientoReq[K]) => setReq((r) => ({ ...r, [k]: v }))
  const pred = useAction(api.predecirRendimiento)
  const res = pred.data

  const importancia = res
    ? Object.entries(res.importancia_variables)
        .map(([k, v]) => ({ variable: NOMBRES_VAR[k] ?? k, peso: v }))
        .sort((a, b) => b.peso - a.peso)
    : []

  return (
    <>
      <PageHead eyebrow="Módulo 1 · Aprendizaje supervisado" title="Predicción de rendimiento agrícola">
        Random Forest Regressor que estima el rendimiento (t/ha) a partir de las condiciones climáticas de El Niño.
      </PageHead>

      <div className="grid grid-side">
        <Card title="Condiciones climáticas" icon={<Gauge size={18} />}>
          <div className="field">
            <label>Cultivo</label>
            <CultivoToggle value={req.cultivo} onChange={(c) => set('cultivo', c)} />
          </div>
          <div className="field">
            <label>Mes del año</label>
            <select value={req.mes} onChange={(e) => set('mes', Number(e.target.value))}>
              {MESES.map((m, i) => <option key={m} value={i + 1}>{m}</option>)}
            </select>
            <div className="hint">Meses críticos de lluvia en Piura: febrero a abril.</div>
          </div>
          <Slider label="Índice Costero El Niño (ICEN)" unit=" °C" value={req.icen} min={-1} max={3.5} step={0.1}
            hint="> 1.5 °C Niño moderado · > 2.0 °C Niño fuerte/extraordinario" onChange={(v) => set('icen', v)} />
          <Slider label="Anomalía TSM Niño 1+2" unit=" °C" value={req.anomalia_tsm} min={-1.5} max={4} step={0.1}
            onChange={(v) => set('anomalia_tsm', v)} />
          <Slider label="Precipitación mensual" unit=" mm" value={req.lluvia_mm} min={0} max={800} step={10}
            onChange={(v) => set('lluvia_mm', v)} />
          <Slider label="Temperatura máxima promedio" unit=" °C" value={req.temp_max} min={28} max={38} step={0.5}
            onChange={(v) => set('temp_max', v)} />
          <button className="btn" disabled={pred.loading} onClick={() => pred.run(req)}>
            {pred.loading ? <div className="spinner" /> : <Play size={16} />} Predecir rendimiento
          </button>
        </Card>

        <div className="stack">
          {pred.error && <Alert tone="critico">{pred.error}</Alert>}
          {!res && !pred.error && (
            <Card><Empty icon={<Sprout size={40} />}>Ajusta las condiciones climáticas y presiona <strong>Predecir rendimiento</strong>.</Empty></Card>
          )}
          {res && (
            <>
              <div className="grid grid-2">
                <Card title={`Rendimiento estimado · ${res.cultivo === 'Limon' ? 'Limón' : 'Arroz'}`}>
                  <div className="big-number">{fmt(res.rendimiento_estimado_tha)} <small>t/ha</small></div>
                  <p className="muted" style={{ marginBottom: 0 }}>Promedio normal en Piura: {fmt(res.rendimiento_historico_base, 1)} t/ha</p>
                </Card>
                <Card title="Pérdida estimada">
                  <div className="big-number" style={{ color: COLORES[res.nivel] }}>-{fmt(res.perdida_estimada_pct, 1)}<small> %</small></div>
                  <p style={{ marginBottom: 0 }}><span className={`badge ${res.nivel}`}>{res.alerta_riesgo}</span></p>
                </Card>
              </div>
              {res.nivel === 'critico' && (
                <Alert tone="critico"><strong>Medidas urgentes:</strong> evacuar drenes parcelarios, postergar la fertilización nitrogenada y habilitar motobombas.</Alert>
              )}
              <Card title="Importancia de variables en el modelo" sub="Peso relativo de cada variable en las decisiones del Random Forest.">
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={importancia} layout="vertical" margin={{ left: 20, right: 30 }}>
                    <XAxis type="number" tick={{ fontSize: 11 }} domain={[0, 1]} />
                    <YAxis type="category" dataKey="variable" tick={{ fontSize: 12 }} width={110} />
                    <Tooltip formatter={(v) => `${(Number(v) * 100).toFixed(1)} %`} />
                    <Bar dataKey="peso" radius={[0, 6, 6, 0]}>
                      {importancia.map((d, i) => <Cell key={d.variable} fill={i === 0 ? COLORES.ok : '#a5d6a7'} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            </>
          )}
        </div>
      </div>
    </>
  )
}
