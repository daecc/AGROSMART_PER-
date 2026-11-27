import { useMemo, useState } from 'react'
import { LineChart as LineIcon, Play, TrendingUp } from 'lucide-react'
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api, type Cultivo, type Escenario } from '../api'
import { Alert, Card, CultivoToggle, Empty, PageHead, Slider, COLORES, fmt, useAction, useApi } from '../components/ui'

export default function PreciosPage() {
  const [cultivo, setCultivo] = useState<Cultivo>('Limon')
  const [meses, setMeses] = useState(6)
  const [escenario, setEscenario] = useState<Escenario>('Fuerte')
  const escenarios = useApi(api.escenarios, [])
  const pron = useAction(api.pronosticar)
  const res = pron.data

  const serie = useMemo(() => {
    if (!res) return []
    const hist = res.historico.map((p) => ({ fecha: p.fecha, historico: p.precio_soles_kg }))
    const ult = res.historico[res.historico.length - 1]
    // el pronóstico arranca en el último punto histórico para que la línea sea continua
    const fut = [{ fecha: ult.fecha, pronostico: ult.precio_soles_kg }, ...res.pronostico.map((p) => ({ fecha: p.fecha, pronostico: p.precio_soles_kg }))]
    const map = new Map<string, Record<string, number | string>>()
    for (const row of [...hist, ...fut]) map.set(row.fecha, { ...(map.get(row.fecha) ?? {}), ...row })
    return [...map.values()]
  }, [res])

  const ultimoHist = res?.historico.at(-1)?.precio_soles_kg
  const finalPron = res?.pronostico.at(-1)?.precio_soles_kg

  return (
    <>
      <PageHead eyebrow="Módulo 2 · Series temporales" title="Pronóstico de precios en chacra">
        Gradient Boosting autorregresivo (rezagos de precio, ICEN y lluvia) con el ICEN del escenario ENFEN como variable exógena.
      </PageHead>

      <div className="grid grid-side">
        <Card title="Configuración" icon={<LineIcon size={18} />}>
          <div className="field">
            <label>Producto</label>
            <CultivoToggle value={cultivo} onChange={setCultivo} />
          </div>
          <Slider label="Horizonte de pronóstico" unit=" meses" value={meses} min={3} max={12} step={1} onChange={setMeses} />
          <div className="field">
            <label>Escenario climático (ENFEN)</label>
            <select value={escenario} onChange={(e) => setEscenario(e.target.value as Escenario)}>
              {(escenarios.data ?? []).map((e) => <option key={e.id} value={e.id}>{e.nombre}</option>)}
            </select>
          </div>
          <button className="btn" disabled={pron.loading} onClick={() => pron.run(cultivo, meses, escenario)}>
            {pron.loading ? <div className="spinner" /> : <Play size={16} />} Proyectar precios
          </button>
          {res?.metricas && (
            <p className="hint muted" style={{ fontSize: 12.5, marginBottom: 0 }}>
              Error del modelo en el período de prueba: MAPE {fmt(res.metricas.mape_pct)} % · RMSE S/ {fmt(res.metricas.rmse_soles)}/kg
            </p>
          )}
        </Card>

        <div className="stack">
          {(pron.error || escenarios.error) && <Alert tone="critico">{pron.error || escenarios.error}</Alert>}
          {!res && !pron.error && (
            <Card><Empty icon={<TrendingUp size={40} />}>Elige un escenario y presiona <strong>Proyectar precios</strong>.</Empty></Card>
          )}
          {res && ultimoHist !== undefined && finalPron !== undefined && (
            <>
              <div className="grid grid-3">
                <div className="card kpi neutro"><div className="label">Último precio observado</div><div className="value">S/ {fmt(ultimoHist)}</div><div className="detail">{res.historico.at(-1)?.fecha}</div></div>
                <div className={`card kpi ${finalPron > ultimoHist * 1.15 ? 'critico' : finalPron > ultimoHist ? 'alerta' : 'ok'}`}>
                  <div className="label">Precio proyectado ({res.pronostico.at(-1)?.fecha})</div><div className="value">S/ {fmt(finalPron)}</div>
                  <div className="detail">{fmt(((finalPron / ultimoHist) - 1) * 100, 1)} % vs último observado</div>
                </div>
                <div className="card kpi neutro"><div className="label">Escenario</div><div className="value" style={{ fontSize: 20 }}>{res.escenario.id}</div><div className="detail">ICEN supuesto: +{res.escenario.icen} °C</div></div>
              </div>
              <Card title={`Precio en chacra de ${res.cultivo === 'Limon' ? 'limón' : 'arroz'} (S/ por kg)`}>
                <ResponsiveContainer width="100%" height={340}>
                  <LineChart data={serie} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                    <CartesianGrid stroke="#eef2ed" vertical={false} />
                    <XAxis dataKey="fecha" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} domain={['auto', 'auto']} />
                    <Tooltip formatter={(v) => `S/ ${Number(v).toFixed(2)}`} />
                    <Legend />
                    <Line dataKey="historico" name="Histórico" stroke={COLORES.azul} strokeWidth={2} dot={{ r: 3 }} connectNulls />
                    <Line dataKey="pronostico" name={`Pronóstico (${res.escenario.id})`} stroke={COLORES.critico} strokeWidth={2} strokeDasharray="6 4" dot={{ r: 3 }} connectNulls />
                  </LineChart>
                </ResponsiveContainer>
              </Card>
              <Card title="Valores pronosticados">
                <table>
                  <thead><tr><th>Mes</th><th className="num">Precio (S/ por kg)</th></tr></thead>
                  <tbody>{res.pronostico.map((p) => <tr key={p.fecha}><td>{p.fecha}</td><td className="num">{fmt(p.precio_soles_kg)}</td></tr>)}</tbody>
                </table>
              </Card>
            </>
          )}
        </div>
      </div>
    </>
  )
}
