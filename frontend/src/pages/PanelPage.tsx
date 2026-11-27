import { useState } from 'react'
import { Activity } from 'lucide-react'
import {
  CartesianGrid, ComposedChart, Legend, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { api, type Cultivo } from '../api'
import { Alert, Card, CultivoToggle, Loading, PageHead, COLORES, useApi } from '../components/ui'

export default function PanelPage() {
  const [cultivo, setCultivo] = useState<Cultivo>('Limon')
  const { data, error, loading } = useApi(() => api.panel(cultivo), [cultivo])

  return (
    <>
      <PageHead eyebrow="Panel general" title="Monitoreo de El Niño Costero en Piura">
        Estado climático más reciente de la tabla maestra y su impacto histórico en el rendimiento y el precio de los cultivos foco.
      </PageHead>

      {error && <Alert tone="critico">{error}</Alert>}
      {loading && !data && <Loading />}

      {data && (
        <div className="stack">
          <Alert tone={data.evento_nino === 'Normal' ? 'ok' : data.evento_nino === 'Débil' ? 'alerta' : 'critico'}>
            <strong>Estado ENFEN ({data.ultimo_mes}):</strong> condición <strong>{data.evento_nino}</strong> según el
            Índice Costero El Niño (ICEN).
          </Alert>

          <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
            <CultivoToggle value={cultivo} onChange={setCultivo} />
          </div>

          <div className="grid grid-4">
            {data.kpis.map((k) => (
              <div key={k.etiqueta} className={`card kpi ${k.tono}`}>
                <div className="label">{k.etiqueta}</div>
                <div className="value">{k.valor}</div>
                <div className="detail">{k.detalle}</div>
              </div>
            ))}
          </div>

          <Card title="Rendimiento vs. Índice Costero El Niño (2007 – 2026)" icon={<Activity size={18} />}
            sub="Las caídas de rendimiento coinciden con los picos del ICEN: 2017 (Niño Costero), 2023 y 2026.">
            <ResponsiveContainer width="100%" height={340}>
              <ComposedChart data={data.serie} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                <CartesianGrid stroke="#eef2ed" vertical={false} />
                <XAxis dataKey="fecha" tick={{ fontSize: 11 }} minTickGap={40} />
                <YAxis yAxisId="r" tick={{ fontSize: 11 }} label={{ value: 't/ha', angle: -90, position: 'insideLeft', fontSize: 12 }} />
                <YAxis yAxisId="i" orientation="right" tick={{ fontSize: 11 }} label={{ value: 'ICEN °C', angle: 90, position: 'insideRight', fontSize: 12 }} />
                <Tooltip formatter={(v) => Number(v).toFixed(2)} />
                <Legend />
                <ReferenceLine yAxisId="i" y={1.5} stroke={COLORES.alerta} strokeDasharray="4 4" label={{ value: 'Moderado', fontSize: 10, fill: COLORES.alerta, position: 'insideTopRight' }} />
                <ReferenceLine yAxisId="i" y={2.0} stroke={COLORES.critico} strokeDasharray="4 4" label={{ value: 'Fuerte', fontSize: 10, fill: COLORES.critico, position: 'insideTopRight' }} />
                <Line yAxisId="r" dataKey="rendimiento_tha" name="Rendimiento (t/ha)" stroke={COLORES.ok} strokeWidth={2} dot={false} />
                <Line yAxisId="i" dataKey="icen" name="ICEN (°C)" stroke={COLORES.critico} strokeWidth={1.5} strokeDasharray="5 3" dot={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </Card>

          <div className="grid grid-3">
            <Card title="Años de impacto severo">
              <p className="muted" style={{ margin: 0 }}>Caídas drásticas de rendimiento en 2017 (Niño Costero), 2023 y el evento actual 2026.</p>
            </Card>
            <Card title="Limón sutil">
              <p className="muted" style={{ margin: 0 }}>El exceso de humedad asfixia las raíces en Chulucanas y Tambogrande y aborta la floración.</p>
            </Card>
            <Card title="Arroz cáscara">
              <p className="muted" style={{ margin: 0 }}>Las inundaciones del Bajo Piura anegan parcelas en maduración y favorecen plagas bacterianas (<em>Xanthomonas</em>).</p>
            </Card>
          </div>
        </div>
      )}
    </>
  )
}
