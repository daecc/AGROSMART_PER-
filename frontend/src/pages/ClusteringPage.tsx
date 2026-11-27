import { Map as MapIcon, ScatterChart as ScatterIcon } from 'lucide-react'
import {
  CartesianGrid, LabelList, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from 'recharts'
import { api, type Provincia } from '../api'
import { Alert, Card, Loading, PageHead, fmt, useApi } from '../components/ui'

export default function ClusteringPage() {
  const { data, error, loading } = useApi(api.clustering, [])

  return (
    <>
      <PageHead eyebrow="Módulo 4 · Aprendizaje no supervisado" title="Zonificación de vulnerabilidad provincial">
        K-Means (k = 3) sobre lluvia máxima histórica, área inundable, pérdida de cosecha y déficit de drenaje de las 8 provincias de Piura.
      </PageHead>

      {error && <Alert tone="critico">{error}</Alert>}
      {loading && <Loading />}

      {data && (
        <div className="stack">
          <div className="grid grid-3">
            {data.grupos.map((g) => (
              <div key={g.nivel} className="card" style={{ borderTop: `4px solid ${g.color}` }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontWeight: 700 }}>
                  <span className="dot" style={{ background: g.color }} />{g.nivel}
                </div>
                <div style={{ fontSize: 18, fontWeight: 700, margin: '8px 0 6px' }}>{g.provincias.join(', ')}</div>
                <p className="muted" style={{ margin: 0, fontSize: 13.5 }}>{g.accion}</p>
              </div>
            ))}
          </div>

          <div className="grid grid-2">
            <Card title="Espacio PCA (2 componentes principales)" icon={<ScatterIcon size={18} />}
              sub={data.silueta != null ? `Coeficiente de silueta: ${fmt(data.silueta, 3)}` : undefined}>
              <ResponsiveContainer width="100%" height={360}>
                <ScatterChart margin={{ top: 16, right: 30, left: 0, bottom: 10 }}>
                  <CartesianGrid stroke="#eef2ed" />
                  <XAxis type="number" dataKey="pca_x" name="PC1" tick={{ fontSize: 11 }} label={{ value: 'PC1 · exposición y daño', position: 'insideBottom', offset: -4, fontSize: 12 }} />
                  <YAxis type="number" dataKey="pca_y" name="PC2" tick={{ fontSize: 11 }} label={{ value: 'PC2 · drenaje', angle: -90, position: 'insideLeft', fontSize: 12 }} />
                  <ZAxis range={[220, 220]} />
                  <Tooltip content={<TooltipProvincia />} />
                  {data.grupos.map((g) => (
                    <Scatter key={g.nivel} name={g.nivel} data={data.provincias.filter((p) => p.nivel_vulnerabilidad === g.nivel)}
                      fill={g.color} stroke="#1c2b22" strokeWidth={0.6}>
                      <LabelList dataKey="provincia" position="top" style={{ fontSize: 11, fontWeight: 600, fill: '#1c2b22' }} />
                    </Scatter>
                  ))}
                </ScatterChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Detalle por provincia" icon={<MapIcon size={18} />}>
              <table>
                <thead>
                  <tr><th>Provincia</th><th>Cultivo predominante</th><th className="num">Pérdida %</th><th className="num">Inundable %</th></tr>
                </thead>
                <tbody>
                  {[...data.provincias].sort((a, b) => b.perdida_cosecha_nino_pct - a.perdida_cosecha_nino_pct).map((p) => (
                    <tr key={p.ubigeo}>
                      <td><span className="dot" style={{ background: p.color_alerta, marginRight: 8 }} />{p.provincia}</td>
                      <td className="muted">{p.cultivo_predominante}</td>
                      <td className="num">{fmt(p.perdida_cosecha_nino_pct, 1)}</td>
                      <td className="num">{fmt(p.pct_area_inundable, 1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          </div>
        </div>
      )}
    </>
  )
}

function TooltipProvincia({ active, payload }: { active?: boolean; payload?: { payload: Provincia }[] }) {
  if (!active || !payload?.length) return null
  const p = payload[0].payload
  return (
    <div className="card" style={{ padding: 12, fontSize: 13 }}>
      <strong>{p.provincia}</strong> · {p.valle_principal}
      <div className="muted">{p.nivel_vulnerabilidad}</div>
      <div>Lluvia máx.: {p.lluvia_max_historica_mm} mm · Déficit drenaje: {p.deficit_drenaje_pct} %</div>
    </div>
  )
}
