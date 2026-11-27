import { lazy, Suspense } from 'react'
import { NavLink, Navigate, Route, Routes } from 'react-router-dom'
import { FileText, LayoutDashboard, Leaf, Map, Sprout, TrendingUp } from 'lucide-react'
import { Loading } from './components/ui'

// Cada módulo se descarga solo cuando se visita (code splitting)
const PanelPage = lazy(() => import('./pages/PanelPage'))
const RendimientoPage = lazy(() => import('./pages/RendimientoPage'))
const PreciosPage = lazy(() => import('./pages/PreciosPage'))
const VisionPage = lazy(() => import('./pages/VisionPage'))
const ClusteringPage = lazy(() => import('./pages/ClusteringPage'))
const ResumenPage = lazy(() => import('./pages/ResumenPage'))

const NAV = [
  { to: '/', label: 'Panel general', icon: LayoutDashboard, num: null },
  { to: '/rendimiento', label: 'Predicción de rendimiento', icon: Sprout, num: 1 },
  { to: '/precios', label: 'Pronóstico de precios', icon: TrendingUp, num: 2 },
  { to: '/vision', label: 'Diagnóstico de hojas', icon: Leaf, num: 3 },
  { to: '/clustering', label: 'Vulnerabilidad provincial', icon: Map, num: 4 },
  { to: '/resumen', label: 'Resumen de comunicados', icon: FileText, num: 5 },
]

export default function App() {
  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-logo">🌾</div>
          <div>
            <h1>AgroSmart Perú</h1>
            <p>Inteligencia ante El Niño Costero</p>
          </div>
        </div>
        <nav className="nav">
          {NAV.map(({ to, label, icon: Icon, num }) => (
            <NavLink key={to} to={to} end={to === '/'}>
              {num ? <span className="num">{num}</span> : <Icon size={18} />}
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-foot">
          <p><strong>Área de estudio:</strong> Región Piura (costa norte)</p>
          <p><strong>Cultivos:</strong> limón sutil y arroz cáscara</p>
          <p><strong>Fuentes:</strong> BCRP, NOAA, IGP (ICEN), SENAMHI, MIDAGRI, ENFEN</p>
        </div>
      </aside>
      <main className="main">
        <Suspense fallback={<Loading />}>
          <Routes>
            <Route path="/" element={<PanelPage />} />
            <Route path="/rendimiento" element={<RendimientoPage />} />
            <Route path="/precios" element={<PreciosPage />} />
            <Route path="/vision" element={<VisionPage />} />
            <Route path="/clustering" element={<ClusteringPage />} />
            <Route path="/resumen" element={<ResumenPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Suspense>
      </main>
    </div>
  )
}
