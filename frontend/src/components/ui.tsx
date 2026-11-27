import { useEffect, useState, type ReactNode } from 'react'
import { AlertTriangle, CheckCircle2, Info, OctagonAlert } from 'lucide-react'
import type { Cultivo, Nivel } from '../api'

export function PageHead({ eyebrow, title, children }: { eyebrow: string; title: string; children?: ReactNode }) {
  return (
    <header className="page-head">
      <div className="eyebrow">{eyebrow}</div>
      <h2>{title}</h2>
      {children && <p>{children}</p>}
    </header>
  )
}

export function Card({ title, icon, sub, children, className = '' }: {
  title?: string; icon?: ReactNode; sub?: ReactNode; children: ReactNode; className?: string
}) {
  return (
    <section className={`card ${className}`}>
      {title && <h3>{icon}{title}</h3>}
      {sub && <p className="sub">{sub}</p>}
      {children}
    </section>
  )
}

const ICONOS = { ok: CheckCircle2, alerta: AlertTriangle, critico: OctagonAlert, info: Info }

export function Alert({ tone, children }: { tone: Nivel | 'info'; children: ReactNode }) {
  const Icon = ICONOS[tone]
  return <div className={`alert ${tone}`}><Icon size={18} /><div>{children}</div></div>
}

export function Loading({ text = 'Cargando…' }: { text?: string }) {
  return <div className="loading"><div className="spinner dark" />{text}</div>
}

export function Empty({ icon, children }: { icon: ReactNode; children: ReactNode }) {
  return <div className="empty">{icon}<div>{children}</div></div>
}

export function CultivoToggle({ value, onChange }: { value: Cultivo; onChange: (c: Cultivo) => void }) {
  return (
    <div className="segmented" role="tablist" aria-label="Cultivo">
      {(['Limon', 'Arroz'] as Cultivo[]).map((c) => (
        <button key={c} type="button" className={value === c ? 'on' : ''} onClick={() => onChange(c)}>
          {c === 'Limon' ? 'Limón sutil' : 'Arroz cáscara'}
        </button>
      ))}
    </div>
  )
}

export function Slider({ label, value, min, max, step, unit = '', hint, onChange }: {
  label: string; value: number; min: number; max: number; step: number; unit?: string; hint?: string
  onChange: (v: number) => void
}) {
  return (
    <div className="field">
      <label>{label}<span>{value}{unit}</span></label>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
      {hint && <div className="hint">{hint}</div>}
    </div>
  )
}

/** Carga datos al montar o cuando cambian las dependencias. */
export function useApi<T>(fn: () => Promise<T>, deps: unknown[]) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  useEffect(() => {
    let vivo = true
    setLoading(true)
    setError(null)
    fn()
      .then((d) => { if (vivo) setData(d) })
      .catch((e: Error) => { if (vivo) setError(e.message) })
      .finally(() => { if (vivo) setLoading(false) })
    return () => { vivo = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
  return { data, error, loading }
}

/** Ejecuta una acción (botón) controlando estado de carga y error. */
export function useAction<A extends unknown[], T>(fn: (...args: A) => Promise<T>) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const run = async (...args: A) => {
    setLoading(true)
    setError(null)
    try {
      setData(await fn(...args))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }
  return { data, error, loading, run, setData }
}

export const COLORES = { ok: '#2e7d32', alerta: '#b26a00', critico: '#c62828', azul: '#1565c0', gris: '#90a4ae' }
export const fmt = (n: number, d = 2) => n.toLocaleString('es-PE', { minimumFractionDigits: d, maximumFractionDigits: d })
