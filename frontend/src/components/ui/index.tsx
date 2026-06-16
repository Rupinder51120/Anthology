import { type ReactNode, type CSSProperties } from 'react'
import { glass } from '../../lib/theme'

// ── Card ─────────────────────────────────────────────────────
export function Card({ children, style }: { children: ReactNode; style?: CSSProperties }) {
  return (
    <div style={{
      ...glass,
      borderRadius: 'var(--radius-lg)',
      ...style,
    }}>
      {children}
    </div>
  )
}

// ── Button ───────────────────────────────────────────────────
interface BtnProps {
  children: ReactNode
  onClick?: () => void
  variant?: 'primary' | 'ghost' | 'danger'
  size?: 'sm' | 'md'
  disabled?: boolean
  type?: 'button' | 'submit'
  style?: CSSProperties
}

export function Button({ children, onClick, variant = 'primary', size = 'md', disabled, type = 'button', style }: BtnProps) {
  const base: CSSProperties = {
    display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
    gap: 6, border: 'none', cursor: disabled ? 'not-allowed' : 'pointer',
    borderRadius: 'var(--radius-md)', fontWeight: 600,
    fontSize: size === 'sm' ? 12 : 13,
    padding: size === 'sm' ? '5px 12px' : '8px 18px',
    transition: 'all 0.15s',
    opacity: disabled ? 0.5 : 1,
    fontFamily: 'var(--font-sans)',
  }
  const variants: Record<string, CSSProperties> = {
    primary: {
      background: 'var(--color-accent)',
      color: '#fff',
      boxShadow: '0 4px 12px var(--color-accent-dim)',
    },
    ghost: {
      background: 'var(--glass-bg)',
      backdropFilter: 'var(--glass-blur)',
      WebkitBackdropFilter: 'var(--glass-blur)',
      color: 'var(--color-text)',
      border: 'var(--glass-border)',
      boxShadow: 'var(--glass-shadow)',
    },
    danger: {
      background: 'var(--color-danger-dim)',
      color: 'var(--color-danger)',
      border: '1px solid rgba(255,59,48,0.20)',
    },
  }
  return (
    <button type={type} onClick={onClick} disabled={disabled} style={{ ...base, ...variants[variant], ...style }}>
      {children}
    </button>
  )
}

// ── Badge ────────────────────────────────────────────────────
export function Badge({ children, color = 'accent' }: { children: ReactNode; color?: 'accent' | 'green' | 'yellow' | 'muted' }) {
  const colors: Record<string, CSSProperties> = {
    accent: { background: 'var(--color-accent-dim)',  color: 'var(--color-accent)',  border: '1px solid var(--color-accent-border)' },
    green:  { background: 'var(--color-success-dim)', color: 'var(--color-success)', border: '1px solid rgba(48,209,88,0.20)' },
    yellow: { background: 'var(--color-warning-dim)', color: 'var(--color-warning)', border: '1px solid rgba(255,159,10,0.20)' },
    muted:  { background: 'rgba(0,0,0,0.04)',         color: 'var(--color-muted)',   border: '1px solid var(--color-border)' },
  }
  return (
    <span style={{
      fontSize: 11, fontWeight: 600, padding: '2px 8px',
      borderRadius: 6, display: 'inline-flex', alignItems: 'center',
      ...colors[color],
    }}>
      {children}
    </span>
  )
}

// ── Input ────────────────────────────────────────────────────
interface InputProps {
  value: string
  onChange: (v: string) => void
  placeholder?: string
  onKeyDown?: (e: React.KeyboardEvent) => void
  style?: CSSProperties
  autoFocus?: boolean
}
export function Input({ value, onChange, placeholder, onKeyDown, style, autoFocus }: InputProps) {
  return (
    <input
      value={value}
      onChange={e => onChange(e.target.value)}
      placeholder={placeholder}
      onKeyDown={onKeyDown}
      autoFocus={autoFocus}
      style={{
        background: 'var(--glass-bg)',
        backdropFilter: 'var(--glass-blur)',
        WebkitBackdropFilter: 'var(--glass-blur)',
        border: 'var(--glass-border)',
        borderRadius: 'var(--radius-md)',
        boxShadow: 'var(--glass-shadow)',
        color: 'var(--color-text)',
        padding: '8px 12px',
        fontSize: 13,
        outline: 'none',
        width: '100%',
        fontFamily: 'var(--font-sans)',
        ...style,
      }}
    />
  )
}

// ── Spinner ──────────────────────────────────────────────────
export function Spinner({ size = 20 }: { size?: number }) {
  return (
    <div style={{
      width: size, height: size,
      border: `2px solid var(--color-border)`,
      borderTop: `2px solid var(--color-accent)`,
      borderRadius: '50%',
      animation: 'spin 0.7s linear infinite',
    }} />
  )
}

// ── Empty state ──────────────────────────────────────────────
export function Empty({ icon, title, sub }: { icon: ReactNode; title: string; sub?: string }) {
  return (
    <div style={{
      textAlign: 'center', padding: '56px 24px',
      color: 'var(--color-muted)',
    }}>
      <div style={{ fontSize: 40, marginBottom: 14, opacity: 0.5 }}>{icon}</div>
      <div style={{ fontWeight: 700, color: 'var(--color-text)', marginBottom: 6, fontSize: 15 }}>{title}</div>
      {sub && <div style={{ fontSize: 13, color: 'var(--color-muted)' }}>{sub}</div>}
    </div>
  )
}
