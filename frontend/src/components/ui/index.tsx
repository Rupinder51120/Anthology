import { type ReactNode, type CSSProperties } from 'react'

// ── Card ─────────────────────────────────────────────────────
export function Card({ children, style }: { children: ReactNode; style?: CSSProperties }) {
  return (
    <div style={{
      background: 'var(--color-surface)',
      border: '1px solid var(--color-border)',
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
  }
  const variants: Record<string, CSSProperties> = {
    primary: { background: 'var(--color-accent)', color: '#fff' },
    ghost:   { background: 'var(--color-surface2)', color: 'var(--color-text)', border: '1px solid var(--color-border)' },
    danger:  { background: '#ef444422', color: 'var(--color-danger)', border: '1px solid #ef444444' },
  }
  return (
    <button type={type} onClick={onClick} disabled={disabled} style={{ ...base, ...variants[variant], ...style }}>
      {children}
    </button>
  )
}

// ── Badge ────────────────────────────────────────────────────
export function Badge({ children, color = 'accent' }: { children: ReactNode; color?: 'accent' | 'green' | 'yellow' }) {
  const colors: Record<string, CSSProperties> = {
    accent: { background: 'var(--color-accent-dim)', color: 'var(--color-accent)' },
    green:  { background: '#22c55e22', color: '#22c55e' },
    yellow: { background: '#f59e0b22', color: '#f59e0b' },
  }
  return (
    <span style={{
      fontSize: 11, fontWeight: 600, padding: '2px 8px',
      borderRadius: 20, ...colors[color],
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
        background: 'var(--color-surface2)',
        border: '1px solid var(--color-border)',
        borderRadius: 'var(--radius-md)',
        color: 'var(--color-text)',
        padding: '8px 12px',
        fontSize: 13,
        outline: 'none',
        width: '100%',
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
    <div style={{ textAlign: 'center', padding: '48px 24px', color: 'var(--color-muted)' }}>
      <div style={{ fontSize: 40, marginBottom: 12 }}>{icon}</div>
      <div style={{ fontWeight: 600, color: 'var(--color-text)', marginBottom: 4 }}>{title}</div>
      {sub && <div style={{ fontSize: 12 }}>{sub}</div>}
    </div>
  )
}
