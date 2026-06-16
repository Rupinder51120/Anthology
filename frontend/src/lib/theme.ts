import type { CSSProperties } from 'react'

export const glass: CSSProperties = {
  background: 'var(--glass-bg)',
  backdropFilter: 'var(--glass-blur)',
  WebkitBackdropFilter: 'var(--glass-blur)',
  border: 'var(--glass-border)',
  boxShadow: 'var(--glass-shadow)',
}

export const glassCard: CSSProperties = {
  ...glass,
  borderRadius: 'var(--radius-lg)',
  padding: 20,
}

export const glassCardSm: CSSProperties = {
  ...glass,
  borderRadius: 'var(--radius-md)',
  padding: 14,
}

export const badge = (color: 'blue' | 'green' | 'orange' | 'purple' | 'muted'): CSSProperties => {
  const map = {
    blue:   { color: 'var(--color-blue)',   background: 'var(--color-blue-dim)',   border: '1px solid var(--color-accent-border)' },
    green:  { color: 'var(--color-green)',  background: 'var(--color-green-dim)',  border: '1px solid rgba(48,209,88,0.20)' },
    orange: { color: 'var(--color-orange)', background: 'var(--color-orange-dim)', border: '1px solid rgba(255,159,10,0.20)' },
    purple: { color: 'var(--color-purple)', background: 'var(--color-purple-dim)', border: '1px solid rgba(191,90,242,0.20)' },
    muted:  { color: 'var(--color-muted)',  background: 'rgba(0,0,0,0.04)',        border: '1px solid var(--color-border)' },
  }
  return {
    display: 'inline-flex', alignItems: 'center',
    fontSize: 11, fontWeight: 600,
    padding: '2px 8px', borderRadius: 6,
    ...map[color],
  }
}

export const actionColors = [
  { color: 'var(--color-blue)',   bg: 'var(--color-blue-dim)'   },
  { color: 'var(--color-green)',  bg: 'var(--color-green-dim)'  },
  { color: 'var(--color-orange)', bg: 'var(--color-orange-dim)' },
  { color: 'var(--color-purple)', bg: 'var(--color-purple-dim)' },
]

export const hoverLift = {
  onMouseEnter: (e: React.MouseEvent<HTMLElement>) => {
    e.currentTarget.style.transform = 'translateY(-3px)'
    e.currentTarget.style.boxShadow = 'var(--glass-shadow-hover)'
  },
  onMouseLeave: (e: React.MouseEvent<HTMLElement>) => {
    e.currentTarget.style.transform = 'translateY(0)'
    e.currentTarget.style.boxShadow = 'var(--glass-shadow)'
  },
}

export const hoverSlide = {
  onMouseEnter: (e: React.MouseEvent<HTMLElement>) => {
    e.currentTarget.style.transform = 'translateX(4px)'
  },
  onMouseLeave: (e: React.MouseEvent<HTMLElement>) => {
    e.currentTarget.style.transform = 'translateX(0)'
  },
}
