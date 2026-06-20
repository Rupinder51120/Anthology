import { NavLink } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  Home, Search, MessageSquare, Upload,
  Library, Layers, Settings, Activity, Compass, BarChart2
} from 'lucide-react'
import { getStats } from '../../api/client'
import { glass } from '../../lib/theme'
import MoodToggle from '../ui/MoodToggle'

const nav = [
  { to: '/',            icon: Home,          label: 'Home' },
  { to: '/search',      icon: Search,        label: 'Search Papers' },
  { to: '/chat',        icon: MessageSquare, label: 'Chat Assistant' },
  { to: '/library',     icon: Library,       label: 'My Library' },
  { to: '/upload',      icon: Upload,        label: 'Upload Papers' },
  { to: '/discover',    icon: Compass,       label: 'Discovery' },
  { to: '/collections', icon: Layers,        label: 'Collections' },
  { to: '/benchmark',   icon: BarChart2,     label: 'Benchmark' },
  { to: '/settings',    icon: Settings,      label: 'Settings' },
]

export default function Sidebar() {
  return (
    <aside style={{
      width: 220, minHeight: '100vh',
      ...glass,
      borderRadius: 0,
      borderRight: '1px solid var(--color-border)',
      borderTop: 'none', borderBottom: 'none', borderLeft: 'none',
      display: 'flex', flexDirection: 'column',
      padding: '0 0 20px 0',
      flexShrink: 0, position: 'sticky', top: 0, height: '100vh',
    }}>
      {/* Logo */}
      <div style={{ padding: '20px 20px 16px', borderBottom: '1px solid var(--color-border)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 32, height: 32, borderRadius: 9,
            background: 'var(--mood-btn-grad)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 15, fontWeight: 800, color: '#fff',
            boxShadow: '0 4px 12px var(--mood-glow-sm)',
          }}>A</div>
          <div>
            <div style={{ fontWeight: 700, fontSize: 14, color: 'var(--color-text)', letterSpacing: '-0.01em' }}>Anthology</div>
            <div style={{ fontSize: 10, color: 'var(--color-muted)', fontWeight: 500 }}>AI Research Assistant</div>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav style={{ flex: 1, padding: '10px 8px', display: 'flex', flexDirection: 'column', gap: 2, overflowY: 'auto' }}>
        {nav.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            style={({ isActive }) => ({
              display: 'flex', alignItems: 'center', gap: 10,
              padding: '8px 12px', borderRadius: 'var(--radius-md)',
              color: isActive ? 'var(--mood-accent)' : 'var(--color-muted)',
              background: isActive ? 'var(--mood-badge-bg)' : 'transparent',
              border: isActive ? '1px solid var(--mood-badge-border)' : '1px solid transparent',
              textDecoration: 'none', fontSize: 13, fontWeight: isActive ? 600 : 500,
              transition: 'all 0.15s',
            })}
          >
            <Icon size={15} />
            {label}
          </NavLink>
        ))}
      </nav>

      {/* Stats */}
      <LiveStats />

      {/* Mood toggle */}
      <div style={{
        margin: '12px 10px 0',
        padding: '12px 14px',
        background: 'rgba(0,0,0,0.02)',
        borderRadius: 'var(--radius-md)',
        border: '1px solid var(--color-border)',
      }}>
        <div style={{
          fontSize: 10, fontWeight: 600, color: 'var(--color-muted)',
          textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8,
        }}>Mood</div>
        <MoodToggle />
      </div>
    </aside>
  )
}

function LiveStats() {
  const { data: stats } = useQuery({ queryKey: ['stats'], queryFn: getStats, staleTime: 60_000 })

  const items = stats
    ? [
        { label: 'Papers',  value: stats.total_papers.toLocaleString() },
        { label: 'Chunks',  value: (stats.total_chunks / 1000).toFixed(1) + 'k' },
        { label: 'Queries', value: stats.total_queries.toLocaleString() },
      ]
    : [
        { label: 'Papers',  value: '—' },
        { label: 'Chunks',  value: '—' },
        { label: 'Queries', value: '—' },
      ]

  return (
    <div style={{
      margin: '0 10px',
      padding: '12px',
      background: 'rgba(0,0,0,0.03)',
      borderRadius: 'var(--radius-md)',
      border: '1px solid var(--color-border)',
    }}>
      <div style={{
        display: 'flex', alignItems: 'center', gap: 5, marginBottom: 10,
        color: 'var(--color-muted)', fontSize: 10,
        fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em',
      }}>
        <Activity size={10} /> Research Stats
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 6 }}>
        {items.map(s => (
          <div key={s.label} style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--color-text)', letterSpacing: '-0.01em' }}>{s.value}</div>
            <div style={{ fontSize: 9, color: 'var(--color-muted)', fontWeight: 500, marginTop: 1 }}>{s.label}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
