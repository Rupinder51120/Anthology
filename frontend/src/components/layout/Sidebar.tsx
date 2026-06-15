import { NavLink } from 'react-router-dom'
import {
  Home, Search, MessageSquare, BookOpen,
  Upload, Library, Layers, Settings, Activity,
} from 'lucide-react'

const nav = [
  { to: '/',         icon: Home,           label: 'Home' },
  { to: '/search',   icon: Search,         label: 'Search Papers' },
  { to: '/chat',     icon: MessageSquare,  label: 'Chat Assistant' },
  { to: '/library',  icon: Library,        label: 'My Library' },
  { to: '/upload',   icon: Upload,         label: 'Upload Papers' },
  { to: '/collections', icon: Layers,      label: 'Collections' },
  { to: '/settings', icon: Settings,       label: 'Settings' },
]

export default function Sidebar() {
  return (
    <aside style={{
      width: 220,
      minHeight: '100vh',
      background: 'var(--color-surface)',
      borderRight: '1px solid var(--color-border)',
      display: 'flex',
      flexDirection: 'column',
      padding: '0 0 24px 0',
      flexShrink: 0,
    }}>
      {/* Logo */}
      <div style={{ padding: '20px 20px 16px', borderBottom: '1px solid var(--color-border)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 32, height: 32, borderRadius: 8,
            background: 'var(--color-accent)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 16, fontWeight: 700, color: '#fff',
          }}>A</div>
          <div>
            <div style={{ fontWeight: 700, fontSize: 14, color: 'var(--color-text)' }}>Anthology</div>
            <div style={{ fontSize: 11, color: 'var(--color-muted)' }}>AI Research Assistant</div>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav style={{ flex: 1, padding: '12px 8px', display: 'flex', flexDirection: 'column', gap: 2 }}>
        {nav.map(({ to, icon: Icon, label }) => (
          <NavLink key={to} to={to} end={to === '/'} style={({ isActive }) => ({
            display: 'flex', alignItems: 'center', gap: 10,
            padding: '8px 12px', borderRadius: 8,
            color: isActive ? '#fff' : 'var(--color-muted)',
            background: isActive ? 'var(--color-accent)' : 'transparent',
            textDecoration: 'none', fontSize: 13, fontWeight: 500,
            transition: 'all 0.15s',
          })}>
            <Icon size={16} />
            {label}
          </NavLink>
        ))}
      </nav>

      {/* Stats */}
      <StatsBar />
    </aside>
  )
}

function StatsBar() {
  return (
    <div style={{
      margin: '0 12px',
      padding: '12px',
      background: 'var(--color-surface2)',
      borderRadius: 10,
      border: '1px solid var(--color-border)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 10, color: 'var(--color-muted)', fontSize: 11 }}>
        <Activity size={12} /> Research Stats
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
        {[
          { label: 'Papers', value: '122' },
          { label: 'Chunks', value: '15.3k' },
          { label: 'Queries', value: '98%' },
        ].map(s => (
          <div key={s.label} style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--color-text)' }}>{s.value}</div>
            <div style={{ fontSize: 10, color: 'var(--color-muted)' }}>{s.label}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
