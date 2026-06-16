import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { Search, MessageSquare, Upload, BookOpen, Zap, Database, Activity, ArrowRight } from 'lucide-react'
import { getStats, getPapers } from '../api/client'
import { Spinner } from '../components/ui'
import { glass, glassCard, hoverLift, hoverSlide } from '../lib/theme'
import type { StatsResponse } from '../api/client'

const actions = [
  { icon: Search,        label: 'Smart Search',  sub: 'Find papers by concept',   to: '/search',  color: 'var(--color-blue)',   bg: 'var(--color-blue-dim)'   },
  { icon: MessageSquare, label: 'Chat with AI',  sub: 'Ask research questions',   to: '/chat',    color: 'var(--color-green)',  bg: 'var(--color-green-dim)'  },
  { icon: Upload,        label: 'Upload Papers', sub: 'Add PDFs to your library', to: '/upload',  color: 'var(--color-orange)', bg: 'var(--color-orange-dim)' },
  { icon: BookOpen,      label: 'My Library',    sub: 'Browse your collection',   to: '/library', color: 'var(--color-purple)', bg: 'var(--color-purple-dim)' },
]

const statDef = [
  { icon: BookOpen, label: 'Papers',  key: 'total_papers',  fmt: (v: number) => v.toString(),                color: 'var(--color-blue)',   bg: 'var(--color-blue-dim)'   },
  { icon: Database, label: 'Chunks',  key: 'total_chunks',  fmt: (v: number) => v.toLocaleString(),          color: 'var(--color-green)',  bg: 'var(--color-green-dim)'  },
  { icon: Zap,      label: 'Vectors', key: 'vector_chunks', fmt: (v: number) => v.toLocaleString(),          color: 'var(--color-orange)', bg: 'var(--color-orange-dim)' },
  { icon: Activity, label: 'Queries', key: 'total_queries', fmt: (v: number) => v.toString(),                color: 'var(--color-purple)', bg: 'var(--color-purple-dim)' },
] satisfies Array<{
  icon: typeof BookOpen
  label: string
  key: keyof Pick<StatsResponse, 'total_papers' | 'total_chunks' | 'vector_chunks' | 'total_queries'>
  fmt: (v: number) => string
  color: string
  bg: string
}>

export default function HomePage() {
  const nav = useNavigate()
  const { data: stats } = useQuery({ queryKey: ['stats'], queryFn: getStats })
  const { data: papers } = useQuery({ queryKey: ['papers'], queryFn: getPapers })

  return (
    <div style={{ padding: '32px 36px', maxWidth: 1100, margin: '0 auto' }}>

      {/* Hero */}
      <div style={{
        ...glass,
        borderRadius: 'var(--radius-xl)',
        padding: '48px 48px 40px',
        marginBottom: 24,
        position: 'relative',
        overflow: 'hidden',
      }}>
        {/* Decorative blobs — use CSS vars for colors */}
        <div style={{ position: 'absolute', top: -60, right: -60, width: 240, height: 240, borderRadius: '50%', background: 'radial-gradient(circle, var(--color-blue-dim) 0%, transparent 70%)', pointerEvents: 'none' }} />
        <div style={{ position: 'absolute', bottom: -40, left: 100, width: 180, height: 180, borderRadius: '50%', background: 'radial-gradient(circle, var(--color-green-dim) 0%, transparent 70%)', pointerEvents: 'none' }} />

        <div style={{ position: 'relative', zIndex: 1 }}>
          <div style={{
            display: 'inline-flex', alignItems: 'center', gap: 6,
            background: 'var(--color-accent-dim)', border: '1px solid var(--color-accent-border)',
            borderRadius: 20, padding: '3px 12px', fontSize: 11, fontWeight: 600,
            color: 'var(--color-accent)', marginBottom: 16, letterSpacing: '0.04em',
          }}>
            ✦ AI Research Assistant
          </div>

          <div style={{ fontSize: 36, fontWeight: 800, marginBottom: 10, lineHeight: 1.15, color: 'var(--color-text)', letterSpacing: '-0.02em' }}>
            Discover. Understand.<br />
            <span style={{ color: 'var(--color-accent)' }}>Innovate.</span>
          </div>

          <div style={{ color: 'var(--color-muted)', fontSize: 15, marginBottom: 28, maxWidth: 480 }}>
            Your AI-powered research companion for academic papers and scientific knowledge.
          </div>

          {/* Search */}
          <div style={{ display: 'flex', gap: 10, maxWidth: 520 }}>
            <div style={{ flex: 1, position: 'relative' }}>
              <Search size={14} style={{ position: 'absolute', left: 14, top: '50%', transform: 'translateY(-50%)', color: 'var(--color-muted)', pointerEvents: 'none' }} />
              <input
                placeholder="Search papers, concepts, authors..."
                onKeyDown={e => e.key === 'Enter' && nav('/search')}
                style={{
                  width: '100%', background: 'var(--glass-bg)',
                  backdropFilter: 'var(--glass-blur)',
                  WebkitBackdropFilter: 'var(--glass-blur)',
                  border: 'var(--glass-border)',
                  borderRadius: 'var(--radius-md)',
                  color: 'var(--color-text)',
                  padding: '11px 14px 11px 38px',
                  fontSize: 13, outline: 'none',
                  boxShadow: 'var(--glass-shadow)',
                  fontFamily: 'var(--font-sans)',
                }}
              />
            </div>
            <button onClick={() => nav('/search')} style={{
              background: 'var(--color-accent)', border: 'none',
              borderRadius: 'var(--radius-md)',
              color: '#fff', padding: '11px 22px', fontSize: 13,
              fontWeight: 600, cursor: 'pointer',
              boxShadow: '0 4px 12px var(--color-accent-dim)',
              fontFamily: 'var(--font-sans)',
              transition: 'all 0.15s',
            }}>
              Search
            </button>
          </div>

          <div style={{ marginTop: 14, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {['"transformer architecture"', '"graph neural networks"', '"federated learning"'].map(s => (
              <button key={s} onClick={() => nav('/search')} style={{
                background: 'rgba(0,0,0,0.04)', border: '1px solid var(--color-border)',
                borderRadius: 20, padding: '3px 12px', fontSize: 11,
                color: 'var(--color-muted)', cursor: 'pointer',
                fontFamily: 'var(--font-sans)', transition: 'all 0.15s',
              }}>
                Try {s}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Stats */}
      {stats && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 24 }}>
          {statDef.map(s => (
            <div key={s.label} style={{ ...glassCard, display: 'flex', alignItems: 'center', gap: 14 }}>
              <div style={{ width: 44, height: 44, borderRadius: 12, background: s.bg, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                <s.icon size={20} color={s.color} />
              </div>
              <div>
                <div style={{ fontSize: 22, fontWeight: 800, color: 'var(--color-text)', letterSpacing: '-0.02em' }}>
                  {s.fmt(stats[s.key])}
                </div>
                <div style={{ fontSize: 11, color: 'var(--color-muted)', fontWeight: 500 }}>{s.label}</div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Quick Actions */}
      <div style={{ marginBottom: 24 }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--color-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 12 }}>
          Quick Actions
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
          {actions.map(a => (
            <button
              key={a.label}
              onClick={() => nav(a.to)}
              style={{ ...glassCard, cursor: 'pointer', textAlign: 'left', border: 'var(--glass-border)', transition: 'transform 0.2s, box-shadow 0.2s', display: 'flex', flexDirection: 'column', gap: 10 }}
              {...hoverLift}
            >
              <div style={{ width: 40, height: 40, borderRadius: 11, background: a.bg, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <a.icon size={18} color={a.color} />
              </div>
              <div>
                <div style={{ fontWeight: 700, fontSize: 13, color: 'var(--color-text)', marginBottom: 2 }}>{a.label}</div>
                <div style={{ fontSize: 11, color: 'var(--color-muted)' }}>{a.sub}</div>
              </div>
              <ArrowRight size={14} color={a.color} style={{ marginTop: 'auto' }} />
            </button>
          ))}
        </div>
      </div>

      {/* Recent Papers */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--color-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Recent Papers
          </div>
          <button onClick={() => nav('/library')} style={{
            fontSize: 12, color: 'var(--color-accent)', background: 'none',
            border: 'none', cursor: 'pointer', fontWeight: 600,
            display: 'flex', alignItems: 'center', gap: 4,
            fontFamily: 'var(--font-sans)',
          }}>
            View all <ArrowRight size={12} />
          </button>
        </div>

        {papers ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {papers.papers.slice(0, 5).map(p => (
              <div
                key={p.id}
                style={{ ...glass, borderRadius: 'var(--radius-md)', padding: '14px 18px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer', transition: 'transform 0.15s' }}
                {...hoverSlide}
              >
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: 'var(--color-text)' }}>{p.title}</div>
                  <div style={{ fontSize: 11, color: 'var(--color-muted)' }}>{p.authors} · {p.year}</div>
                </div>
                <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexShrink: 0, marginLeft: 16 }}>
                  {p.year && <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--color-accent)', background: 'var(--color-accent-dim)', border: '1px solid var(--color-accent-border)', borderRadius: 6, padding: '2px 8px' }}>{p.year}</span>}
                  <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--color-success)', background: 'var(--color-success-dim)', border: '1px solid rgba(48,209,88,0.20)', borderRadius: 6, padding: '2px 8px' }}>Indexed</span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div style={{ display: 'flex', justifyContent: 'center', padding: 32 }}><Spinner /></div>
        )}
      </div>
    </div>
  )
}
