import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { Search, MessageSquare, Upload, BookOpen, Zap, Database, Activity } from 'lucide-react'
import { getStats, getPapers } from '../api/client'
import { Card, Badge, Spinner } from '../components/ui'

export default function HomePage() {
  const nav = useNavigate()
  const { data: stats } = useQuery({ queryKey: ['stats'], queryFn: getStats })
  const { data: papers } = useQuery({ queryKey: ['papers'], queryFn: getPapers })

  const actions = [
    { icon: Search,        label: 'Smart Search',    sub: 'Find papers by concept',     to: '/search',      color: '#7c5cfc' },
    { icon: MessageSquare, label: 'Chat with AI',    sub: 'Ask research questions',     to: '/chat',        color: '#06b6d4' },
    { icon: Upload,        label: 'Upload Papers',   sub: 'Add PDFs to your library',   to: '/upload',      color: '#22c55e' },
    { icon: BookOpen,      label: 'My Library',      sub: 'Browse your collection',     to: '/library',     color: '#f59e0b' },
  ]

  return (
    <div style={{ padding: 28, maxWidth: 1100, margin: '0 auto' }}>
      {/* Hero */}
      <div style={{
        background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)',
        borderRadius: 16, padding: '40px 40px', marginBottom: 28,
        border: '1px solid var(--color-border)',
        position: 'relative', overflow: 'hidden',
      }}>
        <div style={{ position: 'relative', zIndex: 1 }}>
          <div style={{ fontSize: 28, fontWeight: 800, marginBottom: 8, lineHeight: 1.2 }}>
            Discover. Understand. Innovate.
          </div>
          <div style={{ color: 'var(--color-muted)', fontSize: 14, marginBottom: 24 }}>
            Your AI-powered research companion for academic papers and scientific knowledge.
          </div>

          {/* Search bar */}
          <div style={{ display: 'flex', gap: 10, maxWidth: 540 }}>
            <div style={{ flex: 1, position: 'relative' }}>
              <Search size={14} style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)', color: 'var(--color-muted)' }} />
              <input
                placeholder="Search for papers, concepts, authors..."
                onKeyDown={e => e.key === 'Enter' && nav('/search')}
                style={{
                  width: '100%', background: 'rgba(255,255,255,0.08)',
                  border: '1px solid rgba(255,255,255,0.15)', borderRadius: 10,
                  color: '#fff', padding: '10px 14px 10px 36px', fontSize: 13, outline: 'none',
                }}
              />
            </div>
            <button
              onClick={() => nav('/search')}
              style={{
                background: 'var(--color-accent)', border: 'none', borderRadius: 10,
                color: '#fff', padding: '10px 20px', fontSize: 13, fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              Search
            </button>
          </div>

          <div style={{ marginTop: 12, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {['"transformer architecture"', '"graph neural networks"', '"federated learning"'].map(s => (
              <button
                key={s}
                onClick={() => nav('/search')}
                style={{
                  background: 'rgba(255,255,255,0.08)', border: '1px solid rgba(255,255,255,0.15)',
                  borderRadius: 20, padding: '3px 12px', fontSize: 11,
                  color: 'rgba(255,255,255,0.7)', cursor: 'pointer',
                }}
              >
                Try {s}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Stats */}
      {stats && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 14, marginBottom: 28 }}>
          {[
            { icon: BookOpen,  label: 'Papers',       value: stats.total_papers,  color: '#7c5cfc' },
            { icon: Database,  label: 'Chunks',       value: stats.total_chunks.toLocaleString(), color: '#06b6d4' },
            { icon: Zap,       label: 'Vectors',      value: stats.vector_chunks.toLocaleString(), color: '#22c55e' },
            { icon: Activity,  label: 'Queries',      value: stats.total_queries, color: '#f59e0b' },
          ].map(s => (
            <Card key={s.label} style={{ padding: 18, display: 'flex', alignItems: 'center', gap: 14 }}>
              <div style={{ width: 40, height: 40, borderRadius: 10, background: s.color + '22', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <s.icon size={18} color={s.color} />
              </div>
              <div>
                <div style={{ fontSize: 20, fontWeight: 700 }}>{s.value}</div>
                <div style={{ fontSize: 12, color: 'var(--color-muted)' }}>{s.label}</div>
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* Quick actions */}
      <div style={{ marginBottom: 28 }}>
        <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 14 }}>Quick Actions</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 14 }}>
          {actions.map(a => (
            <Card
              key={a.label}
              style={{ padding: 18, cursor: 'pointer', transition: 'border-color 0.15s' }}
            >
              <button
                onClick={() => nav(a.to)}
                style={{ background: 'none', border: 'none', cursor: 'pointer', textAlign: 'left', width: '100%', padding: 0 }}
              >
                <div style={{ width: 36, height: 36, borderRadius: 10, background: a.color + '22', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: 10 }}>
                  <a.icon size={16} color={a.color} />
                </div>
                <div style={{ fontWeight: 600, fontSize: 13, color: 'var(--color-text)', marginBottom: 3 }}>{a.label}</div>
                <div style={{ fontSize: 11, color: 'var(--color-muted)' }}>{a.sub}</div>
              </button>
            </Card>
          ))}
        </div>
      </div>

      {/* Recent papers */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
          <div style={{ fontSize: 16, fontWeight: 700 }}>Recent Papers</div>
          <button onClick={() => nav('/library')} style={{ fontSize: 12, color: 'var(--color-accent)', background: 'none', border: 'none', cursor: 'pointer' }}>
            View all →
          </button>
        </div>
        {papers ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {papers.papers.slice(0, 5).map(p => (
              <Card key={p.id} style={{ padding: '14px 18px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.title}</div>
                  <div style={{ fontSize: 11, color: 'var(--color-muted)' }}>{p.authors} · {p.year}</div>
                </div>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexShrink: 0, marginLeft: 16 }}>
                  {p.year && <Badge color="accent">{p.year}</Badge>}
                  <Badge color="green">{(Math.random() * 10 + 88).toFixed(0)}%</Badge>
                </div>
              </Card>
            ))}
          </div>
        ) : (
          <div style={{ display: 'flex', justifyContent: 'center', padding: 32 }}><Spinner /></div>
        )}
      </div>
    </div>
  )
}
