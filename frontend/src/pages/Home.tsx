import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import {
  Search, MessageSquare, Upload, BookOpen, Compass,
  Zap, Database, Activity, ArrowRight, ArrowUpRight, Sparkles
} from 'lucide-react'
import { getStats, getPapers } from '../api/client'
import { Spinner } from '../components/ui'
import { glass, glassCard, hoverLift, hoverSlide } from '../lib/theme'
import MoodToggle from '../components/ui/MoodToggle'
import type { StatsResponse } from '../api/client'

const actions = [
  { icon: Compass,       label: 'Search Research',  sub: 'Find new papers on arXiv + OpenAlex', to: '/discover', color: 'var(--color-pink)',   bg: 'var(--color-pink-dim)',   grad: 'linear-gradient(135deg,rgba(244,114,182,0.15),rgba(244,114,182,0.04))' },
  { icon: Search,        label: 'Search My Papers', sub: 'Search your indexed research corpus', to: '/search',   color: 'var(--color-blue)',   bg: 'var(--color-blue-dim)',   grad: 'linear-gradient(135deg,rgba(91,158,249,0.15),rgba(91,158,249,0.04))' },
  { icon: MessageSquare, label: 'Chat',             sub: 'Ask research questions',               to: '/chat',     color: 'var(--color-green)',  bg: 'var(--color-green-dim)',  grad: 'linear-gradient(135deg,rgba(52,211,153,0.15),rgba(52,211,153,0.04))' },
  { icon: Upload,        label: 'Upload',           sub: 'Add PDFs to your library',              to: '/upload',   color: 'var(--color-orange)', bg: 'var(--color-orange-dim)', grad: 'linear-gradient(135deg,rgba(251,191,36,0.15),rgba(251,191,36,0.04))' },
  { icon: BookOpen,      label: 'Library',          sub: 'Browse your collection',                to: '/library',  color: 'var(--color-purple)', bg: 'var(--color-purple-dim)', grad: 'linear-gradient(135deg,rgba(167,139,250,0.15),rgba(167,139,250,0.04))' },
]

const statDef = [
  { icon: BookOpen, label: 'Papers',  key: 'total_papers',  fmt: (v: number) => v.toString(),         color: 'var(--color-blue)',   bg: 'var(--color-blue-dim)'   },
  { icon: Database, label: 'Chunks',  key: 'total_chunks',  fmt: (v: number) => v.toLocaleString(),   color: 'var(--color-green)',  bg: 'var(--color-green-dim)'  },
  { icon: Zap,      label: 'Vectors', key: 'vector_chunks', fmt: (v: number) => v.toLocaleString(),   color: 'var(--color-orange)', bg: 'var(--color-orange-dim)' },
  { icon: Activity, label: 'Queries', key: 'total_queries', fmt: (v: number) => v.toString(),         color: 'var(--color-purple)', bg: 'var(--color-purple-dim)' },
] satisfies Array<{
  icon: typeof BookOpen
  label: string
  key: keyof Pick<StatsResponse, 'total_papers'|'total_chunks'|'vector_chunks'|'total_queries'>
  fmt: (v: number) => string
  color: string
  bg: string
}>

const chips = ['"transformer architecture"', '"graph neural networks"', '"federated learning"']

function HeroIllustration() {
  return (
    <svg
      width="190" height="170" viewBox="0 0 200 180"
      style={{ position: 'absolute', right: 20, bottom: -14, opacity: 0.95, pointerEvents: 'none' }}
    >
      <defs>
        <linearGradient id="paperGrad" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" style={{ stopColor: 'var(--mood-accent)' }} stopOpacity={0.92} />
          <stop offset="100%" style={{ stopColor: 'var(--mood-accent2)' }} stopOpacity={0.75} />
        </linearGradient>
      </defs>
      {/* back paper */}
      <g transform="rotate(-8 100 90)">
        <rect x="40" y="20" width="110" height="140" rx="14" fill="white" stroke="var(--color-border)" />
        <line x1="58" y1="50" x2="132" y2="50" stroke="var(--color-border)" strokeWidth="3" strokeLinecap="round" />
        <line x1="58" y1="64" x2="132" y2="64" stroke="var(--color-border)" strokeWidth="3" strokeLinecap="round" />
      </g>
      {/* front paper — accent gradient */}
      <g transform="rotate(6 100 90)">
        <rect x="46" y="30" width="108" height="138" rx="14" fill="url(#paperGrad)" />
        <line x1="64" y1="60" x2="130" y2="60" stroke="rgba(255,255,255,0.6)" strokeWidth="3" strokeLinecap="round" />
        <line x1="64" y1="74" x2="118" y2="74" stroke="rgba(255,255,255,0.6)" strokeWidth="3" strokeLinecap="round" />
        <line x1="64" y1="88" x2="124" y2="88" stroke="rgba(255,255,255,0.45)" strokeWidth="3" strokeLinecap="round" />
      </g>
      {/* sparkle badge */}
      <circle cx="150" cy="38" r="20" fill="white" stroke="var(--color-border)" />
      <path d="M150 28l3 7 7 3-7 3-3 7-3-7-7-3 7-3z" fill="var(--mood-accent)" />
    </svg>
  )
}

export default function HomePage() {
  const nav = useNavigate()
  const { data: stats } = useQuery({ queryKey: ['stats'], queryFn: getStats })
  const { data: papers } = useQuery({ queryKey: ['papers'], queryFn: getPapers })

  return (
    <div style={{ padding: '32px 36px', maxWidth: 1100, margin: '0 auto' }}>

      {/* ── Hero ── */}
      <div style={{
        ...glass,
        borderRadius: 'var(--radius-xl)',
        padding: '48px 48px 40px',
        marginBottom: 20,
        position: 'relative',
        overflow: 'hidden',
        background: 'linear-gradient(135deg, rgba(255,255,255,0.82) 0%, rgba(255,255,255,0.65) 100%)',
      }}>
        {/* Floating mood orbs */}
        <div style={{
          position: 'absolute', top: -80, right: -40, width: 280, height: 280,
          borderRadius: '50%',
          background: 'radial-gradient(circle, var(--mood-orb1) 0%, transparent 70%)',
          animation: 'floatOrb 7s ease-in-out infinite',
          pointerEvents: 'none',
        }} />
        <div style={{
          position: 'absolute', bottom: -60, left: 60, width: 200, height: 200,
          borderRadius: '50%',
          background: 'radial-gradient(circle, var(--mood-orb2) 0%, transparent 70%)',
          animation: 'floatOrb 9s ease-in-out infinite reverse',
          pointerEvents: 'none',
        }} />
        <HeroIllustration />

        <div style={{ position: 'relative', zIndex: 1 }}>
          {/* Badge row + mood toggle */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
            <div style={{
              display: 'inline-flex', alignItems: 'center', gap: 6,
              background: 'var(--mood-badge-bg)',
              border: '1px solid var(--mood-badge-border)',
              borderRadius: 'var(--radius-pill)',
              padding: '4px 14px', fontSize: 11, fontWeight: 600,
              color: 'var(--mood-accent)', letterSpacing: '0.04em',
            }}>
              <Sparkles size={11} /> AI Research Assistant
            </div>
            <MoodToggle />
          </div>

          {/* Headline */}
          <div style={{
            fontSize: 38, fontWeight: 800, lineHeight: 1.12,
            color: 'var(--color-text)', letterSpacing: '-0.025em', marginBottom: 12,
          }}>
            Discover. Understand.<br />
            <span style={{
              background: 'var(--mood-btn-grad)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
            }}>Innovate.</span>
          </div>

          <div style={{ color: 'var(--color-muted)', fontSize: 15, marginBottom: 24, maxWidth: 520, lineHeight: 1.6 }}>
            Discover new research from arXiv and OpenAlex, or search and chat with the papers already in your library.
          </div>

          {/* Two explicit entry points — no single ambiguous search bar */}
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            <button
              onClick={() => nav('/discover')}
              className="btn-mood"
              style={{
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '12px 22px', fontSize: 13.5, fontWeight: 600,
                borderRadius: 'var(--radius-pill)',
              }}
            >
              <Compass size={15} /> Search Research
              <span style={{ fontWeight: 400, opacity: 0.85, fontSize: 12 }}>· arXiv + OpenAlex</span>
            </button>
            <button
              onClick={() => nav('/search')}
              style={{
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '12px 22px', fontSize: 13.5, fontWeight: 600,
                borderRadius: 'var(--radius-pill)',
                background: 'rgba(255,255,255,0.85)',
                backdropFilter: 'blur(20px)',
                WebkitBackdropFilter: 'blur(20px)',
                border: '1.5px solid var(--mood-badge-border)',
                color: 'var(--color-text)',
                cursor: 'pointer',
                fontFamily: 'var(--font-sans)',
              }}
            >
              <Search size={15} color="var(--mood-accent)" /> Search My Papers
              <span style={{ fontWeight: 400, color: 'var(--color-muted)', fontSize: 12 }}>· your indexed corpus</span>
            </button>
          </div>

          {/* Chip suggestions — external research topics */}
          <div style={{ marginTop: 16, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {chips.map(s => (
              <button key={s} onClick={() => nav('/discover')} style={{
                background: 'rgba(255,255,255,0.6)',
                backdropFilter: 'blur(8px)',
                border: '1px solid var(--color-border)',
                borderRadius: 'var(--radius-pill)',
                padding: '4px 14px', fontSize: 11,
                color: 'var(--color-muted)', cursor: 'pointer',
                fontFamily: 'var(--font-sans)',
                transition: 'all 0.15s',
              }}
                onMouseEnter={e => {
                  e.currentTarget.style.borderColor = 'var(--mood-accent)'
                  e.currentTarget.style.color = 'var(--mood-accent)'
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.borderColor = 'var(--color-border)'
                  e.currentTarget.style.color = 'var(--color-muted)'
                }}
              >
                Try {s}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* ── Stats bento ── */}
      {stats && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 20 }}>
          {statDef.map(s => (
            <div key={s.label} style={{
              ...glassCard,
              display: 'flex', alignItems: 'center', gap: 14,
              transition: 'transform 0.2s, box-shadow 0.2s',
              background: 'rgba(255,255,255,0.78)',
            }}
              {...hoverLift}
            >
              <div style={{
                width: 46, height: 46, borderRadius: '50%',
                background: `radial-gradient(circle at 35% 30%, ${s.color}2e, ${s.color}10 70%)`,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                flexShrink: 0,
                boxShadow: `0 0 0 1px ${s.bg}, 0 6px 16px ${s.bg}`,
              }}>
                <s.icon size={20} color={s.color} />
              </div>
              <div>
                <div style={{ fontSize: 24, fontWeight: 800, color: 'var(--color-text)', letterSpacing: '-0.03em', lineHeight: 1 }}>
                  {s.fmt(stats[s.key])}
                </div>
                <div style={{ fontSize: 11, color: 'var(--color-muted)', fontWeight: 500, marginTop: 3 }}>{s.label}</div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── Quick Actions ── */}
      <div style={{ marginBottom: 24 }}>
        <div style={{
          fontSize: 11, fontWeight: 700, color: 'var(--color-muted)',
          textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 12,
        }}>
          Quick Actions
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 12 }}>
          {actions.map(a => (
            <button
              key={a.label}
              onClick={() => nav(a.to)}
              style={{
                ...glassCard,
                position: 'relative',
                cursor: 'pointer', textAlign: 'left',
                display: 'flex', flexDirection: 'column', gap: 14,
                background: a.grad,
                border: '1px solid rgba(255,255,255,0.7)',
                transition: 'transform 0.2s, box-shadow 0.2s',
                paddingTop: 22,
              }}
              {...hoverLift}
            >
              <div style={{
                position: 'absolute', top: 14, right: 14,
                width: 28, height: 28, borderRadius: '50%',
                background: 'rgba(255,255,255,0.92)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
                border: '1px solid rgba(255,255,255,0.9)',
              }}>
                <ArrowUpRight size={14} color={a.color} />
              </div>

              <div style={{
                width: 50, height: 50, borderRadius: '50%',
                background: `radial-gradient(circle at 35% 30%, ${a.color}2e, ${a.color}10 70%)`,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                boxShadow: `0 0 0 1px ${a.bg}, 0 8px 22px ${a.bg}`,
              }}>
                <a.icon size={20} color={a.color} />
              </div>
              <div>
                <div style={{ fontWeight: 700, fontSize: 13, color: 'var(--color-text)', marginBottom: 3 }}>{a.label}</div>
                <div style={{ fontSize: 11, color: 'var(--color-muted)', lineHeight: 1.4 }}>{a.sub}</div>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* ── Recent Papers ── */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--color-muted)', textTransform: 'uppercase', letterSpacing: '0.07em' }}>
            Recent Papers
          </div>
          <button onClick={() => nav('/library')} style={{
            fontSize: 12, color: 'var(--mood-accent)', background: 'none',
            border: 'none', cursor: 'pointer', fontWeight: 600,
            display: 'flex', alignItems: 'center', gap: 4,
            fontFamily: 'var(--font-sans)',
          }}>
            View all <ArrowRight size={12} />
          </button>
        </div>

        {papers ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {papers.papers.slice(0, 5).map((p, i) => (
              <div
                key={p.id}
                onClick={() => nav(`/papers/${p.id}`)}
                style={{
                  ...glass,
                  borderRadius: 'var(--radius-md)',
                  padding: '14px 18px',
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  cursor: 'pointer',
                  transition: 'transform 0.15s, box-shadow 0.15s',
                  animation: `fadeUp 0.3s ease ${i * 0.05}s both`,
                }}
                {...hoverSlide}
              >
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{
                    fontWeight: 600, fontSize: 13, marginBottom: 3,
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                    color: 'var(--color-text)',
                  }}>{p.title}</div>
                  <div style={{ fontSize: 11, color: 'var(--color-muted)' }}>{p.authors} · {p.year}</div>
                </div>
                <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexShrink: 0, marginLeft: 16 }}>
                  {p.year && (
                    <span style={{
                      fontSize: 11, fontWeight: 600,
                      color: 'var(--mood-accent)',
                      background: 'var(--mood-badge-bg)',
                      border: '1px solid var(--mood-badge-border)',
                      borderRadius: 'var(--radius-sm)', padding: '2px 8px',
                    }}>{p.year}</span>
                  )}
                  <span style={{
                    fontSize: 11, fontWeight: 600,
                    color: 'var(--color-success)',
                    background: 'var(--color-success-dim)',
                    border: '1px solid rgba(52,211,153,0.22)',
                    borderRadius: 'var(--radius-sm)', padding: '2px 8px',
                  }}>Indexed</span>
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
