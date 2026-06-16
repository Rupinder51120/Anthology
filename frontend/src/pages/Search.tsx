import { useState } from 'react'
import { Search, FileText, Calendar } from 'lucide-react'
import { useMutation } from '@tanstack/react-query'
import { searchPapers } from '../api/client'
import { Button, Badge, Spinner, Empty } from '../components/ui'
import { glass, glassCard } from '../lib/theme'

export default function SearchPage() {
  const [q, setQ] = useState('')
  const search = useMutation({ mutationFn: () => searchPapers(q, 15) })

  return (
    <div style={{ padding: '32px 36px', maxWidth: 900, margin: '0 auto' }}>
      <div style={{ marginBottom: 28 }}>
        <h1 style={{ fontSize: 22, fontWeight: 800, color: 'var(--color-text)', letterSpacing: '-0.02em', marginBottom: 4 }}>Search Papers</h1>
        <p style={{ color: 'var(--color-muted)', fontSize: 13 }}>Semantic search across all indexed chunks</p>
      </div>

      {/* Search bar */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 24 }}>
        <div style={{ flex: 1, position: 'relative' }}>
          <Search size={15} style={{ position: 'absolute', left: 14, top: '50%', transform: 'translateY(-50%)', color: 'var(--color-muted)', pointerEvents: 'none' }} />
          <input
            value={q}
            onChange={e => setQ(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && q.trim() && search.mutate()}
            placeholder="Search papers, concepts, authors..."
            style={{
              width: '100%', background: 'var(--glass-bg)',
              backdropFilter: 'var(--glass-blur)',
              WebkitBackdropFilter: 'var(--glass-blur)',
              border: 'var(--glass-border)',
              borderRadius: 'var(--radius-md)',
              boxShadow: 'var(--glass-shadow)',
              color: 'var(--color-text)',
              padding: '11px 14px 11px 42px',
              fontSize: 14, outline: 'none',
              fontFamily: 'var(--font-sans)',
            }}
          />
        </div>
        <Button onClick={() => q.trim() && search.mutate()} disabled={!q.trim() || search.isPending}>
          {search.isPending ? <Spinner size={14} /> : <><Search size={14} /> Search</>}
        </Button>
      </div>

      {/* Suggestions */}
      {!search.data && !search.isPending && (
        <div style={{ marginBottom: 24 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--color-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 10 }}>Try searching</div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {['"transformer architecture"', '"graph neural networks"', '"federated learning"', '"attention mechanism"'].map(s => (
              <button
                key={s}
                onClick={() => setQ(s.replace(/"/g, ''))}
                style={{
                  background: 'var(--glass-bg)',
                  backdropFilter: 'var(--glass-blur)',
                  WebkitBackdropFilter: 'var(--glass-blur)',
                  border: 'var(--glass-border)',
                  borderRadius: 20, padding: '5px 14px', fontSize: 12,
                  color: 'var(--color-muted)', cursor: 'pointer',
                  fontFamily: 'var(--font-sans)',
                  transition: 'all 0.15s',
                }}
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      )}

      {search.isPending && (
        <div style={{ display: 'flex', justifyContent: 'center', padding: 48 }}><Spinner size={32} /></div>
      )}

      {search.data && (
        <>
          <div style={{ fontSize: 12, color: 'var(--color-muted)', marginBottom: 14, fontWeight: 500 }}>
            {search.data.total} results for <strong style={{ color: 'var(--color-text)' }}>"{q}"</strong>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {search.data.results.length === 0
              ? <Empty icon="🔍" title="No results" sub="Try different keywords" />
              : search.data.results.map((r, i) => (
                <div key={i} style={{
                  ...glass, borderRadius: 'var(--radius-lg)', padding: 18,
                  transition: 'transform 0.15s',
                }}
                onMouseEnter={e => (e.currentTarget as HTMLElement).style.transform = 'translateX(4px)'}
                onMouseLeave={e => (e.currentTarget as HTMLElement).style.transform = 'translateX(0)'}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
                    <div style={{ fontWeight: 700, fontSize: 14, color: 'var(--color-text)', lineHeight: 1.4 }}>{r.title}</div>
                    {r.score && <Badge color="green">{(r.score * 100).toFixed(0)}%</Badge>}
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--color-muted)', marginBottom: 10, display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                    <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}><FileText size={11} /> {r.authors}</span>
                    {r.year && <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}><Calendar size={11} /> {r.year}</span>}
                    <Badge color="accent">{r.section}</Badge>
                  </div>
                  <div style={{ fontSize: 13, color: 'var(--color-text)', lineHeight: 1.6, opacity: 0.75, borderTop: '1px solid var(--color-border)', paddingTop: 10 }}>
                    {r.text}
                  </div>
                </div>
              ))
            }
          </div>
        </>
      )}
    </div>
  )
}
