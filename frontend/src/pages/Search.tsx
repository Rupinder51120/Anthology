import { useState } from 'react'
import { Search, FileText, Calendar } from 'lucide-react'
import { useMutation } from '@tanstack/react-query'
import { searchPapers } from '../api/client'
import { Card, Button, Badge, Spinner, Empty } from '../components/ui'

export default function SearchPage() {
  const [q, setQ] = useState('')

  const search = useMutation({ mutationFn: () => searchPapers(q, 15) })

  return (
    <div style={{ padding: 28, maxWidth: 900, margin: '0 auto' }}>
      <div style={{ marginBottom: 28 }}>
        <h1 style={{ fontSize: 22, fontWeight: 700, marginBottom: 4 }}>Search Papers</h1>
        <p style={{ color: 'var(--color-muted)', fontSize: 13 }}>Semantic search across all indexed chunks</p>
      </div>

      {/* Search bar */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 28 }}>
        <div style={{ flex: 1, position: 'relative' }}>
          <Search size={15} style={{ position: 'absolute', left: 14, top: '50%', transform: 'translateY(-50%)', color: 'var(--color-muted)' }} />
          <input
            value={q}
            onChange={e => setQ(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && q.trim() && search.mutate()}
            placeholder="Search papers, concepts, authors..."
            style={{
              width: '100%', background: 'var(--color-surface)',
              border: '1px solid var(--color-border)', borderRadius: 10,
              color: 'var(--color-text)', padding: '11px 14px 11px 40px',
              fontSize: 14, outline: 'none',
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
          <div style={{ fontSize: 12, color: 'var(--color-muted)', marginBottom: 10 }}>Try:</div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {['"transformer architecture"', '"graph neural networks"', '"federated learning"', '"attention mechanism"'].map(s => (
              <button
                key={s}
                onClick={() => { setQ(s.replace(/"/g, '')); }}
                style={{
                  background: 'var(--color-surface)', border: '1px solid var(--color-border)',
                  borderRadius: 20, padding: '4px 14px', fontSize: 12,
                  color: 'var(--color-muted)', cursor: 'pointer',
                }}
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Results */}
      {search.isPending && (
        <div style={{ display: 'flex', justifyContent: 'center', padding: 48 }}><Spinner size={32} /></div>
      )}

      {search.data && (
        <>
          <div style={{ fontSize: 13, color: 'var(--color-muted)', marginBottom: 16 }}>
            {search.data.total} results found
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {search.data.results.length === 0
              ? <Empty icon="🔍" title="No results" sub="Try different keywords" />
              : search.data.results.map((r, i) => (
                <Card key={i} style={{ padding: 18 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 6 }}>
                    <div style={{ fontWeight: 600, fontSize: 14, color: 'var(--color-text)', lineHeight: 1.4 }}>{r.title}</div>
                    {r.score && (
                      <Badge color="green">{(r.score * 100).toFixed(0)}%</Badge>
                    )}
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--color-muted)', marginBottom: 8, display: 'flex', gap: 12 }}>
                    <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}><FileText size={11} /> {r.authors}</span>
                    {r.year && <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}><Calendar size={11} /> {r.year}</span>}
                    <Badge color="accent">{r.section}</Badge>
                  </div>
                  <div style={{ fontSize: 13, color: 'var(--color-text)', lineHeight: 1.6, opacity: 0.8 }}>
                    {r.text}
                  </div>
                </Card>
              ))
            }
          </div>
        </>
      )}
    </div>
  )
}
