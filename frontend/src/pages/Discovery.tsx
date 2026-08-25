import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Search, ExternalLink, BookOpen, Users, Calendar, TrendingUp, Zap, Plus, MessageSquare } from 'lucide-react'
import { useMutation } from '@tanstack/react-query'
import { Button, Spinner, Empty, Badge } from '../components/ui'
import { glass, glassCard } from '../lib/theme'
import { discoverPapers, addExternalPaper } from '../api/client'
import type { DiscoveryPaper, IngestResult } from '../api/client'

const SUGGESTED = [
  'retrieval augmented generation',
  'vision transformers',
  'graph neural networks',
  'diffusion models',
  'federated learning',
  'large language models',
]

export default function DiscoveryPage() {
  const [q, setQ] = useState('')
  const [tab, setTab] = useState<'combined' | 'arxiv' | 's2'>('combined')

  const search = useMutation({ mutationFn: () => discoverPapers(q.trim()) })

  const results = search.data
    ? (tab === 'combined' ? search.data.combined : tab === 'arxiv' ? search.data.arxiv : search.data.s2)
    : []

  return (
    <div style={{ padding: '32px 36px', maxWidth: 960, margin: '0 auto' }}>
      {/* Header */}
      <div style={{ marginBottom: 28 }}>
        <h1 style={{ fontSize: 22, fontWeight: 800, color: 'var(--color-text)', letterSpacing: '-0.02em', marginBottom: 4 }}>
          Search
        </h1>
        <p style={{ color: 'var(--color-muted)', fontSize: 13 }}>
          Discover research from arXiv and OpenAlex
        </p>
      </div>

      {/* Search bar */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 16 }}>
        <div style={{ flex: 1, position: 'relative' }}>
          <Search size={15} style={{ position: 'absolute', left: 14, top: '50%', transform: 'translateY(-50%)', color: 'var(--color-muted)', pointerEvents: 'none' }} />
          <input
            value={q}
            onChange={e => setQ(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && q.trim() && search.mutate()}
            placeholder="Search for papers e.g. 'attention mechanisms in transformers'..."
            style={{
              width: '100%',
              background: 'var(--glass-bg)',
              backdropFilter: 'var(--glass-blur)',
              WebkitBackdropFilter: 'var(--glass-blur)',
              border: 'var(--glass-border)',
              borderRadius: 'var(--radius-md)',
              boxShadow: 'var(--glass-shadow)',
              color: 'var(--color-text)',
              padding: '12px 14px 12px 42px',
              fontSize: 14, outline: 'none',
              fontFamily: 'var(--font-sans)',
            }}
          />
        </div>
        <Button onClick={() => q.trim() && search.mutate()} disabled={!q.trim() || search.isPending}>
          {search.isPending ? <Spinner size={14} /> : <><Zap size={14} /> Discover</>}
        </Button>
      </div>

      {/* Suggestions */}
      {!search.data && !search.isPending && (
        <div style={{ marginBottom: 32 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--color-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 10 }}>
            Trending topics
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {SUGGESTED.map(s => (
              <button
                key={s}
                onClick={() => { setQ(s); }}
                style={{
                  background: 'var(--glass-bg)',
                  backdropFilter: 'var(--glass-blur)',
                  WebkitBackdropFilter: 'var(--glass-blur)',
                  border: 'var(--glass-border)',
                  borderRadius: 20, padding: '5px 14px', fontSize: 12,
                  color: 'var(--color-muted)', cursor: 'pointer',
                  fontFamily: 'var(--font-sans)', transition: 'all 0.15s',
                }}
                onMouseEnter={e => { (e.currentTarget as HTMLElement).style.color = 'var(--color-accent)'; (e.currentTarget as HTMLElement).style.borderColor = 'var(--color-accent)' }}
                onMouseLeave={e => { (e.currentTarget as HTMLElement).style.color = 'var(--color-muted)'; (e.currentTarget as HTMLElement).style.borderColor = '' }}
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      )}

      {search.isPending && (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12, padding: 64 }}>
          <Spinner size={32} />
          <div style={{ fontSize: 13, color: 'var(--color-muted)' }}>Searching ArXiv + OpenAlex...</div>
        </div>
      )}

      {search.data && (
        <>
          {/* Stats bar */}
          <div style={{ ...glassCard, display: 'flex', gap: 24, alignItems: 'center', marginBottom: 20, padding: '12px 20px' }}>
            <div style={{ fontSize: 13, color: 'var(--color-muted)' }}>
              Found <strong style={{ color: 'var(--color-text)' }}>{search.data.total}</strong> papers for <strong style={{ color: 'var(--color-accent)' }}>"{search.data.query}"</strong>
            </div>
            <div style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
              <span style={{ fontSize: 11, color: 'var(--color-muted)' }}>ArXiv: {search.data.arxiv.length}</span>
              <span style={{ color: 'var(--color-border)' }}>·</span>
              <span style={{ fontSize: 11, color: 'var(--color-muted)' }}>OpenAlex: {search.data.s2.length}</span>
            </div>
          </div>

          {/* Tabs */}
          <div style={{ display: 'flex', gap: 4, marginBottom: 20, background: 'var(--glass-bg)', borderRadius: 'var(--radius-md)', padding: 4, border: 'var(--glass-border)', width: 'fit-content' }}>
            {(['combined', 'arxiv', 's2'] as const).map(t => (
              <button
                key={t}
                onClick={() => setTab(t)}
                style={{
                  padding: '6px 16px', borderRadius: 'var(--radius-sm)', border: 'none',
                  background: tab === t ? 'var(--color-accent)' : 'transparent',
                  color: tab === t ? '#fff' : 'var(--color-muted)',
                  fontSize: 12, fontWeight: 600, cursor: 'pointer',
                  fontFamily: 'var(--font-sans)', transition: 'all 0.15s',
                }}
              >
                {t === 'combined' ? 'All' : t === 'arxiv' ? 'ArXiv' : 'OpenAlex'}
              </button>
            ))}
          </div>

          {/* Results */}
          {results.length === 0
            ? <Empty icon="🔍" title="No results" sub="Try a different query" />
            : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                {results.map((p, i) => (
                  <PaperCard key={`${p.arxiv_id ?? p.s2_id ?? i}`} paper={p} />
                ))}
              </div>
            )
          }
        </>
      )}
    </div>
  )
}

type AddState = 'idle' | 'adding' | 'processing' | 'done' | 'error'

function PaperCard({ paper }: { paper: DiscoveryPaper }) {
  const nav = useNavigate()
  const [state, setState] = useState<AddState>('idle')
  const [result, setResult] = useState<IngestResult | null>(null)
  const [error, setError] = useState('')
  const [startedAt, setStartedAt] = useState<number | null>(null)
  const [, forceTick] = useState(0)

  // Live elapsed-time readout while we're waiting on ingestion.
  useEffect(() => {
    if (state !== 'processing') return
    const t = setInterval(() => forceTick(x => x + 1), 1000)
    return () => clearInterval(t)
  }, [state])

  const handleAdd = async () => {
    if (!paper.pdf_url) return
    setState('adding')
    setError('')
    setStartedAt(Date.now())
    // The download step is fast; anything past a few seconds is really the
    // parse/embed pipeline. No real phase signal exists for a single
    // synchronous request, so this is an honest best-effort label switch,
    // not a fake progress bar.
    const toProcessing = setTimeout(() => setState('processing'), 3500)
    try {
      const r = await addExternalPaper(paper.pdf_url, paper.title)
      clearTimeout(toProcessing)
      setResult(r)
      setState('done')
    } catch (e: unknown) {
      clearTimeout(toProcessing)
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(detail ?? 'Could not add this paper. Try again.')
      setState('error')
    }
  }

  return (
    <div style={{
      ...glass,
      borderRadius: 'var(--radius-lg)', padding: 18,
      transition: 'transform 0.15s',
    }}
    onMouseEnter={e => (e.currentTarget as HTMLElement).style.transform = 'translateX(3px)'}
    onMouseLeave={e => (e.currentTarget as HTMLElement).style.transform = 'translateX(0)'}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 12, marginBottom: 8 }}>
        <div style={{ fontWeight: 700, fontSize: 14, color: 'var(--color-text)', lineHeight: 1.4, flex: 1 }}>
          {paper.title}
        </div>
        <Badge color={paper.source === 'arxiv' ? 'accent' : 'green'}>
          {paper.source === 'arxiv' ? 'ArXiv' : 'OpenAlex'}
        </Badge>
      </div>

      <div style={{ display: 'flex', gap: 16, fontSize: 12, color: 'var(--color-muted)', marginBottom: 10, flexWrap: 'wrap' }}>
        {paper.authors && (
          <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <Users size={11} /> {paper.authors}
          </span>
        )}
        {paper.year && (
          <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <Calendar size={11} /> {paper.year}
          </span>
        )}
        {paper.citation_count != null && paper.citation_count > 0 && (
          <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <TrendingUp size={11} /> {paper.citation_count.toLocaleString()} citations
          </span>
        )}
      </div>

      {paper.abstract && (
        <div style={{
          fontSize: 13, color: 'var(--color-text)', lineHeight: 1.6,
          opacity: 0.75, marginBottom: 12,
          display: '-webkit-box', WebkitLineClamp: 3,
          WebkitBoxOrient: 'vertical', overflow: 'hidden',
        }}>
          {paper.abstract}
        </div>
      )}

      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
        {state === 'idle' && (
          <Button
            size="sm"
            onClick={handleAdd}
            disabled={!paper.pdf_url}
            style={!paper.pdf_url ? { opacity: 0.5, cursor: 'not-allowed' } : undefined}
          >
            <Plus size={12} /> {paper.pdf_url ? 'Add to Anthology' : 'No PDF available'}
          </Button>
        )}
        {state === 'adding' && (
          <Button size="sm" disabled><Spinner size={12} /> Adding…</Button>
        )}
        {state === 'processing' && (
          <Button size="sm" disabled>
            <Spinner size={12} /> Processing…{startedAt ? ` ${Math.round((Date.now() - startedAt) / 1000)}s` : ''}
          </Button>
        )}
        {state === 'done' && result && (
          <>
            <Badge color="green">Added ✓</Badge>
            <Button variant="ghost" size="sm" onClick={() => nav(`/papers/${result.paper_id}`)}>
              <BookOpen size={12} /> View Paper
            </Button>
            <Button variant="ghost" size="sm" onClick={() => nav(`/papers/${result.paper_id}`, { state: { tab: 'chat' } })}>
              <MessageSquare size={12} /> Chat Now
            </Button>
          </>
        )}
        {state === 'error' && (
          <>
            <span style={{ fontSize: 11.5, color: 'var(--color-danger)' }}>{error}</span>
            <Button variant="ghost" size="sm" onClick={handleAdd}>Retry</Button>
          </>
        )}

        <a href={paper.url} target="_blank" rel="noopener noreferrer" style={{ textDecoration: 'none' }}>
          <Button variant="ghost" size="sm">
            <BookOpen size={12} /> View
          </Button>
        </a>
        {paper.pdf_url && (
          <a href={paper.pdf_url} target="_blank" rel="noopener noreferrer" style={{ textDecoration: 'none' }}>
            <Button variant="ghost" size="sm">
              <ExternalLink size={12} /> PDF
            </Button>
          </a>
        )}
      </div>
    </div>
  )
}
