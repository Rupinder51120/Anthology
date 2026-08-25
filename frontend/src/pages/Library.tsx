import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { Search, BookOpen, Calendar, Hash, ExternalLink } from 'lucide-react'
import { getPapers } from '../api/client'
import { Button, Badge, Input, Empty, Spinner } from '../components/ui'
import { glass } from '../lib/theme'
import type { Paper } from '../api/client'

export default function LibraryPage() {
  const [q, setQ] = useState('')
  const nav = useNavigate()
  const { data, isLoading } = useQuery({ queryKey: ['papers'], queryFn: getPapers })
  const papers = (data?.papers ?? []).filter(p =>
    !q || p.title.toLowerCase().includes(q.toLowerCase()) ||
    p.authors.toLowerCase().includes(q.toLowerCase())
  )
  return (
    <div style={{ padding: '32px 36px', maxWidth: 1100, margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <div>
          <h1 style={{ fontSize: 22, fontWeight: 800, color: 'var(--color-text)', letterSpacing: '-0.02em' }}>My Library</h1>
          <p style={{ color: 'var(--color-muted)', fontSize: 13, marginTop: 2 }}>{data?.total ?? 0} papers indexed</p>
        </div>
        <Button variant="primary" onClick={() => nav('/upload')}>+ Upload PDF</Button>
      </div>
      <div style={{ marginBottom: 20, position: 'relative' }}>
        <Search size={14} style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)', color: 'var(--color-muted)', pointerEvents: 'none' }} />
        <Input value={q} onChange={setQ} placeholder="Filter by title or author..." style={{ paddingLeft: 34 }} />
      </div>
      {isLoading ? (
        <div style={{ display: 'flex', justifyContent: 'center', padding: 48 }}><Spinner size={32} /></div>
      ) : papers.length === 0 ? (
        <Empty icon="📚" title="No papers yet" sub="Upload your first PDF to get started" />
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 14 }}>
          {papers.map(p => <PaperCard key={p.id} paper={p} />)}
        </div>
      )}
    </div>
  )
}

function PaperCard({ paper }: { paper: Paper }) {
  const nav = useNavigate()
  return (
    <div
      onClick={() => nav(`/papers/${paper.id}`)}
      style={{ ...glass, borderRadius: 'var(--radius-lg)', padding: 18, display: 'flex', flexDirection: 'column', gap: 10, cursor: 'pointer', transition: 'transform 0.15s, box-shadow 0.15s' }}
      onMouseEnter={e => { (e.currentTarget as HTMLElement).style.transform = 'translateY(-2px)'; (e.currentTarget as HTMLElement).style.boxShadow = 'var(--glass-shadow-hover)' }}
      onMouseLeave={e => { (e.currentTarget as HTMLElement).style.transform = 'translateY(0)'; (e.currentTarget as HTMLElement).style.boxShadow = 'var(--glass-shadow)' }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 8 }}>
        <div style={{ fontSize: 13, fontWeight: 600, lineHeight: 1.4, color: 'var(--color-text)' }}>{paper.title}</div>
        <Badge color={paper.indexed ? 'green' : 'yellow'}>{paper.indexed ? 'Indexed' : 'Pending'}</Badge>
      </div>
      <div style={{ fontSize: 12, color: 'var(--color-muted)', lineHeight: 1.4 }}>{paper.authors}</div>
      {paper.abstract && <div style={{ fontSize: 12, color: 'var(--color-muted)', lineHeight: 1.5, display: '-webkit-box', WebkitLineClamp: 3, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>{paper.abstract}</div>}
      <div style={{ display: 'flex', gap: 10, alignItems: 'center', fontSize: 11, color: 'var(--color-muted)', marginTop: 2 }}>
        {paper.year && <span style={{ display: 'flex', alignItems: 'center', gap: 3 }}><Calendar size={11} /> {paper.year}</span>}
        <span style={{ display: 'flex', alignItems: 'center', gap: 3 }}><Hash size={11} /> {paper.chunk_count} chunks</span>
        {paper.topic && <Badge color="accent">{paper.topic}</Badge>}
      </div>
      <div style={{ display: 'flex', gap: 8, marginTop: 2 }} onClick={e => e.stopPropagation()}>
        <Button variant="ghost" size="sm" style={{ flex: 1 }} onClick={() => nav(`/papers/${paper.id}`)}>
          <BookOpen size={12} /> Chat
        </Button>
        {paper.url && <Button variant="ghost" size="sm" onClick={() => window.open(paper.url, '_blank')}><ExternalLink size={12} /></Button>}
      </div>
    </div>
  )
}
