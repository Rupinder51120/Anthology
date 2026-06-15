import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Search, BookOpen, Calendar, Hash, ExternalLink } from 'lucide-react'
import { getPapers, uploadPaper } from '../api/client'
import { Card, Button, Badge, Input, Empty, Spinner } from '../components/ui'
import type { Paper } from '../api/client'

export default function LibraryPage() {
  const [q, setQ] = useState('')
  const [dragging, setDragging] = useState(false)
  const [uploadProgress, setUploadProgress] = useState<Record<string, number>>({})
  const qc = useQueryClient()

  const { data, isLoading } = useQuery({ queryKey: ['papers'], queryFn: getPapers })

  const uploadMut = useMutation({
    mutationFn: (file: File) => uploadPaper(file, pct => setUploadProgress(p => ({ ...p, [file.name]: pct }))),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['papers'] }); setUploadProgress({}) },
  })

  const papers = (data?.papers ?? []).filter(p =>
    !q || p.title.toLowerCase().includes(q.toLowerCase()) ||
    p.authors.toLowerCase().includes(q.toLowerCase())
  )

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault(); setDragging(false)
    const files = Array.from(e.dataTransfer.files).filter(f => f.name.endsWith('.pdf'))
    files.forEach(f => uploadMut.mutate(f))
  }

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? [])
    files.forEach(f => uploadMut.mutate(f))
  }

  return (
    <div style={{ padding: 28, maxWidth: 1100, margin: '0 auto' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <div>
          <h1 style={{ fontSize: 22, fontWeight: 700 }}>My Library</h1>
          <p style={{ color: 'var(--color-muted)', fontSize: 13, marginTop: 2 }}>
            {data?.total ?? 0} papers indexed
          </p>
        </div>
        <label style={{ cursor: 'pointer' }}>
          <Button variant="primary">+ Upload PDF</Button>
          <input type="file" accept=".pdf" multiple onChange={handleFileInput} style={{ display: 'none' }} />
        </label>
      </div>

      {/* Search */}
      <div style={{ marginBottom: 20, position: 'relative' }}>
        <Search size={14} style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)', color: 'var(--color-muted)' }} />
        <Input value={q} onChange={setQ} placeholder="Search papers, authors..." style={{ paddingLeft: 34 }} />
      </div>

      {/* Drop zone */}
      <div
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        style={{
          border: `2px dashed ${dragging ? 'var(--color-accent)' : 'var(--color-border)'}`,
          borderRadius: 12, padding: '20px', textAlign: 'center',
          marginBottom: 24, transition: 'all 0.2s',
          background: dragging ? 'var(--color-accent-dim)' : 'transparent',
          color: 'var(--color-muted)', fontSize: 13,
        }}
      >
        📄 Drag &amp; drop PDF files here or click Upload PDF above
      </div>

      {/* Upload progress */}
      {Object.keys(uploadProgress).length > 0 && (
        <Card style={{ padding: 16, marginBottom: 20 }}>
          <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 10 }}>Processing Queue</div>
          {Object.entries(uploadProgress).map(([name, pct]) => (
            <div key={name} style={{ marginBottom: 8 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
                <span style={{ color: 'var(--color-muted)' }}>{name}</span>
                <span style={{ color: pct === 100 ? 'var(--color-success)' : 'var(--color-accent)' }}>
                  {pct === 100 ? '✓ Done' : `${pct}%`}
                </span>
              </div>
              <div style={{ height: 4, background: 'var(--color-border)', borderRadius: 2 }}>
                <div style={{ height: '100%', width: `${pct}%`, background: 'var(--color-accent)', borderRadius: 2, transition: 'width 0.3s' }} />
              </div>
            </div>
          ))}
        </Card>
      )}

      {/* Papers grid */}
      {isLoading ? (
        <div style={{ display: 'flex', justifyContent: 'center', padding: 48 }}><Spinner size={32} /></div>
      ) : papers.length === 0 ? (
        <Empty icon="📚" title="No papers yet" sub="Upload your first PDF to get started" />
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 16 }}>
          {papers.map(p => <PaperCard key={p.id} paper={p} />)}
        </div>
      )}
    </div>
  )
}

function PaperCard({ paper }: { paper: Paper }) {
  return (
    <Card style={{ padding: 18, display: 'flex', flexDirection: 'column', gap: 10, transition: 'border-color 0.15s' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 8 }}>
        <div style={{ fontSize: 13, fontWeight: 600, lineHeight: 1.4, color: 'var(--color-text)' }}>
          {paper.title}
        </div>
        <Badge color={paper.indexed ? 'green' : 'yellow'}>
          {paper.indexed ? 'Indexed' : 'Pending'}
        </Badge>
      </div>

      <div style={{ fontSize: 12, color: 'var(--color-muted)', lineHeight: 1.4 }}>
        {paper.authors}
      </div>

      {paper.abstract && (
        <div style={{ fontSize: 12, color: 'var(--color-muted)', lineHeight: 1.5, display: '-webkit-box', WebkitLineClamp: 3, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
          {paper.abstract}
        </div>
      )}

      <div style={{ display: 'flex', gap: 12, alignItems: 'center', fontSize: 11, color: 'var(--color-muted)', marginTop: 4 }}>
        {paper.year && (
          <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <Calendar size={11} /> {paper.year}
          </span>
        )}
        <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <Hash size={11} /> {paper.chunk_count} chunks
        </span>
        {paper.topic && <Badge color="accent">{paper.topic}</Badge>}
      </div>

      <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
        <Button variant="ghost" size="sm" style={{ flex: 1 }}>
          <BookOpen size={12} /> Chat
        </Button>
        {paper.url && (
          <Button variant="ghost" size="sm" onClick={() => window.open(paper.url, '_blank')}>
            <ExternalLink size={12} />
          </Button>
        )}
      </div>
    </Card>
  )
}
