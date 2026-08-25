import { useState, useCallback, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Upload, File, CheckCircle, XCircle, Loader, MessageSquare, BookOpen } from 'lucide-react'
import { useQueryClient } from '@tanstack/react-query'
import { uploadPaper } from '../api/client'
import type { IngestResult } from '../api/client'
import { Button } from '../components/ui'
import { glass } from '../lib/theme'

interface UploadItem {
  file: File
  status: 'pending' | 'uploading' | 'processing' | 'done' | 'error'
  progress: number
  processingStartedAt?: number
  result?: IngestResult
  error?: string
}

export default function UploadPage() {
  const [items, setItems] = useState<UploadItem[]>([])
  const [dragging, setDragging] = useState(false)
  const [, forceTick] = useState(0)
  const qc = useQueryClient()
  const nav = useNavigate()

  const update = (name: string, patch: Partial<UploadItem>) =>
    setItems(prev => prev.map(i => i.file.name === name ? { ...i, ...patch } : i))

  // Keep the elapsed-time readout live while anything is being parsed/embedded.
  useEffect(() => {
    if (!items.some(i => i.status === 'processing')) return
    const t = setInterval(() => forceTick(x => x + 1), 1000)
    return () => clearInterval(t)
  }, [items])

  const addFiles = useCallback((files: File[]) => {
    const pdfs = files.filter(f => f.name.endsWith('.pdf'))
    setItems(prev => [
      ...prev,
      ...pdfs.filter(f => !prev.find(i => i.file.name === f.name))
        .map(f => ({ file: f, status: 'pending' as const, progress: 0 })),
    ])
  }, [])

  const uploadAll = async () => {
    const pending = items.filter(i => i.status === 'pending')
    for (const item of pending) {
      update(item.file.name, { status: 'uploading' })
      try {
        const result = await uploadPaper(item.file, pct => {
          if (pct >= 100) {
            // Bytes are on the server now -- everything after this is
            // parsing/enrichment/embedding, which has no progress signal,
            // only elapsed time.
            update(item.file.name, { progress: 100, status: 'processing', processingStartedAt: Date.now() })
          } else {
            update(item.file.name, { progress: pct })
          }
        })
        update(item.file.name, { status: 'done', result, progress: 100 })
        qc.invalidateQueries({ queryKey: ['papers'] })
      } catch (e: unknown) {
        const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
          ?? (e instanceof Error ? e.message : 'Upload failed')
        update(item.file.name, { status: 'error', error: msg })
      }
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault(); setDragging(false)
    addFiles(Array.from(e.dataTransfer.files))
  }

  const pendingCount = items.filter(i => i.status === 'pending').length

  return (
    <div style={{ padding: '32px 36px', maxWidth: 700, margin: '0 auto' }}>
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 22, fontWeight: 800, color: 'var(--color-text)', letterSpacing: '-0.02em', marginBottom: 4 }}>Upload Papers</h1>
        <p style={{ color: 'var(--color-muted)', fontSize: 13 }}>Add PDF papers to your research library</p>
      </div>

      {/* Drop zone */}
      <label>
        <input type="file" accept=".pdf" multiple onChange={e => addFiles(Array.from(e.target.files ?? []))} style={{ display: 'none' }} />
        <div
          onDragOver={e => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          style={{
            border: `2px dashed ${dragging ? 'var(--color-accent)' : 'var(--color-border)'}`,
            borderRadius: 'var(--radius-xl)', padding: '52px 24px',
            textAlign: 'center', cursor: 'pointer',
            transition: 'all 0.2s', marginBottom: 20,
            background: dragging ? 'var(--color-accent-dim)' : 'var(--glass-bg)',
            backdropFilter: 'var(--glass-blur)',
            WebkitBackdropFilter: 'var(--glass-blur)',
          }}
        >
          <Upload
            size={36}
            color={dragging ? 'var(--color-accent)' : 'var(--color-muted)'}
            style={{ margin: '0 auto 14px' }}
          />
          <div style={{ fontWeight: 700, marginBottom: 4, color: 'var(--color-text)', fontSize: 15 }}>
            Drag &amp; drop PDF files here
          </div>
          <div style={{ color: 'var(--color-muted)', fontSize: 13 }}>or click to browse</div>
          <div style={{ color: 'var(--color-subtle)', fontSize: 11, marginTop: 8 }}>Supports PDF only · Max 50MB per file</div>
        </div>
      </label>

      {/* File list */}
      {items.length > 0 && (
        <div style={{ ...glass, borderRadius: 'var(--radius-lg)', marginBottom: 16 }}>
          <div style={{
            padding: '14px 18px', borderBottom: '1px solid var(--color-border)',
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          }}>
            <div style={{ fontWeight: 700, fontSize: 14, color: 'var(--color-text)' }}>Processing Queue</div>
            <div style={{ fontSize: 12, color: 'var(--color-muted)' }}>{items.length} files</div>
          </div>
          <div style={{ padding: '6px 0' }}>
            {items.map(item => (
              <div key={item.file.name} style={{ padding: '10px 18px', borderBottom: '1px solid var(--color-border)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
                  <File size={15} color="var(--color-muted)" />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 13, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: 'var(--color-text)' }}>{item.file.name}</div>
                    <div style={{ fontSize: 11, color: 'var(--color-muted)' }}>{(item.file.size / 1024 / 1024).toFixed(1)} MB</div>
                  </div>
                  <StatusIcon status={item.status} />
                </div>

                {item.status === 'uploading' && (
                  <>
                    <div style={{ fontSize: 11, color: 'var(--color-accent)', marginBottom: 4 }}>Uploading… {item.progress}%</div>
                    <div style={{ height: 3, background: 'var(--color-border)', borderRadius: 2, overflow: 'hidden' }}>
                      <div style={{ height: '100%', width: `${item.progress}%`, background: 'var(--color-accent)', borderRadius: 2, transition: 'width 0.3s' }} />
                    </div>
                  </>
                )}

                {item.status === 'processing' && (
                  <div style={{ fontSize: 11, color: 'var(--color-accent)' }}>
                    Parsing &amp; embedding…{' '}
                    {item.processingStartedAt && `${Math.round((Date.now() - item.processingStartedAt) / 1000)}s elapsed`}
                    <div style={{ color: 'var(--color-subtle)', marginTop: 2 }}>
                      Complex papers with many tables or figures can take several minutes.
                    </div>
                  </div>
                )}

                {item.status === 'done' && item.result && (
                  <div style={{ marginTop: 4 }}>
                    <div style={{ fontSize: 11, color: 'var(--color-success)', marginBottom: 6 }}>
                      ✓ Indexed · {item.result.chunks} chunks · {item.result.figures} figures · {item.result.tables} tables
                    </div>
                    <div style={{ display: 'flex', gap: 8 }}>
                      <Button variant="ghost" size="sm" onClick={() => nav(`/papers/${item.result!.paper_id}`)}>
                        <BookOpen size={12} /> View Paper
                      </Button>
                      <Button variant="ghost" size="sm" onClick={() => nav(`/papers/${item.result!.paper_id}`, { state: { tab: 'chat' } })}>
                        <MessageSquare size={12} /> Chat Now
                      </Button>
                    </div>
                  </div>
                )}

                {item.status === 'error' && (
                  <div style={{ fontSize: 11, color: 'var(--color-danger)', marginTop: 4 }}>✗ {item.error}</div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {pendingCount > 0 && (
        <Button onClick={uploadAll} style={{ width: '100%', justifyContent: 'center' }}>
          <Upload size={14} /> Upload {pendingCount} file{pendingCount > 1 ? 's' : ''}
        </Button>
      )}
    </div>
  )
}

function StatusIcon({ status }: { status: UploadItem['status'] }) {
  if (status === 'done')       return <CheckCircle size={16} color="var(--color-success)" />
  if (status === 'error')      return <XCircle size={16} color="var(--color-danger)" />
  if (status === 'uploading' || status === 'processing') return <Loader size={16} color="var(--color-accent)" style={{ animation: 'spin 1s linear infinite' }} />
  return <div style={{ width: 16, height: 16, borderRadius: '50%', background: 'var(--color-border)' }} />
}
