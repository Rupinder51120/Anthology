import { useState, useCallback } from 'react'
import { Upload, File, CheckCircle, XCircle, Loader } from 'lucide-react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { uploadPaper } from '../api/client'
import { Card, Button } from '../components/ui'

interface UploadItem {
  file: File
  status: 'pending' | 'uploading' | 'done' | 'error'
  progress: number
  result?: Record<string, unknown>
  error?: string
}

export default function UploadPage() {
  const [items, setItems] = useState<UploadItem[]>([])
  const [dragging, setDragging] = useState(false)
  const qc = useQueryClient()

  const update = (name: string, patch: Partial<UploadItem>) =>
    setItems(prev => prev.map(i => i.file.name === name ? { ...i, ...patch } : i))

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
        const result = await uploadPaper(item.file, pct => update(item.file.name, { progress: pct }))
        update(item.file.name, { status: 'done', result, progress: 100 })
        qc.invalidateQueries({ queryKey: ['papers'] })
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : 'Upload failed'
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
    <div style={{ padding: 28, maxWidth: 700, margin: '0 auto' }}>
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 22, fontWeight: 700, marginBottom: 4 }}>Upload Papers</h1>
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
            borderRadius: 16, padding: '48px 24px', textAlign: 'center',
            cursor: 'pointer', transition: 'all 0.2s',
            background: dragging ? 'var(--color-accent-dim)' : 'var(--color-surface)',
            marginBottom: 20,
          }}
        >
          <Upload size={36} color={dragging ? 'var(--color-accent)' : 'var(--color-muted)'} style={{ margin: '0 auto 12px' }} />
          <div style={{ fontWeight: 600, marginBottom: 4 }}>Drag &amp; drop PDF files here</div>
          <div style={{ color: 'var(--color-muted)', fontSize: 13 }}>or click to browse</div>
          <div style={{ color: 'var(--color-muted)', fontSize: 11, marginTop: 8 }}>Supports PDF only · Max 50MB per file</div>
        </div>
      </label>

      {/* File list */}
      {items.length > 0 && (
        <Card style={{ marginBottom: 20 }}>
          <div style={{ padding: '14px 18px', borderBottom: '1px solid var(--color-border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ fontWeight: 600, fontSize: 14 }}>Processing Queue</div>
            <div style={{ fontSize: 12, color: 'var(--color-muted)' }}>{items.length} files</div>
          </div>
          <div style={{ padding: '8px 0' }}>
            {items.map(item => (
              <div key={item.file.name} style={{ padding: '10px 18px', borderBottom: '1px solid var(--color-border)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: item.status === 'uploading' ? 6 : 0 }}>
                  <File size={16} color="var(--color-muted)" />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 13, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{item.file.name}</div>
                    <div style={{ fontSize: 11, color: 'var(--color-muted)' }}>{(item.file.size / 1024 / 1024).toFixed(1)} MB</div>
                  </div>
                  <StatusIcon status={item.status} />
                </div>
                {item.status === 'uploading' && (
                  <div style={{ height: 3, background: 'var(--color-border)', borderRadius: 2, overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${item.progress}%`, background: 'var(--color-accent)', borderRadius: 2, transition: 'width 0.3s' }} />
                  </div>
                )}
                {item.status === 'done' && item.result && (
                  <div style={{ fontSize: 11, color: 'var(--color-success)', marginTop: 4 }}>
                    ✓ Ingested · {String(item.result.chunks ?? 0)} chunks · {String(item.result.figures ?? 0)} figures · {String(item.result.tables ?? 0)} tables
                  </div>
                )}
                {item.status === 'error' && (
                  <div style={{ fontSize: 11, color: 'var(--color-danger)', marginTop: 4 }}>✗ {item.error}</div>
                )}
              </div>
            ))}
          </div>
        </Card>
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
  if (status === 'done')     return <CheckCircle size={16} color="var(--color-success)" />
  if (status === 'error')    return <XCircle size={16} color="var(--color-danger)" />
  if (status === 'uploading') return <Loader size={16} color="var(--color-accent)" style={{ animation: 'spin 1s linear infinite' }} />
  return <div style={{ width: 16, height: 16, borderRadius: '50%', background: 'var(--color-border)' }} />
}
