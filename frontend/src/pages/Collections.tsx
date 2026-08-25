import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Plus, Trash2, BookOpen, X, Check } from 'lucide-react'
import { listCollections, createCollection, deleteCollection, getCollectionPapers, addPaperToCollection, removePaperFromCollection, getPapers } from '../api/client'
import { Button, Spinner, Empty } from '../components/ui'

const COLORS = ['#7c5cfc', '#06b6d4', '#22c55e', '#f59e0b', '#ef4444', '#ec4899', '#8b5cf6', '#14b8a6']

export default function CollectionsPage() {
  const [creating, setCreating] = useState(false)
  const [newName, setNewName] = useState('')
  const [newDesc, setNewDesc] = useState('')
  const [newColor, setNewColor] = useState(COLORS[0])
  const [activeCol, setActiveCol] = useState<string | null>(null)
  const [addingPaper, setAddingPaper] = useState(false)
  const qc = useQueryClient()

  const { data: collections = [], isLoading } = useQuery({ queryKey: ['collections'], queryFn: listCollections })
  const { data: colPapers = [] } = useQuery({
    queryKey: ['collection-papers', activeCol],
    queryFn: () => getCollectionPapers(activeCol!),
    enabled: !!activeCol,
  })
  const { data: allPapersData } = useQuery({
    queryKey: ['papers'],
    queryFn: getPapers,
    enabled: addingPaper,
  })
  const allPapers = allPapersData?.papers ?? []

  const createCol = useMutation({
    mutationFn: () => createCollection(newName.trim(), newDesc.trim() || undefined, newColor),
    onSuccess: () => { setCreating(false); setNewName(''); setNewDesc(''); qc.invalidateQueries({ queryKey: ['collections'] }) },
  })

  const delCol = useMutation({
    mutationFn: deleteCollection,
    onSuccess: (_, id) => { if (id === activeCol) setActiveCol(null); qc.invalidateQueries({ queryKey: ['collections'] }) },
  })

  const addPaper = useMutation({
    mutationFn: (paperId: string) => { console.log('[DIAG] addPaper mutationFn called', { activeCol, paperId }); return addPaperToCollection(activeCol!, paperId) },
    onSuccess: (data) => { console.log('[DIAG] addPaper SUCCESS', data); qc.invalidateQueries({ queryKey: ['collection-papers', activeCol] }); qc.invalidateQueries({ queryKey: ['collections'] }) },
    onError: (err) => console.error('[DIAG] addPaper ERROR', err),
  })

  const removePaper = useMutation({
    mutationFn: (paperId: string) => { console.log('[DIAG] removePaper mutationFn called', { activeCol, paperId }); return removePaperFromCollection(activeCol!, paperId) },
    onSuccess: (data) => { console.log('[DIAG] removePaper SUCCESS', data); qc.invalidateQueries({ queryKey: ['collection-papers', activeCol] }); qc.invalidateQueries({ queryKey: ['collections'] }) },
    onError: (err) => console.error('[DIAG] removePaper ERROR', err),
  })

  const activeCollection = collections.find(c => c.id === activeCol)
  const colPaperIds = new Set(colPapers.map(p => p.id))

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      {/* Left: collection list */}
      <div style={{ width: 280, borderRight: '1px solid var(--color-border)', background: 'var(--color-surface)', display: 'flex', flexDirection: 'column' }}>
        <div style={{ padding: '20px 16px 12px', borderBottom: '1px solid var(--color-border)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ fontWeight: 800, fontSize: 15 }}>Collections</div>
          <Button size="sm" onClick={() => setCreating(true)}>
            <Plus size={13} /> New
          </Button>
        </div>

        {/* Create form */}
        {creating && (
          <div style={{ padding: '12px 16px', borderBottom: '1px solid var(--color-border)', background: 'var(--color-surface2)' }}>
            <input
              value={newName}
              onChange={e => setNewName(e.target.value)}
              placeholder="Collection name"
              autoFocus
              style={{ width: '100%', background: 'var(--color-bg)', border: '1px solid var(--color-border)', borderRadius: 8, padding: '7px 10px', fontSize: 13, color: 'var(--color-text)', outline: 'none', marginBottom: 8, boxSizing: 'border-box' }}
            />
            <input
              value={newDesc}
              onChange={e => setNewDesc(e.target.value)}
              placeholder="Description (optional)"
              style={{ width: '100%', background: 'var(--color-bg)', border: '1px solid var(--color-border)', borderRadius: 8, padding: '7px 10px', fontSize: 12, color: 'var(--color-text)', outline: 'none', marginBottom: 10, boxSizing: 'border-box' }}
            />
            <div style={{ display: 'flex', gap: 6, marginBottom: 10 }}>
              {COLORS.map(c => (
                <button key={c} onClick={() => setNewColor(c)} style={{ width: 20, height: 20, borderRadius: '50%', background: c, border: c === newColor ? '2px solid white' : '2px solid transparent', cursor: 'pointer', outline: c === newColor ? `2px solid ${c}` : 'none' }} />
              ))}
            </div>
            <div style={{ display: 'flex', gap: 6 }}>
              <Button size="sm" onClick={() => createCol.mutate()} disabled={!newName.trim() || createCol.isPending}>
                {createCol.isPending ? <Spinner size={12} /> : <><Check size={12} /> Create</>}
              </Button>
              <Button variant="ghost" size="sm" onClick={() => setCreating(false)}><X size={12} /> Cancel</Button>
            </div>
          </div>
        )}

        <div style={{ flex: 1, overflowY: 'auto', padding: 8 }}>
          {isLoading && <div style={{ padding: 20, textAlign: 'center' }}><Spinner size={20} /></div>}
          {!isLoading && collections.length === 0 && (
            <Empty icon="📁" title="No collections" sub="Create one to organize your papers" />
          )}
          {collections.map(col => (
            <div
              key={col.id}
              onClick={() => setActiveCol(col.id)}
              style={{
                display: 'flex', alignItems: 'center', gap: 10, padding: '10px 12px',
                borderRadius: 10, cursor: 'pointer', marginBottom: 2,
                background: col.id === activeCol ? col.color + '18' : 'transparent',
                border: col.id === activeCol ? `1px solid ${col.color}44` : '1px solid transparent',
                transition: 'all 0.15s',
              }}
              onMouseEnter={e => { if (col.id !== activeCol) (e.currentTarget as HTMLElement).style.background = 'var(--color-surface2)' }}
              onMouseLeave={e => { if (col.id !== activeCol) (e.currentTarget as HTMLElement).style.background = 'transparent' }}
            >
              <div style={{ width: 10, height: 10, borderRadius: '50%', background: col.color, flexShrink: 0 }} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--color-text)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{col.name}</div>
                <div style={{ fontSize: 11, color: 'var(--color-muted)' }}>{col.paper_count} papers</div>
              </div>
              <button onClick={e => { e.stopPropagation(); delCol.mutate(col.id) }} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--color-muted)', padding: 2 }}>
                <Trash2 size={11} />
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Right: papers in collection */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        {!activeCol ? (
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Empty icon="📂" title="Select a collection" sub="Click a collection to view its papers" />
          </div>
        ) : (
          <>
            <div style={{ padding: '16px 24px', borderBottom: '1px solid var(--color-border)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: 'var(--color-surface)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <div style={{ width: 12, height: 12, borderRadius: '50%', background: activeCollection?.color }} />
                <div>
                  <div style={{ fontWeight: 700, fontSize: 15 }}>{activeCollection?.name}</div>
                  {activeCollection?.description && <div style={{ fontSize: 12, color: 'var(--color-muted)' }}>{activeCollection.description}</div>}
                </div>
              </div>
              <Button size="sm" onClick={() => setAddingPaper(v => !v)}>
                <Plus size={13} /> Add Papers
              </Button>
            </div>

            {/* Add papers panel */}
            {addingPaper && (
              <div style={{ padding: '12px 24px', borderBottom: '1px solid var(--color-border)', background: 'var(--color-surface2)', maxHeight: 260, overflowY: 'auto' }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--color-muted)', marginBottom: 8 }}>Click to add/remove from collection</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                  {allPapers.map(p => {
                    const inCol = colPaperIds.has(p.id)
                    return (
                      <button
                        key={p.id}
                        onClick={() => { console.log('[DIAG] row clicked', { id: p.id, inCol, activeCol }); inCol ? removePaper.mutate(p.id) : addPaper.mutate(p.id) }}
                        style={{
                          display: 'flex', alignItems: 'center', gap: 10, padding: '8px 12px',
                          background: inCol ? (activeCollection?.color ?? '#7c5cfc') + '18' : 'var(--color-bg)',
                          border: `1px solid ${inCol ? (activeCollection?.color ?? '#7c5cfc') + '44' : 'var(--color-border)'}`,
                          borderRadius: 8, cursor: 'pointer', textAlign: 'left',
                          fontFamily: 'var(--font-sans)', transition: 'all 0.15s',
                        }}
                      >
                        <div style={{ width: 16, height: 16, borderRadius: 4, background: inCol ? activeCollection?.color : 'var(--color-border)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                          {inCol && <Check size={10} color="#fff" />}
                        </div>
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--color-text)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.title}</div>
                          <div style={{ fontSize: 11, color: 'var(--color-muted)' }}>{p.authors} · {p.year}</div>
                        </div>
                      </button>
                    )
                  })}
                </div>
              </div>
            )}

            <div style={{ flex: 1, overflowY: 'auto', padding: '20px 24px' }}>
              {colPapers.length === 0 ? (
                <Empty icon="📄" title="No papers yet" sub="Click 'Add Papers' to add papers to this collection" />
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {colPapers.map(p => (
                    <div key={p.id} style={{ background: 'var(--color-surface)', border: '1px solid var(--color-border)', borderRadius: 12, padding: '14px 16px', display: 'flex', alignItems: 'center', gap: 12, borderLeft: `3px solid ${activeCollection?.color}` }}>
                      <BookOpen size={16} color={activeCollection?.color} style={{ flexShrink: 0 }} />
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--color-text)', marginBottom: 2 }}>{p.title}</div>
                        <div style={{ fontSize: 11, color: 'var(--color-muted)' }}>{p.authors} · {p.year}</div>
                      </div>
                      <button onClick={() => removePaper.mutate(p.id)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--color-muted)', padding: 4 }}>
                        <X size={13} />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
