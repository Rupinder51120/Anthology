import { useState, useRef, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useParams, useNavigate, useLocation } from 'react-router-dom'
import {
  ArrowLeft, ExternalLink, Hash, Calendar, FileText,
  MessageSquare, Send, BookOpen, Layers,
} from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { getPaper, streamQueryFetch, listCollections, addPaperToCollection } from '../api/client'
import { Badge, Spinner, Empty } from '../components/ui'
import { glass, glassCard } from '../lib/theme'
import type { Citation } from '../api/client'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  streaming?: boolean
}

function formatResponse(text: string): string {
  return text
    .replace(/##\s*(EXPLANATION|KEY INSIGHT|EVIDENCE FROM PAPERS|Sources Used)/g, '\n\n## $1')
    .replace(/(?<!\n)(EXPLANATION|KEY INSIGHT|EVIDENCE FROM PAPERS|Sources Used)(?!\w)/g, '\n\n## $1\n')
    .replace(/\* ([^\n])/g, '\n* $1')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

export default function PaperView() {
  const { id } = useParams<{ id: string }>()
  const nav = useNavigate()
  const location = useLocation()
  const initialTab = (location.state as { tab?: 'overview' | 'chat' } | null)?.tab ?? 'overview'
  const [tab, setTab] = useState<'overview' | 'chat'>(initialTab)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  const { data: paper, isLoading, error } = useQuery({
    queryKey: ['paper', id],
    queryFn: () => getPaper(id!),
    enabled: !!id,
  })

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const send = async () => {
    const q = input.trim()
    if (!q || streaming || !paper) return
    setInput('')

    const userMsg: Message = { id: crypto.randomUUID(), role: 'user', content: q }
    const asstMsg: Message = { id: crypto.randomUUID(), role: 'assistant', content: '', streaming: true }
    setMessages(prev => [...prev, userMsg, asstMsg])
    setStreaming(true)

    let full = ''
    try {
      await streamQueryFetch(
        q, 5,
        (token) => {
          full += token
          setMessages(prev => prev.map(m =>
            m.id === asstMsg.id ? { ...m, content: full } : m
          ))
        },
        () => {
          setMessages(prev => prev.map(m =>
            m.id === asstMsg.id ? { ...m, streaming: false } : m
          ))
          setStreaming(false)
        },
        paper.id,
      )
    } catch {
      setMessages(prev => prev.map(m =>
        m.id === asstMsg.id
          ? { ...m, content: 'Failed to get a response.', streaming: false }
          : m
      ))
      setStreaming(false)
    }
  }

  if (isLoading) return (
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '60vh' }}>
      <Spinner size={32} />
    </div>
  )

  if (error || !paper) return (
    <div style={{ padding: 32 }}>
      <Empty icon="📄" title="Paper not found" sub="This paper may have been removed." />
    </div>
  )

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' }}>

      {/* Top bar */}
      <div style={{
        ...glass,
        borderRadius: 0, borderLeft: 'none', borderRight: 'none', borderTop: 'none',
        padding: '14px 28px',
        display: 'flex', alignItems: 'center', gap: 16,
        flexShrink: 0,
      }}>
        <button
          onClick={() => nav('/library')}
          style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--color-muted)', display: 'flex', alignItems: 'center', gap: 6, fontSize: 13, fontFamily: 'var(--font-sans)' }}
        >
          <ArrowLeft size={15} /> Library
        </button>
        <div style={{ width: 1, height: 16, background: 'var(--color-border)' }} />
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontWeight: 700, fontSize: 14, color: 'var(--color-text)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {paper.title}
          </div>
        </div>
        {paper.url && (
          <button onClick={() => window.open(paper.url, '_blank')} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--color-muted)' }}>
            <ExternalLink size={15} />
          </button>
        )}
      </div>

      {/* Tabs */}
      <div style={{
        display: 'flex', gap: 4, padding: '12px 28px',
        borderBottom: '1px solid var(--color-border)',
        flexShrink: 0,
      }}>
        {([['overview', BookOpen, 'Overview'], ['chat', MessageSquare, 'Chat']] as const).map(([key, Icon, label]) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            style={{
              display: 'flex', alignItems: 'center', gap: 7,
              padding: '7px 16px', borderRadius: 'var(--radius-md)',
              border: tab === key ? '1px solid var(--color-accent-border)' : '1px solid transparent',
              background: tab === key ? 'var(--color-accent-dim)' : 'transparent',
              color: tab === key ? 'var(--color-accent)' : 'var(--color-muted)',
              fontSize: 13, fontWeight: tab === key ? 600 : 500,
              cursor: 'pointer', fontFamily: 'var(--font-sans)',
              transition: 'all 0.15s',
            }}
          >
            <Icon size={14} /> {label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflow: 'hidden', display: 'flex' }}>

        {tab === 'overview' && (
          <div style={{ flex: 1, overflowY: 'auto', padding: '28px' }}>
            <div style={{ maxWidth: 800, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 20 }}>

              {/* Meta card */}
              <div style={{ ...glassCard }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16, marginBottom: 16 }}>
                  <h1 style={{ fontSize: 20, fontWeight: 800, color: 'var(--color-text)', lineHeight: 1.3, letterSpacing: '-0.02em' }}>
                    {paper.title}
                  </h1>
                  <Badge color={paper.indexed ? 'green' : 'yellow'}>
                    {paper.indexed ? 'Indexed' : 'Pending'}
                  </Badge>
                </div>

                <div style={{ fontSize: 13, color: 'var(--color-muted)', marginBottom: 16 }}>
                  {paper.authors}
                </div>

                <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', fontSize: 12, color: 'var(--color-muted)' }}>
                  {paper.year && (
                    <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                      <Calendar size={12} /> {paper.year}
                    </span>
                  )}
                  <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                    <Hash size={12} /> {paper.chunk_count} chunks
                  </span>
                  <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                    <FileText size={12} /> {paper.filename}
                  </span>
                  {paper.topic && <Badge color="accent">{paper.topic}</Badge>}
                </div>

                {paper.url && (
                  <div style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid var(--color-border)' }}>
                    <a
                      href={paper.url}
                      target="_blank"
                      rel="noreferrer"
                      style={{ fontSize: 12, color: 'var(--color-accent)', display: 'flex', alignItems: 'center', gap: 5, textDecoration: 'none', fontWeight: 600 }}
                    >
                      <ExternalLink size={12} /> View on ArXiv
                    </a>
                  </div>
                )}
              </div>

              {/* Abstract */}
              {paper.abstract && (
                <div style={{ ...glassCard }}>
                  <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--color-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 12 }}>
                    Abstract
                  </div>
                  <p style={{ fontSize: 14, color: 'var(--color-text)', lineHeight: 1.7, opacity: 0.85 }}>
                    {paper.abstract}
                  </p>
                </div>
              )}

              {/* Quick actions */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <button
                  onClick={() => setTab('chat')}
                  style={{
                    ...glassCard,
                    cursor: 'pointer', border: 'var(--glass-border)',
                    display: 'flex', alignItems: 'center', gap: 14,
                    transition: 'transform 0.15s, box-shadow 0.15s',
                  }}
                  onMouseEnter={e => { (e.currentTarget as HTMLElement).style.transform = 'translateY(-2px)'; (e.currentTarget as HTMLElement).style.boxShadow = 'var(--glass-shadow-hover)' }}
                  onMouseLeave={e => { (e.currentTarget as HTMLElement).style.transform = 'translateY(0)'; (e.currentTarget as HTMLElement).style.boxShadow = 'var(--glass-shadow)' }}
                >
                  <div style={{ width: 40, height: 40, borderRadius: 11, background: 'var(--color-blue-dim)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                    <MessageSquare size={18} color="var(--color-blue)" />
                  </div>
                  <div>
                    <div style={{ fontWeight: 700, fontSize: 13, color: 'var(--color-text)' }}>Chat about this paper</div>
                    <div style={{ fontSize: 11, color: 'var(--color-muted)' }}>Ask questions, get explanations</div>
                  </div>
                </button>

                <AddToCollectionWidget paperId={paper.id} />
              </div>
            </div>
          </div>
        )}

        {tab === 'chat' && (
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            {/* Context banner */}
            <div style={{
              padding: '10px 24px',
              background: 'var(--color-accent-dim)',
              borderBottom: '1px solid var(--color-accent-border)',
              fontSize: 12, color: 'var(--color-accent)', fontWeight: 500,
              display: 'flex', alignItems: 'center', gap: 8,
            }}>
              <FileText size={13} />
              Chatting about: <strong>{paper.title.slice(0, 60)}{paper.title.length > 60 ? '…' : ''}</strong>
            </div>

            {/* Messages */}
            <div style={{ flex: 1, overflowY: 'auto', padding: '24px' }}>
              {messages.length === 0 && (
                <PaperChatWelcome title={paper.title} onPrompt={p => setInput(p)} />
              )}
              {messages.map(msg => (
                <PaperChatMessage key={msg.id} msg={msg} />
              ))}
              <div ref={bottomRef} />
            </div>

            {/* Input */}
            <div style={{
              padding: '16px 24px',
              borderTop: '1px solid var(--color-border)',
              ...glass, borderRadius: 0, borderLeft: 'none', borderRight: 'none', borderBottom: 'none',
            }}>
              <div style={{
                display: 'flex', gap: 10,
                background: 'var(--glass-bg)',
                backdropFilter: 'var(--glass-blur)',
                WebkitBackdropFilter: 'var(--glass-blur)',
                border: '1px solid var(--color-border)',
                borderRadius: 12, padding: '8px 12px',
              }}>
                <textarea
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }}
                  placeholder={`Ask about "${paper.title.slice(0, 40)}…"`}
                  rows={1}
                  style={{
                    flex: 1, background: 'transparent', border: 'none',
                    color: 'var(--color-text)', fontSize: 13, resize: 'none',
                    outline: 'none', lineHeight: 1.5, maxHeight: 120,
                    overflowY: 'auto', fontFamily: 'var(--font-sans)',
                  }}
                />
                <button
                  onClick={send}
                  disabled={!input.trim() || streaming}
                  style={{
                    background: input.trim() && !streaming ? 'var(--color-accent)' : 'var(--color-border)',
                    border: 'none', borderRadius: 8, padding: '6px 12px',
                    color: '#fff', cursor: 'pointer', transition: 'all 0.15s',
                    display: 'flex', alignItems: 'center',
                  }}
                >
                  {streaming ? <Spinner size={15} /> : <Send size={15} />}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function PaperChatMessage({ msg }: { msg: Message }) {
  const isUser = msg.role === 'user'
  return (
    <div style={{
      marginBottom: 18,
      display: 'flex', flexDirection: isUser ? 'row-reverse' : 'row',
      gap: 10, alignItems: 'flex-start',
    }}>
      <div style={{
        width: 28, height: 28, borderRadius: '50%', flexShrink: 0,
        background: isUser ? 'var(--color-accent)' : 'var(--glass-bg)',
        border: '1px solid var(--color-border)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: 11, fontWeight: 700, color: isUser ? '#fff' : 'var(--color-muted)',
      }}>
        {isUser ? 'U' : 'A'}
      </div>
      <div style={{ maxWidth: '78%' }}>
        <div style={{
          background: isUser ? 'var(--color-accent)' : 'var(--glass-bg)',
          backdropFilter: isUser ? 'none' : 'var(--glass-blur)',
          WebkitBackdropFilter: isUser ? 'none' : 'var(--glass-blur)',
          border: isUser ? 'none' : 'var(--glass-border)',
          borderRadius: isUser ? '12px 12px 4px 12px' : '12px 12px 12px 4px',
          padding: '12px 16px',
          color: isUser ? '#fff' : 'var(--color-text)',
          fontSize: 13, lineHeight: 1.65,
        }}>
          {isUser ? (
            <span style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</span>
          ) : (
            <div className="prose">
              <ReactMarkdown
                components={{
                  h2: ({ children }) => (
                    <div style={{
                      fontSize: 13, fontWeight: 700, color: 'var(--color-accent)',
                      textTransform: 'uppercase', letterSpacing: '0.08em',
                      marginTop: 16, marginBottom: 8, paddingBottom: 4,
                      borderBottom: '1px solid var(--color-accent-border)',
                      display: 'flex', alignItems: 'center', gap: 6,
                    }}>
                      <div style={{ width: 3, height: 13, background: 'var(--color-accent)', borderRadius: 2 }} />
                      {children}
                    </div>
                  ),
                  p: ({ children }) => (
                    <p style={{ margin: '0 0 10px 0', lineHeight: 1.7 }}>{children}</p>
                  ),
                  strong: ({ children }) => (
                    <strong style={{ color: 'var(--color-text)', fontWeight: 700 }}>{children}</strong>
                  ),
                  ul: ({ children }) => (
                    <ul style={{ margin: '6px 0 10px 0', paddingLeft: 18, display: 'flex', flexDirection: 'column', gap: 3 }}>
                      {children}
                    </ul>
                  ),
                  li: ({ children }) => (
                    <li style={{ lineHeight: 1.6, color: 'var(--color-text)' }}>{children}</li>
                  ),
                }}
              >
                {msg.streaming ? msg.content : formatResponse(msg.content || '▌')}
              </ReactMarkdown>
            </div>
          )}
        </div>
        {msg.citations && msg.citations.length > 0 && (
          <div style={{ marginTop: 6, display: 'flex', gap: 5, flexWrap: 'wrap' }}>
            {msg.citations.slice(0, 3).map((c, i) => (
              <span key={i} style={{
                fontSize: 10, fontWeight: 600, padding: '2px 8px', borderRadius: 6,
                background: 'var(--color-accent-dim)', color: 'var(--color-accent)',
                border: '1px solid var(--color-accent-border)',
              }}>
                {c.section}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function PaperChatWelcome({ title, onPrompt }: { title: string; onPrompt: (p: string) => void }) {
  const short = title.length > 50 ? title.slice(0, 50) + '…' : title
  const prompts = [
    `What is the main contribution of this paper?`,
    `Explain the methodology used`,
    `What are the key results and findings?`,
    `What are the limitations of this work?`,
  ]
  return (
    <div style={{ textAlign: 'center', padding: '40px 24px', maxWidth: 560, margin: '0 auto' }}>
      <div style={{
        width: 56, height: 56, borderRadius: 16,
        background: 'var(--color-blue-dim)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        margin: '0 auto 16px',
      }}>
        <FileText size={26} color="var(--color-blue)" />
      </div>
      <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--color-text)', marginBottom: 6 }}>
        Chat about this paper
      </div>
      <div style={{ fontSize: 13, color: 'var(--color-muted)', marginBottom: 24, lineHeight: 1.5 }}>
        "{short}"
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        {prompts.map(p => (
          <button
            key={p}
            onClick={() => onPrompt(p)}
            style={{
              background: 'var(--glass-bg)',
              backdropFilter: 'var(--glass-blur)',
              WebkitBackdropFilter: 'var(--glass-blur)',
              border: 'var(--glass-border)',
              borderRadius: 10, padding: '10px 14px',
              textAlign: 'left', color: 'var(--color-text)',
              fontSize: 12, cursor: 'pointer',
              transition: 'all 0.15s', lineHeight: 1.4,
              fontFamily: 'var(--font-sans)',
            }}
          >
            {p}
          </button>
        ))}
      </div>
    </div>
  )
}

function AddToCollectionWidget({ paperId }: { paperId: string }) {
  const [open, setOpen] = useState(false)
  const [added, setAdded] = useState<Set<string>>(new Set())
  const qc = useQueryClient()

  const { data: collections = [] } = useQuery({
    queryKey: ['collections'],
    queryFn: listCollections,
    enabled: open,
  })

  const add = useMutation({
    mutationFn: (colId: string) => addPaperToCollection(colId, paperId),
    onSuccess: (_, colId) => {
      setAdded(prev => new Set(prev).add(colId))
      qc.invalidateQueries({ queryKey: ['collections'] })
    },
  })

  return (
    <div style={{ position: 'relative' }}>
      <div
        onClick={() => setOpen(v => !v)}
        style={{
          display: 'flex', alignItems: 'center', gap: 14, cursor: 'pointer',
          padding: '12px 16px', borderRadius: 12,
          background: 'var(--glass-bg)',
          backdropFilter: 'var(--glass-blur)',
          border: open ? '1px solid var(--color-accent)' : 'var(--glass-border)',
          transition: 'all 0.15s',
        }}
      >
        <div style={{ width: 40, height: 40, borderRadius: 11, background: 'var(--color-purple-dim)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
          <Layers size={18} color="var(--color-purple)" />
        </div>
        <div>
          <div style={{ fontWeight: 700, fontSize: 13, color: 'var(--color-text)' }}>Add to Collection</div>
          <div style={{ fontSize: 11, color: 'var(--color-muted)' }}>{open ? 'Select a collection' : 'Organize this paper'}</div>
        </div>
      </div>

      {open && (
        <div style={{
          position: 'absolute', top: '110%', left: 0, right: 0, zIndex: 50,
          background: 'var(--color-surface)', border: '1px solid var(--color-border)',
          borderRadius: 12, overflow: 'hidden', boxShadow: '0 8px 32px rgba(0,0,0,0.2)',
        }}>
          {collections.length === 0 ? (
            <div style={{ padding: '14px 16px', fontSize: 12, color: 'var(--color-muted)', textAlign: 'center' }}>
              No collections yet — create one first
            </div>
          ) : (
            collections.map(col => {
              const isAdded = added.has(col.id)
              return (
                <button
                  key={col.id}
                  onClick={() => !isAdded && add.mutate(col.id)}
                  style={{
                    width: '100%', display: 'flex', alignItems: 'center', gap: 10,
                    padding: '10px 14px', background: isAdded ? col.color + '15' : 'transparent',
                    border: 'none', borderBottom: '1px solid var(--color-border)',
                    cursor: isAdded ? 'default' : 'pointer', fontFamily: 'var(--font-sans)',
                    transition: 'background 0.15s',
                  }}
                  onMouseEnter={e => { if (!isAdded) (e.currentTarget as HTMLElement).style.background = 'var(--color-surface2)' }}
                  onMouseLeave={e => { if (!isAdded) (e.currentTarget as HTMLElement).style.background = 'transparent' }}
                >
                  <div style={{ width: 10, height: 10, borderRadius: '50%', background: col.color, flexShrink: 0 }} />
                  <span style={{ fontSize: 13, color: 'var(--color-text)', flex: 1, textAlign: 'left' }}>{col.name}</span>
                  {isAdded && <span style={{ fontSize: 11, color: col.color, fontWeight: 600 }}>✓ Added</span>}
                </button>
              )
            })
          )}
        </div>
      )}
    </div>
  )
}
