import { useState, useRef, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Send, Zap, FileText, RotateCcw, BookOpen, Brain, GitBranch, Sparkles, Plus, Trash2, MessageSquare } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { useChatStore } from '../store'
import { streamQueryFetch, createSession, listSessions, addSessionMessage, getSessionMessages, deleteSession, getStats } from '../api/client'
import type { Citation } from '../api/client'
import { Button, Spinner } from '../components/ui'

function formatResponse(text: string): string {
  return text
    .replace(/##\s*(EXPLANATION|KEY INSIGHT|EVIDENCE FROM PAPERS|Sources Used)/g, '\n\n## $1')
    .replace(/(?<!\n)(EXPLANATION|KEY INSIGHT|EVIDENCE FROM PAPERS|Sources Used)(?!\w)/g, '\n\n## $1\n')
    .replace(/\* ([^\n])/g, '\n* $1')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

export default function ChatPage() {
  const [input, setInput] = useState('')
  const [selectedCitations, setSelectedCitations] = useState<Citation[]>([])
  const [status, setStatus] = useState('')
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const qc = useQueryClient()
  const { messages, streaming, addMessage, updateLastMessage, finalizeMessage, setStreaming, clearMessages, setMessages } = useChatStore()

  const { data: sessions = [] } = useQuery({ queryKey: ['sessions'], queryFn: listSessions })
  const { data: stats } = useQuery({ queryKey: ['stats'], queryFn: getStats })

  const newSession = useMutation({
    mutationFn: () => createSession('New Session'),
    onSuccess: (s) => { setActiveSessionId(s.id); clearMessages(); qc.invalidateQueries({ queryKey: ['sessions'] }) },
  })

  const delSession = useMutation({
    mutationFn: deleteSession,
    onSuccess: (_, id) => {
      if (id === activeSessionId) { setActiveSessionId(null); clearMessages() }
      qc.invalidateQueries({ queryKey: ['sessions'] })
    },
  })

  useEffect(() => {
    if (!activeSessionId) return
    getSessionMessages(activeSessionId).then(msgs => {
      setMessages(msgs.map(m => ({ id: m.id, role: m.role as 'user' | 'assistant', content: m.content, streaming: false })))
    })
  }, [activeSessionId])

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  const send = async () => {
    const q = input.trim()
    if (!q || streaming) return
    setInput('')

    let sid = activeSessionId
    if (!sid) {
      const s = await createSession(q.slice(0, 60))
      sid = s.id
      setActiveSessionId(sid)
      qc.invalidateQueries({ queryKey: ['sessions'] })
    }

    addMessage({ id: crypto.randomUUID(), role: 'user', content: q })
    if (sid) addSessionMessage(sid, 'user', q)
    setStreaming(true)

    addMessage({ id: crypto.randomUUID(), role: 'assistant', content: '', streaming: true })
    let full = ''
    await streamQueryFetch(
      q, 5,
      (token) => { full += token; setStatus(''); updateLastMessage(full) },
      () => {
        finalizeMessage({})
        setStreaming(false)
        setStatus('')
        if (sid && full) {
          addSessionMessage(sid, 'assistant', full)
          qc.invalidateQueries({ queryKey: ['sessions'] })
        }
      },
      undefined,
      (s) => setStatus(s),
    )
  }

  const footerNote = stats
    ? `Enter to send · Shift+Enter for new line · ${stats.total_papers} papers indexed`
    : 'Enter to send · Shift+Enter for new line'

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      {/* Sessions sidebar */}
      <div style={{ width: 220, borderRight: '1px solid var(--color-border)', background: 'var(--color-surface)', display: 'flex', flexDirection: 'column', flexShrink: 0 }}>
        <div style={{ padding: '14px 12px 10px', borderBottom: '1px solid var(--color-border)' }}>
          <button
            onClick={() => newSession.mutate()}
            style={{
              width: '100%', display: 'flex', alignItems: 'center', gap: 8,
              background: 'linear-gradient(135deg, #7c5cfc22, #06b6d422)',
              border: '1px solid var(--color-accent)44',
              borderRadius: 10, padding: '8px 12px', cursor: 'pointer',
              color: 'var(--color-accent)', fontSize: 12, fontWeight: 700,
              fontFamily: 'var(--font-sans)',
            }}
          >
            <Plus size={13} /> New Chat
          </button>
        </div>
        <div style={{ flex: 1, overflowY: 'auto', padding: '8px' }}>
          {sessions.length === 0 && (
            <div style={{ fontSize: 11, color: 'var(--color-muted)', textAlign: 'center', padding: '20px 8px' }}>No sessions yet</div>
          )}
          {sessions.map(s => (
            <div
              key={s.id}
              onClick={() => setActiveSessionId(s.id)}
              style={{
                display: 'flex', alignItems: 'center', gap: 6,
                padding: '8px 10px', borderRadius: 8, cursor: 'pointer',
                background: s.id === activeSessionId ? 'var(--color-accent)18' : 'transparent',
                border: s.id === activeSessionId ? '1px solid var(--color-accent)33' : '1px solid transparent',
                marginBottom: 2, transition: 'all 0.15s',
              }}
              onMouseEnter={e => { if (s.id !== activeSessionId) (e.currentTarget as HTMLElement).style.background = 'var(--color-surface2)' }}
              onMouseLeave={e => { if (s.id !== activeSessionId) (e.currentTarget as HTMLElement).style.background = 'transparent' }}
            >
              <MessageSquare size={11} color={s.id === activeSessionId ? 'var(--color-accent)' : 'var(--color-muted)'} style={{ flexShrink: 0 }} />
              <span style={{ fontSize: 12, flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: s.id === activeSessionId ? 'var(--color-text)' : 'var(--color-muted)' }}>
                {s.title}
              </span>
              <button
                onClick={e => { e.stopPropagation(); delSession.mutate(s.id) }}
                style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--color-muted)', padding: 2, display: 'flex', flexShrink: 0 }}
              >
                <Trash2 size={10} />
              </button>
            </div>
          ))}
        </div>
        {stats && (
          <div style={{ padding: '10px 14px', borderTop: '1px solid var(--color-border)', fontSize: 10, color: 'var(--color-muted)' }}>
            {stats.total_papers} papers · {stats.total_chunks.toLocaleString()} chunks
          </div>
        )}
      </div>

      {/* Main chat */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        <div style={{ padding: '14px 24px', borderBottom: '1px solid var(--color-border)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: 'var(--color-surface)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{ width: 36, height: 36, borderRadius: 10, background: 'linear-gradient(135deg, #7c5cfc, #06b6d4)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Brain size={18} color="#fff" />
            </div>
            <div>
              <div style={{ fontWeight: 700, fontSize: 15 }}>Research Chat</div>
              <div style={{ fontSize: 11, color: 'var(--color-muted)' }}>
                {activeSessionId ? (sessions.find(s => s.id === activeSessionId)?.title || 'Session') : 'AI-powered research assistant'}
              </div>
            </div>
          </div>
          <Button variant="ghost" size="sm" onClick={() => { clearMessages(); setActiveSessionId(null) }}>
            <RotateCcw size={13} /> Clear
          </Button>
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: '24px 28px' }}>
          {messages.length === 0 && <WelcomeScreen onPrompt={p => setInput(p)} paperCount={stats?.total_papers} />}
          {messages.map(msg => (
            <MessageBubble key={msg.id} msg={msg} onCitationClick={setSelectedCitations} />
          ))}
          {streaming && messages[messages.length - 1]?.role !== 'assistant' && (
            <div style={{ display: 'flex', gap: 10, alignItems: 'center', color: 'var(--color-muted)', fontSize: 13, marginBottom: 20 }}>
              <div style={{ width: 32, height: 32, borderRadius: '50%', background: 'var(--color-surface2)', border: '1px solid var(--color-border)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Spinner size={14} />
              </div>
              <div style={{ background: 'var(--color-surface2)', border: '1px solid var(--color-border)', borderRadius: '12px 12px 12px 4px', padding: '10px 16px', display: 'flex', alignItems: 'center', gap: 8 }}>
                <Sparkles size={13} color="var(--color-accent)" />
                {status || 'Searching papers...'}
              </div>
            </div>
          )}
          {streaming && status && messages[messages.length - 1]?.role === 'assistant' && messages[messages.length - 1]?.content === '' && (
            <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 8, paddingLeft: 44 }}>
              <div style={{ fontSize: 11, color: 'var(--color-accent)', display: 'flex', alignItems: 'center', gap: 5 }}>
                <Sparkles size={10} color="var(--color-accent)" /> {status}
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        <div style={{ padding: '16px 24px', borderTop: '1px solid var(--color-border)', background: 'var(--color-surface)' }}>
          <div style={{ display: 'flex', gap: 10, background: 'var(--color-surface2)', border: `1px solid ${input ? 'var(--color-accent)' : 'var(--color-border)'}`, borderRadius: 14, padding: '10px 14px', transition: 'border-color 0.2s' }}>
            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }}
              placeholder="Ask anything about your research papers..."
              rows={1}
              style={{ flex: 1, background: 'transparent', border: 'none', color: 'var(--color-text)', fontSize: 13, resize: 'none', outline: 'none', lineHeight: 1.6, maxHeight: 120, overflowY: 'auto' }}
            />
            <button
              onClick={send}
              disabled={!input.trim() || streaming}
              style={{
                background: input.trim() && !streaming ? 'linear-gradient(135deg, #7c5cfc, #6b4ef0)' : 'var(--color-border)',
                border: 'none', borderRadius: 10, padding: '6px 14px', color: '#fff',
                cursor: input.trim() && !streaming ? 'pointer' : 'not-allowed',
                transition: 'all 0.2s', display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, fontWeight: 600,
              }}
            >
              {streaming ? <Spinner size={14} /> : <><Send size={14} /> Send</>}
            </button>
          </div>
          <div style={{ fontSize: 11, color: 'var(--color-muted)', marginTop: 8, textAlign: 'center' }}>{footerNote}</div>
        </div>
      </div>

      {selectedCitations.length > 0 && <CitationsPanel citations={selectedCitations} onClose={() => setSelectedCitations([])} />}
    </div>
  )
}

function MessageBubble({ msg, onCitationClick }: {
  msg: { id: string; role: string; content: string; citations?: Citation[]; latency_ms?: number; streaming?: boolean }
  onCitationClick: (c: Citation[]) => void
}) {
  const isUser = msg.role === 'user'
  if (isUser) {
    return (
      <div style={{ marginBottom: 24, display: 'flex', justifyContent: 'flex-end' }}>
        <div style={{ maxWidth: '70%', background: 'linear-gradient(135deg, #7c5cfc, #6b4ef0)', borderRadius: '16px 16px 4px 16px', padding: '12px 16px', color: '#fff', fontSize: 13, lineHeight: 1.6, boxShadow: '0 4px 20px #7c5cfc33' }}>
          {msg.content}
        </div>
      </div>
    )
  }

  const renderedContent = msg.streaming ? msg.content : formatResponse(msg.content)

  return (
    <div style={{ marginBottom: 28, display: 'flex', gap: 12, alignItems: 'flex-start' }}>
      <div style={{ width: 32, height: 32, borderRadius: '50%', flexShrink: 0, background: 'linear-gradient(135deg, #7c5cfc22, #06b6d422)', border: '1px solid var(--color-accent)44', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Brain size={15} color="var(--color-accent)" />
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ background: 'var(--color-surface2)', border: '1px solid var(--color-border)', borderRadius: '4px 16px 16px 16px', padding: '16px 20px', color: 'var(--color-text)', fontSize: 13, lineHeight: 1.7 }}>
          <div className="markdown-body">
            <ReactMarkdown
              components={{
                h2: ({ children }) => (
                  <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--color-accent)', textTransform: 'uppercase', letterSpacing: '0.08em', marginTop: 20, marginBottom: 8, paddingBottom: 4, borderBottom: '1px solid var(--color-accent)33', display: 'flex', alignItems: 'center', gap: 6 }}>
                    <div style={{ width: 3, height: 14, background: 'var(--color-accent)', borderRadius: 2 }} />
                    {children}
                  </div>
                ),
                h3: ({ children }) => <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--color-text)', marginTop: 14, marginBottom: 6 }}>{children}</div>,
                p: ({ children }) => <p style={{ margin: '0 0 12px 0', lineHeight: 1.7 }}>{children}</p>,
                strong: ({ children }) => <strong style={{ color: 'var(--color-text)', fontWeight: 700 }}>{children}</strong>,
                ul: ({ children }) => <ul style={{ margin: '8px 0 12px 0', paddingLeft: 20, display: 'flex', flexDirection: 'column', gap: 4 }}>{children}</ul>,
                li: ({ children }) => <li style={{ lineHeight: 1.6, color: 'var(--color-text)' }}>{children}</li>,
                code: ({ children }) => <code style={{ background: 'var(--color-bg)', border: '1px solid var(--color-border)', borderRadius: 4, padding: '1px 6px', fontSize: 12, fontFamily: 'monospace', color: 'var(--color-accent)' }}>{children}</code>,
                blockquote: ({ children }) => <blockquote style={{ borderLeft: '3px solid var(--color-accent)', margin: '12px 0', paddingLeft: 14, color: 'var(--color-muted)', fontStyle: 'italic' }}>{children}</blockquote>,
              }}
            >
              {renderedContent}
            </ReactMarkdown>
            {msg.streaming && <span style={{ display: 'inline-block', width: 2, height: 14, background: 'var(--color-accent)', marginLeft: 2, animation: 'blink 1s infinite', verticalAlign: 'middle' }} />}
          </div>
        </div>
        {msg.citations && msg.citations.length > 0 && (
          <div style={{ marginTop: 10, display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
            <span style={{ fontSize: 11, color: 'var(--color-muted)', display: 'flex', alignItems: 'center', gap: 4 }}>
              <BookOpen size={11} /> Sources:
            </span>
            {msg.citations.slice(0, 4).map((c, i) => (
              <button key={i} onClick={() => onCitationClick(msg.citations!)} style={{ background: 'var(--color-surface)', border: '1px solid var(--color-accent)44', borderRadius: 20, padding: '3px 12px', fontSize: 11, color: 'var(--color-accent)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 5 }}>
                <FileText size={10} />
                {c.title.slice(0, 30)}{c.title.length > 30 ? '…' : ''}
                {c.year && <span style={{ color: 'var(--color-muted)' }}>· {c.year}</span>}
              </button>
            ))}
            {msg.citations.length > 4 && (
              <button onClick={() => onCitationClick(msg.citations!)} style={{ background: 'transparent', border: '1px solid var(--color-border)', borderRadius: 20, padding: '3px 10px', fontSize: 11, color: 'var(--color-muted)', cursor: 'pointer' }}>
                +{msg.citations.length - 4} more
              </button>
            )}
          </div>
        )}
        {msg.latency_ms && (
          <div style={{ fontSize: 10, color: 'var(--color-muted)', marginTop: 6, display: 'flex', alignItems: 'center', gap: 4 }}>
            <Zap size={10} /> {msg.latency_ms.toFixed(0)}ms
          </div>
        )}
      </div>
    </div>
  )
}

function CitationsPanel({ citations, onClose }: { citations: Citation[]; onClose: () => void }) {
  return (
    <div style={{ width: 340, borderLeft: '1px solid var(--color-border)', background: 'var(--color-surface)', overflowY: 'auto', padding: 20, display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <div style={{ fontWeight: 700, fontSize: 14, display: 'flex', alignItems: 'center', gap: 8 }}>
          <BookOpen size={15} color="var(--color-accent)" /> Sources ({citations.length})
        </div>
        <button onClick={onClose} style={{ background: 'none', border: 'none', color: 'var(--color-muted)', cursor: 'pointer', fontSize: 20, lineHeight: 1 }}>×</button>
      </div>
      {citations.map((c, i) => (
        <div key={i} style={{ background: 'var(--color-surface2)', border: '1px solid var(--color-border)', borderRadius: 12, padding: 14, borderLeft: '3px solid var(--color-accent)' }}>
          <div style={{ fontSize: 11, color: 'var(--color-accent)', fontWeight: 600, marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.06em' }}>[{i + 1}] {c.section}</div>
          <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--color-text)', marginBottom: 4, lineHeight: 1.4 }}>{c.title}</div>
          <div style={{ fontSize: 11, color: 'var(--color-muted)', marginBottom: 8 }}>{c.authors} · {c.year}</div>
          {c.score !== null && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <div style={{ flex: 1, height: 3, background: 'var(--color-border)', borderRadius: 2 }}>
                <div style={{ height: '100%', width: `${Math.min(c.score * 100, 100)}%`, background: 'var(--color-accent)', borderRadius: 2 }} />
              </div>
              <span style={{ fontSize: 10, color: 'var(--color-muted)' }}>{(c.score * 100).toFixed(0)}%</span>
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

function WelcomeScreen({ onPrompt, paperCount }: { onPrompt: (p: string) => void; paperCount?: number }) {
  const categories = [
    { icon: Brain, color: '#7c5cfc', label: 'Explain Concepts', prompts: ['What is retrieval augmented generation?', 'Explain the transformer architecture'] },
    { icon: GitBranch, color: '#06b6d4', label: 'Compare Papers', prompts: ['Compare attention mechanisms across papers', 'What are differences between GAN variants?'] },
    { icon: BookOpen, color: '#22c55e', label: 'Deep Dive', prompts: ['What are the key findings on federated learning?', 'Summarize recent work on graph neural networks'] },
    { icon: Sparkles, color: '#f59e0b', label: 'Discover', prompts: ['What papers discuss self-supervised learning?', 'Find papers on vision transformers'] },
  ]
  return (
    <div style={{ maxWidth: 700, margin: '0 auto', padding: '32px 0' }}>
      <div style={{ textAlign: 'center', marginBottom: 40 }}>
        <div style={{ width: 64, height: 64, borderRadius: 20, margin: '0 auto 16px', background: 'linear-gradient(135deg, #7c5cfc22, #06b6d422)', border: '1px solid var(--color-accent)33', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Brain size={28} color="var(--color-accent)" />
        </div>
        <div style={{ fontSize: 24, fontWeight: 800, marginBottom: 8, letterSpacing: '-0.02em' }}>What would you like to explore?</div>
        <div style={{ color: 'var(--color-muted)', fontSize: 14, lineHeight: 1.6 }}>
          {paperCount ? `Ask questions across ${paperCount} indexed papers.` : 'Ask questions across your indexed library.'}
        </div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
        {categories.map(cat => (
          <div key={cat.label} style={{ background: 'var(--color-surface)', border: '1px solid var(--color-border)', borderRadius: 14, overflow: 'hidden' }}>
            <div style={{ padding: '12px 16px', borderBottom: '1px solid var(--color-border)', display: 'flex', alignItems: 'center', gap: 8, background: cat.color + '11' }}>
              <cat.icon size={14} color={cat.color} />
              <span style={{ fontSize: 12, fontWeight: 700, color: cat.color }}>{cat.label}</span>
            </div>
            <div style={{ padding: 8, display: 'flex', flexDirection: 'column', gap: 4 }}>
              {cat.prompts.map(p => (
                <button key={p} onClick={() => onPrompt(p)}
                  style={{ background: 'transparent', border: 'none', borderRadius: 8, padding: '8px 10px', textAlign: 'left', color: 'var(--color-muted)', fontSize: 12, cursor: 'pointer', lineHeight: 1.4 }}
                  onMouseEnter={e => { (e.target as HTMLElement).style.background = 'var(--color-surface2)'; (e.target as HTMLElement).style.color = 'var(--color-text)' }}
                  onMouseLeave={e => { (e.target as HTMLElement).style.background = 'transparent'; (e.target as HTMLElement).style.color = 'var(--color-muted)' }}
                >→ {p}</button>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
