import axios from 'axios'

const api = axios.create({ baseURL: '/api/v1' })

// ── Types ────────────────────────────────────────────────────

export interface Citation {
  title: string
  authors: string
  year: number | null
  section: string
  filename: string
  score: number | null
}

export interface QueryResponse {
  question: string
  answer: string
  citations: Citation[]
  chunks_used: number
  response_type: string
  tokens_used: number
  latency_ms: number
  query_id: string
}

export interface Paper {
  id: string
  title: string
  authors: string
  year: number | null
  abstract: string
  topic: string
  filename: string
  chunk_count: number
  indexed: boolean
  created_at: string
  url?: string
  arxiv_id?: string
}

export interface SearchResult {
  title: string
  authors: string
  year: number | null
  section: string
  score: number | null
  text: string
  filename: string
}

export interface StatsResponse {
  total_papers: number
  total_chunks: number
  vector_chunks: number
  embedding_dim: number
  total_queries: number
}

// ── Query ────────────────────────────────────────────────────

export const queryPapers = (question: string, top_k = 5, session_id = 'default', paper_id?: string) =>
  api.post<QueryResponse>('/query', { question, top_k, session_id, paper_id }).then(r => r.data)

export const streamQuery = (question: string, top_k = 5): EventSource => {
  // POST-based SSE via fetch
  return new EventSource(`/api/v1/query/stream?question=${encodeURIComponent(question)}&top_k=${top_k}`)
}

export const streamQueryFetch = async (
  question: string,
  top_k = 5,
  onToken: (token: string) => void,
  onDone: () => void,
  paper_id?: string,
  onStatus?: (text: string) => void,
) => {
  const res = await fetch('/api/v1/query/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, top_k, paper_id }),
  })
  const reader = res.body!.getReader()
  const decoder = new TextDecoder()
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    const chunk = decoder.decode(value)
    const lines = chunk.split('\n')
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const raw = line.slice(6)
        if (raw === '[DONE]') { onDone(); return }
        try {
          const evt = JSON.parse(raw)
          if (evt.type === 'status') { onStatus?.(evt.text) }
          else if (evt.type === 'token') { onToken(evt.text) }
        } catch { onToken(raw) }
      }
    }
  }
  onDone()
}

// ── Papers ───────────────────────────────────────────────────

export const getPapers = () =>
  api.get<{ papers: Paper[]; total: number }>('/papers').then(r => r.data)

export const getPaper = (id: string) =>
  api.get<Paper>(`/papers/${id}`).then(r => r.data)

export const uploadPaper = (file: File, onProgress?: (pct: number) => void) => {
  const form = new FormData()
  form.append('file', file)
  return api.post('/papers/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: e => onProgress?.(Math.round((e.loaded * 100) / (e.total ?? 1))),
  }).then(r => r.data)
}

export const syncPapers = () =>
  api.post('/papers/sync').then(r => r.data)

// ── Search ───────────────────────────────────────────────────

export const searchPapers = (query: string, top_k = 10) =>
  api.post<{ query: string; results: SearchResult[]; total: number }>(
    '/search', { query, top_k }
  ).then(r => r.data)

// ── Stats ────────────────────────────────────────────────────

export const getStats = () =>
  api.get<StatsResponse>('/stats').then(r => r.data)

// ── Health ───────────────────────────────────────────────────

export const getHealth = () =>
  api.get('/health').then(r => r.data)

// ── Feedback ─────────────────────────────────────────────────

export const submitFeedback = (query_id: string, rating: number, comment?: string) =>
  api.post('/feedback', { query_id, rating, comment }).then(r => r.data)

// ── Flowchart ────────────────────────────────────────────────

export const generateFlowchart = (query: string, explanation: string) =>
  api.post<{ diagram: string; success: boolean }>('/flowchart', { query, explanation }).then(r => r.data)

// ── Sessions ─────────────────────────────────────────────────

export interface Session {
  id: string
  title: string
  created_at: string
  updated_at: string
}

export interface SessionMessage {
  id: string
  role: string
  content: string
  created_at: string
}

export const createSession = (title = 'New Session') =>
  api.post<Session>('/sessions', { title }).then(r => r.data)

export const listSessions = () =>
  api.get<Session[]>('/sessions').then(r => r.data)

export const getSessionMessages = (id: string) =>
  api.get<SessionMessage[]>(`/sessions/${id}/messages`).then(r => r.data)

export const addSessionMessage = (id: string, role: string, content: string) =>
  api.post(`/sessions/${id}/messages`, { role, content }).then(r => r.data)

export const deleteSession = (id: string) =>
  api.delete(`/sessions/${id}`).then(r => r.data)
