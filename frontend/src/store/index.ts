import { create } from 'zustand'
import type { Citation } from '../api/client'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  latency_ms?: number
  query_id?: string
  streaming?: boolean
}

interface ChatStore {
  messages: Message[]
  streaming: boolean
  sessionId: string
  addMessage: (msg: Message) => void
  updateLastMessage: (content: string) => void
  finalizeMessage: (extra: Partial<Message>) => void
  setStreaming: (v: boolean) => void
  clearMessages: () => void
  setMessages: (msgs: Message[]) => void
}

export const useChatStore = create<ChatStore>((set) => ({
  messages: [],
  streaming: false,
  sessionId: crypto.randomUUID(),
  addMessage: (msg) => set(s => ({ messages: [...s.messages, msg] })),
  updateLastMessage: (content) => set(s => {
    const msgs = [...s.messages]
    if (msgs.length) msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], content }
    return { messages: msgs }
  }),
  finalizeMessage: (extra) => set(s => {
    const msgs = [...s.messages]
    if (msgs.length) msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], ...extra, streaming: false }
    return { messages: msgs }
  }),
  setStreaming: (v) => set({ streaming: v }),
  clearMessages: () => set({ messages: [] }),
  setMessages: (msgs) => set({ messages: msgs }),
}))

interface PaperStore {
  selectedPaperId: string | null
  setSelectedPaper: (id: string | null) => void
}

export const usePaperStore = create<PaperStore>((set) => ({
  selectedPaperId: null,
  setSelectedPaper: (id) => set({ selectedPaperId: id }),
}))
