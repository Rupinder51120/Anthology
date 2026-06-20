import { createContext, useContext, useState, useEffect, type ReactNode } from 'react'
import type { Mood } from './theme'

interface MoodCtx { mood: Mood; setMood: (m: Mood) => void }
const MoodContext = createContext<MoodCtx>({ mood: 'blue', setMood: () => {} })

export function MoodProvider({ children }: { children: ReactNode }) {
  const [mood, setMoodState] = useState<Mood>(() => {
    return (localStorage.getItem('anthology-mood') as Mood) || 'blue'
  })

  const setMood = (m: Mood) => {
    setMoodState(m)
    localStorage.setItem('anthology-mood', m)
  }

  useEffect(() => {
    document.documentElement.setAttribute('data-mood', mood)
  }, [mood])

  return <MoodContext.Provider value={{ mood, setMood }}>{children}</MoodContext.Provider>
}

export const useMood = () => useContext(MoodContext)
