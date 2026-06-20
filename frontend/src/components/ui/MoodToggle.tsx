import { useMood } from '../../lib/MoodContext'
import type { Mood } from '../../lib/theme'

const moods: { value: Mood; label: string; color: string }[] = [
  { value: 'blue',   label: '🫐', color: '#5b9ef9' },
  { value: 'pink',   label: '🌸', color: '#f472b6' },
  { value: 'purple', label: '🔮', color: '#a78bfa' },
]

export default function MoodToggle() {
  const { mood, setMood } = useMood()

  return (
    <div style={{
      display: 'flex', gap: 4, alignItems: 'center',
      background: 'rgba(0,0,0,0.04)',
      border: '1px solid var(--color-border)',
      borderRadius: 'var(--radius-pill)',
      padding: '3px',
    }}>
      {moods.map(m => (
        <button
          key={m.value}
          onClick={() => setMood(m.value)}
          title={m.value}
          style={{
            width: 26, height: 26,
            borderRadius: 'var(--radius-pill)',
            border: mood === m.value ? `2px solid ${m.color}` : '2px solid transparent',
            background: mood === m.value ? `${m.color}22` : 'transparent',
            cursor: 'pointer', fontSize: 13,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            transition: 'all 0.2s',
            boxShadow: mood === m.value ? `0 0 8px ${m.color}55` : 'none',
          }}
        >
          {m.label}
        </button>
      ))}
    </div>
  )
}
