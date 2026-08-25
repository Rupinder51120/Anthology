import { Settings } from 'lucide-react'
import { glass } from '../lib/theme'

export function SettingsPage() {
  const sections = [
    { label: 'Interface', items: ['Theme: Glass UI', 'Default view: Research dashboard'] },
    { label: 'Retrieval', items: ['Embeddings: SPECTER2', 'Search: pgvector + full-text + RRF'] },
    { label: 'Generation', items: ['Primary LLM: Groq (cloud)', 'Swappable provider: Ollama (local) via USE_GROQ config'] },
  ]

  return (
    <div style={{ padding: '32px 36px', maxWidth: 640 }}>
      <h1 style={{ fontSize: 22, fontWeight: 800, color: 'var(--color-text)', letterSpacing: '-0.02em', marginBottom: 8 }}>Settings</h1>
      <p style={{ color: 'var(--color-muted)', fontSize: 13, marginBottom: 24 }}>Static project settings shown for portfolio clarity.</p>

      {sections.map(section => (
        <div key={section.label} style={{ ...glass, borderRadius: 'var(--radius-lg)', padding: 20, marginBottom: 14 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontWeight: 700, marginBottom: 14, fontSize: 14, color: 'var(--color-text)' }}>
            <Settings size={15} color="var(--color-accent)" />
            {section.label}
          </div>
          {section.items.map(item => {
            const [name, value] = item.split(':')
            return (
              <div key={item} style={{
                padding: '10px 0', borderBottom: '1px solid var(--color-border)',
                fontSize: 13, display: 'flex', justifyContent: 'space-between', gap: 16,
              }}>
                <span style={{ color: 'var(--color-muted)' }}>{name}</span>
                <span style={{ color: 'var(--color-text)', fontWeight: 500, textAlign: 'right' }}>{value?.trim()}</span>
              </div>
            )
          })}
        </div>
      ))}
    </div>
  )
}
