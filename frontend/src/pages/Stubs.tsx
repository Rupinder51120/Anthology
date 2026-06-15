import { Layers } from 'lucide-react'
import { Empty } from '../components/ui'

export function CollectionsPage() {
  return (
    <div style={{ padding: 28 }}>
      <h1 style={{ fontSize: 22, fontWeight: 700, marginBottom: 4 }}>Collections</h1>
      <p style={{ color: 'var(--color-muted)', fontSize: 13, marginBottom: 28 }}>Group papers into research collections</p>
      <Empty icon={<Layers size={40} />} title="No collections yet" sub="Coming soon — group papers by topic or project" />
    </div>
  )
}

export function SettingsPage() {
  return (
    <div style={{ padding: 28, maxWidth: 600 }}>
      <h1 style={{ fontSize: 22, fontWeight: 700, marginBottom: 24 }}>Settings</h1>
      {[
        { label: 'Appearance', items: ['Theme: Dark', 'Accent Color: Purple'] },
        { label: 'Search Preferences', items: ['Default Results: 10', 'Default Sort: Relevance'] },
        { label: 'API Keys', items: ['Groq API Key: configured', 'Cohere API Key: configured', 'Langfuse: enabled'] },
      ].map(section => (
        <div key={section.label} style={{
          background: 'var(--color-surface)', border: '1px solid var(--color-border)',
          borderRadius: 12, padding: 20, marginBottom: 16,
        }}>
          <div style={{ fontWeight: 700, marginBottom: 12 }}>{section.label}</div>
          {section.items.map(item => (
            <div key={item} style={{
              padding: '8px 0', borderBottom: '1px solid var(--color-border)',
              fontSize: 13, color: 'var(--color-muted)', display: 'flex',
              justifyContent: 'space-between',
            }}>
              <span>{item.split(':')[0]}</span>
              <span style={{ color: 'var(--color-text)' }}>{item.split(':')[1]?.trim()}</span>
            </div>
          ))}
        </div>
      ))}
    </div>
  )
}
