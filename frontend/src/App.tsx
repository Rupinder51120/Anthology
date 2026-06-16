import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import Sidebar from './components/layout/Sidebar'
import HomePage from './pages/Home'
import ChatPage from './pages/Chat'
import LibraryPage from './pages/Library'
import SearchPage from './pages/Search'
import UploadPage from './pages/Upload'
import PaperView from './pages/PaperView'
import { CollectionsPage, SettingsPage } from './pages/Stubs'
import DiscoveryPage from './pages/Discovery'

const qc = new QueryClient({ defaultOptions: { queries: { staleTime: 30_000, retry: 1 } } })

export default function App() {
  return (
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <div style={{ display: 'flex', minHeight: '100vh', background: 'var(--color-bg)' }}>
          <Sidebar />
          <main style={{ flex: 1, minWidth: 0, overflowY: 'auto' }}>
            <Routes>
              <Route path="/"            element={<HomePage />} />
              <Route path="/chat"        element={<ChatPage />} />
              <Route path="/library"     element={<LibraryPage />} />
              <Route path="/papers/:id"  element={<PaperView />} />
              <Route path="/search"      element={<SearchPage />} />
              <Route path="/discover"    element={<DiscoveryPage />} />
              <Route path="/upload"      element={<UploadPage />} />
              <Route path="/collections" element={<CollectionsPage />} />
              <Route path="/settings"    element={<SettingsPage />} />
            </Routes>
          </main>
        </div>
        <style>{`
          @keyframes spin { to { transform: rotate(360deg); } }
          .prose h2 { font-size:15px; font-weight:700; margin:12px 0 6px; color:var(--color-text); }
          .prose h3 { font-size:14px; font-weight:700; margin:10px 0 4px; color:var(--color-text); }
          .prose p  { margin:6px 0; line-height:1.65; }
          .prose ul,.prose ol { padding-left:20px; margin:6px 0; }
          .prose li { margin:3px 0; }
          .prose strong { font-weight:700; }
          .prose code { background:rgba(0,0,0,0.06); border-radius:4px; padding:1px 5px; font-size:12px; }
        `}</style>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
