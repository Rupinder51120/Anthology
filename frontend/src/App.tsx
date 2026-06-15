import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import Sidebar from './components/layout/Sidebar'
import HomePage from './pages/Home'
import ChatPage from './pages/Chat'
import LibraryPage from './pages/Library'
import SearchPage from './pages/Search'
import UploadPage from './pages/Upload'
import { CollectionsPage, SettingsPage } from './pages/Stubs'

const qc = new QueryClient({
  defaultOptions: { queries: { staleTime: 30_000, retry: 1 } },
})

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
              <Route path="/search"      element={<SearchPage />} />
              <Route path="/upload"      element={<UploadPage />} />
              <Route path="/collections" element={<CollectionsPage />} />
              <Route path="/settings"    element={<SettingsPage />} />
            </Routes>
          </main>
        </div>
        <style>{`
          @keyframes spin { to { transform: rotate(360deg); } }
        `}</style>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
