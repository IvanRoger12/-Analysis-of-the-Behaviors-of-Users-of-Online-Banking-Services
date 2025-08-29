import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import UploadPage from './pages/UploadPage'
import PreviewPage from './pages/PreviewPage'
import CleaningPage from './pages/CleaningPage'
import DashboardPage from './pages/DashboardPage'
import InsightsPage from './pages/InsightsPage'
import ChatPage from './pages/ChatPage'
import ExportPage from './pages/ExportPage'
import StatusBar from './components/StatusBar'
import DemoPage from './pages/DemoPage'

function NavItem({ to, label }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `btn btn-ghost btn-sm font-mono ${isActive ? 'text-secondary' : ''}`
      }
    >
      {label}
    </NavLink>
  )
}

function AppLayout({ children }) {
  const [lang, setLang] = useState('fr')
  const { i18n } = useTranslation()
  return (
    <div className="min-h-dvh flex flex-col">
      <div className="navbar bg-base-200/60 backdrop-blur border-b border-base-300 sticky top-0 z-50">
        <div className="flex-1">
          <span className="text-xl font-display tracking-widest">Data Cleaner <span className="text-secondary">AI</span></span>
        </div>
        <div className="flex gap-1">
          <NavItem to="/" label="Upload" />
          <NavItem to="/preview" label="Preview" />
          <NavItem to="/dashboard" label="Dashboard" />
          <NavItem to="/cleaning" label="Cleaning" />
          <NavItem to="/insights" label="Insights" />
          <NavItem to="/chat" label="Chat" />
          <NavItem to="/export" label="Export" />
          <NavItem to="/demo" label="Demo" />
        </div>
        <div className="ml-2">
          <div className="join join-horizontal">
            <button className={`btn btn-xs join-item ${lang==='fr'?'btn-secondary':''}`} onClick={()=>{setLang('fr'); i18n.changeLanguage('fr')}}>FR</button>
            <button className={`btn btn-xs join-item ${lang==='en'?'btn-secondary':''}`} onClick={()=>{setLang('en'); i18n.changeLanguage('en')}}>EN</button>
          </div>
        </div>
        <div className="ml-4 hidden md:block">
          <StatusBar />
        </div>
      </div>
      <main className="container mx-auto p-4 flex-1">
        {children}
      </main>
      <footer className="footer footer-center p-4 bg-base-200/60 border-t border-base-300">
        <aside>
          <p className="font-mono">Status: <span className="badge badge-success badge-outline">Live</span> <span className="badge badge-warning badge-outline">Processing</span></p>
        </aside>
      </footer>
    </div>
  )
}

function Placeholder({ title, description }) {
  return (
    <div className="prose prose-invert max-w-none">
      <h1 className="font-display">{title}</h1>
      <p>{description}</p>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
        {[1,2,3,4].map((i)=> (
          <div key={i} className="stat shadow-xl bg-base-200/60">
            <div className="stat-title">KPI {i}</div>
            <div className="stat-value text-secondary">{Math.round(Math.random()*100)}%</div>
            <div className="stat-desc">Animated metric</div>
          </div>
        ))}
      </div>
    </div>
  )
}

function App() {
  return (
    <BrowserRouter>
      <AppLayout>
        <Routes>
          <Route path="/" element={<UploadPage />} />
          <Route path="/preview" element={<PreviewPage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/cleaning" element={<CleaningPage />} />
          <Route path="/insights" element={<InsightsPage />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/export" element={<ExportPage />} />
          <Route path="/demo" element={<DemoPage />} />
        </Routes>
      </AppLayout>
    </BrowserRouter>
  )
}

export default App
