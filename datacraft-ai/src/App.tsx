// no React import needed for jsx in TS with vite config
import UploadDropzone from './components/UploadDropzone'
import DataPreview from './components/DataPreview'

function App() {
  return (
    <div className="min-h-screen bg-black/95 text-white">
      <div className="mx-auto max-w-5xl px-4 py-12">
        <header className="mb-10">
          <h1 className="text-3xl md:text-4xl font-semibold tracking-tight">
            DataCraft AI — Ingestion & Cleaning
          </h1>
          <p className="text-zinc-400 mt-2">
            Chargez vos fichiers (CSV, Excel, JSON, XML, TXT) — 50MB max
          </p>
        </header>

        <section>
          <UploadDropzone />
          <DataPreview />
        </section>
      </div>
    </div>
  )
}

export default App
