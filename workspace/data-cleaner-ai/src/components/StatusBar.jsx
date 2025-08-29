import { useDataStore } from '../store/useDataStore'

export default function StatusBar() {
  const status = useDataStore(s=>s.status)
  const error = useDataStore(s=>s.error)
  return (
    <div className="flex items-center gap-2 text-sm">
      <span className="badge badge-outline">{status}</span>
      {status==='processing' && <progress className="progress progress-secondary w-40" />}
      {error && <span className="badge badge-error badge-outline">{error}</span>}
    </div>
  )
}

