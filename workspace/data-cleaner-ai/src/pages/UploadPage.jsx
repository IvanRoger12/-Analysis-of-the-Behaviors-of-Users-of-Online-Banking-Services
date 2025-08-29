import UploadArea from '../components/UploadArea'
import PreviewTable from '../components/PreviewTable'
import { useDataStore } from '../store/useDataStore'

export default function UploadPage() {
  const status = useDataStore(s=>s.status)
  const error = useDataStore(s=>s.error)
  return (
    <div className="space-y-6">
      <UploadArea />
      {status==='processing' && <progress className="progress progress-secondary w-full" />}
      {error && <div className="alert alert-error">{error}</div>}
      <PreviewTable />
    </div>
  )
}

