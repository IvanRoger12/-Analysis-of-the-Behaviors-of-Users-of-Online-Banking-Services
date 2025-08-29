import { useEffect, useState } from 'react'
import { useDataStore } from '../store/useDataStore'
import { detectColumnTypes } from '../lib/typeDetect'
import PreviewTable from '../components/PreviewTable'

export default function PreviewPage() {
  const rows = useDataStore(s=>s.previewRows)
  const setSchema = useDataStore(s=>s.setSchema)
  const schema = useDataStore(s=>s.schema)
  const columns = useDataStore(s=>s.columns)
  const [ready, setReady] = useState(false)

  useEffect(()=>{
    if (rows?.length) {
      const types = detectColumnTypes(rows)
      setSchema(types)
      setReady(true)
    }
  },[rows, setSchema])

  return (
    <div className="space-y-4">
      <div className="card bg-base-200/40">
        <div className="card-body">
          <h2 className="card-title">Schéma détecté</h2>
          {!schema && <span className="loading loading-dots loading-md" />}
          {schema && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {columns.map(c => (
                <div key={c} className="badge badge-outline font-mono">{c}: {schema[c]}</div>
              ))}
            </div>
          )}
        </div>
      </div>
      <PreviewTable />
    </div>
  )
}

