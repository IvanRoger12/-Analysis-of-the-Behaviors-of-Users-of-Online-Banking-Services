import { useMemo, useState } from 'react'
import { useDataStore } from '../store/useDataStore'

export default function PreviewTable() {
  const rows = useDataStore(s=>s.previewRows)
  const columns = useDataStore(s=>s.columns)
  const [page, setPage] = useState(1)
  const pageSize = 25
  const totalPages = Math.max(1, Math.ceil(Math.min(100, rows.length) / pageSize))

  const pageRows = useMemo(()=>{
    const start = (page-1) * pageSize
    return rows.slice(start, start + pageSize)
  },[rows, page])

  if (!rows?.length) return <div className="alert alert-info">Aucune donnée à prévisualiser.</div>

  return (
    <div className="overflow-x-auto">
      <table className="table table-zebra">
        <thead>
          <tr>
            {columns.map(col => (
              <th key={col} className="font-mono text-xs">{col}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {pageRows.map((r, idx) => (
            <tr key={idx}>
              {columns.map(c => (
                <td key={c} className="text-sm">
                  {r?.[c] !== undefined && r?.[c] !== null ? String(r[c]) : <span className="opacity-50">—</span>}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="join mt-4">
        <button className="btn join-item" onClick={()=>setPage(p=>Math.max(1,p-1))}>«</button>
        <button className="btn join-item">Page {page} / {totalPages}</button>
        <button className="btn join-item" onClick={()=>setPage(p=>Math.min(totalPages,p+1))}>»</button>
      </div>
    </div>
  )
}

