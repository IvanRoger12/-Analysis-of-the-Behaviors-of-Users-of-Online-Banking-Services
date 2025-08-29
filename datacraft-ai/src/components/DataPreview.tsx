import { useMemo, useState } from 'react'
import { useDatasetStore } from '../store'

const PAGE_SIZE = 25

export function DataPreview() {
  const rows = useDatasetStore((s) => s.previewRows)
  const columns = useDatasetStore((s) => s.meta?.columns ?? [])
  const [page, setPage] = useState(0)

  const totalPages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE))
  const pageRows = useMemo(() => rows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE), [rows, page])

  if (rows.length === 0) return null

  return (
    <div className="mt-8 rounded-xl border border-zinc-800 overflow-hidden">
      <div className="overflow-auto">
        <table className="min-w-full text-sm">
          <thead className="bg-zinc-900/60">
            <tr>
              {columns.map((c) => (
                <th key={c} className="px-3 py-2 text-left font-medium text-zinc-300 whitespace-nowrap border-b border-zinc-800">{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {pageRows.map((r, i) => (
              <tr key={i} className={i % 2 ? 'bg-zinc-900/30' : ''}>
                {columns.map((c) => (
                  <td key={c} className="px-3 py-2 border-b border-zinc-900/60 text-zinc-200 whitespace-nowrap max-w-[320px] truncate" title={String(r[c] ?? '')}>
                    {String(r[c] ?? '')}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex items-center justify-between px-3 py-2 bg-zinc-950/60">
        <div className="text-xs text-zinc-400">Page {page + 1} / {totalPages}</div>
        <div className="flex gap-2">
          <button className="rounded-md bg-zinc-800 px-3 py-1 text-xs disabled:opacity-50" disabled={page === 0} onClick={() => setPage((p) => Math.max(0, p - 1))}>Précédent</button>
          <button className="rounded-md bg-zinc-800 px-3 py-1 text-xs disabled:opacity-50" disabled={page + 1 >= totalPages} onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}>Suivant</button>
        </div>
      </div>
    </div>
  )
}

export default DataPreview

