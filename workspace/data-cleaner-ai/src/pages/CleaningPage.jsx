import { useEffect, useState } from 'react'
import { useDataStore } from '../store/useDataStore'
import { deduplicateRows, imputeMissing, standardizeFormats, normalizeNamesAddresses, detectAnomalies, qualityScore } from '../lib/cleaning'

export default function CleaningPage() {
  const raw = useDataStore(s=>s.rawData)
  const setCleaned = useDataStore(s=>s.setCleaned)
  const setQuality = useDataStore(s=>s.setQuality)
  const [anoms, setAnoms] = useState([])
  const [dedupedCount, setDedupedCount] = useState(0)

  useEffect(()=>{
    if (!raw?.length) return
    let rows = deduplicateRows(raw)
    setDedupedCount(raw.length - rows.length)
    rows = imputeMissing(rows)
    rows = standardizeFormats(rows)
    rows = normalizeNamesAddresses(rows)
    const anomalies = detectAnomalies(rows)
    setAnoms(anomalies)
    setCleaned(rows)
    setQuality(qualityScore(rows))
  },[raw, setCleaned, setQuality])

  return (
    <div className="grid gap-6 md:grid-cols-3">
      <div className="stats shadow bg-base-200/60 md:col-span-3">
        <div className="stat">
          <div className="stat-title">Doublons supprimés</div>
          <div className="stat-value text-secondary">{dedupedCount}</div>
        </div>
        <div className="stat">
          <div className="stat-title">Anomalies détectées</div>
          <div className="stat-value text-warning">{anoms.length}</div>
        </div>
        <div className="stat">
          <div className="stat-title">Score Qualité</div>
          <div className="stat-value text-success">{useDataStore.getState().qualityScore ?? 0}%</div>
        </div>
      </div>
      <div className="card bg-base-200/40 md:col-span-3">
        <div className="card-body">
          <h2 className="card-title">Anomalies (z-score &gt; 3)</h2>
          {!anoms.length && <div className="opacity-70">Aucune anomalie majeure.</div>}
          {!!anoms.length && (
            <div className="overflow-x-auto">
              <table className="table table-sm">
                <thead>
                  <tr>
                    <th>Ligne</th>
                    <th>Colonne</th>
                    <th>Valeur</th>
                    <th>Type</th>
                  </tr>
                </thead>
                <tbody>
                  {anoms.slice(0,100).map((a,i)=> (
                    <tr key={i}>
                      <td>{a.row}</td>
                      <td className="font-mono text-xs">{a.col}</td>
                      <td>{String(a.value)}</td>
                      <td><span className="badge badge-warning badge-outline">{a.type}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

