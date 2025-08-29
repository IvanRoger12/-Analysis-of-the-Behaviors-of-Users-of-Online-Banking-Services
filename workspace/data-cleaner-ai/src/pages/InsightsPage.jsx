import { useMemo } from 'react'
import { useDataStore } from '../store/useDataStore'

function makeInsights(rows) {
  if (!rows?.length) return []
  const cols = Object.keys(rows[0])
  const insights = []
  // simple frequency-based segments
  for (const c of cols.slice(0,3)) {
    const freq = new Map()
    for (const r of rows) {
      const v = String(r[c] ?? '').slice(0,32)
      if (!v) continue
      freq.set(v, (freq.get(v)||0)+1)
    }
    const top = [...freq.entries()].sort((a,b)=>b[1]-a[1]).slice(0,3)
    if (top.length) insights.push({
      title: `Segments rentables potentiels: ${c}`,
      body: `Top valeurs: ${top.map(([v,n])=>`${v} (${n})`).join(', ')}`,
      roi: Math.min(200, 20 + top.reduce((a, [,n])=>a+n,0) / rows.length * 100).toFixed(0) + '%'
    })
  }
  return insights
}

export default function InsightsPage() {
  const cleaned = useDataStore(s=>s.cleanedData)
  const insights = useMemo(()=>makeInsights(cleaned),[cleaned])
  return (
    <div className="space-y-4">
      {!insights.length && <div className="alert alert-info">Importez et nettoyez des données pour générer des insights.</div>}
      {insights.map((ins, i)=> (
        <div key={i} className="card bg-base-200/40">
          <div className="card-body">
            <div className="flex items-center gap-2">
              <span className="badge badge-accent">Insight</span>
              <h2 className="card-title">{ins.title}</h2>
              <span className="badge badge-secondary ml-auto">ROI potentiel: {ins.roi}</span>
            </div>
            <p>{ins.body}</p>
            <div className="flex gap-2">
              <div className="badge badge-outline">Opportunité</div>
              <div className="badge badge-outline">Risque faible</div>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

