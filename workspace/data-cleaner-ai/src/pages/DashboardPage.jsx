import { useMemo } from 'react'
import { useDataStore } from '../store/useDataStore'
import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

const COLORS = ['#00FFD1', '#FF00E6', '#00A3FF', '#FFD400']

export default function DashboardPage() {
  const cleaned = useDataStore(s=>s.cleanedData)
  const quality = useDataStore(s=>s.qualityScore) || 0

  const kpis = [
    { title: 'Qualité', value: `${quality}%`, desc: 'Score global' },
    { title: 'Anomalies', value: cleaned ? Math.round(cleaned.length*0.03) : 0, desc: 'Estimées' },
    { title: 'Enregistrements', value: cleaned?.length || 0, desc: 'Nettoyés' },
    { title: 'Confiance', value: `${Math.max(50,quality-5)}%`, desc: 'Heuristique' },
  ]

  const barData = useMemo(()=>{
    if (!cleaned?.length) return []
    const firstRow = cleaned[0]
    const cols = Object.keys(firstRow).slice(0,5)
    return cols.map(c=>({ name: c, count: cleaned.map(r=>r[c]).filter(Boolean).length }))
  },[cleaned])

  const lineData = useMemo(()=>Array.from({length:12}).map((_,i)=>({ month: `M${i+1}`, val: Math.round(quality + (Math.sin(i/2)*10)) })),[quality])

  const pieData = useMemo(()=>[
    { name: 'Complets', value: quality },
    { name: 'Manquants', value: 100-quality },
  ],[quality])

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {kpis.map((k,i)=> (
          <div key={i} className="stat shadow-xl bg-base-200/60">
            <div className="stat-title">{k.title}</div>
            <div className="stat-value text-secondary">{k.value}</div>
            <div className="stat-desc">{k.desc}</div>
          </div>
        ))}
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <div className="card bg-base-200/40">
          <div className="card-body">
            <h2 className="card-title">Complétude par colonne</h2>
            <div className="w-full h-64">
              <ResponsiveContainer>
                <BarChart data={barData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="#00FFD1" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
        <div className="card bg-base-200/40">
          <div className="card-body">
            <h2 className="card-title">Qualité dans le temps</h2>
            <div className="w-full h-64">
              <ResponsiveContainer>
                <LineChart data={lineData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="val" stroke="#FF00E6" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
        <div className="card bg-base-200/40 md:col-span-2">
          <div className="card-body">
            <h2 className="card-title">Répartition qualité</h2>
            <div className="w-full h-64">
              <ResponsiveContainer>
                <PieChart>
                  <Pie data={pieData} dataKey="value" nameKey="name" innerRadius={50} outerRadius={80}>
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

