import { useDataStore } from '../store/useDataStore'

const demoRows = Array.from({length: 150}).map((_,i)=> ({
  id: i+1,
  email: `user${i+1}@exemple.com`,
  phone: `+33 6 ${String(100000 + i).slice(0,6)}`,
  date: `2024-0${(i%9)+1}-0${(i%27)+1}`,
  amount: (Math.random()*1000).toFixed(2),
  segment: ['A','B','C'][i%3]
}))

export default function DemoPage() {
  const setRaw = useDataStore(s=>s.setRaw)
  const setPreview = useDataStore(s=>s.setPreview)
  const setColumns = useDataStore(s=>s.setColumns)
  const setStatus = useDataStore(s=>s.setStatus)

  function loadDemo() {
    setStatus('processing')
    setTimeout(()=>{
      setRaw(demoRows)
      setPreview(demoRows.slice(0,100))
      setColumns(Object.keys(demoRows[0]))
      setStatus('ready')
    }, 500)
  }

  return (
    <div className="hero min-h-[50vh] bg-base-200/40 rounded-xl">
      <div className="hero-content text-center">
        <div className="max-w-md">
          <div className="badge badge-info mb-2">Demo</div>
          <h1 className="text-4xl font-display">Auto-Play Démo</h1>
          <p className="py-4">Charge un dataset réaliste et simule des résultats.</p>
          <div className="flex gap-2 justify-center">
            <button className="btn btn-secondary" onClick={loadDemo}>Lancer la démo</button>
            <div className="badge badge-outline">Premium</div>
            <div className="badge badge-outline">VIP</div>
          </div>
        </div>
      </div>
    </div>
  )
}

