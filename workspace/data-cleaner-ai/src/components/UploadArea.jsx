import { useCallback, useRef, useState } from 'react'
import Papa from 'papaparse'
import * as XLSX from 'xlsx'
import { useDataStore } from '../store/useDataStore'

const MAX_SIZE_BYTES = 50 * 1024 * 1024
const SUPPORTED = ['text/csv','application/vnd.ms-excel','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet','application/json','text/plain','application/xml','text/xml']

function parseJSON(text) {
  try { return JSON.parse(text) } catch { return null }
}

function parseXML(text) {
  const parser = new DOMParser()
  const xml = parser.parseFromString(text, 'application/xml')
  const rows = []
  xml.querySelectorAll('*:root > *').forEach(node => {
    const obj = {}
    node.childNodes.forEach(child => {
      if (child.nodeType === 1) obj[child.nodeName] = child.textContent
    })
    if (Object.keys(obj).length) rows.push(obj)
  })
  return rows
}

export default function UploadArea() {
  const inputRef = useRef(null)
  const [dragOver, setDragOver] = useState(false)
  const setStatus = useDataStore(s=>s.setStatus)
  const setFile = useDataStore(s=>s.setFile)
  const setRaw = useDataStore(s=>s.setRaw)
  const setPreview = useDataStore(s=>s.setPreview)
  const setColumns = useDataStore(s=>s.setColumns)
  const setError = useDataStore(s=>s.setError)

  const handleFiles = useCallback(async (files) => {
    const file = files?.[0]
    if (!file) return
    if (file.size > MAX_SIZE_BYTES) { setError('Fichier > 50MB'); return }
    if (!SUPPORTED.includes(file.type) && !file.name.match(/\.(csv|xlsx?|json|xml|txt)$/i)) { setError('Format non supporté'); return }
    setStatus('processing'); setError(null); setFile(file)

    const ext = (file.name.split('.').pop() || '').toLowerCase()
    const text = await file.text()
    let rows = []

    if (ext === 'csv' || file.type === 'text/csv') {
      const res = Papa.parse(text, { header: true, dynamicTyping: false, skipEmptyLines: true })
      rows = res.data
    } else if (ext === 'xlsx' || ext === 'xls') {
      const buf = await file.arrayBuffer()
      const wb = XLSX.read(buf, { type: 'array' })
      const ws = wb.Sheets[wb.SheetNames[0]]
      rows = XLSX.utils.sheet_to_json(ws, { defval: null })
    } else if (ext === 'json') {
      rows = parseJSON(text) || []
    } else if (ext === 'xml') {
      rows = parseXML(text)
    } else if (ext === 'txt') {
      rows = text.split(/\r?\n/).filter(Boolean).map((v, i) => ({ line: i+1, value: v }))
    }

    const preview = rows.slice(0, 100)
    const columns = preview.length ? Object.keys(preview[0]) : []
    setRaw(rows); setPreview(preview); setColumns(columns); setStatus('ready')
  }, [setStatus, setFile, setRaw, setPreview, setColumns, setError])

  return (
    <div className={`p-8 border-2 border-dashed rounded-2xl transition ${dragOver? 'border-secondary bg-secondary/10' : 'border-base-300 bg-base-200/40'}`}
      onDragOver={(e)=>{e.preventDefault(); setDragOver(true)}}
      onDragLeave={()=>setDragOver(false)}
      onDrop={(e)=>{ e.preventDefault(); setDragOver(false); handleFiles(e.dataTransfer.files) }}
    >
      <div className="text-center space-y-3">
        <h2 className="text-2xl font-display">Glissez-déposez vos fichiers</h2>
        <p className="opacity-70">CSV, Excel, JSON, XML, TXT • ≤ 50MB</p>
        <div className="flex justify-center gap-2">
          <button className="btn btn-secondary" onClick={()=>inputRef.current?.click()}>Choisir un fichier</button>
          <input ref={inputRef} type="file" className="hidden" onChange={(e)=>handleFiles(e.target.files)} />
        </div>
      </div>
    </div>
  )
}

