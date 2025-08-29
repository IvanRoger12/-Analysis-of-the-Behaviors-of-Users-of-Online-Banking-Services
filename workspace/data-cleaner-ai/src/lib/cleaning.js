import dayjs from 'dayjs'
import { isEqual } from 'lodash'

const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
const phoneRegex = /^[+]?[(]?[0-9]{1,4}[)]?[-\s0-9]{5,}$/

export function deduplicateRows(rows) {
  const seen = new Set()
  const unique = []
  for (const r of rows) {
    const key = JSON.stringify(r)
    if (!seen.has(key)) { seen.add(key); unique.push(r) }
  }
  return unique
}

export function imputeMissing(rows) {
  if (!rows.length) return rows
  const cols = Object.keys(rows[0])
  const stats = {}
  for (const c of cols) {
    const vals = rows.map(r=>r[c]).filter(v=>v!==null && v!==undefined && String(v).trim()!=='')
    const numbers = vals.map(Number).filter(v=>!isNaN(v))
    const modeMap = new Map()
    for (const v of vals) modeMap.set(v, (modeMap.get(v)||0)+1)
    const mode = vals.length ? [...modeMap.entries()].sort((a,b)=>b[1]-a[1])[0][0] : null
    const mean = numbers.length ? numbers.reduce((a,b)=>a+b,0)/numbers.length : null
    stats[c] = { mean, mode }
  }
  return rows.map(r=>{
    const n = { ...r }
    for (const [c, s] of Object.entries(stats)) {
      if (n[c]===null || n[c]===undefined || String(n[c]).trim()==='') {
        n[c] = s.mean ?? s.mode ?? null
      }
    }
    return n
  })
}

export function standardizeFormats(rows) {
  return rows.map(r=>{
    const n = { ...r }
    for (const k of Object.keys(n)) {
      const v = n[k]
      if (typeof v === 'string') {
        const t = v.trim()
        if (emailRegex.test(t)) n[k] = t.toLowerCase()
        else if (phoneRegex.test(t)) n[k] = t.replace(/\D/g,'').replace(/(\d{2})(?=\d)/g,'$1 ').trim()
        else if (dayjs(t).isValid()) n[k] = dayjs(t).format('YYYY-MM-DD')
      }
    }
    return n
  })
}

export function normalizeNamesAddresses(rows) {
  return rows.map(r=>{
    const n = { ...r }
    for (const k of Object.keys(n)) {
      if (k.toLowerCase().includes('nom') || k.toLowerCase().includes('name')) {
        n[k] = String(n[k]||'').replace(/\s+/g,' ').trim()
      }
      if (k.toLowerCase().includes('adresse') || k.toLowerCase().includes('address')) {
        n[k] = String(n[k]||'').replace(/\s+/g,' ').trim()
      }
    }
    return n
  })
}

export function detectAnomalies(rows) {
  const anomalies = []
  if (!rows.length) return anomalies
  const cols = Object.keys(rows[0])
  for (const c of cols) {
    const nums = rows.map(r=>Number(r[c])).filter(v=>!isNaN(v))
    if (nums.length>4) {
      const mean = nums.reduce((a,b)=>a+b,0)/nums.length
      const variance = nums.reduce((a,b)=>a + Math.pow(b-mean,2),0)/nums.length
      const sd = Math.sqrt(variance)
      rows.forEach((r, idx)=>{
        const val = Number(r[c])
        if (!isNaN(val) && Math.abs(val-mean) > 3*sd) anomalies.push({ row: idx, col: c, value: val, type: 'zscore' })
      })
    }
  }
  return anomalies
}

export function qualityScore(rows) {
  if (!rows.length) return 0
  const totalCells = rows.length * Object.keys(rows[0]).length
  let missing = 0
  for (const r of rows) {
    for (const v of Object.values(r)) {
      if (v===null || v===undefined || String(v).trim()==='') missing++
    }
  }
  const completeness = 1 - missing/Math.max(1,totalCells)
  return Math.round(100 * completeness)
}

