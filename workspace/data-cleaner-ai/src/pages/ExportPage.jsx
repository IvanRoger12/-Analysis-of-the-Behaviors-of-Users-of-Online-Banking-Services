import { saveAs } from 'file-saver'
import * as XLSX from 'xlsx'
import jsPDF from 'jspdf'
import html2canvas from 'html2canvas'
import { useRef } from 'react'
import { useDataStore } from '../store/useDataStore'

export default function ExportPage() {
  const cleaned = useDataStore(s=>s.cleanedData)
  const quality = useDataStore(s=>s.qualityScore)
  const ref = useRef(null)

  function exportCSV() {
    if (!cleaned?.length) return
    const ws = XLSX.utils.json_to_sheet(cleaned)
    const wb = XLSX.utils.book_new()
    XLSX.utils.book_append_sheet(wb, ws, 'Cleaned')
    const csv = XLSX.utils.sheet_to_csv(ws)
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
    saveAs(blob, 'cleaned.csv')
  }

  function exportXLSX() {
    if (!cleaned?.length) return
    const ws = XLSX.utils.json_to_sheet(cleaned)
    const wb = XLSX.utils.book_new()
    XLSX.utils.book_append_sheet(wb, ws, 'Cleaned')
    const out = XLSX.write(wb, { bookType: 'xlsx', type: 'array' })
    saveAs(new Blob([out], { type: 'application/octet-stream' }), 'cleaned.xlsx')
  }

  async function exportPDF() {
    const node = ref.current
    if (!node) return
    const canvas = await html2canvas(node)
    const imgData = canvas.toDataURL('image/png')
    const pdf = new jsPDF({ unit: 'px', format: 'a4' })
    const pageWidth = pdf.internal.pageSize.getWidth()
    const ratio = pageWidth / canvas.width
    pdf.addImage(imgData, 'PNG', 0, 0, pageWidth, canvas.height * ratio)
    pdf.save('rapport.pdf')
  }

  return (
    <div className="space-y-4">
      <div className="join">
        <button className="btn join-item" onClick={exportCSV}>Export CSV</button>
        <button className="btn join-item" onClick={exportXLSX}>Export Excel</button>
        <button className="btn join-item" onClick={exportPDF}>Rapport PDF</button>
      </div>
      <div ref={ref} className="card bg-base-200/40">
        <div className="card-body">
          <h2 className="card-title">Résumé Business</h2>
          <p>Score qualité global: {quality ?? 0}%</p>
          <p>Données nettoyées: {cleaned?.length || 0} enregistrements</p>
        </div>
      </div>
    </div>
  )
}

