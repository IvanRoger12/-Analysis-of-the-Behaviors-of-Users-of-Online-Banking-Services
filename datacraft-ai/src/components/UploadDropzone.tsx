import { useCallback, useRef, useState } from 'react'
import { useDatasetStore } from '../store'
import { buildMeta, parseFile } from '../lib/ingest'
import { basicStructureValidation } from '../lib/validate'
import type { SupportedFormat } from '../types'

const ACCEPTED_EXTENSIONS = ['.csv', '.xls', '.xlsx', '.json', '.xml', '.txt']
const MAX_SIZE_BYTES = 50 * 1024 * 1024

function detectFormatFromName(filename: string): SupportedFormat | null {
  const lower = filename.toLowerCase()
  if (lower.endsWith('.csv')) return 'csv'
  if (lower.endsWith('.xls') || lower.endsWith('.xlsx')) return 'excel'
  if (lower.endsWith('.json')) return 'json'
  if (lower.endsWith('.xml')) return 'xml'
  if (lower.endsWith('.txt')) return 'txt'
  return null
}

export function UploadDropzone() {
  const [dragActive, setDragActive] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [status, setStatus] = useState<'idle' | 'parsing' | 'ready' | 'error'>('idle')
  const inputRef = useRef<HTMLInputElement | null>(null)
  const setDataset = useDatasetStore((s) => s.setDataset)
  const setIssues = useDatasetStore((s) => s.setIssues)

  const validateFiles = useCallback((fileList: FileList) => {
    setErrorMessage(null)
    const files = Array.from(fileList)
    for (const file of files) {
      if (file.size > MAX_SIZE_BYTES) {
        setErrorMessage('Fichier trop volumineux (max 50MB)')
        return false
      }
      const lowerName = file.name.toLowerCase()
      const ok = ACCEPTED_EXTENSIONS.some((ext) => lowerName.endsWith(ext))
      if (!ok) {
        setErrorMessage('Format non supporté. CSV, Excel, JSON, XML, TXT')
        return false
      }
    }
    return true
  }, [])

  const handleFiles = useCallback(async (files: FileList) => {
    try {
      setStatus('parsing')
      const primary = files[0]
      if (!primary) return
      const format = detectFormatFromName(primary.name)
      if (!format) throw new Error('Format non supporté')
      const { rows, columns } = await parseFile(primary)
      const meta = buildMeta(primary, format, rows, columns)
      const issues = basicStructureValidation(rows, columns, format)
      setDataset(meta, rows.slice(0, 100))
      setIssues(issues)
      setStatus('ready')
    } catch (err) {
      console.error(err)
      setErrorMessage((err as Error).message || 'Erreur lors du parsing')
      setStatus('error')
    }
  }, [setDataset, setIssues])

  const onDrop = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    event.stopPropagation()
    setDragActive(false)
    if (event.dataTransfer?.files && event.dataTransfer.files.length > 0) {
      if (!validateFiles(event.dataTransfer.files)) return
      void handleFiles(event.dataTransfer.files)
    }
  }, [handleFiles, validateFiles])

  const onChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    if (!event.target.files) return
    if (!validateFiles(event.target.files)) return
    void handleFiles(event.target.files)
  }, [handleFiles, validateFiles])

  return (
    <div>
      <div
        onDragEnter={(e) => { e.preventDefault(); setDragActive(true) }}
        onDragOver={(e) => { e.preventDefault(); setDragActive(true) }}
        onDragLeave={(e) => { e.preventDefault(); setDragActive(false) }}
        onDrop={onDrop}
        className={
          'relative flex flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed p-10 transition ' +
          (dragActive ? 'border-fuchsia-400 bg-fuchsia-500/10' : 'border-zinc-700 bg-zinc-900/60 hover:border-zinc-500')
        }
      >
        <div className="pointer-events-none absolute -inset-1 rounded-2xl bg-gradient-to-br from-fuchsia-500/20 via-cyan-500/10 to-transparent blur-3xl" />
        <div className="relative z-10 text-center">
          <div className="text-xl font-medium">Glissez & déposez vos fichiers</div>
          <div className="text-zinc-400 text-sm">ou cliquez pour parcourir</div>
        </div>

        <input
          ref={inputRef}
          type="file"
          multiple
          accept={ACCEPTED_EXTENSIONS.join(',')}
          onChange={onChange}
          className="absolute inset-0 h-full w-full cursor-pointer opacity-0"
        />
      </div>

      {status === 'parsing' && (
        <div className="mt-4 text-sm text-zinc-300">Parsing en cours…</div>
      )}
      {errorMessage && (
        <div className="mt-4 rounded-lg border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-300">
          {errorMessage}
        </div>
      )}
    </div>
  )
}

export default UploadDropzone

