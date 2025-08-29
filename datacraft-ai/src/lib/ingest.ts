import Papa from 'papaparse'
import * as XLSX from 'xlsx'
import dayjs from 'dayjs'
import type { IngestionResult, SupportedFormat, ColumnProfile, DetectedColumnType, DatasetMeta } from '../types'

function detectFormatFromName(filename: string): SupportedFormat | null {
  const lower = filename.toLowerCase()
  if (lower.endsWith('.csv')) return 'csv'
  if (lower.endsWith('.xls') || lower.endsWith('.xlsx')) return 'excel'
  if (lower.endsWith('.json')) return 'json'
  if (lower.endsWith('.xml')) return 'xml'
  if (lower.endsWith('.txt')) return 'txt'
  return null
}

export async function parseFile(file: File): Promise<IngestionResult> {
  const format = detectFormatFromName(file.name)
  if (!format) throw new Error('Unsupported format')

  if (format === 'csv' || format === 'txt') {
    const text = await file.text()
    const result = Papa.parse<Record<string, unknown>>(text, { header: true, dynamicTyping: false, skipEmptyLines: true })
    if (result.errors.length) {
      throw new Error(result.errors[0].message)
    }
    const rows = (result.data as Record<string, unknown>[]).filter(Boolean)
    const columns = result.meta.fields ?? Object.keys(rows[0] ?? {})
    return { rows, columns }
  }

  if (format === 'json') {
    const text = await file.text()
    const json = JSON.parse(text)
    const rows = Array.isArray(json) ? json : (json?.data ?? [])
    const columns = Object.keys(rows[0] ?? {})
    return { rows, columns }
  }

  if (format === 'xml') {
    const text = await file.text()
    const parser = new DOMParser()
    const doc = parser.parseFromString(text, 'application/xml')
    const nodes = Array.from(doc.getElementsByTagName('row'))
    const rows = nodes.map((node) => {
      const obj: Record<string, unknown> = {}
      Array.from(node.children).forEach((child) => {
        obj[child.tagName] = child.textContent
      })
      return obj
    })
    const columns = Object.keys(rows[0] ?? {})
    return { rows, columns }
  }

  if (format === 'excel') {
    const data = new Uint8Array(await file.arrayBuffer())
    const workbook = XLSX.read(data, { type: 'array' })
    const first = workbook.SheetNames[0]
    const sheet = workbook.Sheets[first]
    const rows: Record<string, unknown>[] = XLSX.utils.sheet_to_json(sheet, { defval: null })
    const columns = Object.keys(rows[0] ?? {})
    return { rows, columns }
  }

  throw new Error('Unhandled format')
}

function scoreTypeConfidence(scoreParts: number[]): number {
  if (scoreParts.length === 0) return 0
  const sum = scoreParts.reduce((a, b) => a + b, 0)
  return Math.min(1, sum / scoreParts.length)
}

function detectValueType(value: unknown): DetectedColumnType {
  if (value === null || value === undefined || value === '') return 'string'
  if (typeof value === 'boolean') return 'boolean'
  if (typeof value === 'number') return Number.isInteger(value) ? 'integer' : 'number'
  const str = String(value).trim()
  if (str === '') return 'string'
  if (/^(true|false)$/i.test(str)) return 'boolean'
  if (/^[-+]?\d+$/.test(str)) return 'integer'
  if (/^[-+]?\d*\.\d+$/.test(str)) return 'number'
  if (/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(str)) return 'email'
  if (/^\+?[0-9()\s-]{6,}$/.test(str)) return 'phone'
  if (dayjs(str).isValid()) return 'date'
  return 'string'
}

export function profileColumns(rows: Record<string, unknown>[], columns: string[]): ColumnProfile[] {
  const sampleSize = Math.min(rows.length, 1000)
  const sampled = rows.slice(0, sampleSize)
  return columns.map((name) => {
    const values = sampled.map((r) => r[name])
    const nonNull = values.filter((v) => v !== null && v !== undefined && v !== '')
    const nullRatio = values.length === 0 ? 0 : 1 - nonNull.length / values.length
    const unique = new Set(nonNull.map((v) => String(v))).size
    const uniqueRatio = nonNull.length === 0 ? 0 : unique / nonNull.length
    const typeVotes: Record<DetectedColumnType, number> = {
      string: 0, number: 0, integer: 0, boolean: 0, date: 0, email: 0, phone: 0,
    }
    for (const v of nonNull) {
      const t = detectValueType(v)
      typeVotes[t]++
    }
    const best = Object.entries(typeVotes).sort((a, b) => b[1] - a[1])[0] as [DetectedColumnType, number] | undefined
    const type = best ? best[0] : 'string'
    const confidence = scoreTypeConfidence([nonNull.length === 0 ? 0 : best ? best[1] / nonNull.length : 0, 1 - nullRatio])
    const sampleValues = values.slice(0, 5) as Array<string | number | boolean | null>
    return { name, type, confidence, nullRatio, uniqueRatio, sampleValues }
  })
}

export function buildMeta(file: File, format: SupportedFormat, rows: Record<string, unknown>[], columns: string[]): DatasetMeta {
  const columnProfiles = profileColumns(rows, columns)
  return {
    filename: file.name,
    bytes: file.size,
    format,
    columns,
    columnProfiles,
    numRows: rows.length,
  }
}

