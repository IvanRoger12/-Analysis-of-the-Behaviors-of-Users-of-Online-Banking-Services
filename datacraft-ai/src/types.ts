export type SupportedFormat = 'csv' | 'excel' | 'json' | 'xml' | 'txt'

export type DetectedColumnType =
  | 'string'
  | 'number'
  | 'integer'
  | 'boolean'
  | 'date'
  | 'email'
  | 'phone'

export interface ColumnProfile {
  name: string
  type: DetectedColumnType
  confidence: number // 0-1
  nullRatio: number // 0-1
  uniqueRatio: number // 0-1
  sampleValues: Array<string | number | boolean | null>
}

export interface DatasetMeta {
  filename: string
  bytes: number
  format: SupportedFormat
  columns: string[]
  columnProfiles: ColumnProfile[]
  numRows: number
}

export interface ValidationIssue {
  level: 'info' | 'warning' | 'error'
  message: string
  column?: string
}

export interface DatasetState {
  meta: DatasetMeta | null
  previewRows: Record<string, unknown>[] // capped to 100
  issues: ValidationIssue[]
  qualityScore: number // 0-100
  duplicateRatio: number // 0-1
  anomalyCount: number
  updatedAt: number | null
}

export interface IngestionResult {
  rows: Record<string, unknown>[]
  columns: string[]
}

