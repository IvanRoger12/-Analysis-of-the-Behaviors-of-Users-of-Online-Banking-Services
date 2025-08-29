import type { ValidationIssue, SupportedFormat } from '../types'

export function basicStructureValidation(rows: Record<string, unknown>[], columns: string[], format: SupportedFormat): ValidationIssue[] {
  const issues: ValidationIssue[] = []
  if (columns.length === 0) {
    issues.push({ level: 'error', message: 'Aucune colonne détectée' })
  }
  if (rows.length === 0) {
    issues.push({ level: 'warning', message: 'Aucune ligne de données' })
  }
  // Simple heuristic: flag if any row misses more than half columns
  const threshold = Math.ceil(columns.length / 2)
  let sparseRows = 0
  for (const row of rows.slice(0, 1000)) {
    let missing = 0
    for (const c of columns) if (row[c] === null || row[c] === undefined || row[c] === '') missing++
    if (missing > threshold) sparseRows++
  }
  if (sparseRows > 0) {
    issues.push({ level: 'warning', message: `${sparseRows} lignes très incomplètes (sur échantillon)` })
  }
  // Format-specific checks
  if (format === 'csv' && rows.length > 0 && columns.length > 0) {
    const firstRowCols = Object.keys(rows[0]).length
    if (firstRowCols !== columns.length) {
      issues.push({ level: 'info', message: 'Colonnes irrégulières détectées (CSV)' })
    }
  }
  return issues
}

