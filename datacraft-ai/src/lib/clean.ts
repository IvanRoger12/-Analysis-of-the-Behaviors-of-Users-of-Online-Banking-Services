import dayjs from 'dayjs'
import type { ColumnProfile } from '../types'

export function standardizeEmails(value: unknown): unknown {
  if (value == null) return value
  const str = String(value).trim().toLowerCase()
  return str
}

export function standardizePhones(value: unknown): unknown {
  if (value == null) return value
  const digits = String(value).replace(/\D+/g, '')
  if (digits.length === 0) return null
  return `+${digits}`
}

export function standardizeDates(value: unknown): unknown {
  if (value == null || value === '') return value
  const d = dayjs(String(value))
  return d.isValid() ? d.toISOString() : value
}

export function imputeMissing(value: unknown, fallback: unknown): unknown {
  return value == null || value === '' ? fallback : value
}

export function computeQualityScore(profiles: ColumnProfile[], duplicateRatio: number, anomalyCount: number): number {
  const completeness = 1 - average(profiles.map((p) => p.nullRatio))
  const typing = average(profiles.map((p) => p.confidence))
  const uniqueness = average(profiles.map((p) => p.uniqueRatio))
  const duplicatePenalty = 1 - Math.min(1, duplicateRatio)
  const anomalyPenalty = 1 - Math.min(1, anomalyCount / 10)
  const score = 100 * clamp((0.35 * completeness + 0.35 * typing + 0.2 * uniqueness + 0.05 * duplicatePenalty + 0.05 * anomalyPenalty), 0, 1)
  return Math.round(score)
}

function average(values: number[]): number {
  if (values.length === 0) return 0
  return values.reduce((a, b) => a + b, 0) / values.length
}

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v))
}

