import { create } from 'zustand'
import type { DatasetState, DatasetMeta, ValidationIssue } from './types'

interface Actions {
  reset: () => void
  setDataset: (meta: DatasetMeta, previewRows: Record<string, unknown>[]) => void
  setIssues: (issues: ValidationIssue[]) => void
  setQuality: (qualityScore: number, duplicateRatio: number, anomalyCount: number) => void
}

const initialState: DatasetState = {
  meta: null,
  previewRows: [],
  issues: [],
  qualityScore: 0,
  duplicateRatio: 0,
  anomalyCount: 0,
  updatedAt: null,
}

export const useDatasetStore = create<DatasetState & Actions>((set) => ({
  ...initialState,
  reset: () => set({ ...initialState }),
  setDataset: (meta, previewRows) => set({ meta, previewRows, updatedAt: Date.now() }),
  setIssues: (issues) => set({ issues, updatedAt: Date.now() }),
  setQuality: (qualityScore, duplicateRatio, anomalyCount) =>
    set({ qualityScore, duplicateRatio, anomalyCount, updatedAt: Date.now() }),
}))

