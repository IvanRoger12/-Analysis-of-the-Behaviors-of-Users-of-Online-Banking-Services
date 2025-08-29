import { create } from 'zustand'

export const useDataStore = create((set, get) => ({
  rawFile: null,
  rawData: null,
  columns: [],
  previewRows: [],
  cleanedData: null,
  schema: null,
  qualityScore: null,
  status: 'idle',
  error: null,

  setStatus: (status) => set({ status }),
  setError: (error) => set({ error }),
  setFile: (file) => set({ rawFile: file }),
  setRaw: (raw) => set({ rawData: raw }),
  setPreview: (rows) => set({ previewRows: rows }),
  setColumns: (cols) => set({ columns: cols }),
  setSchema: (schema) => set({ schema }),
  setCleaned: (rows) => set({ cleanedData: rows }),
  setQuality: (score) => set({ qualityScore: score }),
  reset: () => set({ rawFile: null, rawData: null, columns: [], previewRows: [], cleanedData: null, schema: null, qualityScore: null, status:'idle', error:null })
}))

