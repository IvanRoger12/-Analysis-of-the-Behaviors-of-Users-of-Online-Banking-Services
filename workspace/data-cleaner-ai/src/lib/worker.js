/* Web worker for heavy operations (placeholder) */
self.onmessage = (e) => {
  const { op, payload } = e.data || {}
  if (op === 'noop') {
    self.postMessage({ ok: true, payload })
  }
}

