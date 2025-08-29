import dayjs from 'dayjs'

const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
const phoneRegex = /^[+]?[(]?[0-9]{1,4}[)]?[-\s0-9]{5,}$/

export function detectColumnTypes(rows) {
  if (!rows?.length) return {}
  const sample = rows.slice(0, 100)
  const columns = Object.keys(sample[0] || {})
  const types = {}

  for (const col of columns) {
    const values = sample.map(r => r?.[col]).filter(v => v !== null && v !== undefined && String(v).trim() !== '')
    const checks = {
      number: values.every(v => !isNaN(Number(v))),
      integer: values.every(v => Number.isInteger(Number(v))),
      date: values.every(v => dayjs(v).isValid()),
      email: values.every(v => String(v).match(emailRegex)),
      phone: values.every(v => String(v).match(phoneRegex)),
      boolean: values.every(v => ['true','false','0','1'].includes(String(v).toLowerCase())),
    }
    let t = 'string'
    if (checks.email) t = 'email'
    else if (checks.phone) t = 'phone'
    else if (checks.boolean) t = 'boolean'
    else if (checks.integer) t = 'integer'
    else if (checks.number) t = 'number'
    else if (checks.date) t = 'date'
    types[col] = t
  }
  return types
}

