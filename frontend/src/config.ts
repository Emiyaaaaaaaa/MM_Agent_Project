const API_BASE = import.meta.env.VITE_API_BASE || ''
const WS_BASE = import.meta.env.VITE_WS_BASE || ''

export const getApiUrl = (path: string) => `${API_BASE}${path}`

export const getWsUrl = (path: string) => {
  if (WS_BASE) {
    return `${WS_BASE}${path}`
  }
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
  return `${protocol}://${window.location.host}${path}`
}
