const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

async function requestJson(path) {
    const response = await fetch(`${API_BASE}${path}`)
    if (!response.ok) {
        throw new Error(`Request failed: ${response.status}`)
    }
    return response.json()
}

export const apiBase = API_BASE

export function getOverview() {
    return requestJson('/api/overview')
}

export function getAlerts(limit = 200) {
    return requestJson(`/api/alerts?levels=medium,high&limit=${limit}`)
}

export function getXaiSamples(limit = 80) {
    return requestJson(`/api/xai/samples?limit=${limit}`)
}

export function getXaiDetail(id) {
    return requestJson(`/api/xai/detail/${id}`)
}

export function getAiInsights(type = '', limit = 30) {
    const q = type ? `insight_type=${encodeURIComponent(type)}&` : ''
    return requestJson(`/api/ai/insights?${q}limit=${limit}`)
}

export function getModelMetrics() {
    return requestJson('/api/model/metrics')
}
