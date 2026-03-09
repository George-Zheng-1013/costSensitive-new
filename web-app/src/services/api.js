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

export function getXaiExplain(id, refresh = false) {
    const r = refresh ? '1' : '0'
    return requestJson(`/api/xai/explain/${id}?refresh=${r}`)
}

export function getAiInsights(type = '', limit = 30) {
    const q = type ? `insight_type=${encodeURIComponent(type)}&` : ''
    return requestJson(`/api/ai/insights?${q}limit=${limit}`)
}

export function getModelMetrics() {
    return requestJson('/api/model/metrics')
}

export function getSourceHeatmap(scope = 'global', limit = 120) {
    const s = scope === 'china' ? 'china' : 'global'
    return requestJson(`/api/geo/source-heatmap?scope=${encodeURIComponent(s)}&levels=medium,high&limit=${limit}`)
}

export function getGeoDrilldown({
    scope = 'global',
    countryCode = '',
    region = '',
    city = '',
    ipLimit = 30,
    alertLimit = 60,
} = {}) {
    const s = scope === 'china' ? 'china' : 'global'
    const query = [
        `scope=${encodeURIComponent(s)}`,
        `country_code=${encodeURIComponent(countryCode)}`,
        `region=${encodeURIComponent(region)}`,
        `city=${encodeURIComponent(city)}`,
        `ip_limit=${encodeURIComponent(ipLimit)}`,
        `alert_limit=${encodeURIComponent(alertLimit)}`,
    ].join('&')
    return requestJson(`/api/geo/source-drilldown?${query}`)
}
