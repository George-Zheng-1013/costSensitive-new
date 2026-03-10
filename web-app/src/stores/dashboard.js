import { defineStore } from 'pinia'
import {
    getAiInsights,
    getAlerts,
    getModelMetrics,
    getGeoDrilldown,
    getUnknownClusterSummary,
    getUnknownClusterTrend,
    getOverview,
    rebuildUnknownClusters,
    getSourceHeatmap,
    getXaiDetail,
    getXaiExplain,
    getXaiSamples,
} from '../services/api'

export const useDashboardStore = defineStore('dashboard', {
    state: () => ({
        loading: false,
        overview: {
            total: 0,
            medium_high: 0,
            unknown_count: 0,
            security_score: 0,
            level_dist: { low: 0, medium: 0, high: 0 },
            trend: [],
            top_ip: [],
        },
        alerts: [],
        xaiSamples: [],
        xaiDetail: null,
        xaiDetailLoading: false,
        xaiExplain: null,
        xaiExplainLoading: false,
        xaiCompareItems: [],
        xaiCompareLoading: false,
        aiAlertInsights: [],
        aiBehaviorInsights: [],
        modelMetrics: { num_classes: 11, radar: { labels: [], baseline: [], netguard: [] } },
        geoLoading: false,
        geoHeatmap: {
            global: { points: [], stats: {}, updated_at: '' },
            china: { points: [], stats: {}, updated_at: '' },
        },
        geoDrilldownLoading: false,
        geoDrilldown: {
            area: null,
            top_ips: [],
            recent_alerts: [],
            updated_at: '',
        },
        unknownClusterLoading: false,
        unknownClusterSummary: {
            generated_at: '',
            total_unknown: 0,
            noise_count: 0,
            clusters: [],
            spikes: [],
        },
        unknownClusterTrend: {
            cluster_ids: [],
            series: [],
        },
        wsStatus: {
            overview: 'idle',
            alerts: 'idle',
            ai: 'idle',
        },
    }),

    getters: {
        mediumHighRatio(state) {
            if (!state.overview.total) {
                return 0
            }
            return Number(((state.overview.medium_high / state.overview.total) * 100).toFixed(2))
        },
    },

    actions: {
        _normalizeRadar(radar) {
            const src = radar && typeof radar === 'object' ? radar : {}
            const labelsRaw = Array.isArray(src.labels) ? src.labels : []
            const toNumbers = (arr) => {
                if (!Array.isArray(arr)) return []
                return arr.map((v) => {
                    const n = Number(v)
                    return Number.isFinite(n) ? n : 0
                })
            }

            const baseline = toNumbers(src.baseline)
            const netguard = toNumbers(src.netguard)
            const dim = Math.max(labelsRaw.length, baseline.length, netguard.length, 1)

            const labels = Array.from({ length: dim }, (_, i) => {
                const x = labelsRaw[i]
                return typeof x === 'string' && x.trim() ? x : `Metric-${i + 1}`
            })

            const pad = (arr) => {
                const out = arr.slice(0, dim)
                while (out.length < dim) out.push(0)
                return out
            }

            return {
                labels,
                baseline: pad(baseline),
                netguard: pad(netguard),
            }
        },

        _normalizeModelMetrics(metrics) {
            const src = metrics && typeof metrics === 'object' ? metrics : {}
            const numClasses = Number(src.num_classes)
            return {
                ...src,
                num_classes: Number.isFinite(numClasses) ? numClasses : 11,
                radar: this._normalizeRadar(src.radar),
            }
        },

        _normalizePacketContrib(xaiDetail) {
            const payload = xaiDetail?.packet_contrib
            if (Array.isArray(payload)) {
                return payload.map((row, idx) => {
                    const x = row && typeof row === 'object' ? row : {}
                    const value = Number(x.importance ?? x.score_drop ?? 0)
                    return {
                        packet_index: Number.isFinite(Number(x.packet_index)) ? Number(x.packet_index) : idx,
                        importance: Number.isFinite(value) ? value : 0,
                    }
                })
            }
            if (payload && typeof payload === 'object' && Array.isArray(payload.scores)) {
                return payload.scores.map((v, idx) => {
                    const n = Number(v)
                    return {
                        packet_index: idx,
                        importance: Number.isFinite(n) ? n : 0,
                    }
                })
            }
            return []
        },

        _normalizeHeatmapMatrix(xaiDetail) {
            const payload = xaiDetail?.byte_heatmap
            const matrix = Array.isArray(payload?.byte_heatmap)
                ? payload.byte_heatmap
                : (Array.isArray(payload) ? payload : [])

            if (!Array.isArray(matrix)) {
                return []
            }
            return matrix.map((row) =>
                (Array.isArray(row) ? row : []).map((v) => {
                    const n = Number(v)
                    return Number.isFinite(n) ? n : 0
                }),
            )
        },

        _buildXaiModel(raw) {
            const detail = raw && typeof raw === 'object' ? raw : {}
            const packetContrib = this._normalizePacketContrib(detail)
            const byteHeatmapMatrix = this._normalizeHeatmapMatrix(detail)
            return {
                ...detail,
                packetContrib,
                byteHeatmapMatrix,
            }
        },

        _dedupeById(rows) {
            const seen = new Set()
            const out = []
            for (const row of rows) {
                const id = row?.id
                if (id == null) {
                    out.push(row)
                    continue
                }
                if (seen.has(id)) {
                    continue
                }
                seen.add(id)
                out.push(row)
            }
            return out
        },

        async bootstrap() {
            this.loading = true
            try {
                const [overviewRes, alertsRes, samplesRes, alertInsightsRes, behaviorInsightsRes, metricsRes] =
                    await Promise.all([
                        getOverview(),
                        getAlerts(250),
                        getXaiSamples(120),
                        getAiInsights('alert', 50),
                        getAiInsights('behavior', 50),
                        getModelMetrics(),
                    ])

                this.overview = overviewRes
                this.alerts = alertsRes.items || []
                this.xaiSamples = samplesRes.items || []
                this.aiAlertInsights = alertInsightsRes.items || []
                this.aiBehaviorInsights = behaviorInsightsRes.items || []
                this.modelMetrics = this._normalizeModelMetrics(metricsRes || this.modelMetrics)
                try {
                    this.geoHeatmap.global = await getSourceHeatmap('global', 120)
                } catch {
                    // Keep dashboard usable even if geo service is unavailable.
                }
                try {
                    await this.loadUnknownClusters({ rebuild: false })
                } catch {
                    // Keep dashboard usable even if cluster service is unavailable.
                }
            } finally {
                this.loading = false
            }
        },

        async loadUnknownClusters({ rebuild = false } = {}) {
            this.unknownClusterLoading = true
            try {
                if (rebuild) {
                    const rebuilt = await rebuildUnknownClusters({
                        eps: 0.2,
                        minSamples: 4,
                        metric: 'cosine',
                        l2Normalize: true,
                    })
                    this.unknownClusterSummary = rebuilt?.summary || this.unknownClusterSummary
                } else {
                    this.unknownClusterSummary = await getUnknownClusterSummary()
                }
                this.unknownClusterTrend = await getUnknownClusterTrend(48)
                return {
                    summary: this.unknownClusterSummary,
                    trend: this.unknownClusterTrend,
                }
            } finally {
                this.unknownClusterLoading = false
            }
        },

        async loadGeoHeatmap(scope = 'global', force = false) {
            const s = scope === 'china' ? 'china' : 'global'
            const existing = this.geoHeatmap[s]
            if (!force && Array.isArray(existing?.points) && existing.points.length > 0) {
                return existing
            }
            this.geoLoading = true
            try {
                const payload = await getSourceHeatmap(s, 120)
                this.geoHeatmap[s] = payload || { points: [], stats: {}, updated_at: '' }
                return this.geoHeatmap[s]
            } finally {
                this.geoLoading = false
            }
        },

        clearGeoDrilldown() {
            this.geoDrilldown = {
                area: null,
                top_ips: [],
                recent_alerts: [],
                updated_at: '',
            }
        },

        async loadGeoDrilldown({ scope = 'global', countryCode = '', region = '', city = '' } = {}) {
            this.geoDrilldownLoading = true
            try {
                const payload = await getGeoDrilldown({
                    scope,
                    countryCode,
                    region,
                    city,
                    ipLimit: 30,
                    alertLimit: 60,
                })
                this.geoDrilldown = {
                    area: payload?.area || { country_code: countryCode, region, city },
                    top_ips: payload?.top_ips || [],
                    recent_alerts: payload?.recent_alerts || [],
                    updated_at: payload?.updated_at || '',
                }
                return this.geoDrilldown
            } finally {
                this.geoDrilldownLoading = false
            }
        },

        async loadXaiDetail(id) {
            if (!id) {
                this.xaiDetail = null
                return
            }
            this.xaiDetailLoading = true
            try {
                const payload = await getXaiDetail(id)
                this.xaiDetail = this._buildXaiModel(payload)
            } finally {
                this.xaiDetailLoading = false
            }
        },

        async loadXaiExplain(id, refresh = false) {
            if (!id) {
                this.xaiExplain = null
                return null
            }
            this.xaiExplainLoading = true
            try {
                const payload = await getXaiExplain(id, refresh)
                this.xaiExplain = payload || null
                return this.xaiExplain
            } finally {
                this.xaiExplainLoading = false
            }
        },

        async loadXaiCompare(ids) {
            const useIds = Array.isArray(ids)
                ? [...new Set(ids.map((x) => Number(x)).filter((x) => Number.isFinite(x) && x > 0))]
                : []

            if (useIds.length === 0) {
                this.xaiCompareItems = []
                return []
            }

            this.xaiCompareLoading = true
            try {
                const rows = await Promise.all(
                    useIds.map(async (id) => {
                        const raw = await getXaiDetail(id)
                        const x = this._buildXaiModel(raw)
                        const contrib = Array.isArray(x.packetContrib) ? x.packetContrib : []
                        const sorted = [...contrib].sort((a, b) => Number(b.importance || 0) - Number(a.importance || 0))
                        const topPacket = sorted.length > 0 ? Number(sorted[0].packet_index) : -1
                        const topImportance = sorted.length > 0 ? Number(sorted[0].importance || 0) : 0
                        return {
                            id,
                            threat_category: String(x.threat_category || '-'),
                            alert_level: String(x.alert_level || '-'),
                            confidence: Number(x.confidence || 0),
                            risk_score: Number(x.risk_score || 0),
                            is_unknown: Number(x.is_unknown || 0),
                            top_packet: topPacket,
                            top_importance: topImportance,
                            matrix_rows: Array.isArray(x.byteHeatmapMatrix) ? x.byteHeatmapMatrix.length : 0,
                        }
                    }),
                )
                this.xaiCompareItems = rows
                return rows
            } finally {
                this.xaiCompareLoading = false
            }
        },

        applyOverview(payload) {
            this.overview = payload
        },

        async refreshOverview() {
            this.overview = await getOverview()
        },

        async refreshAlerts() {
            const res = await getAlerts(250)
            this.alerts = this._dedupeById(res.items || [])
        },

        async refreshAi() {
            const [alertRes, behaviorRes] = await Promise.all([
                getAiInsights('alert', 50),
                getAiInsights('behavior', 50),
            ])
            this.aiAlertInsights = this._dedupeById(alertRes.items || [])
            this.aiBehaviorInsights = this._dedupeById(behaviorRes.items || [])
        },

        prependAlerts(items) {
            if (!Array.isArray(items) || items.length === 0) {
                return
            }
            const merged = [...items.reverse(), ...this.alerts]
            this.alerts = this._dedupeById(merged).slice(0, 300)
        },

        prependAi(items) {
            if (!Array.isArray(items) || items.length === 0) {
                return
            }
            for (const row of items.reverse()) {
                if (row.insight_type === 'behavior') {
                    this.aiBehaviorInsights = this._dedupeById([row, ...this.aiBehaviorInsights]).slice(0, 80)
                } else {
                    this.aiAlertInsights = this._dedupeById([row, ...this.aiAlertInsights]).slice(0, 80)
                }
            }
        },

        setWsStatus(channel, status) {
            this.wsStatus[channel] = status
        },
    },
})
