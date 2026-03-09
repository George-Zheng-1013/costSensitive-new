import { defineStore } from 'pinia'
import {
    getAiInsights,
    getAlerts,
    getModelMetrics,
    getOverview,
    getXaiDetail,
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
        aiAlertInsights: [],
        aiBehaviorInsights: [],
        modelMetrics: { num_classes: 11, radar: { labels: [], baseline: [], netguard: [] } },
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
                this.modelMetrics = metricsRes || this.modelMetrics
            } finally {
                this.loading = false
            }
        },

        async loadXaiDetail(id) {
            if (!id) {
                this.xaiDetail = null
                return
            }
            this.xaiDetail = await getXaiDetail(id)
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
