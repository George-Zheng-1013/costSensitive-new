<script setup>
import { computed, onMounted, ref, watch } from 'vue'

import LevelPieChart from '../components/charts/LevelPieChart.vue'
import SourceGeoMap from '../components/charts/SourceGeoMap.vue'
import TopIpBarChart from '../components/charts/TopIpBarChart.vue'
import TrendLineChart from '../components/charts/TrendLineChart.vue'
import RadarChart from '../components/charts/RadarChart.vue'
import { useDashboardStore } from '../stores/dashboard'

const store = useDashboardStore()
const geoScope = ref('global')

const scoreTag = computed(() => {
    const x = Number(store.overview.security_score || 0)
    if (x >= 85) return 'success'
    if (x >= 65) return 'warning'
    return 'danger'
})

const geoPayload = computed(() => {
    const s = geoScope.value === 'china' ? 'china' : 'global'
    return store.geoHeatmap?.[s] || { points: [], stats: {}, updated_at: '' }
})

const geoPointCount = computed(() => Number(geoPayload.value?.stats?.point_count || 0))
const geoResolvedCount = computed(() => Number(geoPayload.value?.stats?.resolved_alert_count || 0))
const clusterSummary = computed(() => store.unknownClusterSummary || { clusters: [], spikes: [] })
const clusterRows = computed(() => {
    const rows = Array.isArray(clusterSummary.value?.clusters) ? clusterSummary.value.clusters : []
    const hintMap = store.unknownClusterAiHints?.by_id || {}
    return rows.slice(0, 8).map((row) => {
        const cid = String(row?.cluster_id || '')
        const hint = hintMap[cid] || null
        return {
            ...row,
            ai_hint: hint,
        }
    })
})
const clusterSeries = computed(() => {
    const rows = Array.isArray(store.unknownClusterTrend?.series) ? store.unknownClusterTrend.series : []
    const clusterIds = Array.isArray(store.unknownClusterTrend?.cluster_ids) ? store.unknownClusterTrend.cluster_ids : []
    if (rows.length === 0 || clusterIds.length === 0) return []
    const latest = rows[rows.length - 1] || {}
    return clusterIds.slice(0, 4).map((id) => {
        const value = Number(latest?.[id] || 0)
        const max = Math.max(1, ...rows.map((x) => Number(x?.[id] || 0)))
        return {
            id,
            value,
            pct: (value / max) * 100,
        }
    })
})
const unknownSpikeCount = computed(() => Number(clusterSummary.value?.spikes?.length || 0))
const clusterHintSourceText = computed(() => {
    const source = String(store.unknownClusterAiHints?.source || 'none')
    if (source === 'llm') return 'AI研判'
    if (source === 'mixed') return 'AI+规则研判'
    if (source === 'rule') return '规则兜底研判'
    return '暂无研判'
})

function riskTagType(level) {
    const lv = String(level || '').toLowerCase()
    if (lv === 'critical' || lv === 'high') return 'danger'
    if (lv === 'medium') return 'warning'
    return 'success'
}
const geoAreaLabel = computed(() => {
    const area = store.geoDrilldown?.area
    if (!area) return '未选择'
    const parts = [area.country_code, area.region, area.city].filter(Boolean)
    return parts.length > 0 ? parts.join(' / ') : '未选择'
})

async function refreshGeo() {
    await store.loadGeoHeatmap(geoScope.value, true)
}

async function refreshUnknownClusters(rebuild = false) {
    await store.loadUnknownClusters({ rebuild })
}

async function onGeoPointClick(payload) {
    await store.loadGeoDrilldown({
        scope: geoScope.value,
        countryCode: payload?.country_code || '',
        region: payload?.region || '',
        city: payload?.city || '',
    })
}

onMounted(async () => {
    await store.loadGeoHeatmap(geoScope.value, false)
    await store.loadUnknownClusters({ rebuild: false })
})

watch(
    () => geoScope.value,
    async () => {
        await store.loadGeoHeatmap(geoScope.value, false)
        store.clearGeoDrilldown()
    },
)
</script>

<template>
    <div class="overview-grid">
        <section class="card metric-card">
            <div class="muted">总流量记录</div>
            <div class="metric-value">{{ store.overview.total }}</div>
        </section>

        <section class="card metric-card">
            <div class="muted">中高危告警</div>
            <div class="metric-value">{{ store.overview.medium_high }}</div>
            <div class="muted">占比 {{ store.mediumHighRatio }}%</div>
        </section>

        <section class="card metric-card">
            <div class="muted">Unknown 检测数</div>
            <div class="metric-value">{{ store.overview.unknown_count }}</div>
        </section>

        <section class="card metric-card">
            <div class="muted">安全评分</div>
            <div class="metric-value">{{ Number(store.overview.security_score || 0).toFixed(1) }}</div>
            <el-tag :type="scoreTag">实时更新</el-tag>
        </section>

        <section class="card chart-card two-col">
            <h3 class="card-title">24 小时趋势</h3>
            <TrendLineChart :data="store.overview.trend" />
        </section>

        <section class="card chart-card">
            <h3 class="card-title">告警等级分布</h3>
            <LevelPieChart :level-dist="store.overview.level_dist" />
        </section>

        <section class="card chart-card">
            <h3 class="card-title">高风险源 IP TOP10</h3>
            <TopIpBarChart :data="store.overview.top_ip" />
        </section>

        <section class="card chart-card two-col">
            <h3 class="card-title">模型综合指标</h3>
            <p class="muted">当前分类体系: {{ Number(store.modelMetrics.num_classes || 11) }} 类</p>
            <RadarChart :radar="store.modelMetrics.radar" />
        </section>

        <section class="card chart-card two-col">
            <div class="cluster-head">
                <h3 class="card-title">Unknown Cluster 监控</h3>
                <div class="cluster-actions">
                    <el-tag type="danger" v-if="unknownSpikeCount > 0">突增簇 {{ unknownSpikeCount }}</el-tag>
                    <el-button size="small" :loading="store.unknownClusterLoading"
                        @click="refreshUnknownClusters(true)">
                        重建聚类
                    </el-button>
                    <el-button size="small" :loading="store.unknownClusterLoading"
                        @click="refreshUnknownClusters(false)">
                        刷新
                    </el-button>
                </div>
            </div>
            <div class="cluster-meta muted">
                <span>未知样本: {{ Number(clusterSummary.total_unknown || 0) }}</span>
                <span>噪声点: {{ Number(clusterSummary.noise_count || 0) }}</span>
                <span>簇研判: {{ clusterHintSourceText }}</span>
                <span>更新时间: {{ clusterSummary.generated_at || '-' }}</span>
            </div>
            <div class="cluster-grid">
                <div>
                    <el-table :data="clusterRows" height="220" stripe v-loading="store.unknownClusterLoading">
                        <el-table-column prop="cluster_id" label="簇ID" min-width="130" />
                        <el-table-column prop="size" label="当前规模" width="90" />
                        <el-table-column prop="growth" label="增长" width="80" />
                        <el-table-column label="突增" width="80">
                            <template #default="scope">
                                <el-tag :type="scope.row.is_spike ? 'danger' : 'info'">
                                    {{ scope.row.is_spike ? '是' : '否' }}
                                </el-tag>
                            </template>
                        </el-table-column>
                    </el-table>
                    <div class="cluster-ai-list">
                        <div v-for="row in clusterRows" :key="`hint-${row.cluster_id}`" class="cluster-ai-row">
                            <div class="cluster-ai-row-head">
                                <span class="cluster-ai-id">{{ row.cluster_id }}</span>
                                <el-tag size="small" :type="riskTagType(row.ai_hint?.risk_level)">
                                    {{ String(row.ai_hint?.risk_level || 'low').toUpperCase() }}
                                </el-tag>
                                <span class="cluster-ai-type">{{ row.ai_hint?.possible_type || '未知加密流量' }}</span>
                            </div>
                            <div class="cluster-ai-summary muted">
                                {{ row.ai_hint?.summary || '研判生成中，请稍后刷新。' }}
                            </div>
                        </div>
                        <el-empty v-if="clusterRows.length === 0" description="暂无簇研判" :image-size="56" />
                    </div>
                </div>
                <div class="cluster-spark">
                    <h4 class="card-title">最新簇强度</h4>
                    <div v-for="row in clusterSeries" :key="row.id" class="cluster-spark-row">
                        <span class="cluster-id" :title="row.id">{{ row.id }}</span>
                        <div class="cluster-bar-wrap">
                            <div class="cluster-bar" :style="{ width: `${row.pct}%` }" />
                        </div>
                        <span class="cluster-val">{{ row.value }}</span>
                    </div>
                    <el-empty v-if="clusterSeries.length === 0" description="暂无聚类趋势" :image-size="68" />
                </div>
            </div>
        </section>

        <section class="card chart-card geo-card all-col">
            <div class="geo-head">
                <h3 class="card-title">流量源 IP 全球/全国热力图</h3>
                <div class="geo-controls">
                    <el-select v-model="geoScope" class="geo-select" size="small">
                        <el-option label="全球" value="global" />
                        <el-option label="全国" value="china" />
                    </el-select>
                    <el-button size="small" :loading="store.geoLoading" @click="refreshGeo">刷新</el-button>
                </div>
            </div>
            <div class="geo-meta muted">
                <span>已定位点位: {{ geoPointCount }}</span>
                <span>已解析异常条数: {{ geoResolvedCount }}</span>
                <span>更新时间: {{ geoPayload.updated_at || '-' }}</span>
            </div>
            <SourceGeoMap :scope="geoScope" :data="geoPayload" @point-click="onGeoPointClick" />

            <div class="geo-drill-head">
                <h4>区域下钻</h4>
                <span class="muted">当前区域: {{ geoAreaLabel }}</span>
                <span class="muted">更新时间: {{ store.geoDrilldown.updated_at || '-' }}</span>
            </div>

            <div class="geo-drill-grid">
                <div>
                    <h4 class="card-title">Top 源 IP</h4>
                    <el-table :data="store.geoDrilldown.top_ips" height="240" stripe
                        v-loading="store.geoDrilldownLoading">
                        <el-table-column prop="ip" label="IP" min-width="170" />
                        <el-table-column prop="count" label="异常数" width="90" />
                        <el-table-column prop="region" label="区域" min-width="120" />
                        <el-table-column prop="city" label="城市" min-width="120" />
                    </el-table>
                </div>
                <div>
                    <h4 class="card-title">最近告警样本</h4>
                    <el-table :data="store.geoDrilldown.recent_alerts" height="240" stripe
                        v-loading="store.geoDrilldownLoading">
                        <el-table-column prop="timestamp" label="时间" width="170" />
                        <el-table-column prop="src_ip" label="源 IP" min-width="140" />
                        <el-table-column prop="threat_category" label="类别" min-width="150" />
                        <el-table-column prop="alert_level" label="等级" width="90" />
                    </el-table>
                </div>
            </div>
        </section>
    </div>
</template>

<style scoped>
.overview-grid {
    display: grid;
    gap: 12px;
    grid-template-columns: repeat(4, minmax(0, 1fr));
}

.metric-card,
.chart-card {
    padding: 14px;
}

.all-col {
    grid-column: 1 / -1;
}

.geo-card {
    overflow: hidden;
}

.cluster-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
}

.cluster-actions {
    display: flex;
    align-items: center;
    gap: 8px;
}

.cluster-meta {
    margin-top: 6px;
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
}

.cluster-grid {
    margin-top: 8px;
    display: grid;
    grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr);
    gap: 10px;
}

.cluster-spark {
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 10px;
    min-width: 0;
}

.cluster-spark-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 1.5fr auto;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
}

.cluster-id {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.cluster-bar-wrap {
    height: 8px;
    border-radius: 999px;
    background: #e8eef8;
    overflow: hidden;
}

.cluster-bar {
    height: 100%;
    background: linear-gradient(90deg, #f5a524, #d14b4b);
}

.cluster-val {
    color: #1f2a44;
    font-weight: 600;
}

.cluster-ai-list {
    margin-top: 8px;
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 8px 10px;
}

.cluster-ai-row {
    padding: 8px 0;
    border-bottom: 1px dashed var(--line);
}

.cluster-ai-row:last-child {
    border-bottom: none;
}

.cluster-ai-row-head {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}

.cluster-ai-id {
    font-weight: 600;
    color: #1f2a44;
}

.cluster-ai-type {
    color: #1f2a44;
}

.cluster-ai-summary {
    margin-top: 4px;
    line-height: 1.5;
    word-break: break-word;
}

.geo-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
}

.geo-controls {
    display: flex;
    align-items: center;
    gap: 8px;
}

.geo-select {
    width: 120px;
}

.geo-meta {
    margin-top: 6px;
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
}

.geo-drill-head {
    margin-top: 10px;
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-items: center;
}

.geo-drill-head h4 {
    margin: 0;
}

.geo-drill-grid {
    margin-top: 8px;
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
}

.two-col {
    grid-column: span 2;
}

@media (max-width: 1260px) {
    .overview-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .two-col {
        grid-column: span 2;
    }
}

@media (max-width: 760px) {
    .overview-grid {
        grid-template-columns: 1fr;
    }

    .two-col {
        grid-column: span 1;
    }

    .all-col {
        grid-column: span 1;
    }

    .geo-head {
        flex-direction: column;
        align-items: flex-start;
    }

    .geo-drill-grid {
        grid-template-columns: 1fr;
    }

    .cluster-grid {
        grid-template-columns: 1fr;
    }
}
</style>
