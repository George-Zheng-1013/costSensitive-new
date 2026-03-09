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
const geoAreaLabel = computed(() => {
  const area = store.geoDrilldown?.area
  if (!area) return '未选择'
  const parts = [area.country_code, area.region, area.city].filter(Boolean)
  return parts.length > 0 ? parts.join(' / ') : '未选择'
})

async function refreshGeo() {
  await store.loadGeoHeatmap(geoScope.value, true)
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
          <el-table :data="store.geoDrilldown.top_ips" height="240" stripe v-loading="store.geoDrilldownLoading">
            <el-table-column prop="ip" label="IP" min-width="170" />
            <el-table-column prop="count" label="异常数" width="90" />
            <el-table-column prop="region" label="区域" min-width="120" />
            <el-table-column prop="city" label="城市" min-width="120" />
          </el-table>
        </div>
        <div>
          <h4 class="card-title">最近告警样本</h4>
          <el-table :data="store.geoDrilldown.recent_alerts" height="240" stripe v-loading="store.geoDrilldownLoading">
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
}
</style>
