<script setup>
import { computed } from 'vue'

import LevelPieChart from '../components/charts/LevelPieChart.vue'
import TopIpBarChart from '../components/charts/TopIpBarChart.vue'
import TrendLineChart from '../components/charts/TrendLineChart.vue'
import RadarChart from '../components/charts/RadarChart.vue'
import { useDashboardStore } from '../stores/dashboard'

const store = useDashboardStore()

const scoreTag = computed(() => {
  const x = Number(store.overview.security_score || 0)
  if (x >= 85) return 'success'
  if (x >= 65) return 'warning'
  return 'danger'
})
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
      <RadarChart :radar="store.modelMetrics.radar" />
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
}
</style>
