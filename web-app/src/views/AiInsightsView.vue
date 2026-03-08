<script setup>
import { computed } from 'vue'
import { useDashboardStore } from '../stores/dashboard'

const store = useDashboardStore()

const alertRows = computed(() => store.aiAlertInsights)
const behaviorRows = computed(() => store.aiBehaviorInsights)

function parseActions(value) {
  if (Array.isArray(value)) return value
  if (!value) return []
  try {
    return JSON.parse(value)
  } catch {
    return []
  }
}

function levelTag(level) {
  if (level === 'high') return 'danger'
  if (level === 'medium') return 'warning'
  return 'info'
}
</script>

<template>
  <div class="insight-grid">
    <section class="card panel">
      <h3 class="card-title">AI 告警研判</h3>
      <el-timeline>
        <el-timeline-item v-for="item in alertRows" :key="`a-${item.id}`" :timestamp="item.timestamp">
          <div class="entry-row">
            <el-tag :type="levelTag(item.risk_level)">{{ item.risk_level || 'unknown' }}</el-tag>
            <b>{{ item.title || '未命名告警事件' }}</b>
          </div>
          <p class="muted">{{ item.summary }}</p>
          <p class="muted">场景: {{ item.scenario || '-' }}</p>
          <div class="action-line">
            <el-tag v-for="(a, idx) in parseActions(item.actions || item.actions_json)" :key="idx" size="small">
              {{ a }}
            </el-tag>
          </div>
        </el-timeline-item>
      </el-timeline>
    </section>

    <section class="card panel">
      <h3 class="card-title">已知流量行为分析</h3>
      <el-table :data="behaviorRows" height="650" stripe>
        <el-table-column prop="timestamp" label="时间" width="170" />
        <el-table-column prop="title" label="行为标签" width="180" />
        <el-table-column prop="scenario" label="场景" width="170" />
        <el-table-column prop="summary" label="行为说明" min-width="240" show-overflow-tooltip />
        <el-table-column prop="risk_level" label="风险" width="90" />
      </el-table>
    </section>
  </div>
</template>

<style scoped>
.insight-grid {
  display: grid;
  grid-template-columns: 1.05fr 1fr;
  gap: 12px;
}

.panel {
  padding: 14px;
}

.entry-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.action-line {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

@media (max-width: 980px) {
  .insight-grid {
    grid-template-columns: 1fr;
  }
}
</style>
