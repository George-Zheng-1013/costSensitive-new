<script setup>
import { computed, ref } from 'vue'
import { useDashboardStore } from '../stores/dashboard'

const store = useDashboardStore()

const levelFilter = ref('all')
const keyword = ref('')
const unknownOnly = ref(false)
const selectedInsight = ref(null)
const detailVisible = ref(false)

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
  if (level === 'high' || level === 'critical') return 'danger'
  if (level === 'medium') return 'warning'
  return 'info'
}

const alertRows = computed(() => {
  const rows = Array.isArray(store.aiAlertInsights) ? store.aiAlertInsights : []
  const kw = String(keyword.value || '').trim().toLowerCase()

  return rows.filter((row) => {
    const level = String(row?.risk_level || 'unknown').toLowerCase()
    if (levelFilter.value !== 'all' && level !== levelFilter.value) {
      return false
    }

    if (unknownOnly.value) {
      const text = String(row?.summary || '') + ' ' + String(row?.title || '')
      if (!text.toLowerCase().includes('unknown')) {
        return false
      }
    }

    if (!kw) return true
    const hay = [
      row?.title,
      row?.scene,
      row?.scenario,
      row?.summary,
      row?.risk_level,
    ]
      .filter(Boolean)
      .join(' ')
      .toLowerCase()
    return hay.includes(kw)
  })
})

const behaviorRows = computed(() => {
  const rows = Array.isArray(store.aiBehaviorInsights) ? store.aiBehaviorInsights : []
  return rows
})

const insightStats = computed(() => {
  const rows = Array.isArray(store.aiAlertInsights) ? store.aiAlertInsights : []
  const total = rows.length
  let high = 0
  let medium = 0
  let confSum = 0
  let confN = 0
  let latest = '-'

  for (const r of rows) {
    const level = String(r?.risk_level || '').toLowerCase()
    if (level === 'high' || level === 'critical') high += 1
    if (level === 'medium') medium += 1
    const c = Number(r?.confidence)
    if (Number.isFinite(c)) {
      confSum += c
      confN += 1
    }
    const ts = String(r?.created_at || r?.timestamp || '').trim()
    if (ts && ts > latest) latest = ts
  }

  return {
    total,
    high,
    medium,
    avgConfidence: confN > 0 ? confSum / confN : 0,
    latest,
  }
})

const behaviorTagDist = computed(() => {
  const rows = Array.isArray(store.aiBehaviorInsights) ? store.aiBehaviorInsights : []
  const dist = {}
  for (const r of rows) {
    const key = String(r?.title || r?.behavior_tag || '未标注').trim() || '未标注'
    dist[key] = (dist[key] || 0) + 1
  }
  const list = Object.entries(dist).map(([label, value]) => ({ label, value }))
  list.sort((a, b) => b.value - a.value)
  const max = list.length > 0 ? list[0].value : 1
  return list.map((x) => ({
    ...x,
    pct: max > 0 ? (x.value / max) * 100 : 0,
  }))
})

function openDetail(item) {
  selectedInsight.value = item
  detailVisible.value = true
}
</script>

<template>
  <div class="insight-page">
    <section class="card panel topbar">
      <h3 class="card-title">AI 研判总览</h3>
      <div class="stat-grid">
        <div class="stat-item">
          <div class="muted">研判总数</div>
          <div class="stat-value">{{ insightStats.total }}</div>
        </div>
        <div class="stat-item">
          <div class="muted">高危数量</div>
          <div class="stat-value danger">{{ insightStats.high }}</div>
        </div>
        <div class="stat-item">
          <div class="muted">中危数量</div>
          <div class="stat-value warning">{{ insightStats.medium }}</div>
        </div>
        <div class="stat-item">
          <div class="muted">平均置信度</div>
          <div class="stat-value">{{ Number(insightStats.avgConfidence).toFixed(3) }}</div>
        </div>
        <div class="stat-item">
          <div class="muted">最近更新</div>
          <div class="stat-sub">{{ insightStats.latest }}</div>
        </div>
      </div>
    </section>

    <section class="insight-grid">
      <section class="card panel alert-panel">
        <div class="panel-head">
          <h3 class="card-title">AI 告警研判流</h3>
          <div class="filters">
            <el-radio-group v-model="levelFilter" size="small">
              <el-radio-button label="all">全部</el-radio-button>
              <el-radio-button label="high">高危</el-radio-button>
              <el-radio-button label="medium">中危</el-radio-button>
            </el-radio-group>
            <el-input v-model="keyword" size="small" clearable placeholder="关键词筛选" class="filter-input" />
            <el-switch v-model="unknownOnly" inline-prompt active-text="Unknown" inactive-text="全部" />
          </div>
        </div>

        <div class="alert-list">
          <article
            v-for="item in alertRows"
            :key="`a-${item.id}`"
            class="alert-card"
            @click="openDetail(item)"
          >
            <div class="alert-head-row">
              <el-tag :type="levelTag(item.risk_level)">{{ item.risk_level || 'unknown' }}</el-tag>
              <b class="title-text">{{ item.title || '未命名告警事件' }}</b>
              <span class="muted ts">{{ item.created_at || item.timestamp || '-' }}</span>
            </div>
            <p class="summary">{{ item.summary || '-' }}</p>
            <div class="meta-row muted">
              <span>场景: {{ item.scene || item.scenario || '-' }}</span>
              <span>置信: {{ Number(item.confidence || 0).toFixed(3) }}</span>
            </div>
            <div class="action-line">
              <el-tag
                v-for="(a, idx) in parseActions(item.actions || item.actions_json).slice(0, 3)"
                :key="`act-${item.id}-${idx}`"
                size="small"
              >
                {{ a }}
              </el-tag>
            </div>
          </article>

          <el-empty v-if="alertRows.length === 0" description="当前筛选条件下无研判结果" />
        </div>
      </section>

      <section class="card panel behavior-panel">
        <h3 class="card-title">已知流量行为画像</h3>

        <div class="behavior-dist">
          <div v-for="item in behaviorTagDist.slice(0, 6)" :key="`dist-${item.label}`" class="dist-row">
            <span class="dist-label" :title="item.label">{{ item.label }}</span>
            <div class="dist-bar-wrap">
              <div class="dist-bar" :style="{ width: `${item.pct}%` }" />
            </div>
            <span class="dist-value">{{ item.value }}</span>
          </div>
          <el-empty v-if="behaviorTagDist.length === 0" description="暂无行为画像数据" :image-size="68" />
        </div>

        <el-table :data="behaviorRows" height="380" stripe>
          <el-table-column prop="created_at" label="时间" width="170" />
          <el-table-column prop="title" label="行为标签" min-width="120" show-overflow-tooltip />
          <el-table-column prop="scene" label="场景" min-width="100" show-overflow-tooltip />
          <el-table-column prop="risk_level" label="风险" width="90" />
          <el-table-column prop="summary" label="行为说明" min-width="180" show-overflow-tooltip />
        </el-table>
      </section>
    </section>

    <el-drawer v-model="detailVisible" :with-header="false" size="38%" direction="rtl" class="detail-drawer">
      <template #default>
        <div v-if="selectedInsight" class="detail-wrap">
          <h3>研判详情</h3>
          <p><b>标题：</b>{{ selectedInsight.title || '未命名告警事件' }}</p>
          <p><b>风险等级：</b>{{ selectedInsight.risk_level || '-' }}</p>
          <p><b>时间：</b>{{ selectedInsight.created_at || selectedInsight.timestamp || '-' }}</p>
          <p><b>场景：</b>{{ selectedInsight.scene || selectedInsight.scenario || '-' }}</p>
          <p><b>摘要：</b>{{ selectedInsight.summary || '-' }}</p>
          <el-divider />
          <h4>处置建议</h4>
          <ul>
            <li v-for="(a, idx) in parseActions(selectedInsight.actions || selectedInsight.actions_json)" :key="`da-${idx}`">{{ a }}</li>
          </ul>
          <h4>原始记录</h4>
          <pre>{{ JSON.stringify(selectedInsight.raw || selectedInsight, null, 2) }}</pre>
        </div>
      </template>
    </el-drawer>
  </div>
</template>

<style scoped>
.insight-page {
  width: 100%;
  min-width: 0;
  overflow-x: hidden;
  display: grid;
  gap: 12px;
}

.panel {
  padding: 14px;
  min-width: 0;
}

.topbar {
  display: grid;
  gap: 8px;
}

.stat-grid {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 10px;
}

.stat-item {
  padding: 10px;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: #f8fbff;
  min-width: 0;
}

.stat-value {
  font-size: 22px;
  font-weight: 700;
}

.stat-sub {
  font-size: 13px;
  color: var(--text-sub);
  word-break: break-all;
}

.danger {
  color: #d14b4b;
}

.warning {
  color: #f5a524;
}

.insight-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.6fr) minmax(0, 1fr);
  gap: 12px;
  min-width: 0;
}

.alert-panel,
.behavior-panel {
  min-width: 0;
  overflow: hidden;
}

.panel-head {
  display: flex;
  gap: 10px;
  justify-content: space-between;
  align-items: flex-start;
}

.filters {
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
}

.filter-input {
  width: 180px;
}

.alert-list {
  margin-top: 10px;
  display: grid;
  gap: 10px;
  max-height: 640px;
  overflow: auto;
  padding-right: 4px;
}

.alert-card {
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 10px;
  cursor: pointer;
  background: #fbfdff;
}

.alert-card:hover {
  border-color: #9eb8ea;
  background: #f7fbff;
}

.alert-head-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.title-text {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.ts {
  font-size: 12px;
  white-space: nowrap;
}

.summary {
  margin: 8px 0;
  color: var(--text-main);
  line-height: 1.55;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.meta-row {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.action-line {
  margin-top: 8px;
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}

.behavior-dist {
  display: grid;
  gap: 8px;
  margin-bottom: 10px;
}

.dist-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 1.3fr auto;
  align-items: center;
  gap: 8px;
}

.dist-label {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.dist-bar-wrap {
  height: 9px;
  border-radius: 999px;
  background: #e7eefb;
  overflow: hidden;
}

.dist-bar {
  height: 100%;
  background: linear-gradient(90deg, #6ca8ff, #2e6fd8);
}

.dist-value {
  font-weight: 600;
  color: #1f2a44;
}

.detail-wrap pre {
  background: #f4f7fc;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px;
  max-height: 320px;
  overflow: auto;
  font-size: 12px;
}

ul {
  margin: 8px 0;
  padding-left: 18px;
}

@media (max-width: 1280px) {
  .stat-grid {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }

  .insight-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 760px) {
  .stat-grid {
    grid-template-columns: 1fr;
  }

  .panel-head {
    flex-direction: column;
  }

  .filter-input {
    width: 100%;
  }
}
</style>
