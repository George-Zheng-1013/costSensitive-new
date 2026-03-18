<script setup>
import { computed, ref } from 'vue'
import { useDashboardStore } from '../stores/dashboard'

const store = useDashboardStore()
const selectedLevels = ref(['high', 'medium'])
const keyword = ref('')
const timeWindow = ref('24h')

const rows = computed(() => {
  const now = Date.now()
  const kw = keyword.value.trim().toLowerCase()
  const levelSet = new Set(selectedLevels.value)
  const windowMsMap = {
    '15m': 15 * 60 * 1000,
    '1h': 60 * 60 * 1000,
    '6h': 6 * 60 * 60 * 1000,
    '24h': 24 * 60 * 60 * 1000,
    all: null,
  }
  const windowMs = windowMsMap[timeWindow.value]

  return store.alerts.filter((row) => {
    if (!levelSet.has(row.alert_level)) {
      return false
    }

    if (windowMs != null && row.timestamp) {
      const ts = String(row.timestamp).replace(' ', 'T')
      const t = new Date(ts).getTime()
      if (Number.isFinite(t) && now - t > windowMs) {
        return false
      }
    }

    if (!kw) {
      return true
    }

    const haystack = [
      row.src_ip,
      row.dst_ip,
      row.protocol,
      row.threat_category,
      row.message,
      row.explain_reason,
    ]
      .filter(Boolean)
      .join(' ')
      .toLowerCase()
    return haystack.includes(kw)
  })
})

function levelType(level) {
  if (level === 'high') return 'danger'
  if (level === 'medium') return 'warning'
  return 'success'
}
</script>

<template>
  <div class="card table-wrap">
    <h3 class="card-title">实时告警流</h3>
    <div class="toolbar">
      <el-checkbox-group v-model="selectedLevels">
        <el-checkbox label="high">High</el-checkbox>
        <el-checkbox label="medium">Medium</el-checkbox>
        <el-checkbox label="low">Low</el-checkbox>
      </el-checkbox-group>
      <el-input v-model="keyword" clearable placeholder="关键词：IP/类别/理由" class="kw" />
      <el-select v-model="timeWindow" class="window">
        <el-option label="15 分钟" value="15m" />
        <el-option label="1 小时" value="1h" />
        <el-option label="6 小时" value="6h" />
        <el-option label="24 小时" value="24h" />
        <el-option label="全部" value="all" />
      </el-select>
      <span class="muted">当前 {{ rows.length }} 条</span>
    </div>
    <el-table :data="rows" height="680" stripe>
      <el-table-column prop="timestamp" label="时间" width="180" />
      <el-table-column prop="src_ip" label="源 IP" width="150" />
      <el-table-column prop="dst_ip" label="目标 IP" width="150" />
      <el-table-column prop="threat_category" label="类别" min-width="180" />
      <el-table-column label="告警等级" width="110">
        <template #default="scope">
          <el-tag :type="levelType(scope.row.alert_level)">{{ scope.row.alert_level }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="confidence" label="置信度" width="90" />
      <el-table-column prop="is_unknown" label="Unknown" width="100" />
      <el-table-column prop="centroid_distance" label="Distance" width="100" />
      <el-table-column prop="message" label="理由" min-width="280" show-overflow-tooltip />
    </el-table>
  </div>
</template>

<style scoped>
.table-wrap {
  padding: 14px;
}

.toolbar {
  margin-bottom: 12px;
  display: flex;
  gap: 10px;
  align-items: center;
  flex-wrap: wrap;
}

.kw {
  width: 260px;
}

.window {
  width: 130px;
}
</style>
