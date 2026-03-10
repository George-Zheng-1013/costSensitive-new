<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import ByteHeatmapChart from '../components/charts/ByteHeatmapChart.vue'
import PacketContributionChart from '../components/charts/PacketContributionChart.vue'
import { useDashboardStore } from '../stores/dashboard'

const store = useDashboardStore()

const selectedId = ref(null)
const compareIds = ref([])
const selectedPacket = ref(-1)
const byteWindow = ref([0, 255])
const threshold = ref(0)

const detail = computed(() => store.xaiDetail || null)
const explain = computed(() => store.xaiExplain || null)
const hasSelection = computed(() => Number.isFinite(Number(selectedId.value)) && Number(selectedId.value) > 0)

const options = computed(() =>
  (Array.isArray(store.xaiSamples) ? store.xaiSamples : []).map((x) => ({
    value: Number(x.id),
    label: `#${x.id} ${x.title}`,
  })),
)

const packetScores = computed(() =>
  (Array.isArray(detail.value?.packetContrib) ? detail.value.packetContrib : []).map((row) => {
    const n = Number(row?.importance)
    return Number.isFinite(n) ? n : 0
  }),
)

const heatmapMatrix = computed(() =>
  Array.isArray(detail.value?.byteHeatmapMatrix) ? detail.value.byteHeatmapMatrix : [],
)

const maxByteIndex = computed(() => {
  const matrix = heatmapMatrix.value
  let maxLen = 0
  for (const row of matrix) {
    if (Array.isArray(row)) maxLen = Math.max(maxLen, row.length)
  }
  return Math.max(0, maxLen - 1)
})

const ratioText = computed(() => {
  const d = Number(detail.value?.centroid_distance || 0)
  const t = Number(detail.value?.centroid_threshold || 0)
  if (!Number.isFinite(d) || !Number.isFinite(t) || t <= 1e-12) return '-'
  return (d / t).toFixed(3)
})

const topPacketList = computed(() => {
  const ranked = packetScores.value
    .map((v, i) => ({ packet_index: i, importance: v }))
    .sort((a, b) => b.importance - a.importance)
  return ranked.slice(0, 5)
})

const byteHotClusters = computed(() => {
  const matrix = heatmapMatrix.value
  const out = []
  const th = Number(threshold.value || 0)

  for (let p = 0; p < matrix.length; p += 1) {
    const row = Array.isArray(matrix[p]) ? matrix[p] : []
    let start = -1
    let peak = 0
    for (let i = 0; i < row.length; i += 1) {
      const v = Number(row[i] || 0)
      if (v >= th && start < 0) {
        start = i
        peak = v
      } else if (v >= th && start >= 0) {
        peak = Math.max(peak, v)
      } else if (v < th && start >= 0) {
        out.push({ packet_index: p, byte_start: start, byte_end: i - 1, peak })
        start = -1
      }
    }
    if (start >= 0) {
      out.push({ packet_index: p, byte_start: start, byte_end: row.length - 1, peak })
    }
  }

  return out.sort((a, b) => b.peak - a.peak).slice(0, 12)
})

function pickTopPacket() {
  const top = topPacketList.value[0]
  selectedPacket.value = top ? Number(top.packet_index) : -1
}

async function loadSample(id, refreshExplain = false) {
  if (!id) return
  await store.loadXaiDetail(id)
  pickTopPacket()
  const maxIdx = maxByteIndex.value
  byteWindow.value = [0, Math.min(255, maxIdx)]
  threshold.value = 0
  await store.loadXaiExplain(id, refreshExplain)
}

async function refreshExplain() {
  if (!selectedId.value) return
  await store.loadXaiExplain(selectedId.value, true)
}

async function loadCompareRows() {
  await store.loadXaiCompare(compareIds.value)
}

function onHeatCellClick(payload) {
  if (!payload) return
  selectedPacket.value = Number(payload.packetIndex)
}

watch(
  () => selectedId.value,
  async (v) => {
    if (!v) {
      store.xaiDetail = null
      store.xaiExplain = null
      selectedPacket.value = -1
      byteWindow.value = [0, 255]
      threshold.value = 0
      return
    }
    await loadSample(v, false)
  },
)

watch(
  () => compareIds.value,
  async () => {
    await loadCompareRows()
  },
  { deep: true },
)

onMounted(async () => {
  if (!store.xaiSamples || store.xaiSamples.length === 0) {
    await store.bootstrap()
  }
})
</script>

<template>
  <div class="xai-v2">
    <section class="card panel toolbar">
      <div class="toolbar-left">
        <h3 class="card-title">可解释分析工作台</h3>
        <p class="muted">会话级 -> 包级 -> 字节级证据联动，支持 AI 解读与多样本对比。</p>
      </div>
      <div class="toolbar-right">
        <el-select v-model="selectedId" clearable filterable placeholder="选择告警样本" class="w280">
          <el-option v-for="item in options" :key="item.value" :label="item.label" :value="item.value" />
        </el-select>
        <el-button :loading="store.xaiExplainLoading" :disabled="!hasSelection" @click="refreshExplain">刷新 AI 解读</el-button>
      </div>
    </section>

    <section v-if="hasSelection" class="kpi-grid">
      <div class="card panel kpi">
        <div class="muted">最终判定</div>
        <div class="kpi-value">{{ detail?.threat_category || '-' }}</div>
      </div>
      <div class="card panel kpi">
        <div class="muted">置信度</div>
        <div class="kpi-value">{{ Number(detail?.confidence || 0).toFixed(4) }}</div>
      </div>
      <div class="card panel kpi">
        <div class="muted">异常分数</div>
        <div class="kpi-value">{{ Number(detail?.risk_score || 0).toFixed(4) }}</div>
      </div>
      <div class="card panel kpi">
        <div class="muted">Distance/Threshold</div>
        <div class="kpi-value">{{ ratioText }}</div>
      </div>
    </section>

    <section v-if="hasSelection" class="chart-grid">
      <div class="card panel">
        <h3 class="card-title">包级贡献热区</h3>
        <PacketContributionChart :scores="packetScores" :selected-packet="selectedPacket" @packet-click="selectedPacket = $event" />
      </div>

      <div class="card panel">
        <h3 class="card-title">字节级二维热力图</h3>
        <div class="control-row">
          <span class="muted">字节窗口</span>
          <el-slider
            v-model="byteWindow"
            range
            :min="0"
            :max="maxByteIndex"
            :disabled="maxByteIndex <= 0"
            class="window-slider"
          />
          <span class="muted">阈值 {{ Number(threshold).toFixed(3) }}</span>
          <el-slider v-model="threshold" :min="0" :max="1" :step="0.01" class="th-slider" />
        </div>
        <ByteHeatmapChart
          :matrix="heatmapMatrix"
          :byte-start="byteWindow[0]"
          :byte-end="byteWindow[1]"
          :threshold="threshold"
          :selected-packet="selectedPacket"
          @cell-click="onHeatCellClick"
        />
      </div>
    </section>

    <section v-if="hasSelection" class="analysis-grid">
      <div class="card panel">
        <h3 class="card-title">AI 解读与建议</h3>
        <el-skeleton :loading="store.xaiExplainLoading" animated :rows="6">
          <template #default>
            <p><b>摘要：</b>{{ explain?.summary || '暂无' }}</p>
            <p class="muted">来源: {{ explain?.source || '-' }} | 置信: {{ Number(explain?.confidence || 0).toFixed(3) }}</p>
            <el-divider />
            <h4>关键原因</h4>
            <ul>
              <li v-for="(x, idx) in (explain?.why || [])" :key="`w-${idx}`">{{ x }}</li>
            </ul>
            <h4>处置建议</h4>
            <ul>
              <li v-for="(x, idx) in (explain?.actions || [])" :key="`a-${idx}`">{{ x }}</li>
            </ul>
          </template>
        </el-skeleton>
      </div>

      <div class="card panel">
        <h3 class="card-title">可疑模式聚类 (Top12)</h3>
        <el-table :data="byteHotClusters" height="300" stripe>
          <el-table-column prop="packet_index" label="包" width="70" />
          <el-table-column prop="byte_start" label="字节起" width="86" />
          <el-table-column prop="byte_end" label="字节止" width="86" />
          <el-table-column prop="peak" label="峰值强度" />
        </el-table>
      </div>
    </section>

    <section v-if="hasSelection" class="card panel">
      <h3 class="card-title">多样本对比</h3>
      <div class="control-row">
        <el-select v-model="compareIds" multiple filterable collapse-tags placeholder="选择多个样本进行横向对比" class="w420">
          <el-option v-for="item in options" :key="`cmp-${item.value}`" :label="item.label" :value="item.value" />
        </el-select>
        <el-button :loading="store.xaiCompareLoading" @click="loadCompareRows">刷新对比</el-button>
      </div>
      <el-table :data="store.xaiCompareItems" height="300" stripe>
        <el-table-column prop="id" label="样本ID" width="90" />
        <el-table-column prop="threat_category" label="类别" min-width="140" />
        <el-table-column prop="alert_level" label="等级" width="80" />
        <el-table-column prop="confidence" label="置信度" width="100" />
        <el-table-column prop="risk_score" label="风险分" width="100" />
        <el-table-column prop="is_unknown" label="Unknown" width="90" />
        <el-table-column prop="top_packet" label="Top包" width="90" />
        <el-table-column prop="top_importance" label="Top贡献" width="110" />
      </el-table>
    </section>
  </div>
</template>

<style scoped>
.xai-v2 {
  display: grid;
  gap: 12px;
}

.panel {
  padding: 14px;
}

.toolbar {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: flex-start;
}

.toolbar-right {
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
}

.w280 {
  width: 280px;
}

.w420 {
  width: 420px;
}

.kpi-grid {
  display: grid;
  gap: 10px;
  grid-template-columns: repeat(4, minmax(0, 1fr));
}

.kpi {
  display: grid;
  gap: 4px;
}

.kpi-value {
  font-size: 24px;
  font-weight: 700;
  color: #1f2a44;
}

.chart-grid {
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.analysis-grid {
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.control-row {
  display: flex;
  gap: 10px;
  align-items: center;
  flex-wrap: wrap;
}

.window-slider {
  width: 280px;
}

.th-slider {
  width: 120px;
}

ul {
  margin: 8px 0;
  padding-left: 18px;
}

@media (max-width: 1100px) {
  .kpi-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .chart-grid,
  .analysis-grid {
    grid-template-columns: 1fr;
  }

  .toolbar {
    flex-direction: column;
  }
}

@media (max-width: 700px) {
  .kpi-grid {
    grid-template-columns: 1fr;
  }

  .w280,
  .w420 {
    width: 100%;
  }
}
</style>
