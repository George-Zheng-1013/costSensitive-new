<script setup>
import { computed, ref, watch } from 'vue'
import { useDashboardStore } from '../stores/dashboard'

const store = useDashboardStore()
const selectedId = ref(null)

const options = computed(() =>
  (Array.isArray(store.xaiSamples) ? store.xaiSamples : []).map((x) => ({
    value: x.id,
    label: `#${x.id} ${x.title}`,
  })),
)

const packetRows = computed(() => {
  const contrib = store.xaiDetail?.packet_contrib
  if (Array.isArray(contrib)) {
    return contrib
  }
  const scores = Array.isArray(contrib?.scores) ? contrib.scores : []
  return scores.map((score, idx) => ({
    packet_index: idx,
    score_drop: Number(score || 0).toFixed(6),
    importance: Number(score || 0).toFixed(6),
  }))
})

const heatmapRows = computed(() => {
  const heatmap = store.xaiDetail?.byte_heatmap
  if (Array.isArray(heatmap)) {
    return heatmap
  }
  const matrix = Array.isArray(heatmap?.byte_heatmap) ? heatmap.byte_heatmap : []
  return matrix
    .filter((row) => Array.isArray(row) && row.length > 0)
    .map((row, packetIndex) => {
      let maxValue = Number.NEGATIVE_INFINITY
      let maxIndex = 0
      for (let i = 0; i < row.length; i += 1) {
        const val = Number(row[i] || 0)
        if (val > maxValue) {
          maxValue = val
          maxIndex = i
        }
      }
      return {
        packet_index: packetIndex,
        byte_start: maxIndex,
        byte_end: maxIndex,
        importance: Number.isFinite(maxValue) ? maxValue.toFixed(6) : '0.000000',
      }
    })
})

watch(
  () => selectedId.value,
  async (v) => {
    if (!v) return
    await store.loadXaiDetail(v)
  },
)
</script>

<template>
  <div class="xai-grid">
    <section class="card panel">
      <h3 class="card-title">样本选择</h3>
      <el-select v-model="selectedId" filterable placeholder="选择告警样本" style="width: 100%">
        <el-option v-for="item in options" :key="item.value" :label="item.label" :value="item.value" />
      </el-select>
      <p class="muted tip">先在告警页观察实时记录，再选择样本看贡献与热力图。</p>
    </section>

    <section class="card panel">
      <h3 class="card-title">文本证据</h3>
      <el-descriptions v-if="store.xaiDetail" :column="1" border>
        <el-descriptions-item label="类别">{{ store.xaiDetail.threat_category }}</el-descriptions-item>
        <el-descriptions-item label="置信度">{{ store.xaiDetail.confidence }}</el-descriptions-item>
        <el-descriptions-item label="Unknown">{{ store.xaiDetail.is_unknown }}</el-descriptions-item>
        <el-descriptions-item label="阈值理由">{{ store.xaiDetail.message || '-' }}</el-descriptions-item>
      </el-descriptions>
      <div v-else class="muted">请选择样本后查看详情。</div>
    </section>

    <section class="card panel two-col">
      <h3 class="card-title">包级贡献 (Occlusion)</h3>
      <el-table :data="packetRows" height="260" stripe>
        <el-table-column prop="packet_index" label="包序号" width="100" />
        <el-table-column prop="score_drop" label="遮挡后得分下降" width="170" />
        <el-table-column prop="importance" label="贡献度" />
      </el-table>
    </section>

    <section class="card panel two-col">
      <h3 class="card-title">字节热力图 (Grad-CAM)</h3>
      <el-table :data="heatmapRows" height="260" stripe>
        <el-table-column prop="packet_index" label="包序号" width="100" />
        <el-table-column prop="byte_start" label="起始" width="90" />
        <el-table-column prop="byte_end" label="结束" width="90" />
        <el-table-column prop="importance" label="强度" />
      </el-table>
    </section>
  </div>
</template>

<style scoped>
.xai-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.panel {
  padding: 14px;
}

.tip {
  margin-top: 12px;
}

.two-col {
  grid-column: span 2;
}

@media (max-width: 860px) {
  .xai-grid {
    grid-template-columns: 1fr;
  }

  .two-col {
    grid-column: span 1;
  }
}
</style>
