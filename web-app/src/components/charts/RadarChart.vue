<script setup>
import { echarts } from '../../services/echarts'
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'

const props = defineProps({
  radar: {
    type: Object,
    default: () => ({ labels: [], models: { rf: [], cnn: [], netguard: [] } }),
  },
})

const el = ref(null)
let chart = null

function toNumberArray(x) {
  if (!Array.isArray(x)) return []
  return x.map((v) => {
    const n = Number(v)
    return Number.isFinite(n) ? n : 0
  })
}

function normalizeRadarInput(radar) {
  const src = radar && typeof radar === 'object' ? radar : {}
  const labelsRaw = Array.isArray(src.labels) ? src.labels : []
  const modelsSrc = src.models && typeof src.models === 'object' ? src.models : {}
  const legacyBaseline = toNumberArray(src.baseline)
  const rf = toNumberArray(modelsSrc.rf || legacyBaseline)
  const cnn = toNumberArray(modelsSrc.cnn)
  const netguard = toNumberArray(modelsSrc.netguard)

  const dim = Math.max(labelsRaw.length, rf.length, cnn.length, netguard.length, 1)
  const labels = Array.from({ length: dim }, (_, i) => {
    const label = labelsRaw[i]
    return typeof label === 'string' && label.trim() ? label : `Metric-${i + 1}`
  })

  const pad = (arr) => {
    const out = arr.slice(0, dim)
    while (out.length < dim) out.push(0)
    return out
  }

  return {
    labels,
    rf: pad(rf),
    cnn: pad(cnn),
    netguard: pad(netguard),
  }
}

function render() {
  if (!chart) return
  const normalized = normalizeRadarInput(props.radar)
  const labels = normalized.labels

  // Keep radar max stable so all values fit even when backend sends out-of-range data.
  const maxVal = Math.max(
    100,
    ...normalized.rf,
    ...normalized.cnn,
    ...normalized.netguard,
  )

  chart.setOption({
    animation: false,
    color: ['#7E8CA3', '#C9893A', '#2E6FD8'],
    legend: {
      data: ['RF', 'CNN', 'NetGuard (Current)'],
      bottom: 0,
    },
    radar: {
      indicator: labels.map((label) => ({ name: label, max: maxVal })),
      radius: 86,
    },
    series: [
      {
        type: 'radar',
        data: [
          {
            value: normalized.rf,
            name: 'RF',
            symbolSize: 6,
            lineStyle: { width: 2 },
            areaStyle: { color: 'rgba(126,140,163,.12)' },
          },
          {
            value: normalized.cnn,
            name: 'CNN',
            symbolSize: 6,
            lineStyle: { width: 2 },
            areaStyle: { color: 'rgba(201,137,58,.12)' },
          },
          {
            value: normalized.netguard,
            name: 'NetGuard (Current)',
            symbolSize: 8,
            lineStyle: { width: 3.5, shadowBlur: 10, shadowColor: 'rgba(46,111,216,.35)' },
            areaStyle: { color: 'rgba(46,111,216,.24)' },
          },
        ],
      },
    ],
  }, true)
}

onMounted(() => {
  chart = echarts.init(el.value)
  render()
  window.addEventListener('resize', chart.resize)
})

watch(
  () => props.radar,
  () => render(),
  { deep: true },
)

onBeforeUnmount(() => {
  if (chart) {
    window.removeEventListener('resize', chart.resize)
    chart.dispose()
    chart = null
  }
})
</script>

<template>
  <div ref="el" style="height: 280px" />
</template>
