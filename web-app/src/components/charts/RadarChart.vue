<script setup>
import { echarts } from '../../services/echarts'
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'

const props = defineProps({
  radar: {
    type: Object,
    default: () => ({ labels: [], baseline: [], netguard: [] }),
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
  const baseline = toNumberArray(src.baseline)
  const netguard = toNumberArray(src.netguard)

  const dim = Math.max(labelsRaw.length, baseline.length, netguard.length, 1)
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
    baseline: pad(baseline),
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
    ...normalized.baseline,
    ...normalized.netguard,
  )

  chart.setOption({
    animation: false,
    legend: { data: ['Baseline', 'NetGuard'] },
    radar: {
      indicator: labels.map((label) => ({ name: label, max: maxVal })),
      radius: 90,
    },
    series: [
      {
        type: 'radar',
        data: [
          {
            value: normalized.baseline,
            name: 'Baseline',
            areaStyle: { color: 'rgba(245,165,36,.18)' },
          },
          {
            value: normalized.netguard,
            name: 'NetGuard',
            areaStyle: { color: 'rgba(46,111,216,.2)' },
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
