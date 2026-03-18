<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { echarts } from '../../services/echarts'

const props = defineProps({
  scores: {
    type: Array,
    default: () => [],
  },
  selectedPacket: {
    type: Number,
    default: -1,
  },
})
const emit = defineEmits(['packet-click'])

const el = ref(null)
let chart = null

const normalizedScores = computed(() =>
  (Array.isArray(props.scores) ? props.scores : []).map((x) => {
    const n = Number(x)
    return Number.isFinite(n) ? n : 0
  }),
)

function render() {
  if (!chart) return

  const xs = normalizedScores.value
  const labels = xs.map((_, i) => `#${i}`)
  const maxVal = xs.length > 0 ? Math.max(...xs) : 1

  chart.setOption({
    animation: false,
    grid: { left: 50, right: 16, top: 20, bottom: 20 },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      formatter: (params) => {
        const p = Array.isArray(params) ? params[0] : params
        const idx = Number(p?.dataIndex ?? -1)
        const v = Number(p?.value ?? 0)
        return `包 #${idx}<br/>贡献: ${v.toFixed(6)}`
      },
    },
    xAxis: {
      type: 'value',
      min: 0,
      max: Math.max(0.0001, maxVal * 1.15),
      axisLabel: { color: '#5b6784' },
    },
    yAxis: {
      type: 'category',
      data: labels,
      axisLabel: { color: '#5b6784' },
      inverse: true,
    },
    series: [
      {
        type: 'bar',
        data: xs,
        itemStyle: {
          borderRadius: [0, 6, 6, 0],
          color: (p) => {
            if (Number(p.dataIndex) === Number(props.selectedPacket)) {
              return '#d14b4b'
            }
            const val = Number(p.value || 0)
            const t = maxVal > 1e-12 ? val / maxVal : 0
            return t > 0.65 ? '#d14b4b' : t > 0.35 ? '#f5a524' : '#2e6fd8'
          },
        },
      },
    ],
  }, true)
}

onMounted(() => {
  chart = echarts.init(el.value)
  chart.on('click', (params) => {
    const idx = Number(params?.dataIndex ?? -1)
    if (idx >= 0) {
      emit('packet-click', idx)
    }
  })
  render()
  window.addEventListener('resize', chart.resize)
})

watch(
  () => [props.scores, props.selectedPacket],
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
  <div ref="el" style="height: 320px" />
</template>
