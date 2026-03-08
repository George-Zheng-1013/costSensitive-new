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

function render() {
  if (!chart) return
  const labels = props.radar.labels || []
  chart.setOption({
    legend: { data: ['Baseline', 'NetGuard'] },
    radar: {
      indicator: labels.map((label) => ({ name: label, max: 100 })),
      radius: 90,
    },
    series: [
      {
        type: 'radar',
        data: [
          { value: props.radar.baseline || [], name: 'Baseline', areaStyle: { color: 'rgba(245,165,36,.18)' } },
          { value: props.radar.netguard || [], name: 'NetGuard', areaStyle: { color: 'rgba(46,111,216,.2)' } },
        ],
      },
    ],
  })
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
