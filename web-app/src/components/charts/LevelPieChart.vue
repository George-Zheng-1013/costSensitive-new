<script setup>
import { echarts } from '../../services/echarts'
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'

const props = defineProps({
  levelDist: {
    type: Object,
    default: () => ({ low: 0, medium: 0, high: 0 }),
  },
})

const el = ref(null)
let chart = null

function render() {
  if (!chart) return
  chart.setOption({
    tooltip: { trigger: 'item' },
    series: [
      {
        type: 'pie',
        radius: ['36%', '70%'],
        data: [
          { value: props.levelDist.low || 0, name: 'Low', itemStyle: { color: '#36a16b' } },
          { value: props.levelDist.medium || 0, name: 'Medium', itemStyle: { color: '#f5a524' } },
          { value: props.levelDist.high || 0, name: 'High', itemStyle: { color: '#d14b4b' } },
        ],
        label: { formatter: '{b}: {c}' },
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
  () => props.levelDist,
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
