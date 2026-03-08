<script setup>
import { echarts } from '../../services/echarts'
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'

const props = defineProps({
  data: {
    type: Array,
    default: () => [],
  },
})

const el = ref(null)
let chart = null

function render() {
  if (!chart) {
    return
  }
  chart.setOption({
    tooltip: { trigger: 'axis' },
    legend: { data: ['正常', '告警'] },
    xAxis: {
      type: 'category',
      data: props.data.map((x) => (x.hour || '').slice(11, 16)),
      axisLabel: { color: '#5b6784' },
    },
    yAxis: { type: 'value' },
    series: [
      {
        name: '正常',
        type: 'line',
        smooth: true,
        data: props.data.map((x) => x.normal || 0),
        lineStyle: { color: '#3a7ad9' },
        itemStyle: { color: '#3a7ad9' },
      },
      {
        name: '告警',
        type: 'line',
        smooth: true,
        data: props.data.map((x) => x.alert || 0),
        lineStyle: { color: '#d14b4b' },
        itemStyle: { color: '#d14b4b' },
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
  () => props.data,
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
