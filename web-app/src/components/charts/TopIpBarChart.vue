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
  if (!chart) return
  const labels = props.data.map((x) => x.src_ip)
  const values = props.data.map((x) => x.count)
  chart.setOption({
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'value' },
    yAxis: { type: 'category', data: labels, inverse: true },
    series: [
      {
        type: 'bar',
        data: values,
        itemStyle: {
          color: '#2e6fd8',
        },
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
