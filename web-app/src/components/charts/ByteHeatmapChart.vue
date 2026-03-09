<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { echarts } from '../../services/echarts'

const props = defineProps({
  matrix: {
    type: Array,
    default: () => [],
  },
  byteStart: {
    type: Number,
    default: 0,
  },
  byteEnd: {
    type: Number,
    default: 255,
  },
  selectedPacket: {
    type: Number,
    default: -1,
  },
  threshold: {
    type: Number,
    default: 0,
  },
})
const emit = defineEmits(['cell-click'])

const el = ref(null)
let chart = null

const normalizedMatrix = computed(() => {
  const rows = Array.isArray(props.matrix) ? props.matrix : []
  return rows.map((row) =>
    (Array.isArray(row) ? row : []).map((v) => {
      const n = Number(v)
      return Number.isFinite(n) ? n : 0
    }),
  )
})

const sliced = computed(() => {
  const matrix = normalizedMatrix.value
  const maxLen = matrix.reduce((m, r) => Math.max(m, r.length), 0)
  const start = Math.max(0, Math.min(props.byteStart, Math.max(0, maxLen - 1)))
  const end = Math.max(start, Math.min(props.byteEnd, Math.max(0, maxLen - 1)))

  const data = []
  let maxVal = 0
  for (let p = 0; p < matrix.length; p += 1) {
    const row = matrix[p]
    for (let b = start; b <= end; b += 1) {
      const val = Number(row[b] || 0)
      if (val > maxVal) maxVal = val
      if (val >= Number(props.threshold || 0)) {
        data.push([b - start, p, val])
      }
    }
  }

  return {
    start,
    end,
    width: Math.max(0, end - start + 1),
    height: matrix.length,
    data,
    maxVal: maxVal > 0 ? maxVal : 1,
  }
})

function render() {
  if (!chart) return
  const view = sliced.value

  const packetLabels = Array.from({ length: view.height }, (_, i) => `#${i}`)
  const byteLabels = Array.from({ length: view.width }, (_, i) => String(view.start + i))

  chart.setOption({
    animation: false,
    grid: { left: 52, right: 18, top: 20, bottom: 58 },
    tooltip: {
      position: 'top',
      formatter: (p) => {
        const x = Number(p?.data?.[0] ?? 0)
        const y = Number(p?.data?.[1] ?? 0)
        const v = Number(p?.data?.[2] ?? 0)
        return `包 #${y}<br/>字节 ${view.start + x}<br/>强度 ${v.toFixed(6)}`
      },
    },
    xAxis: {
      type: 'category',
      data: byteLabels,
      axisLabel: {
        color: '#5b6784',
        interval: Math.max(0, Math.floor(byteLabels.length / 10)),
      },
      splitArea: { show: false },
    },
    yAxis: {
      type: 'category',
      data: packetLabels,
      axisLabel: {
        color: '#5b6784',
        formatter: (txt, idx) => (idx === Number(props.selectedPacket) ? `*${txt}` : txt),
      },
      inverse: true,
      splitArea: { show: false },
    },
    visualMap: {
      min: 0,
      max: Math.max(0.0001, view.maxVal),
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: 10,
      text: ['高', '低'],
      inRange: {
        color: ['#edf4ff', '#7ec8ff', '#2e6fd8', '#d14b4b'],
      },
      textStyle: { color: '#5b6784' },
    },
    series: [
      {
        type: 'heatmap',
        data: view.data,
        progressive: 2000,
        emphasis: {
          itemStyle: {
            shadowBlur: 8,
            shadowColor: 'rgba(0, 0, 0, 0.35)',
          },
        },
      },
    ],
  }, true)
}

onMounted(() => {
  chart = echarts.init(el.value)
  chart.on('click', (params) => {
    const data = params?.data
    if (!Array.isArray(data) || data.length < 3) return
    const byteOffset = Number(data[0])
    const packetIndex = Number(data[1])
    emit('cell-click', {
      packetIndex,
      byteIndex: sliced.value.start + byteOffset,
      importance: Number(data[2] || 0),
    })
  })
  render()
  window.addEventListener('resize', chart.resize)
})

watch(
  () => [props.matrix, props.byteStart, props.byteEnd, props.selectedPacket, props.threshold],
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
  <div ref="el" style="height: 360px" />
</template>
