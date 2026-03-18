<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { echarts } from '../../services/echarts'
import worldGeoJson from 'geojson-world-map'

const props = defineProps({
  scope: {
    type: String,
    default: 'global',
  },
  data: {
    type: Object,
    default: () => ({ points: [], stats: {}, updated_at: '' }),
  },
})
const emit = defineEmits(['point-click'])

const el = ref(null)
let chart = null

const MAP_SOURCES = {
  world: {
    local: worldGeoJson,
    urls: [
      'https://geo.datav.aliyun.com/areas_v3/bound/world.geo.json',
      'https://geo.datav.aliyun.com/areas_v3/bound/world.json',
    ],
  },
  china: {
    urls: [
      'https://geo.datav.aliyun.com/areas_v3/bound/100000_full.json',
      'https://geo.datav.aliyun.com/areas_v3/bound/100000.json',
    ],
  },
}
const mapReady = {
  world: false,
  china: false,
}

const points = computed(() => {
  const arr = Array.isArray(props.data?.points) ? props.data.points : []
  return arr
    .map((x) => {
      const lon = Number(x.lon)
      const lat = Number(x.lat)
      const val = Number(x.value || 0)
      if (!Number.isFinite(lon) || !Number.isFinite(lat) || !Number.isFinite(val)) {
        return null
      }
      return {
        name: x.name || x.ip || 'unknown',
        value: [lon, lat, val],
        ip: x.ip || '',
        country: x.country || '',
        country_code: x.country_code || '',
        region: x.region || '',
        city: x.city || '',
      }
    })
    .filter(Boolean)
})

async function ensureMapRegistered(mapName) {
  if (mapReady[mapName]) {
    return true
  }
  const source = MAP_SOURCES[mapName]
  if (!source) {
    return false
  }

  const local = source.local
  if (local && (Array.isArray(local?.features) || local?.type === 'FeatureCollection')) {
    echarts.registerMap(mapName, local)
    mapReady[mapName] = true
    return true
  }

  const urls = Array.isArray(source.urls) ? source.urls : []
  for (const url of urls) {
    try {
      const resp = await fetch(url)
      if (!resp.ok) {
        continue
      }
      const geoJson = await resp.json()
      if (!geoJson || !geoJson.features) {
        continue
      }
      echarts.registerMap(mapName, geoJson)
      mapReady[mapName] = true
      return true
    } catch {
      // Try next URL fallback.
    }
  }

  return false
}

async function render() {
  if (!chart) return

  const values = points.value.map((x) => Number(x.value[2] || 0))
  const maxVal = values.length > 0 ? Math.max(...values) : 1
  const mapName = props.scope === 'china' ? 'china' : 'world'
  const ok = await ensureMapRegistered(mapName)

  if (!ok) {
    chart.clear()
    chart.setOption({
      title: {
        text: '地图底图加载失败',
        subtext: '请检查网络或稍后重试',
        left: 'center',
        top: 'center',
        textStyle: { color: '#1f2a44', fontSize: 16 },
        subtextStyle: { color: '#5b6784', fontSize: 12 },
      },
    })
    return
  }

  chart.setOption({
    tooltip: {
      trigger: 'item',
      formatter: (p) => {
        const d = p?.data || {}
        const cnt = Number(d?.value?.[2] || 0)
        const loc = [d.country, d.region, d.city].filter(Boolean).join(' / ')
        return [
          `<b>${d.name || '-'}</b>`,
          `IP: ${d.ip || '-'}`,
          `位置: ${loc || '-'}`,
          `异常数: ${cnt}`,
        ].join('<br/>')
      },
    },
    visualMap: {
      min: 0,
      max: Math.max(1, maxVal),
      orient: 'horizontal',
      left: 20,
      bottom: 8,
      text: ['高', '低'],
      calculable: true,
      inRange: {
        color: ['#7ec8ff', '#2e6fd8', '#d14b4b'],
      },
      textStyle: { color: '#5b6784' },
    },
    geo: {
      map: mapName,
      roam: true,
      zoom: props.scope === 'china' ? 1.08 : 1.0,
      label: { show: false },
      itemStyle: {
        areaColor: '#eef3fb',
        borderColor: '#9fb2d1',
      },
      emphasis: {
        itemStyle: {
          areaColor: '#d9e7fb',
        },
      },
    },
    series: [
      {
        name: '异常流量热区',
        type: 'effectScatter',
        coordinateSystem: 'geo',
        data: points.value,
        symbolSize: (v) => {
          const x = Number(v?.[2] || 0)
          return Math.min(26, 8 + Math.sqrt(Math.max(0, x)) * 2.2)
        },
        rippleEffect: {
          scale: 3,
          brushType: 'stroke',
        },
        itemStyle: {
          color: '#d14b4b',
          shadowBlur: 8,
          shadowColor: 'rgba(209,75,75,0.35)',
        },
      },
    ],
  })
}

onMounted(() => {
  chart = echarts.init(el.value)
  chart.on('click', (params) => {
    const d = params?.data
    if (!d || !d.ip) {
      return
    }
    emit('point-click', {
      ip: d.ip,
      country: d.country || '',
      country_code: d.country_code || '',
      region: d.region || '',
      city: d.city || '',
      value: Number(d?.value?.[2] || 0),
    })
  })
  void render()
  window.addEventListener('resize', chart.resize)
})

watch(
  () => [props.scope, props.data],
  () => {
    void render()
  },
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
