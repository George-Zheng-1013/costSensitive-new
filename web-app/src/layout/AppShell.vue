<script setup>
import { computed, onBeforeUnmount, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'

import { connectStream } from '../services/ws'
import { useDashboardStore } from '../stores/dashboard'

const store = useDashboardStore()
const route = useRoute()
const wsPrev = {
  overview: 'idle',
  alerts: 'idle',
  ai: 'idle',
}

let closeOverview = null
let closeAlerts = null
let closeAi = null

const pageTitle = computed(() => {
  const m = {
    '/overview': '系统总览',
    '/alerts': '告警处置',
    '/xai': '可解释分析',
    '/ai-insights': 'AI 研判',
  }
  return m[route.path] || 'NetGuard'
})

onMounted(async () => {
  try {
    await store.bootstrap()
  } catch (e) {
    ElMessage.error(`初始化失败: ${String(e)}`)
  }

  closeOverview = connectStream(
    '/ws/overview',
    (msg) => {
      if (msg.event === 'overview_update') {
        store.applyOverview(msg.payload)
      }
    },
    async (status) => {
      const prev = wsPrev.overview
      wsPrev.overview = status
      store.setWsStatus('overview', status)
      if (status === 'connected' && (prev === 'closed' || prev === 'error' || prev === 'retrying')) {
        await store.refreshOverview()
      }
    },
  )

  closeAlerts = connectStream(
    '/ws/alerts',
    (msg) => {
      if (msg.event === 'alert_inserted') {
        store.prependAlerts(msg.payload.items)
      }
    },
    async (status) => {
      const prev = wsPrev.alerts
      wsPrev.alerts = status
      store.setWsStatus('alerts', status)
      if (status === 'connected' && (prev === 'closed' || prev === 'error' || prev === 'retrying')) {
        await store.refreshAlerts()
      }
    },
  )

  closeAi = connectStream(
    '/ws/ai',
    (msg) => {
      if (msg.event === 'ai_inserted') {
        store.prependAi(msg.payload.items)
      }
    },
    async (status) => {
      const prev = wsPrev.ai
      wsPrev.ai = status
      store.setWsStatus('ai', status)
      if (status === 'connected' && (prev === 'closed' || prev === 'error' || prev === 'retrying')) {
        await store.refreshAi()
      }
    },
  )
})

onBeforeUnmount(() => {
  closeOverview?.()
  closeAlerts?.()
  closeAi?.()
})
</script>

<template>
  <div class="shell-root">
    <aside class="nav-panel card">
      <h1>NetGuard</h1>
      <p class="muted">网络流量检测控制台</p>
      <el-menu :default-active="$route.path" router class="menu">
        <el-menu-item index="/overview">系统总览</el-menu-item>
        <el-menu-item index="/alerts">告警处置</el-menu-item>
        <el-menu-item index="/xai">可解释分析</el-menu-item>
        <el-menu-item index="/ai-insights">AI 研判</el-menu-item>
      </el-menu>
      <div class="status-block">
        <div>Overview WS: <b>{{ store.wsStatus.overview }}</b></div>
        <div>Alerts WS: <b>{{ store.wsStatus.alerts }}</b></div>
        <div>AI WS: <b>{{ store.wsStatus.ai }}</b></div>
      </div>
    </aside>

    <main class="content-panel">
      <section class="page-head card">
        <h2>{{ pageTitle }}</h2>
      </section>
      <section class="page-body">
        <router-view />
      </section>
    </main>
  </div>
</template>

<style scoped>
.shell-root {
  min-height: 100vh;
  display: grid;
  grid-template-columns: 270px 1fr;
  align-items: start;
  gap: 16px;
  padding: 16px;
}

.nav-panel {
  padding: 18px 14px;
  position: sticky;
  top: 16px;
  z-index: 4;
  align-self: start;
}

.nav-panel h1 {
  margin: 0;
  font-size: 26px;
  font-weight: 700;
}

.menu {
  border-right: 0;
  margin-top: 10px;
}

.status-block {
  margin-top: 18px;
  border-top: 1px solid var(--line);
  padding-top: 12px;
  color: var(--text-sub);
  font-size: 13px;
  display: grid;
  gap: 5px;
}

.content-panel {
  display: grid;
  grid-template-rows: auto 1fr;
  gap: 12px;
  min-width: 0;
  overflow: hidden;
}

.page-head {
  padding: 14px 20px;
}

.page-head h2 {
  margin: 0;
  font-size: 22px;
}

.page-body {
  min-height: 0;
  min-width: 0;
  overflow-x: hidden;
}

@media (max-width: 980px) {
  .shell-root {
    grid-template-columns: 1fr;
  }
}
</style>
