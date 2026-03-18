import { createRouter, createWebHistory } from 'vue-router'

import AppShell from '../layout/AppShell.vue'

const OverviewView = () => import('../views/OverviewView.vue')
const AlertsView = () => import('../views/AlertsView.vue')
const XaiView = () => import('../views/XaiView.vue')
const AiInsightsView = () => import('../views/AiInsightsView.vue')

const routes = [
    {
        path: '/',
        component: AppShell,
        children: [
            { path: '', redirect: '/overview' },
            { path: '/overview', name: 'overview', component: OverviewView },
            { path: '/alerts', name: 'alerts', component: AlertsView },
            { path: '/xai', name: 'xai', component: XaiView },
            { path: '/ai-insights', name: 'ai-insights', component: AiInsightsView },
        ],
    },
]

const router = createRouter({
    history: createWebHistory(),
    routes,
})

export default router
