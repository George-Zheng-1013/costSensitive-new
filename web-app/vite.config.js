import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import vueDevTools from 'vite-plugin-vue-devtools'

// https://vite.dev/config/
export default defineConfig({
    plugins: [
        vue(),
        vueDevTools(),
    ],
    build: {
        chunkSizeWarningLimit: 800,
        rollupOptions: {
            output: {
                manualChunks(id) {
                    if (!id.includes('node_modules')) {
                        return undefined
                    }

                    if (id.includes('node_modules/vue') || id.includes('node_modules/pinia') || id.includes('node_modules/vue-router')) {
                        return 'vendor-vue'
                    }

                    if (id.includes('node_modules/zrender')) {
                        return 'vendor-chart'
                    }
                    if (id.includes('node_modules/echarts')) {
                        return 'vendor-chart'
                    }
                    if (id.includes('node_modules/element-plus')) {
                        return 'vendor-ui-core'
                    }

                    return 'vendor-misc'
                },
            },
        },
    },
    resolve: {
        alias: {
            '@': fileURLToPath(new URL('./src', import.meta.url))
        },
    },
})
