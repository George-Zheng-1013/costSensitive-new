import { apiBase } from './api'

function toWsUrl(path) {
    const base = apiBase.replace(/^http/, 'ws').replace(/\/$/, '')
    return `${base}${path}`
}

export function connectStream(path, onMessage, onStatus) {
    let ws = null
    let timer = null
    let retries = 0
    let manuallyClosed = false

    const connect = () => {
        manuallyClosed = false
        onStatus?.('connecting')
        ws = new WebSocket(toWsUrl(path))

        ws.onopen = () => {
            retries = 0
            onStatus?.('connected')
        }

        ws.onmessage = (event) => {
            try {
                onMessage(JSON.parse(event.data))
            } catch (e) {
                console.error('WS parse error', e)
            }
        }

        ws.onerror = () => {
            onStatus?.('error')
        }

        ws.onclose = () => {
            onStatus?.('closed')
            if (manuallyClosed) {
                return
            }
            onStatus?.('retrying')
            retries += 1
            const delayMs = Math.min(10000, 1200 * (2 ** Math.min(retries, 4)))
            timer = window.setTimeout(connect, delayMs)
        }
    }

    connect()

    return () => {
        manuallyClosed = true
        if (timer) {
            window.clearTimeout(timer)
            timer = null
        }
        if (ws) {
            ws.close()
            ws = null
        }
    }
}
