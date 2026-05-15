import { onBeforeUnmount, ref } from 'vue'
import { getWsUrl } from '../config'
import type { ContentBlock, StagePayload } from '../types/content'
import { textToBlocks } from '../types/content'

type Handlers = {
  onToken?: (payload: { blocks?: ContentBlock[] }) => void
  onProgress?: (payload: {
    node?: string
    status?: string
    payload?: {
      blocks?: ContentBlock[]
      percentage?: number
      stage?: StagePayload
      artifact_url?: string
      artifact_urls?: string[]
      artifact_manifest?: Array<{ kind?: string; path?: string; filename?: string; size?: number; updated_at_ns?: number; url?: string }>
    }
  }) => void
  onFinal?: (payload: { blocks?: ContentBlock[]; paper_blocks?: ContentBlock[]; stage?: StagePayload }) => void
  onIntermediate?: (payload: { node?: string; summary_blocks?: ContentBlock[]; preview_blocks?: ContentBlock[]; ask_continue?: boolean; stage?: StagePayload }) => void
  onError?: (payload: { blocks?: ContentBlock[]; error?: { code?: string; category?: string; retriable?: boolean; detail?: string } }) => void
}

export const useWebSocket = (threadId: string) => {
  const isConnected = ref(false)
  const handlers = ref<Handlers>({})
  const ws = ref<WebSocket | null>(null)

  const connect = () => {
    const socket = new WebSocket(getWsUrl(`/ws/chat/${threadId || 'default_thread'}`))
    ws.value = socket

    socket.onopen = () => {
      isConnected.value = true
    }

    socket.onclose = () => {
      isConnected.value = false
    }

    socket.onerror = () => {
      handlers.value.onError?.({ blocks: textToBlocks('WebSocket 连接异常') })
    }

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        if (data.type === 'TOKEN') {
          handlers.value.onToken?.(data.payload || {})
        } else if (data.type === 'PROGRESS' || data.type === 'STATUS') {
          handlers.value.onProgress?.(data)
        } else if (data.type === 'FINAL') {
          handlers.value.onFinal?.(data.payload || {})
        } else if (data.type === 'INTERMEDIATE') {
          handlers.value.onIntermediate?.(data.payload || {})
        } else if (data.type === 'ERROR') {
          handlers.value.onError?.(data.payload || { blocks: [{ type: 'text', text: '未知错误' }] })
        }
      } catch (error) {
        handlers.value.onError?.({ blocks: [{ type: 'text', text: `消息解析失败: ${String(error)}` }] })
      }
    }
  }

  const setHandlers = (nextHandlers: Handlers) => {
    handlers.value = nextHandlers
  }

  const sendMessage = (payload: Record<string, unknown>) => {
    if (!ws.value || ws.value.readyState !== WebSocket.OPEN) {
      handlers.value.onError?.({ blocks: textToBlocks('WebSocket 未连接') })
      return
    }
    ws.value.send(JSON.stringify(payload))
  }

  connect()

  onBeforeUnmount(() => {
    ws.value?.close()
  })

  return {
    isConnected,
    setHandlers,
    sendMessage,
  }
}
