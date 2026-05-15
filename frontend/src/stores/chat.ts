import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { ContentBlock } from '../types/content'
import { textToBlocks } from '../types/content'

export interface Message {
  role: 'user' | 'ai'
  content_blocks: ContentBlock[]
  node?: string
  stage_id?: string
  stage_label?: string
  created_at: string
  isStreaming?: boolean
  /** 系统/网络错误气泡，用于与正常 Agent 输出区分 */
  is_error?: boolean
}

export const useChatStore = defineStore('chat', () => {
  const messages = ref<Message[]>([])
  const isStreaming = ref(false)
  const isPending = ref(false)
  const pendingPrompt = ref('')

  const addUserMessage = (content: string) => {
    messages.value.push({
      role: 'user',
      content_blocks: textToBlocks(content),
      created_at: new Date().toISOString()
    })
  }

  const addAiMessage = (blocks: ContentBlock[], node?: string, stageId?: string, stageLabel?: string) => {
    messages.value.push({
      role: 'ai',
      content_blocks: blocks || [],
      node,
      stage_id: stageId,
      stage_label: stageLabel,
      created_at: new Date().toISOString()
    })
  }

  const addErrorNotice = (blocks: ContentBlock[], opts?: { code?: string; label?: string }) => {
    messages.value.push({
      role: 'ai',
      content_blocks: blocks || [],
      node: 'System',
      stage_id: opts?.code,
      stage_label: opts?.label || '运行错误',
      created_at: new Date().toISOString(),
      is_error: true,
    })
  }

  const appendToLastMessage = (chunk: string) => {
    const lastMsg = messages.value[messages.value.length - 1]
    const nextChunk = chunk || ''
    if (lastMsg && lastMsg.role === 'ai') {
      const lastBlock = lastMsg.content_blocks[lastMsg.content_blocks.length - 1]
      if (lastBlock && lastBlock.type === 'text') {
        lastBlock.text = (lastBlock.text || '') + nextChunk
      } else {
        lastMsg.content_blocks.push({ type: 'text', text: nextChunk })
      }
    } else {
      addAiMessage(textToBlocks(nextChunk))
    }
  }

  const clearChat = () => {
    messages.value = []
    isStreaming.value = false
    isPending.value = false
    pendingPrompt.value = ''
  }

  const setHistory = (history: Message[]) => {
    messages.value = history
  }

  return {
    messages,
    isStreaming,
    isPending,
    pendingPrompt,
    addUserMessage,
    addAiMessage,
    addErrorNotice,
    appendToLastMessage,
    clearChat,
    setHistory
  }
})
