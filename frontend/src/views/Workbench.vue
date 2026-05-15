<template>
  <div class="workbench-layout">
    <header class="academic-header">
      <div class="logo">
        <n-icon size="24" color="#0969da"><SchoolOutline /></n-icon>
        <span class="title">MCM/ICM 多模态建模工作台</span>
        <n-tag size="small" type="primary" round class="status-tag">
          {{ isConnected ? '已连接' : '连接中...' }}
        </n-tag>
      </div>
      <div class="actions">
        <n-button quaternary size="small">
          <template #icon><n-icon><SettingsOutline /></n-icon></template>
          设置
        </n-button>
        <n-popconfirm @positive-click="handleRestartTask">
          <template #trigger>
            <n-button size="small" type="warning" ghost>重新开始任务</n-button>
          </template>
          将清空该任务内对话历史与流程状态，确认继续吗？
        </n-popconfirm>
        <n-button type="primary" size="small" @click="handleNewTask">
          切换/新建任务
        </n-button>
      </div>
    </header>

    <main class="workbench-content">
      <n-layout has-sider position="static">
        <n-layout-sider
          bordered
          collapse-mode="width"
          :collapsed-width="64"
          :width="280"
          show-trigger="arrow-circle"
          class="side-sider"
        >
          <SidePanel />
        </n-layout-sider>

        <n-layout-content class="chat-content">
          <ChatPanel 
            @send="onUserSend"
          />
        </n-layout-content>

        <n-layout-sider
          bordered
          :width="400"
          class="artifact-sider"
        >
          <ArtifactPanel />
        </n-layout-sider>
      </n-layout>
    </main>
  </div>
</template>

<script setup lang="ts">
import { onMounted } from 'vue'
import { 
  NLayout, NLayoutSider, NLayoutContent, NIcon, NTag, NButton, NPopconfirm, useMessage 
} from 'naive-ui'
import { 
  SchoolOutline, SettingsOutline 
} from '@vicons/ionicons5'

// 组件导入
import SidePanel from '../components/SidePanel.vue'
import ChatPanel from '../components/ChatPanel.vue'
import ArtifactPanel from '../components/ArtifactPanel.vue'

// 状态管理导入
import { useWebSocket } from '../composables/useWebSocket'
import { useChatStore } from '../stores/chat'
import { useTaskStore } from '../stores/task'
import { useUserStore } from '../stores/user'
import { useRouter } from 'vue-router'
import { getApiUrl } from '../config'
import { blocksToText, textToBlocks, type ContentBlock } from '../types/content'

const router = useRouter()
const message = useMessage()
const userStore = useUserStore()
const chatStore = useChatStore()
const taskStore = useTaskStore()

// 使用 Task ID 作为 WebSocket 通讯标识（确保消息持久化到正确线程）
const threadId = taskStore.currentTaskId || 'default_thread'
const { isConnected, sendMessage, setHandlers } = useWebSocket(threadId)

onMounted(async () => {
  // 1. 安全初始化：清空上一个任务的残留消息与状态
  chatStore.clearChat()
  taskStore.clearIntermediatePreviews()

  if (!taskStore.currentTaskId) {
    message.warning('未选择活跃任务，正在跳转配置页...')
    router.push('/setup')
    return
  }

  // 2. 拉取历史消息 (带强力容错)
  try {
    const response = await fetch(getApiUrl(`/api/v1/tasks/${taskStore.currentTaskId}/messages`))
    if (response.ok) {
      const history = await response.json()
      if (Array.isArray(history)) {
        chatStore.setHistory(history.map((m: any) => ({
          role: m.role || 'ai',
          content_blocks: Array.isArray(m.content_blocks) ? m.content_blocks : textToBlocks(m.content_text || ''),
          node: m.node,
          stage_id: m.stage?.id,
          stage_label: m.stage?.label,
          created_at: m.created_at || new Date().toISOString()
        })))
      }
    }
  } catch (err) {
    console.error('Failed to load history', err)
    message.error('无法加载历史对话，但不影响当前使用')
  }

  setHandlers({
    onToken: (_payload) => {
      // 中间节点不在主聊天流展示，最终结果以 FINAL 为准
    },
  onProgress: (data) => {
      if (data.node) {
        const raw = (data.payload || {}) as Record<string, unknown>
        const payload: {
          blocks?: ContentBlock[]
          percentage?: number
          artifact_url?: string
          artifact_urls?: string[]
          artifact_manifest?: Array<{ kind?: string; path?: string; filename?: string; size?: number; updated_at_ns?: number; url?: string }>
        } = {
          blocks: Array.isArray(raw.blocks)
            ? (raw.blocks as ContentBlock[])
            : textToBlocks(String(data.status || '处理中')),
          percentage: typeof raw.percentage === 'number' ? raw.percentage : undefined,
        }
        const art = raw.artifact_url
        if (typeof art === 'string' && art) {
          payload.artifact_url = art.startsWith('http')
            ? `${art.split('?')[0]}?t=${Date.now()}`
            : `${getApiUrl(art)}?t=${Date.now()}`
        }
        const urls = raw.artifact_urls
        if (Array.isArray(urls)) {
          payload.artifact_urls = urls
            .filter((u): u is string => typeof u === 'string' && !!u)
            .map((u) => (u.startsWith('http') ? `${u.split('?')[0]}?t=${Date.now()}` : `${getApiUrl(u)}?t=${Date.now()}`))
        }
        const manifest = raw.artifact_manifest
        if (Array.isArray(manifest)) {
          payload.artifact_manifest = manifest.map((item) => {
            if (!item || typeof item !== 'object') return {}
            const rec = item as Record<string, unknown>
            const u = rec.url
            const normalizedUrl =
              typeof u === 'string' && u
                ? (u.startsWith('http') ? `${u.split('?')[0]}?t=${Date.now()}` : `${getApiUrl(u)}?t=${Date.now()}`)
                : undefined
            return {
              kind: typeof rec.kind === 'string' ? rec.kind : undefined,
              path: typeof rec.path === 'string' ? rec.path : undefined,
              filename: typeof rec.filename === 'string' ? rec.filename : undefined,
              size: typeof rec.size === 'number' ? rec.size : undefined,
              updated_at_ns: typeof rec.updated_at_ns === 'number' ? rec.updated_at_ns : undefined,
              url: normalizedUrl,
            }
          })
        }
        taskStore.setStatus(data.node, payload)
      }
    },
    onFinal: (data) => {
      chatStore.isStreaming = false
      const blocks: ContentBlock[] = data.blocks || []
      if (Array.isArray(data.paper_blocks) && data.paper_blocks.length) {
        taskStore.setPaperDraftBlocks(data.paper_blocks)
      }
      if (blocks.length) {
        chatStore.addAiMessage(blocks, data.stage?.id || 'FINAL', data.stage?.id, data.stage?.label)
      }
      message.success('Agent 计算流程已完成')
    },
    onIntermediate: (data) => {
      chatStore.isStreaming = false
      taskStore.addIntermediatePreview({
        node: data.node,
        summary_blocks: data.summary_blocks,
        preview_blocks: data.preview_blocks,
        stage: data.stage,
      })
      if (data.node === 'Writing' && Array.isArray(data.preview_blocks) && data.preview_blocks.length) {
        taskStore.setPaperDraftBlocks(data.preview_blocks)
      }
      message.info(`${data.node || '当前节点'} 已完成`)
    },
    onError: (err) => {
      chatStore.isStreaming = false
      const errorText = blocksToText(err.blocks || textToBlocks('未知错误'))
      const category = err.error?.category || ''
      const retriable = Boolean(err.error?.retriable)
      const code = err.error?.code || 'WS_ERROR'
      const label = category === 'network' ? '网络/上游异常' : '执行错误'
      chatStore.addErrorNotice(textToBlocks(errorText), { code, label })
      const toastOpts = { duration: 10000 }
      if (category === 'network') {
        message.error(`网络异常${retriable ? '（可重试）' : ''}: ${errorText}`, toastOpts)
        return
      }
      message.error(`实验室通信中断: ${errorText}`, toastOpts)
    }
  })
})

const onUserSend = (payload: { message: string }) => {
  const msgText = payload.message
  if (!isConnected.value) {
    message.error('尚未连接到后端实验室')
    return
  }
  
  chatStore.addUserMessage(msgText)
  chatStore.isStreaming = true
  
  sendMessage({ 
    message: msgText,
    thread_id: taskStore.currentTaskId 
  })
}

const handleNewTask = () => {
  router.push('/setup')
}

const handleRestartTask = async () => {
  if (!userStore.userId || !taskStore.currentTaskId) {
    message.error('缺少用户或任务信息，无法重置')
    return
  }

  try {
    const response = await fetch(
      getApiUrl(`/api/v1/users/${userStore.userId}/tasks/${taskStore.currentTaskId}/restart`),
      { method: 'POST' }
    )
    if (!response.ok) {
      const detail = await response.text()
      throw new Error(detail || '重置失败')
    }

    chatStore.clearChat()
    taskStore.resetNodes()
    taskStore.clearIntermediatePreviews()
    message.success('任务已重新开始，可输入新问题')
  } catch (err) {
    console.error('Failed to restart task', err)
    message.error('任务重置失败，请稍后重试')
  }
}
</script>

<style scoped>
.workbench-layout { height: 100vh; display: flex; flex-direction: column; background-color: #f6f8fa; }
.academic-header { height: 56px; background-color: #ffffff; border-bottom: 1px solid #d0d7de; display: flex; align-items: center; justify-content: space-between; padding: 0 20px; z-index: 100; }
.logo { display: flex; align-items: center; gap: 12px; }
.logo .title { font-weight: 600; font-size: 16px; color: #1f2328; }
.status-tag { margin-left: 8px; font-size: 11px; }
.workbench-content { flex: 1; overflow: hidden; }
.chat-content { background-color: #ffffff; }
:deep(.n-layout) { height: 100%; }
</style>
