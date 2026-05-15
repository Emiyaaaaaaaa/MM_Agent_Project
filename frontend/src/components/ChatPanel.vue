<template>
  <div class="chat-panel">
    <!-- 对话展示区 -->
    <n-scrollbar ref="scrollbarInst" class="message-list-container">
      <div class="message-inner-wrap">
        <div
          v-for="(msg, index) in chatStore.messages"
          :key="index"
          :class="['message-row', msg.role, msg.is_error ? 'error-notice' : '']"
        >
          <!-- 头像区 -->
          <div class="avatar-col">
            <n-avatar round :size="38" :src="msg.role === 'ai' ? aiAvatar : userAvatar" />
          </div>
          
          <!-- 内容区 -->
          <div class="bubble-col">
            <div class="bubble-header">
              <span class="sender-name">{{ msg.role === 'ai' ? 'Research Agent' : 'User Identity' }}</span>
              <n-tag v-if="msg.node" size="tiny" :bordered="false" round class="node-tag">
                {{ translateNode(msg.node) }}
              </n-tag>
            </div>
            <div :class="['bubble-content', 'shadow-sm', msg.is_error ? 'bubble-content--error' : '']">
              <ContentBlocks :blocks="msg.content_blocks" />
              <span v-if="msg.isStreaming" class="streaming-cursor"></span>
            </div>
            <div class="bubble-footer">
              {{ formatDate(msg.created_at) }}
            </div>
          </div>
        </div>
        
        <!-- 占位，防止输入框挡住最后一条消息 -->
        <div class="bottom-spacer"></div>
      </div>
    </n-scrollbar>

    <!-- 悬浮输入区 (Glassmorphism) -->
    <div class="floating-input-wrap">
      <div class="glass-input-card">
        <div class="input-main">
          <!-- 上传入口 -->
          <n-upload
            action="/api/v1/upload"
            :show-file-list="false"
            @finish="onQuickUploadFinish"
          >
            <n-button quaternary circle size="large" class="icon-btn">
              <template #icon><n-icon><AddOutline /></n-icon></template>
            </n-button>
          </n-upload>

          <!-- 输入框 -->
          <n-input
            v-model:value="userInput"
            type="textarea"
            :autosize="{ minRows: 1, maxRows: 8 }"
            placeholder="输入研究指令... (Ctrl + Enter 发送)"
            class="main-textarea"
            @keypress.ctrl.enter="handleSend"
          />

          <!-- 发送按钮 -->
          <n-button 
            type="primary" 
            circle 
            size="large" 
            :disabled="!userInput" 
            @click="handleSend" 
            class="send-float-btn shadow-md"
          >
            <template #icon><n-icon><ArrowUp /></n-icon></template>
          </n-button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick, watch } from 'vue'
import { 
  NScrollbar, NAvatar, NTag, NInput, NButton, NIcon, NUpload, useMessage 
} from 'naive-ui'
import { ArrowUp, AddOutline } from '@vicons/ionicons5'
import { useChatStore } from '../stores/chat'
import ContentBlocks from './ContentBlocks.vue'

const message = useMessage()
const chatStore = useChatStore()
const scrollbarInst = ref<any>(null)
const userInput = ref('')


const aiAvatar = 'https://api.dicebear.com/7.x/bottts/svg?seed=agent&backgroundColor=b6e3f4'
const userAvatar = 'https://api.dicebear.com/7.x/avataaars/svg?seed=user&backgroundColor=ffdfbf'

const translateNode = (node: string) => {
  const map: Record<string, string> = {
    'Analysis': '问题分析',
    'Modeling': '数学建模',
    'Coder': '仿真代码',
    'Reviewer': '逻辑审查',
    'Fusion': '知识检索',
    'System': '系统',
  }
  return map[node] || node
}

const formatDate = (date: any) => {
  if (!date) return ''
  const d = new Date(date)
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

watch(() => chatStore.messages, () => {
  nextTick(() => {
    scrollbarInst.value?.scrollTo({ position: 'bottom', behavior: 'smooth' })
  })
}, { deep: true })

const onQuickUploadFinish = ({ file }: { file: any }) => {
  message.success(`${file.name} 已入库`)
}

const handleSend = () => {
  if (!userInput.value) return
  const text = userInput.value
  userInput.value = ''
  emit('send', { message: text })
}

const emit = defineEmits(['send'])
</script>

<style scoped>
.chat-panel {
  position: relative;
  display: flex;
  flex-direction: column;
  height: 100%;
  background-color: #fcfcfd;
}

.message-list-container {
  flex: 1;
}

.message-inner-wrap {
  padding: 40px 15% 120px 15%; /* 侧边留白增加 Academic 感 */
  display: flex;
  flex-direction: column;
  gap: 32px;
}

/* 消息行基础 */
.message-row {
  display: flex;
  gap: 16px;
  width: 100%;
}

.message-row.user {
  flex-direction: row-reverse;
}

.message-row.error-notice .bubble-content--error {
  border: 1px solid #f85149;
  background-color: #fff5f5;
}

.avatar-col {
  flex-shrink: 0;
}

.bubble-col {
  display: flex;
  flex-direction: column;
  max-width: 85%;
}

.message-row.user .bubble-col {
  align-items: flex-end;
}

.bubble-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 6px;
  font-size: 13px;
  color: #636c76;
}

.sender-name {
  font-weight: 600;
}

.node-tag {
  font-size: 10px;
  background-color: #f1f1f2 !important;
}

/* 核心气泡 */
.bubble-content {
  padding: 18px 24px;
  border-radius: 18px;
  font-size: 15px;
  line-height: 1.65;
  border: 1px solid #d0d7de;
  background-color: #ffffff;
  color: #1f2328;
}

.message-row.user .bubble-content {
  background-color: #1f2328;
  color: #ffffff;
  border: none;
  border-radius: 18px 18px 2px 18px;
}

.message-row.ai .bubble-content {
  border-radius: 18px 18px 18px 2px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.02);
}

.bubble-footer {
  margin-top: 6px;
  font-size: 10px;
  color: #afb8c1;
}

/* Markdown 内部微调 */
.md-body :deep(p) { margin: 0 0 12px 0; }
.md-body :deep(p:last-child) { margin-bottom: 0; }
.md-body :deep(pre) {
  background-color: #f6f8fa;
  padding: 16px;
  border-radius: 8px;
  margin: 12px 0;
  border: 1px solid #d0d7de;
}

/* 流式光标 */
.streaming-cursor {
  display: inline-block;
  width: 6px;
  height: 15px;
  background-color: #0969da;
  margin-left: 4px;
  animation: blink 0.8s infinite;
  vertical-align: middle;
}
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }

/* 悬浮输入区 (核心玻璃拟态) */
.floating-input-wrap {
  position: absolute;
  bottom: 30px;
  left: 0;
  right: 0;
  display: flex;
  justify-content: center;
  padding: 0 20px;
  z-index: 10;
}

.glass-input-card {
  width: 100%;
  max-width: 800px;
  background: rgba(255, 255, 255, 0.85); /* 磨砂玻璃色 */
  backdrop-filter: blur(12px) saturate(180%);
  border: 1px solid rgba(208, 215, 222, 0.5);
  border-radius: 24px;
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.08);
  padding: 12px 16px;
  transition: transform 0.2s;
}

.glass-input-card:focus-within {
  transform: translateY(-2px);
  box-shadow: 0 16px 48px rgba(0, 0, 0, 0.12);
  border-color: rgba(9, 105, 218, 0.3);
}

.input-main {
  display: flex;
  align-items: flex-end;
  gap: 8px;
}

.main-textarea :deep(.n-input-wrapper) {
  background: transparent !important;
  border: none !important;
}

.main-textarea :deep(textarea) {
  font-size: 15px;
  padding: 8px 4px;
}

.icon-btn {
  color: #636c76;
}

.send-float-btn {
  width: 42px;
  height: 42px;
  flex-shrink: 0;
  margin-bottom: 4px;
}

.shadow-md { box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); }

/* 响应式 */
@media (max-width: 768px) {
  .message-inner-wrap { padding: 40px 20px 120px 20px; }
  .glass-input-card { max-width: 100%; border-radius: 16px; }
}
</style>
