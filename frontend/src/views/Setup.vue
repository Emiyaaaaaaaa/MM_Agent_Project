<template>
  <div class="setup-container">
    <!-- 顶部导航 -->
    <div class="top-nav">
      <div class="brand">
        <n-icon size="32" color="#0969da"><SchoolOutline /></n-icon>
        <span class="brand-text">MCM/ICM Research Workbench</span>
      </div>
      <n-button quaternary @click="logout" type="error">退出登录</n-button>
    </div>

    <div class="content-grid">
      <!-- 左侧：新建任务 -->
      <div class="setup-section">
        <div class="section-card academic-card">
          <div class="header">
            <n-icon size="40" color="#0969da"><AddCircleOutline /></n-icon>
            <n-h2>开启新建模课题</n-h2>
            <n-text depth="3">系统将自动为您生成该任务的专属命名空间。</n-text>
          </div>

          <n-form ref="formInst" :model="formValue" :rules="rules">
            <n-form-item label="任务/课题名称 (最多 50 字)" path="taskName">
              <n-input v-model:value="formValue.taskName" placeholder="如: 2026 MCM Problem A - Canal Analysis" maxlength="50" show-count />
            </n-form-item>

            <n-form-item label="演算模型型号" path="model">
              <n-select v-model:value="formValue.model" :options="modelOptions" />
            </n-form-item>

            <n-form-item label="Google API Key" path="apiKey">
              <n-input
                v-model:value="formValue.apiKey"
                type="password"
                show-password-on="click"
                placeholder="在此粘贴密钥 (建议启用权限限制)"
              />
            </n-form-item>

            <div class="actions">
              <n-button
                type="primary"
                size="large"
                block
                :loading="isLoading"
                @click="handleCreate"
              >
                初始化并行研究空间
              </n-button>
            </div>
          </n-form>
        </div>
      </div>

      <!-- 右侧：历史任务列表 -->
      <div class="history-section">
        <div class="history-header">
          <n-text strong style="font-size: 18px">您的历史研究项目</n-text>
          <n-text depth="3" v-if="tasks.length">共 {{ tasks.length }} 个活跃任务</n-text>
        </div>

        <n-scrollbar style="max-height: calc(100vh - 250px)">
          <div class="task-list">
            <!-- 加载状态 -->
            <div v-if="isFetchingTasks" class="empty-state">
              <n-spin size="large" />
            </div>

            <!-- 空状态 -->
            <div v-else-if="tasks.length === 0" class="empty-state">
              <n-icon size="48" color="#d0d7de"><TimeOutline /></n-icon>
              <n-text depth="3">暂无历史任务</n-text>
            </div>

            <!-- 任务卡片流 -->
            <div 
              v-else 
              v-for="task in sortedTasks" 
              :key="task.id" 
              class="task-item academic-card clickable"
              @click="enterTask(task)"
            >
              <div class="task-info">
                <div class="task-title">{{ task.title }}</div>
                <div class="task-meta">
                  <n-tag :bordered="false" type="info" size="small">{{ task.model_id }}</n-tag>
                  <span class="time">{{ formatDate(task.created_at) }}</span>
                </div>
              </div>
              <div class="task-actions">
                <n-button
                  quaternary
                  size="small"
                  type="error"
                  :loading="deletingTaskId === task.id"
                  @click.stop="handleDeleteTask(task)"
                >
                  <template #icon><n-icon><TrashOutline /></n-icon></template>
                </n-button>
                <n-icon size="24" class="arrow"><ArrowForwardOutline /></n-icon>
              </div>
            </div>
          </div>
        </n-scrollbar>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, computed } from 'vue'
import { useRouter } from 'vue-router'
import { 
  NH2, NText, NForm, NFormItem, NInput, NSelect, NIcon, NButton, NScrollbar, NSpin, NTag, useMessage, useDialog
} from 'naive-ui'
import { 
  AddCircleOutline, SettingsOutline, TimeOutline, SchoolOutline, ArrowForwardOutline, TrashOutline 
} from '@vicons/ionicons5'
import { useUserStore } from '../stores/user'
import { useTaskStore } from '../stores/task'
import { getApiUrl } from '../config'

const router = useRouter()
const message = useMessage()
const dialog = useDialog()
const userStore = useUserStore()
const taskStore = useTaskStore()

const isLoading = ref(false)
const isFetchingTasks = ref(false)
const deletingTaskId = ref('')
const tasks = ref<any[]>([])

const formValue = reactive({
  taskName: '',
  apiKey: userStore.apiKey || '',
  model: userStore.selectedModel || 'gemini-2.5-flash-lite'
})

const rules = {
  taskName: { required: true, message: '请定义一个课题名称', trigger: 'blur' },
  apiKey: { required: true, message: 'API Key 必填', trigger: 'blur' }
}

const modelOptions = [
  { label: 'Gemini 3.1 Pro (Preview)', value: 'gemini-3.1-pro-preview' },
  { label: 'Gemini 3 Flash (Preview)', value: 'gemini-3-flash-preview' },
  { label: 'Gemini 2.5 Flash Lite', value: 'gemini-2.5-flash-lite' }
]

// 计算属性：按时间倒序排列
const sortedTasks = computed(() => {
  return [...tasks.value].sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
})

// 获取用户任务列表
const fetchTasks = async () => {
  if (!userStore.userId) return
  isFetchingTasks.value = true
  try {
    const response = await fetch(getApiUrl(`/api/v1/users/${userStore.userId}/tasks`))
    if (response.ok) {
      tasks.value = await response.json()
    }
  } catch (err) {
    console.error('Failed to fetch tasks', err)
  } finally {
    isFetchingTasks.value = false
  }
}

const handleCreate = async () => {
  if (!formValue.apiKey || !formValue.taskName) return
  
  isLoading.value = true
  try {
    const response = await fetch(getApiUrl('/api/v1/tasks'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: userStore.userId,
        title: formValue.taskName,
        api_key: formValue.apiKey,
        model_id: formValue.model
      })
    })
    
    const data = await response.json()
    if (response.ok) {
      console.log('Task Created:', data)
      taskStore.setCurrentTask(data.id, data.title, data.model_id)
      userStore.setApiKey(formValue.apiKey)
      userStore.setSelectedModel(formValue.model)
      message.success('新任务空间初始化成功')
      setTimeout(() => router.push({ name: 'workbench' }), 300)
    } else if (response.status === 400) {
      // 智能处理重名冲突：计算 ID 并尝试直接进入
      const safeTitle = formValue.taskName.replace(/[^a-zA-Z0-9\s-_]/g, '').trim()
      const existingId = `${userStore.userId.slice(0, 6)}_${safeTitle.replace(/\s+/g, '_')}`
      console.log('Task already exists, redirecting to:', existingId)
      
      taskStore.setCurrentTask(existingId, formValue.taskName, formValue.model)
      message.info('检测到已存在同名课题，为您直接打开研究空间')
      setTimeout(() => router.push({ name: 'workbench' }), 300)
    } else {
      message.error(data.detail || '初始化失败')
    }
  } catch (err) {
    message.error('无法通往后端实验室，请检查网络连接')
  } finally {
    isLoading.value = false
  }
}

const enterTask = (task: any) => {
  taskStore.setCurrentTask(task.id, task.title, task.model_id)
  message.loading('正在加载研究上下文...', { duration: 1000 })
  setTimeout(() => router.push({ name: 'workbench' }), 300)
}

const handleDeleteTask = async (task: any) => {
  if (!userStore.userId) return

  dialog.warning({
    title: '确认删除任务',
    content: `确定删除任务「${task.title}」吗？此操作不可恢复。`,
    positiveText: '删除',
    negativeText: '取消',
    onPositiveClick: async () => {
      deletingTaskId.value = task.id
      try {
        const response = await fetch(getApiUrl(`/api/v1/users/${userStore.userId}/tasks/${task.id}`), {
          method: 'DELETE'
        })
        const payload = await response.json()
        if (!response.ok) {
          message.error(payload.detail || '删除失败')
          return
        }
        if (taskStore.currentTaskId === task.id) {
          taskStore.clearCurrentTask()
        }
        tasks.value = tasks.value.filter((item) => item.id !== task.id)
        message.success('任务已删除')
      } catch (err) {
        message.error('删除失败，请检查网络连接')
      } finally {
        deletingTaskId.value = ''
      }
    }
  })
}

const logout = () => {
  userStore.logout()
  router.push('/login')
}

const formatDate = (dateStr: string) => {
  const d = new Date(dateStr)
  return `${d.getMonth() + 1}/${d.getDate()} ${d.getHours()}:${d.getMinutes().toString().padStart(2, '0')}`
}

onMounted(() => {
  fetchTasks()
})
</script>

<style scoped>
.setup-container {
  min-height: 100vh;
  padding: 40px;
  background-color: #f6f8fa;
}

.top-nav {
  display: flex;
  justify-content: space-between;
  align-items: center;
  max-width: 1200px;
  margin: 0 auto 40px;
}

.brand {
  display: flex;
  align-items: center;
  gap: 12px;
}

.brand-text {
  font-size: 20px;
  font-weight: 700;
  color: #1f2328;
}

.content-grid {
  display: grid;
  grid-template-columns: 1fr 400px;
  gap: 40px;
  max-width: 1200px;
  margin: 0 auto;
}

.section-card {
  padding: 48px;
  background: white;
  border-radius: 16px;
  border: 1px solid #d0d7de;
}

.header {
  text-align: center;
  margin-bottom: 40px;
}

.header h2 {
  margin: 16px 0 8px;
  font-weight: 700;
}

.actions {
  margin-top: 32px;
}

.history-header {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin-bottom: 24px;
}

.task-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.task-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px;
  background: white;
  border-radius: 12px;
  border: 1px solid #d0d7de;
  transition: all 0.2s ease;
}

.task-item:hover {
  border-color: #0969da;
  transform: translateX(4px);
  box-shadow: 0 4px 12px rgba(9, 105, 218, 0.1);
}

.task-title {
  font-weight: 600;
  font-size: 16px;
  color: #1f2328;
  margin-bottom: 6px;
}

.task-meta {
  display: flex;
  align-items: center;
  gap: 12px;
}

.time {
  font-size: 12px;
  color: #636c76;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 80px 0;
  background: white;
  border-radius: 12px;
  border: 1px dashed #d0d7de;
}

.arrow {
  color: #d0d7de;
  transition: color 0.2s;
}

.task-item:hover .arrow {
  color: #0969da;
}

.task-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

@media (max-width: 900px) {
  .content-grid {
    grid-template-columns: 1fr;
  }
}
</style>
