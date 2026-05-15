<template>
  <div class="login-container">
    <div class="login-card academic-card">
      <div class="header">
        <n-icon size="64" color="#0969da"><SchoolOutline /></n-icon>
        <n-h1>MCM/ICM 多模态 Agent</n-h1>
        <n-text depth="3">学术建模辅助系统 · {{ isRegister ? '账号注册' : '账号登录' }}</n-text>
      </div>

      <n-form ref="formInst" :model="formValue" :rules="rules">
        <n-form-item label="电子邮箱 (Email)" path="email">
          <n-input 
            v-model:value="formValue.email" 
            placeholder="请输入您的邮箱地址" 
          />
        </n-form-item>

        <n-form-item label="密码 (Password)" path="password">
          <n-input 
            v-model:value="formValue.password" 
            type="password"
            show-password-on="click"
            placeholder="请输入密码" 
          />
        </n-form-item>

        <div class="actions">
          <n-button
            type="primary"
            size="large"
            block
            @click="handleSubmit"
            :loading="isLoading"
          >
            {{ isRegister ? '立即注册' : '登录系统' }}
          </n-button>
          
          <n-button quaternary block @click="isRegister = !isRegister" class="toggle-btn">
            {{ isRegister ? '已有账号？去登录' : '没有账号？去注册' }}
          </n-button>
        </div>
      </n-form>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive } from 'vue'
import { useRouter } from 'vue-router'
import { 
  NH1, NText, NForm, NFormItem, NInput, NButton, NIcon, useMessage 
} from 'naive-ui'
import type { FormRules } from 'naive-ui'
import { SchoolOutline } from '@vicons/ionicons5'
import { useUserStore } from '../stores/user'
import { getApiUrl } from '../config'

const router = useRouter()
const message = useMessage()
const userStore = useUserStore()

const isRegister = ref(false)
const isLoading = ref(false)

const formValue = reactive({
  email: '',
  password: ''
})

const rules: FormRules = {
  email: { required: true, message: '请输入邮箱', trigger: 'blur', type: 'email' },
  password: { required: true, message: '请输入密码', trigger: 'blur' }
}

const handleSubmit = async () => {
  if (!formValue.email || !formValue.password) return
  
  isLoading.value = true
  const endpoint = isRegister.value ? '/api/v1/auth/register' : '/api/v1/auth/login'
  
  try {
    const response = await fetch(getApiUrl(endpoint), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        email: formValue.email,
        password: formValue.password
      })
    })

    const data = await response.json()
    
    if (response.ok) {
      userStore.setUserId(data.id)
      message.success(isRegister.value ? '注册成功，请登录' : '登录成功')
      
      if (isRegister.value) {
        isRegister.value = false
      } else {
        router.push('/setup')
      }
    } else {
      message.error(data.detail || '操作失败')
    }
  } catch (err) {
    message.error('无法连接到后端服务器')
  } finally {
    isLoading.value = false
  }
}
</script>

<style scoped>
.login-container {
  height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: #f6f8fa;
}

.login-card {
  width: 100%;
  max-width: 450px;
  padding: 56px 48px;
  background-color: #ffffff;
  border-radius: 12px;
  border: 1px solid #d0d7de;
}

.header {
  text-align: center;
  margin-bottom: 40px;
}

.header h1 {
  margin: 16px 0 4px;
  font-size: 24px;
  font-weight: 700;
}

.actions {
  margin-top: 12px;
}

.toggle-btn {
  margin-top: 12px;
}
</style>
