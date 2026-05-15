import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useUserStore = defineStore('user', () => {
  const userId = ref(localStorage.getItem('user_id') || '')
  const apiKey = ref(localStorage.getItem('api_key') || '')
  const selectedModel = ref(localStorage.getItem('selected_model') || 'gemini-2.5-flash-lite')

  const setUserId = (id: string) => {
    userId.value = id
    localStorage.setItem('user_id', id)
  }

  const setApiKey = (key: string) => {
    apiKey.value = key
    localStorage.setItem('api_key', key)
  }

  const setSelectedModel = (model: string) => {
    selectedModel.value = model
    localStorage.setItem('selected_model', model)
  }

  const logout = () => {
    userId.value = ''
    apiKey.value = ''
    selectedModel.value = 'gemini-2.5-flash-lite'
    localStorage.removeItem('user_id')
    localStorage.removeItem('api_key')
    localStorage.removeItem('selected_model')
  }

  return {
    userId,
    apiKey,
    selectedModel,
    setUserId,
    setApiKey,
    setSelectedModel,
    logout,
  }
})
