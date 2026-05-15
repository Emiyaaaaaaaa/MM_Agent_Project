import { createRouter, createWebHistory } from 'vue-router'
import LoginView from '../views/LoginView.vue'
import SetupView from '../views/Setup.vue'
import WorkbenchView from '../views/Workbench.vue'
import { useUserStore } from '../stores/user'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', redirect: '/login' },
    { path: '/login', name: 'login', component: LoginView },
    { path: '/setup', name: 'setup', component: SetupView, meta: { requiresAuth: true } },
    { path: '/workbench', name: 'workbench', component: WorkbenchView, meta: { requiresAuth: true } },
  ],
})

router.beforeEach((to) => {
  const userStore = useUserStore()
  if (to.meta.requiresAuth && !userStore.userId) {
    return { name: 'login' }
  }
  if (to.name === 'login' && userStore.userId) {
    return { name: 'setup' }
  }
  return true
})

export default router
