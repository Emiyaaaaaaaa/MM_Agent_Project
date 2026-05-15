<template>
  <n-config-provider :theme-overrides="themeOverrides">
    <n-message-provider>
      <n-dialog-provider>
        <div class="app-container">
          <router-view />
        </div>
      </n-dialog-provider>
    </n-message-provider>
  </n-config-provider>
</template>

<script setup lang="ts">
import { NConfigProvider, NMessageProvider, NDialogProvider, GlobalThemeOverrides } from 'naive-ui'

/**
 * 学术风浅色主题定制 (Light Academic Theme)
 * 对标 Overleaf / arXiv 阅读器风格
 */
const themeOverrides: GlobalThemeOverrides = {
  common: {
    primaryColor: '#0969da',
    primaryColorHover: '#0a7df3',
    primaryColorPressed: '#075bb7',
    bodyColor: '#f6f8fa',
    cardColor: '#ffffff',
    borderColor: '#d0d7de',
    borderRadius: '6px'
  },
  Card: {
    borderRadius: '8px',
    borderColor: '#d0d7de'
  },
  Button: {
    borderRadius: '6px',
    fontWeightStrong: '600'
  }
}
</script>

<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono&display=swap');

html, body, #app {
  height: 100%;
  margin: 0;
  padding: 0;
}

.app-container {
  height: 100vh;
  width: 100vw;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

/* 一些通用的卡片悬浮效果 */
.academic-card {
  transition: box-shadow 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

.academic-card:hover {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
}
</style>
