<template>
  <div class="side-panel">
    <div class="section-header">
      <n-text depth="3" strong>当前任务</n-text>
    </div>
    <div class="task-info">
      <n-h4 prefix="bar" align-text>
        {{ taskStore.currentTaskName }}
      </n-h4>
      <n-text depth="3" class="task-meta">
        模型: {{ taskStore.currentModel }}
      </n-text>
    </div>

    <n-divider />

    <div class="section-header">
      <n-text depth="3" strong>解题流水线</n-text>
    </div>
    <div class="pipeline-list">
      <div 
        v-for="node in taskStore.nodes" 
        :key="node.id" 
        :class="['node-item', node.status]"
      >
        <div class="node-icon">
          <n-spin v-if="node.status === 'running'" size="small" />
          <n-icon v-else-if="node.status === 'done'" color="#1a7f37"><CheckmarkCircleOutline /></n-icon>
          <n-icon v-else depth="3"><EllipseOutline /></n-icon>
        </div>
        <div class="node-content">
          <div class="node-title">{{ node.name }}</div>
          <div class="node-desc" v-if="node.status === 'running'">{{ node.progressMsg }}</div>
          <n-progress
            v-if="node.status === 'running'"
            type="line"
            :percentage="node.progress"
            :show-indicator="false"
            :height="4"
            processing
          />
        </div>
      </div>
    </div>

    <n-divider />

    <div class="section-header">
      <n-text depth="3" strong>中间产物缩略</n-text>
    </div>
    <div class="preview-list" v-if="taskStore.intermediatePreviews.length">
      <div class="preview-card" v-for="(item, idx) in taskStore.intermediatePreviews" :key="`${item.createdAt}-${idx}`">
        <div class="preview-title">{{ item.node }}</div>
        <ContentBlocks :blocks="item.summary_blocks" />
        <div v-if="item.node === 'Review' && item.preview_blocks?.length" class="review-full-wrap">
          <div class="review-full-title">完整审查结果</div>
          <ContentBlocks :blocks="item.preview_blocks" />
        </div>
      </div>
    </div>
    <n-text depth="3" v-else>暂无中间产物</n-text>
  </div>
</template>

<script setup lang="ts">
import { 
  NText, NH4, NDivider, NIcon, NSpin, NProgress 
} from 'naive-ui'
import { 
  CheckmarkCircleOutline, EllipseOutline 
} from '@vicons/ionicons5'
import { useTaskStore } from '../stores/task'
import ContentBlocks from './ContentBlocks.vue'

const taskStore = useTaskStore()
</script>

<style scoped>
.side-panel { padding: 20px; }
.section-header { margin-bottom: 12px; font-size: 11px; letter-spacing: 0.05em; }
.task-info { margin-bottom: 24px; }
.task-meta { font-size: 12px; }
.pipeline-list { display: flex; flex-direction: column; gap: 16px; }
.node-item { display: flex; gap: 12px; padding: 10px; border-radius: 8px; transition: background-color 0.2s; }
.node-item.running { background-color: #ffffff; border: 1px solid #d0d7de; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
.node-icon { width: 20px; display: flex; align-items: center; justify-content: center; }
.node-title { font-size: 13px; font-weight: 600; color: #1f2328; }
.node-desc { font-size: 11px; color: #57606a; margin: 4px 0; }
.node-item.idle .node-title { color: #8c959f; }
.preview-list { display: flex; flex-direction: column; gap: 8px; }
.preview-card { border: 1px solid #d0d7de; border-radius: 8px; padding: 8px 10px; background: #fff; }
.preview-title { font-size: 12px; font-weight: 600; margin-bottom: 4px; color: #1f2328; }
.review-full-wrap { margin-top: 8px; padding-top: 8px; border-top: 1px dashed #d0d7de; }
.review-full-title { font-size: 11px; color: #57606a; margin-bottom: 4px; }
.preview-summary { font-size: 11px; color: #57606a; line-height: 1.4; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
</style>
