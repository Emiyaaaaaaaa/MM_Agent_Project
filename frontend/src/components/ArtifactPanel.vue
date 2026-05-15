<template>
  <div class="artifact-panel">
    <n-tabs type="line" animated class="artifact-tabs">
      <!-- 1. 新增：研究资料管理 (原本在配置页的功能移入此处) -->
      <n-tab-pane name="materials" tab="研究资料">
        <div class="tab-content">
          <div class="artifact-header">
            <n-h3>多模态素材库</n-h3>
            <n-text depth="3">上传 PDF、论文或实验图片以增强 Agent 的 RAG 知识库。</n-text>
          </div>
          
          <div class="upload-area academic-card">
            <n-upload
              multiple
              directory-dnd
              :max="5"
              action="/api/v1/upload"
              @finish="handleUploadFinish"
            >
              <n-upload-dragger>
                <div style="margin-bottom: 12px">
                  <n-icon size="36" :depth="3"><CloudUploadOutline /></n-icon>
                </div>
                <n-text style="font-size: 14px">拖拽至此上传研究素材</n-text>
              </n-upload-dragger>
            </n-upload>
          </div>

          <n-divider />
          <n-list bordered v-if="uploadedFiles.length > 0">
            <n-list-item v-for="file in uploadedFiles" :key="file.name">
              <n-space justify="space-between" align="center" style="width: 100%">
                <n-text>{{ file.name }}</n-text>
                <n-tag size="tiny" type="success">已入库</n-tag>
              </n-space>
            </n-list-item>
          </n-list>
          <n-empty v-else description="暂无素材，请上传以开始分析" />
        </div>
      </n-tab-pane>

      <n-tab-pane name="visuals" tab="可视化结果">
        <div class="tab-content">
          <div class="artifact-header">
            <n-h3>仿真运行结果</n-h3>
          </div>
          <div v-if="artifactsManifest.length > 0" class="artifact-list academic-card">
            <div v-for="(item, idx) in artifactsManifest" :key="`${item.url || item.path || idx}`" class="artifact-item">
              <n-image
                v-if="item.kind === 'image' && item.url"
                width="100%"
                :src="item.url"
                fallback-src="https://via.placeholder.com/600x400?text=图像加载失败"
                show-toolbar-tooltip
              />
              <div v-else class="artifact-link-row">
                <n-text>{{ item.filename || item.path || 'artifact' }}</n-text>
                <a v-if="item.url" :href="item.url" target="_blank" rel="noopener noreferrer">打开</a>
              </div>
            </div>
          </div>
          <div class="plot-container academic-card">
            <n-image
              v-if="plotArtifactUrl"
              width="100%"
              :src="plotArtifactUrl"
              fallback-src="https://via.placeholder.com/600x400?text=等待仿真输出..."
              show-toolbar-tooltip
            />
            <n-empty v-else description="暂无仿真数据" />
          </div>
        </div>
      </n-tab-pane>

      <n-tab-pane name="draft" tab="论文草稿">
        <div class="tab-content">
          <div class="artifact-header flex-header">
            <n-h3>论文实时预览</n-h3>
            <n-button quaternary size="small" type="primary" @click="handleExportPdf">
              <template #icon><n-icon><DownloadOutline /></n-icon></template>
              导出 PDF
            </n-button>
          </div>
          <div class="paper-preview academic-card">
            <ContentBlocks :blocks="paperDraftBlocks" />
            <n-empty v-if="paperDraftBlocks.length === 0" description="论文撰写中..." />
          </div>
        </div>
      </n-tab-pane>
    </n-tabs>
  </div>
</template>

<script setup lang="ts">
import { storeToRefs } from 'pinia'
import { ref } from 'vue'
import { 
  NTabs, NTabPane, NH3, NText, NImage, NEmpty, NButton, NIcon, NUpload, NUploadDragger, NDivider, NList, NListItem, NSpace, NTag, useMessage 
} from 'naive-ui'
import { DownloadOutline, CloudUploadOutline } from '@vicons/ionicons5'
import ContentBlocks from './ContentBlocks.vue'
import { useTaskStore } from '../stores/task'
import { getApiUrl } from '../config'

const message = useMessage()
const taskStore = useTaskStore()
const { plotArtifactUrl, paperDraftBlocks, artifactsManifest } = storeToRefs(taskStore)
const uploadedFiles = ref<any[]>([])

const handleUploadFinish = ({ file }: { file: any }) => {
  uploadedFiles.value.push(file)
  message.success(`${file.name} 已成功入库，Agent 已感知。`)
}

const handleExportPdf = () => {
  const pdfUrl = getApiUrl('/exports/thesis.pdf')
  window.open(pdfUrl, '_blank', 'noopener,noreferrer')
}

</script>

<style scoped>
.artifact-panel { height: 100%; display: flex; flex-direction: column; }
.artifact-tabs :deep(.n-tabs-nav) { padding: 0 20px; background-color: #f6f8fa; }
.tab-content { padding: 24px; }
.upload-area { padding: 12px; margin-bottom: 16px; border: 1px dashed #d0d7de; }
.artifact-header { margin-bottom: 20px; }
.flex-header { display: flex; justify-content: space-between; align-items: center; }
.plot-container { padding: 12px; background: white; }
.artifact-list { padding: 12px; background: white; margin-bottom: 12px; display: flex; flex-direction: column; gap: 12px; }
.artifact-item { border: 1px solid #d0d7de; border-radius: 8px; padding: 8px; }
.artifact-link-row { display: flex; justify-content: space-between; gap: 8px; align-items: center; font-size: 13px; }
.paper-preview { padding: 40px; min-height: 600px; background: white; border: 1px solid #d0d7de; font-family: 'Times New Roman', serif; }
.markdown-body { font-size: 14px; line-height: 1.8; color: #1f2328; }
</style>
