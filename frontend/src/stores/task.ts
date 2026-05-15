import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { ContentBlock, StagePayload } from '../types/content'
import { blocksToText } from '../types/content'

type NodeStatus = 'idle' | 'running' | 'done'

interface PipelineNode {
  id: string
  name: string
  status: NodeStatus
  progress: number
  progressMsg: string
}

interface IntermediatePreview {
  node: string
  summary_blocks: ContentBlock[]
  preview_blocks: ContentBlock[]
  stage?: StagePayload
  createdAt: string
}

export interface ArtifactItem {
  kind?: string
  path?: string
  filename?: string
  size?: number
  updated_at_ns?: number
  url?: string
}

const DEFAULT_NODES: PipelineNode[] = [
  { id: 'Reader', name: '文件读取', status: 'idle', progress: 0, progressMsg: '' },
  { id: 'Fusion', name: '知识融合', status: 'idle', progress: 0, progressMsg: '' },
  { id: 'Analysis', name: '问题分析', status: 'idle', progress: 0, progressMsg: '' },
  { id: 'Modeling', name: '数学建模', status: 'idle', progress: 0, progressMsg: '' },
  { id: 'Coder', name: '仿真编码', status: 'idle', progress: 0, progressMsg: '' },
  { id: 'Review', name: '结果审查', status: 'idle', progress: 0, progressMsg: '' },
  { id: 'Writing', name: '论文写作', status: 'idle', progress: 0, progressMsg: '' },
  { id: 'Export', name: '导出产物', status: 'idle', progress: 0, progressMsg: '' },
]

export const useTaskStore = defineStore('task', () => {
  const currentTaskId = ref(localStorage.getItem('task_id') || '')
  const currentTaskName = ref(localStorage.getItem('task_name') || '')
  const currentModel = ref(localStorage.getItem('task_model') || 'gemini-2.5-flash-lite')
  const nodes = ref<PipelineNode[]>(DEFAULT_NODES.map((item) => ({ ...item })))
  const intermediatePreviews = ref<IntermediatePreview[]>([])
  /** Coder 产出图表 URL（相对路径如 /static/plots/output.png），由 WS PROGRESS 更新 */
  const plotArtifactUrl = ref('')
  const paperDraftBlocks = ref<ContentBlock[]>([])
  const artifactsManifest = ref<ArtifactItem[]>([])

  const setCurrentTask = (id: string, name: string, model: string) => {
    currentTaskId.value = id
    currentTaskName.value = name
    currentModel.value = model
    localStorage.setItem('task_id', id)
    localStorage.setItem('task_name', name)
    localStorage.setItem('task_model', model)
    resetNodes()
    clearIntermediatePreviews()
  }

  const clearCurrentTask = () => {
    currentTaskId.value = ''
    currentTaskName.value = ''
    currentModel.value = 'gemini-2.5-flash-lite'
    localStorage.removeItem('task_id')
    localStorage.removeItem('task_name')
    localStorage.removeItem('task_model')
    resetNodes()
    clearIntermediatePreviews()
  }

  const resetNodes = () => {
    nodes.value = DEFAULT_NODES.map((item) => ({ ...item }))
    plotArtifactUrl.value = ''
    paperDraftBlocks.value = []
    artifactsManifest.value = []
  }

  const setStatus = (nodeId: string, payload: { blocks?: ContentBlock[]; percentage?: number; artifact_url?: string; artifact_urls?: string[]; artifact_manifest?: ArtifactItem[] }) => {
    const node = nodes.value.find((item) => item.id === nodeId)
    if (!node) return
    node.status = 'running'
    node.progressMsg = blocksToText(payload?.blocks || []) || '处理中'
    const percentage = payload?.percentage
    if (typeof percentage === 'number') {
      node.progress = Math.max(0, Math.min(100, percentage))
      if (node.progress >= 100) {
        node.status = 'done'
      }
    }
    if (payload?.artifact_url) {
      plotArtifactUrl.value = payload.artifact_url
    }
    if (Array.isArray(payload?.artifact_manifest) && payload.artifact_manifest.length > 0) {
      artifactsManifest.value = payload.artifact_manifest
      const firstImage = payload.artifact_manifest.find((item) => item?.url && item.kind === 'image')
      if (firstImage?.url) {
        plotArtifactUrl.value = firstImage.url
      }
    } else if (Array.isArray(payload?.artifact_urls) && payload.artifact_urls.length > 0) {
      artifactsManifest.value = payload.artifact_urls.map((url) => ({ kind: 'file', url }))
      plotArtifactUrl.value = payload.artifact_urls[0]
    }
  }

  const markFinished = (nodeId: string) => {
    const node = nodes.value.find((item) => item.id === nodeId)
    if (!node) return
    node.status = 'done'
    node.progress = 100
  }

  const addIntermediatePreview = (payload: { node?: string; summary_blocks?: ContentBlock[]; preview_blocks?: ContentBlock[]; stage?: StagePayload }) => {
    intermediatePreviews.value.unshift({
      node: payload.node || 'Unknown',
      summary_blocks: payload.summary_blocks || [{ type: 'text', text: '阶段已完成，等待确认' }],
      preview_blocks: payload.preview_blocks || [],
      stage: payload.stage,
      createdAt: new Date().toISOString(),
    })
    if (intermediatePreviews.value.length > 12) {
      intermediatePreviews.value = intermediatePreviews.value.slice(0, 12)
    }
  }

  const clearIntermediatePreviews = () => {
    intermediatePreviews.value = []
  }

  const setPaperDraftBlocks = (blocks?: ContentBlock[]) => {
    paperDraftBlocks.value = Array.isArray(blocks) ? blocks : []
  }

  return {
    currentTaskId,
    currentTaskName,
    currentModel,
    nodes,
    intermediatePreviews,
    plotArtifactUrl,
    paperDraftBlocks,
    artifactsManifest,
    setCurrentTask,
    clearCurrentTask,
    resetNodes,
    setStatus,
    markFinished,
    addIntermediatePreview,
    clearIntermediatePreviews,
    setPaperDraftBlocks,
  }
})
