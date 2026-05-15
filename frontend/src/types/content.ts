export type ContentBlockType = 'text' | 'markdown' | 'code' | 'math' | 'image' | 'table'

export interface ContentBlock {
  type: ContentBlockType
  text?: string
  code?: string
  language?: string
  latex?: string
  url?: string
  alt?: string
  headers?: string[]
  rows?: string[][]
}

export interface StagePayload {
  id: string
  label: string
  meta?: Record<string, unknown>
}

export const textToBlocks = (text: string): ContentBlock[] => [{ type: 'text', text }]

export const blocksToText = (blocks: ContentBlock[]): string =>
  (blocks || [])
    .map((block) => {
      if (block.type === 'text' || block.type === 'markdown') return block.text || ''
      if (block.type === 'code') return block.code || ''
      if (block.type === 'math') return block.latex || ''
      if (block.type === 'image') return block.alt || block.url || ''
      if (block.type === 'table') return JSON.stringify(block.rows || [])
      return ''
    })
    .join('\n')
    .trim()
