<template>
  <div class="blocks-wrap">
    <template v-for="(block, idx) in blocks" :key="idx">
      <p v-if="block.type === 'text'" class="block-text">{{ block.text }}</p>
      <div v-else-if="block.type === 'markdown'" class="block-markdown" v-html="renderMarkdownLinks(block.text || '')"></div>
      <pre v-else-if="block.type === 'code'" class="block-code"><code>{{ block.code }}</code></pre>
      <div v-else-if="block.type === 'math'" class="block-math">{{ block.latex }}</div>
      <img v-else-if="block.type === 'image'" class="block-image" :src="block.url" :alt="block.alt || 'image'" />
      <div v-else-if="block.type === 'table'" class="block-table-wrap">
        <table class="block-table">
          <thead v-if="block.headers?.length">
            <tr>
              <th v-for="(head, hidx) in block.headers" :key="hidx">{{ head }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(row, ridx) in block.rows || []" :key="ridx">
              <td v-for="(cell, cidx) in row" :key="cidx">{{ cell }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import type { ContentBlock } from '../types/content'

defineProps<{
  blocks: ContentBlock[]
}>()

const escapeHtml = (input: string): string =>
  input
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')

const renderMarkdownLinks = (text: string): string => {
  const safe = escapeHtml(text || '')
  const withLinks = safe.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+|\/[^)\s]+)\)/g, (_m, label, url) => {
    const href = String(url)
    const target = href.startsWith('http') ? ' target="_blank" rel="noopener noreferrer"' : ''
    return `<a href="${href}"${target}>${label}</a>`
  })
  return withLinks.replace(/\n/g, '<br/>')
}
</script>

<style scoped>
.blocks-wrap { display: flex; flex-direction: column; gap: 8px; }
.block-text { margin: 0; white-space: pre-wrap; }
.block-markdown { white-space: normal; }
.block-markdown :deep(a) { color: #0969da; text-decoration: underline; }
.block-code { margin: 0; padding: 10px; border: 1px solid #d0d7de; border-radius: 8px; background: #f6f8fa; overflow: auto; }
.block-math { font-family: 'Times New Roman', serif; }
.block-image { max-width: 100%; border-radius: 8px; border: 1px solid #d0d7de; }
.block-table-wrap { overflow: auto; }
.block-table { border-collapse: collapse; width: 100%; }
.block-table th, .block-table td { border: 1px solid #d0d7de; padding: 6px 8px; font-size: 12px; }
</style>
