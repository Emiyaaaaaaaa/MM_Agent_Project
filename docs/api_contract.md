# MM Agent API & WS Contract（结构化协议版）

## 1) 通用结构

### `ContentBlock`
- `type: "text" | "markdown" | "code" | "math" | "image" | "table"`
- `text?: string`
- `code?: string`
- `language?: string`
- `latex?: string`
- `url?: string`
- `alt?: string`
- `headers?: string[]`
- `rows?: string[][]`

### `StagePayload`
- `id: string`（机器可判定）
- `label: string`（前端展示）
- `meta?: object`

---

## 2) HTTP API

### `POST /api/v1/auth/register`
- request:
  - `email: string`
  - `password: string`
- response:
  - `id: string`
  - `email: string`
  - `username: string | null`
  - `created_at: string`

### `POST /api/v1/auth/login`
- request:
  - `email: string`
  - `password: string`
- response:
  - same as register response

### `POST /api/v1/tasks`
- request:
  - `user_id: string`
  - `title: string`
  - `api_key?: string`
  - `model_id?: string`
- response:
  - `id: string`
  - `user_id: string`
  - `title: string`
  - `status: string`
  - `model_id: string`
  - `created_at: string`

### `GET /api/v1/users/{user_id}/tasks`
- response: task array

### `GET /api/v1/tasks/{task_id}/messages`
- response item:
  - `id: string`
  - `task_id: string`
  - `role: "user" | "ai"`
  - `content_blocks: ContentBlock[]`
  - `content_text: string`（仅便于调试/检索）
  - `node: string | null`
  - `created_at: string`

### `POST /api/v1/users/{user_id}/tasks/{task_id}/restart`
- purpose:
  - restart task in-place (keep task metadata, clear conversation history and graph checkpoints)
- response:
  - `status: "success"`
  - `message: string`
  - `cleared_messages: number`
  - `cleared_checkpoints: number`

### `POST /api/v1/chat`
- request:
  - `message: string`
  - `thread_id?: string`
  - `human_feedback?: string`
- response:
  - `status: "success"`
  - `agent_response_blocks: ContentBlock[]`
  - `agent_response_text: string`（便于 CLI/日志）

### `POST /api/v1/upload`
- multipart file upload

### `POST /api/v1/search`
- request:
  - `query: string`
  - `top_k?: number`
- response item:
  - `id: string`
  - `content: string`
  - `distance: number`
  - `filename: string`
  - `year: number`
  - `doc_type: string`
  - `image_urls: string[]`

---

## 3) WebSocket

### URL
- `/ws/chat/{client_id}`

### inbound message
- `message?: string`
- `thread_id?: string`
- `human_feedback?: string`

### outbound event（统一结构化 payload）

#### `TOKEN`
```json
{
  "type": "TOKEN",
  "node": "Modeling",
  "payload": {
    "blocks": [{ "type": "text", "text": "..." }]
  }
}
```

#### `STATUS`
```json
{
  "type": "STATUS",
  "node": "Analysis",
  "status": "running",
  "payload": {
    "blocks": [{ "type": "text", "text": "Started: Analysis" }],
    "stage": { "id": "running_analysis", "label": "Started: Analysis", "meta": {} }
  }
}
```

#### `PROGRESS`（from internal bus）
```json
{
  "type": "PROGRESS",
  "node": "Coder",
  "payload": {
    "blocks": [{ "type": "text", "text": "正在执行..." }],
    "percentage": 60,
    "stage": { "id": "progress_coder", "label": "Coder progress", "meta": {} }
  }
}
```

#### `INTERMEDIATE`
```json
{
  "type": "INTERMEDIATE",
  "payload": {
    "node": "Modeling",
    "summary_blocks": [{ "type": "text", "text": "..." }],
    "preview_blocks": [{ "type": "markdown", "text": "..." }],
    "ask_continue": true,
    "stage": { "id": "modeling_completed", "label": "Mathematical Modeling Completed", "meta": {} }
  }
}
```

#### `FINAL`
```json
{
  "type": "FINAL",
  "payload": {
    "blocks": [{ "type": "markdown", "text": "..." }],
    "stage": { "id": "export_completed", "label": "Physical Asset固化完成", "meta": {} }
  }
}
```

#### `ERROR`
```json
{
  "type": "ERROR",
  "payload": {
    "blocks": [{ "type": "text", "text": "执行中断，已保存当前生成内容" }]
  }
}
```
