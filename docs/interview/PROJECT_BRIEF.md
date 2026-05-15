# MM Agent 项目说明（面试版）

## 1. 项目目标
- 面向 MCM/ICM 数学建模场景，提供多模态题目理解、知识检索、建模推导、代码仿真、审查与论文产出的端到端 Agent。
- 核心目标是将人工协作链路工程化为可追踪、可中断、可恢复的状态机流程。

## 2. 系统架构
- 前端：Vue3 + Vite + Pinia + Naive UI。
- 后端：FastAPI + LangGraph + Gemini。
- 存储：
  - SQLite：用户、任务、消息。
  - ChromaDB：本地 RAG 向量检索。
  - SQLite Checkpoint：LangGraph 会话状态恢复。
- 通信：
  - HTTP：鉴权、任务管理、历史消息、上传、检索。
  - WebSocket：Token 流式输出、状态事件、最终结果。
  - 结构化协议：`ContentBlock[] + stage` 作为统一事件与消息载体。

## 3. 后端关键设计
- 路由层拆分为 `backend/routers/*`，`main.py` 仅做应用装配。
- Graph 运行时采用 `GraphRuntime` 托管，避免 import 时创建全局单例带来的生命周期问题。
- 统一序列化层 `backend/services/serialization.py`，将 LangChain Message 转为 JSON 安全类型。
- 统一结构化层：所有节点产出、WS 事件、历史消息回放全部转为 `ContentBlock[]`。
- 任务级 API Key 注入到 `shared_memory`，各节点按任务维度调用模型。
- `Modeler` 增加空输出自愈：同模型重试 + 稳定模型降级兜底。
- `Sandbox` 增加受控自动安装：仅白名单包可安装并自动重跑一次。
- `Coder` 从单图输出升级为动态产物清单：通过执行前后文件快照生成 `artifact_manifest`，并经 WS 实时推送。
- 数据库标准化分阶段推进：
  - Phase 1：补齐 `updated_at / is_deleted / deleted_at`，读路径默认过滤软删除，删除改为软删除。
  - Phase 1.1：启动时自动补列与补索引（SQLite 兼容策略）。
  - Phase 2（已完成执行）：对 `tasks/messages` 关键字段落地 `NOT NULL`，并保留 `*_backup_v1` 备份表以支持回滚。

## 4. 前端关键设计
- 建立统一入口与路由守卫：`main.ts` + `router/index.ts`。
- 将任务上下文与流程状态集中到 `stores/task.ts`。
- WebSocket 抽象为 `composables/useWebSocket.ts`，统一处理 `TOKEN/STATUS/PROGRESS/FINAL/ERROR` 事件。
- UI 渲染从字符串 Markdown 切换到块级渲染组件，支持结构化内容直出。
- 结果面板支持 `artifact_manifest` 动态渲染（多图、多导出文件链接），并兼容旧 `artifact_url`。
- Setup 页支持任务新建、进入、删除。

## 5. 技术选型与取舍（可直接口述）
- 选 FastAPI：快速暴露 HTTP/WS 双通道，类型提示完整，开发效率高。
- 选 LangGraph：天然适合多节点可中断流程，比纯 prompt 链更可控。
- 选 SQLite + ChromaDB：本地化部署成本低，适合比赛/实验环境。
- 选 Vue3 + Pinia：前端状态流清晰，实时流式 UI 开发成本低。
- 主要权衡：
  - 本地优先降低部署门槛，但牺牲了分布式扩展能力。
  - 强状态控制提升可靠性，但增加了运行时与生命周期复杂度。

## 6. 当前能力边界
- 具备可用的任务化对话、流式推理、历史记录、任务删除、任务内重新开始（清空消息 + 清空 checkpoint）、图状态持久化。
- 具备结构化端到端协议（消息/事件/阶段）与模型空输出自愈机制。
- 具备动态实验资产管线：可按上游分析结果生成并展示多算法、多图表、多导出文件。
- 具备数据库演进与回滚能力：已验证结构兼容补齐、非空约束迁移、完整性校验与备份回退路径。
- 若进入生产，需要补齐：统一鉴权令牌、审计日志、限流、监控告警与更完整测试流水线。

## 7. 一句话价值总结（面试可用）
- 将数学建模 Agent 从功能验证阶段提升到工程可维护阶段：具备契约一致性、状态恢复能力、问题闭环机制和可追溯日志。
