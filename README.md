# MCM/ICM Mathematical Modeling Agent

面向数学建模竞赛（MCM/ICM）场景的工程化智能体系统。  
目标是帮助用户完成从赛题理解到论文交付的全流程工作，而不仅是生成零散答案。

## 1. 核心功能

### 1.1 竞赛任务全流程自动化
- 支持从题目输入到论文导出的端到端流程：`Reader -> Fusion -> Analyzer -> Modeler -> Coder -> Reviewer -> Writer -> Exporter`。
- 支持同任务内重启（清空消息 + 清空图检查点），避免历史状态污染。
- 支持 WebSocket 实时过程回传与阶段可观测。

### 1.2 结构化协议驱动
- 后端节点产出、WS 事件、历史消息统一采用 `ContentBlock[] + stage`。
- 前端按块级类型渲染（text/markdown/code/math/image/table），减少字符串拼接与渲染歧义。
- 支持产物清单协议（`artifact_manifest`）表达多图、多文件导出。

### 1.3 建模与写作闭环
- Modeler 节点具备空输出重试与模型回退机制，降低流程中断概率。
- Coder 节点支持受控依赖自动安装（白名单）并重试一次，提升可执行率。
- Writer 节点按章节组织论文内容，结合上游分析与图表产物输出完整草稿。
- Exporter 支持论文与产物导出，形成可交付结果包。

### 1.4 数据与状态可靠性
- 基于 LangGraph + checkpoint 的有状态执行，支持恢复与重入。
- 数据层已落地软删除、字段兼容补齐与 SQLite 约束迁移方案（含回滚路径）。
- 后端具备请求/事件日志能力，便于定位链路问题。

## 2. 与现有开源“MM-Agent：面向真实世界数学建模问题的智能体”的差异

本项目的定位更加偏向可落地交付，强调“帮助用户完成竞赛论文”的业务闭环。

- **目标差异**：不仅解决建模问题，还强调最终论文交付（结构化过程 + 多产物导出 + 可回放）。
- **场景差异**：更聚焦具体数学建模竞赛任务（MCM/ICM）与竞赛文档产出要求。
- **工程差异**：强化了任务重启、结构化协议、数据库迁移、日志可观测、失败自愈与回滚机制。
- **产品化取向**：更关注可维护性、可验收性与可持续迭代，而不仅是模型效果展示。

## 3. 技术架构与选型

### 3.1 技术栈
- 前端：Vue 3 + Vite + Pinia + Naive UI
- 后端：FastAPI + LangGraph + LangChain
- 模型：Gemini 系列
- 存储：
  - SQLite（用户、任务、消息、检查点）
  - ChromaDB（本地 RAG 向量检索）

### 3.2 选型理由
- **FastAPI**：HTTP/WS 双通道开发效率高，类型与依赖注入友好。
- **LangGraph**：适配多节点、有状态、可恢复的编排需求。
- **SQLite + ChromaDB**：本地部署成本低，适合竞赛开发与快速迭代。
- **Vue3 + Pinia**：适配事件驱动 UI 与状态集中管理。

## 4. 模型路线与后续模式

- 当前主模型路线为 **Gemini**（含不同任务阶段的模型分配与回退策略）。
- 后续将新增一个模式：基于 **Gemini Deep Research** 的研究型写作流程。
  - 目标是将论文写作流程做得更工程化、更可追踪、更接近成熟交付标准。
  - 重点提升文献整合、论证链条一致性和章节级可控生成能力。

## 5. 快速启动

### 5.1 环境要求
- Python 3.11+
- Node.js 18+

### 5.2 后端
```bash
poetry install
poetry run python -m backend.main
```

### 5.3 前端
```bash
cd frontend
npm install
npm run dev
```

## 6. 协议与文档

- API/WS 契约：`docs/api_contract.md`
- 面试与问题复盘文档：`docs/interview/`

## 7. 说明

- 运行时大文件（数据库、日志、导出图表）默认不纳入版本控制。
- 数据库迁移脚本位于 `backend/db/migrations/`，包含执行入口与回滚说明。
