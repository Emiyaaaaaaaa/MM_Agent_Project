# MM Agent 问题与修复日志

> 用法：每次线上或联调异常按本格式追加，建议统一采用“问题-根因-动作-结果”的记录结构。

## 记录模板
- 时间：
- 症状：
- 根因：
- 修复动作：
- 验证结果：
- 经验复盘：

---

## 2026-04-21 关键问题记录

### 1) WS 流程中 `AIMessage` 不能 JSON 序列化
- 症状：`Object of type AIMessage is not JSON serializable`
- 根因：直接将 LangChain 消息对象透传到 `send_json` 或响应层。
- 修复动作：
  - 新增统一消息序列化逻辑。
  - WS 与 API 返回前，统一转为字符串/基础 JSON 类型。
- 验证结果：WS `FINAL` 事件可稳定回传，不再抛序列化异常。
- 经验复盘：框架对象与传输对象要解耦，边界层必须统一序列化。

### 2) Graph 编译与运行时生命周期混乱
- 症状：`SqliteSaver does not support async methods`、随后出现 `'_AsyncGeneratorContextManager' object has no attribute 'get_tuple'`
- 根因：
  - 异步事件流用到了 async checkpointer 接口，但初期使用了同步 saver。
  - 后续改为 Async saver 后，又把 async context manager 直接当实例使用。
  - 路由中仍存在 sync/async API 混用（`update_state` vs `aupdate_state`）。
- 修复动作：
  - 重构 `builder.py`，改为纯构建函数。
  - 新建 `GraphRuntime`，由 FastAPI lifespan 托管 startup/shutdown。
  - `chat.py` 从 `app.state` 获取 graph，统一改成异步状态更新。
- 验证结果：后端编译通过，WS 流程不再触发上述异常。
- 经验复盘：含 async 资源的组件必须由应用生命周期托管，不要在 import 阶段初始化。

### 3) 任务级 API Key 与环境变量冲突
- 症状：运行时提示使用 `GOOGLE_API_KEY`，与“任务绑定 key”设计冲突。
- 根因：多个节点保留 env fallback 逻辑。
- 修复动作：
  - 移除运行时 `os.environ.get("GOOGLE_API_KEY")` 回退。
  - 缺少任务 key 时节点显式 REJECTED 并返回清晰提示。
  - RAG 引擎取消默认环境客户端，改为显式 key 注入。
- 验证结果：运行链路仅依赖任务 key，不再隐式使用 env key。
- 经验复盘：多租户/多任务场景下，凭据来源必须单一且可追踪。

### 4) 前后端契约断裂（auth/messages）
- 症状：前端调用接口与后端路由不匹配，出现 404/流程中断。
- 根因：迭代中接口演化未同步，缺少统一契约文档。
- 修复动作：
  - 补齐并统一 auth/tasks/messages/upload/search/ws 契约。
  - 新增 `docs/api_contract.md`。
- 验证结果：前端主流程调用可对齐。
- 经验复盘：接口先契约化，再实现；变更必须同步文档。

### 5) 配置页缺少任务删除功能
- 症状：任务无法在 UI 侧清理，影响多轮测试。
- 根因：只实现了创建/查询，缺少删除链路。
- 修复动作：
  - 后端新增 `DELETE /api/v1/users/{user_id}/tasks/{task_id}`。
  - 前端 Setup 页加入删除按钮与确认对话框（Naive UI dialog）。
  - 删除当前任务时同步清理前端本地状态。
- 验证结果：任务可删除，状态与列表同步更新。
- 经验复盘：任务生命周期应完整覆盖 CRUD，避免“只能增不能减”。

### 6) 同任务多轮调试出现历史状态污染
- 症状：在同一任务持续调试时，节点流转受旧 checkpoint 影响，出现“直接跳阶段”或上下文不符合预期。
- 根因：仅清理前端显示状态，未同步清理任务消息与 LangGraph checkpoint 持久化状态。
- 修复动作：
  - 后端新增 `POST /api/v1/users/{user_id}/tasks/{task_id}/restart`。
  - 重置动作统一执行“清空任务消息 + 删除该 thread_id 的 checkpoints/writes 记录”。
  - 前端工作台新增“重新开始任务”按钮与确认交互，成功后重置聊天与节点状态。
- 验证结果：同任务可原地重开，后续推理不再继承历史残留状态。
- 经验复盘：状态型系统需要“业务重置”与“运行时重置”双通道同时到位，仅清 UI 不足以保证流程一致性。

### 7) 结构化迁移后 RAG 预检入参类型错误
- 症状：日志出现 `RAG Search Runtime Error ... _EmbedContentParametersPrivate`，提示 `contents` 收到 list/dict 而非 string。
- 根因：纯结构化改造后，某些调用链把 `ContentBlock[]` 直接传给了 `rag_engine.search()` 的 embedding 入参。
- 修复动作：
  - 在 `Supervisor` 提取用户消息时统一 `extract_text_content(...)`。
  - 在 `RAGService.search` 内部增加二次文本归一化与空查询保护，形成双保险。
- 验证结果：同链路复测不再出现该类校验错误，Fusion/Analysis 路由恢复正常。
- 经验复盘：结构化协议迁移时，模型输入边界要明确“渲染为文本”的转换点，不能依赖调用方约定。

### 8) Modeler 在 `gemini-3-flash-preview` 下稳定空输出
- 症状：`Modeling output length=0`, `finish_reason=STOP`, `content_type=str`, `candidate_count=0`，前端多次复测一致复现。
- 根因：
  - 上游模型在特定上下文下返回空字符串（应用侧可观测事实）。
  - 预览模型稳定性与返回形态不稳定，且当次 `additional_kwargs.candidates` 不可见，增加排查难度。
- 修复动作：
  - 新增 `modeler_raw_probe` 最小取证脚本，采集 content/meta/candidate 证据。
  - `Modeler` 增加空输出自愈链路：同模型重试一次，仍空则降级 `gemini-2.5-flash-lite` 再试。
  - 失败阶段统一结构化 `stage_id`（`modeling_failed_empty_output`）。
- 验证结果：空输出场景可自动回退并继续流程；日志可追踪到重试与降级动作。
- 经验复盘：对生成模型要做“非异常空响应”兜底，不能只捕获抛异常分支。

### 9) 沙箱依赖缺失导致代码阶段不可执行（`scipy`）
- 症状：Coder 生成代码依赖 `scipy`，执行环境未预装时直接失败。
- 根因：提示词默认技术栈包含 `scipy`，且沙箱此前没有受控自动安装能力。
- 修复动作：
  - `Coder` 提示词改为优先 `numpy/pandas`，并在 `No module named scipy` 时明确要求无 `scipy` 改写（`rankdata -> pandas.Series.rank`）。
  - 沙箱新增“白名单自动安装并重跑一次”能力（带缓存、超时与安装日志）。
- 验证结果：缺库场景可自动修复，或给出明确不可安装原因。
- 经验复盘：运行时安装能力必须受控（白名单+审计），不能开放任意 pip。

### 10) Coder 动态产物在前端显示异常（显示不全/显示错位）
- 症状：
  - 右侧“可视化结果”仅显示单图或不显示多产物。
  - 部分情况下文件链接被当作图片渲染，导致展示异常。
- 根因：
  - 旧链路仅传 `artifact_url`，无法表达“多图+多文件”。
  - `artifact_url` 曾按“第一个 URL”选取，若第一个是导出文件会被前端误当图片。
  - Windows 下路径归属判断使用字符串前缀匹配，存在分隔符/大小写差异导致 `manifest` 识别失败风险。
- 修复动作：
  - 后端消息总线扩展 `artifact_urls` 与 `artifact_manifest` 字段，兼容保留 `artifact_url`。
  - Coder 节点改为“执行前后文件快照”并生成动态 `manifest`（`kind/path/filename/size/url`）。
  - 路径判断改为 `Path.relative_to`，避免 `startswith` 带来的跨平台误判。
  - 主图 URL 选择改为“优先首个 `kind=image`，再退化到首个文件 URL”。
  - 增加兜底：`manifest` 为空但存在 `backend/static/plots/output.png` 时自动补一条图片项。
  - 前端 store 与 `ArtifactPanel` 增加 `artifactsManifest` 渲染，支持多图与文件链接并存。
- 验证结果：
  - 编译校验通过（`coder.py`、`bus.py`）。
  - 前端类型与 lints 校验通过，能同时消费旧 `artifact_url` 和新 `artifact_manifest`。
- 经验复盘：
  - 资产型输出应优先用“清单协议”而不是单一 URL 字段。
  - 展示层必须区分资源类型（image/export/file），避免由 URL 位置推断类型。

### 11) SQLite 非空约束迁移执行问题与修复（v1）
- 症状：
  - 正式执行 `sqlite_not_null_rebuild_v1.sql` 首次失败：`RAISE() may only be used within a trigger-program`。
  - 迁移成功后发现 `tasks/messages` 活动表缺少业务索引，仅剩主键索引。
- 根因：
  - SQLite 中 `RAISE(ABORT, ...)` 不能在普通 `SELECT` 中使用，仅支持触发器程序。
  - SQLite 索引名全局唯一；迁移保留 `*_backup_v1` 表后，旧索引名仍被备份表占用，`CREATE INDEX IF NOT EXISTS` 在新表上被跳过。
- 修复动作：
  - 将 SQL 前置阻断校验从 `SELECT ... RAISE(ABORT, ...)` 改为 `TEMP TABLE + CHECK` 守卫写法。
  - 索引重建策略改为“先 `DROP INDEX IF EXISTS`，再无条件 `CREATE INDEX`”，确保索引绑定到活动表。
  - 增加一键执行入口与回滚文档：
    - `backend/db/migrations/run_sqlite_not_null_rebuild_v1.py`
    - `backend/db/migrations/sqlite_not_null_rebuild_v1_rollback.md`
- 验证结果：
  - 迁移最终执行成功，`tasks/messages` 目标字段已落地 `NOT NULL`。
  - `PRAGMA foreign_key_check` 结果为空；`PRAGMA integrity_check` 返回 `ok`。
  - 业务索引在活动表恢复：`ix_tasks_created_at`、`ix_tasks_user_id`、`ix_messages_created_at`、`ix_messages_task_id`、`ix_messages_id`。
  - 备份表保留：`tasks_backup_v1`、`messages_backup_v1`，可用于快速回滚。
- 经验复盘：
  - SQLite 方言差异必须先做可执行性验证，避免将通用 SQL 写法直接用于迁移脚本。
  - 保留备份表时要考虑“索引名冲突”这类隐藏副作用，避免 `IF NOT EXISTS` 误判成功。
