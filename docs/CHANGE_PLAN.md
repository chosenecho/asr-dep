# 项目修改方案与新增清单

状态：第一批已实现。新增 `app/admission.py`、`app/audio_pipeline.py`、`app/jobs.py`、`app/worker_protocol.py` 四个运行模块，以及准入、音频准备、任务生命周期、Worker 故障、SQLite、上传限制六类专项测试。新增取消接口、三项任务表字段和十二项配置；继续使用现有运行依赖，未新增数据库表。

本文保留原方案作为设计依据；第一批实现细节以 `docs/API.md`、`docs/ARCHITECTURE.md`、`docs/DEPLOYMENT.md` 与源码为准。第二批和条件扩展仍待实施。验证覆盖 CPU/mock、真实进程故障与 ffmpeg；真实 GPU、Linux 进程组和 Modal 云部署需在目标环境验收。

本地验收：`python -m pytest -q` → **80 passed in 10.78s**；`python -m compileall -q app deploy` 与 `git diff --check` 均通过。原有 17 项测试保留，新增 63 项用例。

## 1. 目标与实施范围

保留 FastAPI 单主进程、每卡独立 Worker、单机 SQLite、按路径派发切片、公平轮转和动态合批。围绕现有设计补齐资源上限、故障终结、取消清理和部署契约。

建议按三批实施：

1. 第一批：所有入口统一准入、转码限制、批次超时、Worker 故障处理、任务取消、数据库非阻塞访问。
2. 第二批：固定推理环境、封装优化对齐、补真实模型回归和性能基准。
3. 条件扩展：多租户隔离、Webhook 持久重试、Modal 可靠异步。仅在对应业务需要时加入。

第一批继续使用现有运行依赖，增加内部 Python 模块和测试。分布式队列、独立调度服务、外部数据库均留到存在明确需求时选型。

## 2. 第一批新增文件

### 2.1 app/admission.py：统一资源准入

新增 AdmissionController，集中管理：

- 活跃任务许可：同步、OpenAI、异步共用 MAX_CONCURRENT_JOBS。
- 等待任务名额：等待队列达到 MAX_QUEUED_JOBS 后立即返回服务繁忙。
- 临时存储预留：上传源文件和解码 WAV 分别预留预算，释放时统一归还。
- 服务关闭状态：停止接收新任务，再处理在途任务。

对调用方提供获取/释放任务票据的 Interface；票据由协调模块持有，HTTP 连接结束并不自动释放异步作业占用的配额。第一批限制为单主进程有效，与现有部署模型一致。

multipart 注意事项：FastAPI 的 UploadFile 在进入路由前可能已经完成框架层落盘。任务票据只约束应用接管后的资源；入口另在 app/main.py 增加 ASGI 层请求体字节计数和上传并发控制，对所有三个上传路由生效，并同步配置 Nginx。Content-Length 仅作提前检查，流式计数负责实际上限。Starlette 临时上传目录与 TEMP_DIR 分别计入部署容量预算。

### 2.2 app/audio_pipeline.py：统一音频准备与清理

新增 AudioPreparation，封装“保存上传 → 取得解码许可 → 转码 → WAV 信息验证 → 文件释放”。所有三个上传入口通过同一 Interface 调用。

新增 PreparedAudio 数据结构，包含请求标识、源路径、WAV 路径、时长、实际字节数和配额票据。HTTP 层只处理参数和结果格式。

低层能力继续位于 app/audio.py：

- ffmpeg 增加执行期限。
- 取消或超时时终止子进程，等待退出后清理文件；超过退出宽限期再强制结束。
- 使用最大时长加检测余量限制转码输出；识别超限后返回音频时长错误，避免把截断录音当成成功结果。
- 根据 PCM 输出格式预留空间，同时监控输出增长；完成后核对文件大小和音频时长。
- soundfile 回退路径显式标记为开发用途，生产模式要求 ffmpeg 可用，防止长录音进入整段读内存回退。

文件归 AudioPreparation 管理；释放需等待推理调度模块确认相关读者已结束，或读取进程已被回收。

### 2.3 app/jobs.py：任务生命周期协调

新增 JobCoordinator，管理同步和异步作业的准入、截止时间、状态、取消及清理。异步任务写入 TaskStore；同步请求使用内存状态。

将现有 DualGPUDispatcher 中 enqueue_async_job、_run_async_job 和任务进度持久化迁入此模块。DualGPUDispatcher 保留公平调度、批次收集、结果聚合与 Worker 管理。

新增取消状态流：

```text
queued / decoding / processing → cancelling → cancelled
queued / decoding / processing → failed
processing → completed
```

终态写入采用条件更新，保证完成、失败和取消竞争时只有一次成功终结。失败或取消时先移除尚未派发的批次，再等待在途读者结束，最后释放文件和配额。

第一批重启行为仍为“未完成任务标记失败”，错误码明确为 SERVER_RESTARTED。重启后根据受管目录内的请求标识回收孤儿文件，清理只作用于本应用生成的路径。断点续跑属于后续独立能力。

### 2.4 app/worker_protocol.py：类型化批次协议

新增内部数据结构，替代各处拼接的任意字典：

- BatchRequest：batch_id、job_id、attempt、音频路径、切片、推理选项。
- BatchStarted：batch_id、attempt、gpu_id、worker_generation。
- BatchResult：batch_id、attempt、worker_generation、结果或错误。
- BatchSkipped：因任务取消而跳过的批次确认。

第一批 attempt 固定为 1，自动推理重试默认关闭。generation 用于识别重建前的旧 Worker；旧结果和重复结果只记录日志，不重复修改进度或在途计数。

Worker 预取线程从队列取到每个批次时立即发 BatchStarted，避免预取期间退出导致批次归属不明。该消息证明批次已被领取，并不代表 GPU 已开始计算。

## 3. 第一批修改现有文件

### app/main.py

- 创建并注入 AdmissionController、AudioPreparation、JobCoordinator。
- 三个上传入口统一经过同一资源控制链路。
- 增加 ASGI 上传并发和请求体容量控制，覆盖框架解析前的请求读取。
- 增加取消接口和统一错误映射。
- 原同步、OpenAI 和异步结果格式保持兼容，新增字段采用可选字段。
- lifespan 关闭时先停止准入，再取消/结束作业，随后回收 Worker 和资源。

### app/dispatcher.py

- 增加 Worker 监控协程、批次领取与完成期限检查。
- Worker 退出后终结它已领取的批次；无法确认归属的批次由派发期限兜底。
- 卡死超时后停止对该 Worker 派发并回收进程；将受影响任务标记失败，再按有上限的退避策略重建 Worker。
- 预取线程退出需要通知主循环，避免主循环永久等待 prefetch_queue。
- 只有进程退出和必要子进程已回收后，才确认其持有的文件可释放。
- 删除绕过公平调度的未使用私有提交路径 _submit_chunk_batch。
- 新增取消任务处理：移除未派发项，在途项用共享取消标识由 Worker 在读取和推理前检查；正在执行的 GPU 调用等待结束或期限触发回收。
- 所有完成、取消和迟到结果走同一清算逻辑，在途计数只减少一次。

### app/worker.py

- 读写类型化协议，携带 job_id、attempt 和 Worker generation。
- 预取与主推理循环均检查取消标识，并发回跳过/完成确认。
- 单个任务失败的隔离重试继续保留；自动跨 Worker 重派留到后续。
- 子进程回收要覆盖 vLLM 可能创建的后代进程，Linux GPU 环境单独验收。

### app/store.py

- 提供异步调用 Interface，内部使用一个受控的数据库线程顺序执行现有 SQLite 操作；连接的创建、使用和关闭都在该线程内。
- 将进度回调改为可等待的协程，并合并高频进度写入。
- 任务列表只查询元数据列，按 task_id 单独读取完整结果。
- 通过 PRAGMA user_version 管理顺序迁移，启动时在接收请求前完成迁移。
- 终态采用条件 UPDATE，新增错误码、取消和截止时间字段。
- 数据保留和孤儿清理由周期任务执行，空闲时也能回收。

### app/schemas.py、app/config.py、.env.example

增加取消状态、错误码、健康统计和下文配置项；校验配置之间的关系，包括切片时长与对齐上限、派发片数与推理批大小、任务总期限与批次期限。

### 部署与文档

- docker-compose.yml：API 宿主端口默认绑定 127.0.0.1；Nginx 通过容器网络连接 API。GPU 设备预留与 GPU_IDS 配置统一说明和验证。
- deploy/modal_app.py：第一批设置 ENABLE_ASYNC_TASKS=false，异步任务路由返回明确的部署能力错误；同步和 OpenAI 接口继续提供。
- docs/API.md：补充取消、过载、超时和部署能力响应。
- docs/DEPLOYMENT.md：补充临时存储预算、上传暂存目录、优雅退出和 Modal 同步契约。
- docs/ARCHITECTURE.md：更新资源所有权、批次协议和失败路径。

## 4. 第一批新增配置

以下名称与数值为方案建议，数值需结合部署容量和真实 GPU 时延调优；现有 MAX_CONCURRENT_JOBS、MAX_CONCURRENT_DECODES 等继续使用。

| 新配置 | 建议初值 | 用途 |
| --- | --- | --- |
| MAX_QUEUED_JOBS | 32 | 最多允许等待执行的作业数 |
| MAX_CONCURRENT_UPLOADS | 4 | 解析上传体期间的并发上限 |
| MAX_TEMP_STORAGE_MB | 8192 | 应用临时文件总预算；应小于实际可用空间 |
| MAX_DECODED_AUDIO_MB | 1024 | 每个任务的解码 WAV 上限 |
| MAX_ASYNC_AUDIO_SEC | 28800 | 异步音频最长 8 小时，资源上限取更严格值 |
| DECODE_TIMEOUT_SEC | 600 | 单次转码期限 |
| BATCH_TIMEOUT_SEC | 300 | 批次领取后到完成的最长时间，包含预取等待 |
| BATCH_QUEUE_TIMEOUT_SEC | 600 | 批次派发后等待领取的期限 |
| JOB_TIMEOUT_SEC | 7200 | 作业排队、解码和推理的总期限 |
| WORKER_RESTART_LIMIT | 3 | 单次服务运行中，每卡自动重建次数上限 |
| SHUTDOWN_GRACE_SEC | 30 | 在途作业退出宽限期 |
| ENABLE_ASYNC_TASKS | true；Modal 为 false | 声明部署是否支持异步任务 |

上传总字节上限复用 MAX_UPLOAD_SIZE_MB，并给 multipart 边界与表单元数据设置单独的小额开销上限。过载响应包含 Retry-After；全局资源不足用 503，未来按租户配额限制时用 429。

## 5. 第一批新增数据库字段与接口

### tasks 表增量字段

| 字段 | 类型 | 用途 |
| --- | --- | --- |
| error_code | TEXT，可空 | 机器可读原因，例如 WORKER_EXITED、BATCH_TIMEOUT、SERVER_RESTARTED |
| cancel_requested_at | REAL，可空 | 取消请求的接收时间 |
| deadline_at | REAL，可空 | 作业总期限的墙上时钟时间 |

运行中的超时判断使用单调时钟；deadline_at 用于查询展示和重启审计。旧记录新增字段为 NULL。状态列继续使用 TEXT，响应模型增加 cancelling、cancelled。

第一批新增表数量为 0；运行期批次状态保留在内存中，与“重启后未完成任务失败”的契约一致。

### 新增 HTTP 接口

`POST /v1/asr/tasks/{task_id}/cancel`

- queued 任务撤出等待队列并进入取消流程。
- decoding 任务终止转码进程后取消。
- processing 任务停止继续派发，等待在途批次确认结束或被回收。
- 接收取消返回 202；重复请求返回当前状态；已经 completed/failed 的任务返回 409；任务不存在返回 404。
- 原 DELETE 接口仍负责删除终态记录，取消单独表达为上述接口。

`GET /health` 增加可选统计：waiting_jobs、active_jobs、decoding_jobs、reserved_temp_bytes、worker_restarts。进程存活、模型就绪和批次进展分别报告，避免用 alive 替代可用性。

## 6. 第一批新增测试

| 新文件 | 核心验收 |
| --- | --- |
| tests/test_admission.py | 容量上限、等待/活动票据、存储预留与所有权释放 |
| tests/test_audio_pipeline.py | 解码并发、真实 ffmpeg 时长/输出限制、超时与重复取消回收 |
| tests/test_upload_limits.py | 三入口请求体超限、解析并发、断连/取消 multipart 清理及异步关闭能力 |
| tests/test_job_lifecycle.py | 状态转换、并发重复取消、提交事务取消、上传阶段退出、回收/删除失败保留资源 |
| tests/test_dispatcher_failures.py | Worker 领取后退出、预取线程异常、批次超时、迟到/重复结果、有限重建、计数一致性 |
| tests/test_store.py | 旧库迁移、条件终态更新、保留清理、数据库等待不阻塞事件循环 |

已有 API 集成测试增加取消接口鉴权、终态冲突、配额清零和列表元数据契约。过载、统一音频准备及 Modal 禁用异步能力由对应模块测试覆盖。故障测试使用可控的 Mock Worker 和同步事件触发故障，避免靠固定 sleep 判断并发结果。

验收要求：现有 17 项测试继续通过；每个故障场景在规定期限内返回终态；作业结束后无遗留配额、Future、受管临时文件和解码进程。使用独立临时目录与显式配置隔离本地 .env。

## 7. 第二批新增内容：推理环境与回归

新增 4 个文件：

1. app/aligner_adapter.py：移动官方对齐与优化对齐实现，统一返回词级结果和降级状态；Worker 只调用 align_batch。
2. constraints-gpu.txt：记录经过真实 GPU 验证的 torch、vLLM、qwen-asr、transformers 等版本组合，Docker 与 Modal 共用。具体版本在 GPU 验证后填写。
3. tests/test_aligner_gpu.py：显式 GPU 标记，比较官方与优化路径的词数、时间戳误差、混合语言和不同音频长度；普通 CPU 测试默认跳过。
4. scripts/benchmark.py：记录上传、排队、解码、推理、对齐及端到端时间，输出 JSON；分别报告单任务延迟、聚合吞吐、P50/P95 和显存峰值。

修改 Dockerfile 使用验证过的镜像 digest；Modal 读取统一约束，避免两处维护不同推理依赖。模型来源和 revision 同时记录到基准结果。

新增配置 ALIGNER_MODE=auto|official|optimized：auto 在兼容检查通过时使用优化路径，否则使用官方路径并报告原因。降级以结构化字段显示，调用方可区分词级、切片级时间戳。

## 8. 按场景选做的新增内容

### 8.1 多客户接入：任务归属

- app/auth.py 返回 Principal，不增加独立认证服务。
- 新增身份映射配置：身份标识对应一组 Key，支持轮换时保持相同身份；Key 由 Secret/环境注入，数据库只保存身份标识。
- tasks 新增 owner_id 和查询索引，存储 Interface 强制要求调用身份。
- 新增 tests/test_task_ownership.py，覆盖列表、详情、字幕、取消和删除的跨身份隔离。
- 旧记录迁移到 legacy 身份，默认仅管理员可见；原有共享 Key 行为保留在显式 shared 模式。

### 8.2 业务要求通知可靠送达：Webhook outbox

- 新增 app/webhooks.py，统一连接时目标地址检查、投递和退避重试。
- 新增 webhook_deliveries 表：delivery_id、task_id、status、attempts、next_attempt_at、last_error；完成任务与插入投递记录在同一事务内提交。
- 新增 WEBHOOK_MAX_ATTEMPTS、WEBHOOK_BACKOFF_SEC、WEBHOOK_ENABLED 配置。
- 新增 tests/test_webhooks.py，验证 DNS 地址变化、重复投递、超时重试及重启恢复。
- 交付语义为至少一次；回调携带稳定 delivery_id，由接收方去重。

### 8.3 Modal 需要可靠异步：容器外任务生命周期

这是独立部署扩展：新增 Modal 平台任务入口、持久任务元数据与音频/结果对象存储适配模块，任务领取与状态更新具备幂等性。平台任务持有执行生命周期，API 返回 202 后容器回收仍能查询持久结果。

外部存储选型、账号资源、费用和恢复契约需在该阶段明确后形成单独方案。第一批通过关闭 Modal 异步入口消除错误承诺；仅给 /tmp 数据库换成 Volume 不能完成可靠异步能力。

## 9. 数量汇总与交付顺序

第一批实际新增：4 个运行模块、6 个专项测试文件、1 个取消接口、3 个数据库字段、12 个配置项，新增数据库表为 0。实现也更新了既有 API 集成测试、配置模板、部署模板与使用文档。

第二批拟新增：1 个运行模块、1 个版本约束文件、1 个 GPU 回归测试文件、1 个基准脚本、1 个对齐模式配置。

推荐实施顺序：

1. 资源准入与统一音频准备，打通三个入口。
2. 类型化批次协议、Worker 监控、批次期限和计数清算。
3. JobCoordinator、取消接口、文件所有权和优雅退出。
4. 数据库迁移与异步访问、部署能力开关和文档。
5. 固定推理环境、封装对齐、执行真实 GPU 回归和基准。

每一步单独通过相应测试后进入下一步。迁移前备份 SQLite 数据；回退到旧版前恢复备份，并保留新库用于审计，避免旧版误读新增状态。
