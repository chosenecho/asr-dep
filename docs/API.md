# API 参考文档

本文档描述 `qwen3-asr-server`（Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B 长录音转写服务）的全部 HTTP 接口、参数约定、返回结构与错误码。服务启动后亦可直接访问 FastAPI 自动生成的交互式文档：`http://<HOST>:<PORT>/docs`（Swagger UI）与 `/redoc`。

---

## 1. 认证方式

除 `GET /health` 外，所有 `/v1/*` 接口均受 API Key 鉴权保护（通过 `.env` 中的 `API_KEYS` 配置，支持多个 Key 以英文逗号分隔）。

支持以下两种请求头传递方式（二选一）：

```http
Authorization: Bearer <API_KEY>
```

或：

```http
X-API-Key: <API_KEY>
```

- 当 `API_KEYS` 留空时，鉴权自动关闭（仅建议本地开发使用）。
- 鉴权失败返回 `401 Unauthorized`：

```json
{
  "detail": "Invalid or missing API key."
}
```

> **Modal Serverless 边缘代理鉴权说明**：若通过 [`deploy/modal_app.py`](../deploy/modal_app.py) 部署并保持默认 `REQUIRES_PROXY_AUTH = True`，所有请求（包括 `GET /health`）还需在请求头中附带 `Modal-Key` 与 `Modal-Secret`，未通过边缘网关校验的请求会直接返回 `401` 且不会唤醒 GPU 容器。详见 [部署与运维指南 §7.2](DEPLOYMENT.md#72-modal-边缘网关代理鉴权requires_proxy_authtrue)。

---

## 2. 接口总览

| 方法 | 路径 | 认证 | 说明 |
| :--- | :--- | :---: | :--- |
| `GET` | `/health` | 否 | 服务与双卡 GPU Worker 健康检查 |
| `GET` | `/v1/models` | 是 | OpenAI 兼容模型列表 |
| `POST` | `/v1/asr/tasks` | 是 | 提交长录音异步转写任务（推荐用于长音频） |
| `GET` | `/v1/asr/tasks` | 是 | 查询异步任务列表（支持按状态筛选） |
| `GET` | `/v1/asr/tasks/{task_id}` | 是 | 查询异步任务进度与完整转写结果 |
| `GET` | `/v1/asr/tasks/{task_id}/subtitle` | 是 | 导出已完成任务的 SRT / WebVTT 字幕 |
| `POST` | `/v1/asr/tasks/{task_id}/cancel` | 是 | 请求取消并等待后台回收读者、音频与配额 |
| `DELETE` | `/v1/asr/tasks/{task_id}` | 是 | 删除已结束的异步任务记录 |
| `POST` | `/v1/asr/transcribe` | 是 | 短/中音频同步转写接口（支持 JSON / Text / SRT / VTT） |
| `POST` | `/v1/audio/transcriptions` | 是 | OpenAI Whisper 兼容转写接口 |

---

## 3. 系统与模型接口

### 3.1 `GET /health`

检查服务及各 GPU Worker 进程存活状态。所有 Worker 正常运行时返回 `200 OK`，任一 Worker 异常或未就绪时返回 `503 Service Unavailable`。

**响应示例 (`200 OK`)**

```json
{
  "status": "ok",
  "backend": "vllm",
  "model": "qwen3-asr-1.7b",
  "aligner_enabled": true,
  "workers": [
    { "gpu_id": "0", "pid": 1024, "alive": true },
    { "gpu_id": "1", "pid": 1025, "alive": true }
  ],
  "inflight_batches": 0,
  "active_background_jobs": 0,
  "active_jobs": 0,
  "waiting_jobs": 0,
  "decoding_jobs": 0,
  "reserved_temp_bytes": 0,
  "worker_restarts": 0,
  "readiness": true,
  "unreclaimed_jobs": 0
}
```

### 3.2 `GET /v1/models`

返回当前服务暴露的模型标识（由 `SERVED_MODEL_NAME` 控制）。

**响应示例 (`200 OK`)**

```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen3-asr-1.7b",
      "object": "model",
      "owned_by": "qwen3-asr"
    }
  ]
}
```

---

## 4. 长录音异步任务 API（推荐）

适用于会议录音、访谈、课程、播客等长音视频文件，受上传大小、`MAX_ASYNC_AUDIO_SEC`（默认 8 小时）、解码输出大小和作业截止时间限制。任务状态持久化在 SQLite（WAL 模式）中。Modal 模板设置 `ENABLE_ASYNC_TASKS=false`，提交异步任务会在解析上传前返回 `503`，错误码为 `ASYNC_TASKS_DISABLED`。

### 4.1 `POST /v1/asr/tasks`

提交长录音异步转写任务。请求类型为 `multipart/form-data`。

**表单字段**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| :--- | :--- | :---: | :--- | :--- |
| `file` | `file` | 是 | - | 音视频文件（ffmpeg 支持的格式，如 mp3/m4a/wav/flac/ogg/amr/mp4/webm） |
| `language` | `string` | 否 | `null` | 语言名称或 ISO 代码（如 `Chinese` / `zh` / `en`），留空或 `auto` 自动检测 |
| `context` | `string` | 否 | `""` | 领域热词或上下文提示（如人名、专有名词，逗号或空格分隔） |
| `return_timestamps` | `bool` | 否 | `true` | 是否启用 `Qwen3-ForcedAligner-0.6B` 输出时间戳（设为 `false` 时同步跳过后台对齐梅尔特征预计算，速度更快） |
| `include_words` | `bool` | 否 | `true` | 响应中是否包含字/词级 `words` 列表（关闭可减小 JSON 体积） |
| `webhook_url` | `string` | 否 | `null` | 任务完成或失败后接收 JSON 回调的 `http(s)` URL（禁止回环与内网地址） |

**成功响应 (`202 Accepted`)**

```json
{
  "task_id": "8f2a9c4d1e7b4b3aa6e1f2d9c8b7a654",
  "status": "queued",
  "filename": "meeting_2h.m4a",
  "duration_sec": null,
  "total_chunks": 0,
  "completed_chunks": 0,
  "progress": 0.0,
  "language_requested": "zh",
  "return_timestamps": true,
  "include_words": true,
  "created_at": 1763890000.12,
  "updated_at": 1763890000.12,
  "completed_at": null,
  "error": null,
  "result": null
}
```

### 4.2 `GET /v1/asr/tasks`

按创建时间倒序查询任务列表。

**查询参数**

| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `limit` | `int` | `50` | 返回条数上限，范围 `1 ~ 200` |
| `status` | `string` | `null` | 按状态筛选：`queued` / `decoding` / `processing` / `cancelling` / `cancelled` / `completed` / `failed` |

### 4.3 `GET /v1/asr/tasks/{task_id}`

查询单个任务的实时进度与转写结果。

**任务状态流转**

`queued` → `decoding`（`progress=0.02`）→ `processing`（`0.05 ~ 0.99`，按已完成切片实时推进）→ `completed`（`1.0`）或 `failed`。

活动状态可经 `cancelling` → `cancelled`。任务对象新增 `error_code`（机器可读故障码）、`cancel_requested_at`（接受取消的 Unix 时间）和 `deadline_at`（作业截止的 Unix 时间）。取消和完成采用条件更新；接受取消之后，迟到的完成结果不会覆盖取消状态。`cancelling` 表示后台仍在清理读者，完成清理后才释放文件与资源。服务重启会将未结束任务标记为 `failed / SERVER_RESTARTED`，暂不支持续跑。

任务列表仅返回元数据，`result` 为 `null`；需要完整转写内容请查询单个任务。

**已完成响应示例 (`200 OK`)**

```json
{
  "task_id": "8f2a9c4d1e7b4b3aa6e1f2d9c8b7a654",
  "status": "completed",
  "filename": "meeting_2h.m4a",
  "duration_sec": 35.0,
  "total_chunks": 4,
  "completed_chunks": 4,
  "progress": 1.0,
  "language_requested": "zh",
  "return_timestamps": true,
  "include_words": true,
  "created_at": 1763890000.12,
  "updated_at": 1763890004.85,
  "completed_at": 1763890004.85,
  "error": null,
  "result": {
    "model": "qwen3-asr-1.7b",
    "language": "Chinese",
    "duration_sec": 35.0,
    "processing_time_sec": 4.21,
    "rtf": 0.1203,
    "text": "今天开会。讨论双卡部署！",
    "segments": [
      {
        "id": 0,
        "start": 0.0,
        "end": 1.9,
        "text": "今天开会。",
        "words": [
          { "text": "今", "start": 0.0, "end": 0.4 },
          { "text": "天", "start": 0.5, "end": 0.9 },
          { "text": "开", "start": 1.0, "end": 1.4 },
          { "text": "会", "start": 1.5, "end": 1.9 }
        ]
      }
    ],
    "words": [
      { "text": "今", "start": 0.0, "end": 0.4 }
    ]
  }
}
```

### 4.4 `GET /v1/asr/tasks/{task_id}/subtitle`

导出已完成任务的字幕文件。

**查询参数**

| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `format` | `string` | `srt` | 可选 `srt`（`application/x-subrip`）或 `vtt`（`text/vtt`） |

**常见状态码**
- `200 OK`：返回字幕纯文本
- `404 Not Found`：任务不存在
- `409 Conflict`：任务尚未完成

### 4.5 `DELETE /v1/asr/tasks/{task_id}`

删除任务记录。仅允许删除 `completed`、`failed` 或 `cancelled` 状态的任务；运行中及 `cancelling` 状态返回 `409 Conflict`。

### 4.7 `POST /v1/asr/tasks/{task_id}/cancel`

无需请求体，沿用 API Key 鉴权。接受取消返回 `202 Accepted`，响应结构与单任务查询相同，状态通常为 `cancelling`；清理完成后查询得到 `cancelled`。对 `cancelling` 或 `cancelled` 重复、并发请求均幂等返回 `202`；任务不存在返回 `404`；已完成或失败返回 `409`。

```bash
curl -X POST "http://localhost:8000/v1/asr/tasks/TASK_ID/cancel" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**响应示例 (`200 OK`)**

```json
{
  "deleted": true,
  "task_id": "8f2a9c4d1e7b4b3aa6e1f2d9c8b7a654"
}
```

---

## 5. 同步转写 API

### 5.1 `POST /v1/asr/transcribe`

适合短/中音频（默认时长上限 `MAX_SYNC_AUDIO_SEC=600` 秒，超出返回 `400 Bad Request` 并提示改用 `/v1/asr/tasks`）。请求类型为 `multipart/form-data`。

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| :--- | :--- | :---: | :--- | :--- |
| `file` | `file` | 是 | - | 音视频文件 |
| `language` | `string` | 否 | `null` | 语言代码或名称，留空自动检测 |
| `context` | `string` | 否 | `""` | 热词/上下文提示 |
| `return_timestamps` | `bool` | 否 | `true` | 是否返回时间戳 |
| `include_words` | `bool` | 否 | `true` | 是否返回字/词级时间戳 |
| `response_format` | `string` | 否 | `json` | `json` / `text` / `srt` / `vtt` |

当 `response_format=json` 时，响应体与异步任务中的 `result` 对象结构一致（`TranscriptionResultSchema`）。

---

## 6. OpenAI Whisper 兼容 API

### 6.1 `POST /v1/audio/transcriptions`

兼容 OpenAI Python/Node SDK 的 `audio.transcriptions.create` 接口。同样受 `MAX_SYNC_AUDIO_SEC` 同步时长上限约束。

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| :--- | :--- | :---: | :--- | :--- |
| `file` | `file` | 是 | - | 音视频文件 |
| `model` | `string` | 否 | `qwen3-asr-1.7b` | 兼容字段，实际路由到已加载模型 |
| `language` | `string` | 否 | `null` | 语言代码（如 `zh` / `en`） |
| `prompt` | `string` | 否 | `""` | 映射到内部 `context`（热词提示） |
| `response_format` | `string` | 否 | `json` | `json` / `text` / `srt` / `vtt` / `verbose_json` |
| `timestamp_granularities[]` | `string[]` | 否 | `null` | 可选 `segment`、`word`（可多选） |

**`response_format=verbose_json` 响应示例**

```json
{
  "task": "transcribe",
  "language": "English",
  "duration": 3.0,
  "text": "Hello world. Second sentence.",
  "segments": [
    { "id": 0, "start": 0.0, "end": 1.2, "text": "Hello world." },
    { "id": 1, "start": 1.5, "end": 2.9, "text": "Second sentence." }
  ],
  "words": [
    { "word": "Hello", "start": 0.0, "end": 0.5 },
    { "word": "world", "start": 0.5, "end": 1.2 }
  ]
}
```

---

## 7. Webhook 回调说明

若在 `POST /v1/asr/tasks` 中传入了 `webhook_url`：

1. **SSRF 校验**：提交时校验协议必须为 `http` 或 `https`，且拒绝 `localhost`、回环地址、内网私有网段（如 `10.0.0.0/8`、`172.16.0.0/12`、`192.168.0.0/16`）、链路本地与保留地址；非法地址直接返回 `400 Bad Request`。
2. **回调触发**：任务进入 `completed`、`failed` 或 `cancelled` 状态且资源清理成功后，服务端以 `POST` 方式发送 JSON 请求到 `webhook_url`，请求体与 `GET /v1/asr/tasks/{task_id}` 的返回体完全一致。
3. **超时与容错**：超时时间由 `WEBHOOK_TIMEOUT_SEC`（默认 15s）控制；回调失败仅记录警告日志，不影响任务状态落库。

---

## 8. 语言支持与时间戳降级规则

### 8.1 常用 `language` 参数映射

服务会自动把常见 ISO-639-1 代码转换为 Qwen3-ASR 规范名称，例如：

- `zh` / `zh-cn` / `zh-tw` / `cmn` → `Chinese`
- `en` → `English`
- `yue` → `Cantonese`
- `ja` → `Japanese`、`ko` → `Korean`、`fr` → `French`、`de` → `German`、`es` → `Spanish`、`pt` → `Portuguese`、`it` → `Italian`、`ru` → `Russian` 等（共内置 30+ 常见语种映射；传 `auto` 或留空为自动检测）。

### 8.2 ForcedAligner 支持范围与降级

`Qwen3-ForcedAligner-0.6B` 官方支持以下 **11 种语言**的字/词级时间戳对齐：
`Chinese`、`English`、`Cantonese`、`French`、`German`、`Italian`、`Japanese`、`Korean`、`Portuguese`、`Russian`、`Spanish`。

当某一音频切片出现以下情况时，系统会**自动降级为切片级时间范围**，确保整段长录音任务不会中断：
- 识别结果属于上述 11 种之外的语种；
- 切片为静音或仅含标点符号；
- 单片对齐过程发生异常。

---

## 9. 常见错误码

| HTTP 状态码 | 典型原因 |
| :---: | :--- |
| `400 Bad Request` | 音频损坏/无法解码、同步接口超出 `MAX_SYNC_AUDIO_SEC`、`webhook_url` 触发 SSRF 防护 |
| `401 Unauthorized` | 缺少 API Key 或 API Key 不匹配 |
| `404 Not Found` | 指定的 `task_id` 不存在（或已超过 `TASK_RETENTION_HOURS` 被清理） |
| `409 Conflict` | 导出未完成任务的字幕，或尝试删除仍在运行中的任务 |
| `413 Request Entity Too Large` | 上传文件超过 `MAX_UPLOAD_SIZE_MB`（或 Nginx `client_max_body_size`） |
| `422 Unprocessable Entity` | 必填表单字段缺失或参数类型不合法 |
| `503 Service Unavailable` | 准入、上传并发或存储预算已满（含 `Retry-After`）；Worker 不可用；异步能力关闭；健康检查未就绪 |
| `504 Gateway Timeout` | 同步作业超过总截止时间 |

异步故障在任务的 `error_code` 中报告：`JOB_TIMEOUT`、`WORKER_EXITED`、`WORKER_STARTUP_TIMEOUT`、`BATCH_TIMEOUT`、`BATCH_QUEUE_TIMEOUT`、`WORKER_UNAVAILABLE`、`INFERENCE_FAILED`、`SERVER_RESTARTED`、`CLEANUP_FAILED` 等。`CLEANUP_FAILED` 时保留可能仍被读取的音频和配额，停止新准入，健康检查返回 `503`，等待完整进程回收后再释放。

三个上传入口共用限制。ASGI 层先检查声明长度，再累计实际请求体字节；为 multipart 表单预留 64 KiB 开销，文件本身仍由 `MAX_UPLOAD_SIZE_MB` 校验。框架 multipart 暂存空间需另行规划，详见部署说明。
