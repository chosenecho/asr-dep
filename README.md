# Qwen3-ASR-1.7B 双路 RTX 4090 长录音转写 API 服务

基于 **FastAPI + `qwen-asr[vllm]` + `Qwen3-ForcedAligner-0.6B`** 的生产级语音识别服务，专为 **双路 RTX 4090 (2 × 24GB)**、**长录音离线转写** 与 **句级/字词级时间戳输出** 优化。

## 文档导航

- [API 参考文档](docs/API.md)：全部接口说明、请求/响应示例、OpenAI 兼容层、Webhook 回调与错误码
- [部署与运维指南](docs/DEPLOYMENT.md)：软硬件依赖、完整环境变量表、Docker/Nginx/HTTPS、systemd 裸机部署、显存调优与排障
- [系统架构设计](docs/ARCHITECTURE.md)：双卡数据并行原理、长录音零 OOM 流式链路、时间戳对齐机制与项目目录结构

## 核心架构特性

- **双卡 4090 真并行（数据并行，非 TP）**：每张 4090 启动 1 个独立 `spawn` 进程（`CUDA_VISIBLE_DEVICES=0` 与 `CUDA_VISIBLE_DEVICES=1`），各自常驻 `Qwen3-ASR-1.7B`（vLLM）+ `Qwen3-ForcedAligner-0.6B`。
  - **单个超长录音**自动按低能量静音边界切分为 ≤125s 片段（满足 ForcedAligner 180s 上限），拆成小批次**同时派发给两张 4090 并行处理**，实现单任务近 2× 加速；
  - **多路并发请求**在双卡间自动负载均衡，避免短音频被长任务阻塞。
- **长录音全链路防 OOM**：流式落盘上传 → `ffmpeg` 统一转码 16kHz 单声道（支持 mp3/m4a/mp4/wav/flac/ogg/amr/webm 等）→ 磁盘 Seek 检索低能量切分点 → 按片读取推理 → 自动清理临时文件。
- **双粒度时间戳 + 字幕直出**：
  - `words`：来自 `Qwen3-ForcedAligner-0.6B` 的字/词级时间戳；
  - `segments`：自动将带标点的 ASR 文本与词级时间戳对齐，重建可读的句/子句级时间戳；
  - 内置 **SRT / WebVTT** 字幕导出接口；
  - 单片静音或遇到 ForcedAligner 不支持的语种时**自动降级为切片级时间戳**，绝不拖垮整段长录音任务。
- **对外 API 完备**：
  - 长录音异步任务 API（`POST /v1/asr/tasks` + 进度轮询 + Webhook 回调 + SQLite 持久化）；
  - 短/中音频同步 API（`POST /v1/asr/transcribe`）；
  - OpenAI Whisper 兼容接口（`POST /v1/audio/transcriptions`，支持 `verbose_json` / `srt` / `vtt`）；
  - API Key 鉴权（`Authorization: Bearer` 或 `X-API-Key`）、Webhook SSRF 防护、Nginx 大文件反向代理配置。

## 显存规划（单张 RTX 4090 24GB）

| 组件 | 预计显存 | 说明 |
| :--- | :--- | :--- |
| `Qwen3-ASR-1.7B` (vLLM BF16 + KV Cache) | ~17.3 GB | 默认 `GPU_MEMORY_UTILIZATION=0.72` |
| `Qwen3-ForcedAligner-0.6B` (BF16 + 推理激活) | ~2.5–3.5 GB | 与 ASR 同卡，支持批对齐 |
| 系统与安全余量 | ~3.2–4.2 GB | 防止长音频批次峰值 OOM |

> 若希望进一步提速，可在开启 `USE_FLASH_ATTENTION=true`（官方镜像 `qwenllm/qwen3-asr:latest` 已内置 FlashAttention 2）的前提下微调 `MAX_INFERENCE_BATCH_SIZE=12~16`。

---

## 快速部署（Docker Compose，推荐）

### 1. 准备模型权重（国内推荐 ModelScope）
```bash
pip install -U modelscope
python scripts/download_models.py --source modelscope --output-dir ./models
```
下载完成后目录如下：
```text
models/
├── Qwen3-ASR-1.7B/
└── Qwen3-ForcedAligner-0.6B/
```

### 2. 配置环境变量
```bash
cp .env.example .env
# 务必修改 .env 中的 API_KEYS（对外鉴权密钥）
# 若官方镜像已带 flash-attn，可将 USE_FLASH_ATTENTION 改为 true
```

### 3. 启动双卡服务
```bash
# 仅启动 API（监听 8000 端口）
docker compose up -d --build

# 如需同时启动前置 Nginx 反向代理（监听 80 端口）
docker compose --profile proxy up -d --build

# 查看启动与双卡模型加载日志
docker compose logs -f qwen3-asr-api
```

### 4. 检查双卡健康状态
```bash
curl http://127.0.0.1:8000/health
```
正常返回中 `workers` 应显示 `gpu_id: "0"` 和 `gpu_id: "1"` 均为 `alive: true`。

---

## 宿主机裸机部署（不用 Docker）

```bash
conda create -n qwen3-asr python=3.12 -y
conda activate qwen3-asr

# 1. 安装 qwen-asr (vLLM 后端) 与服务端依赖
pip install -U "qwen-asr[vllm]"
pip install -r requirements.txt
# 确保系统已安装 ffmpeg：sudo apt install -y ffmpeg libsndfile1

# 2. 配置 .env（把 ASR_MODEL_PATH / ALIGNER_MODEL_PATH 指向 ./models/...）
cp .env.example .env

# 3. 启动（注意 --workers 必须为 1，由内部 DualGPUDispatcher 管理双卡进程）
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

---

## API 调用示例

### 1. 长录音异步转写（推荐，支持几小时录音 + 进度查询）
```bash
# 提交任务（支持 mp3 / m4a / wav / mp4 等任意格式）
curl -X POST http://<服务器IP>:8000/v1/asr/tasks \
  -H "Authorization: Bearer sk-asr-change-me-in-production" \
  -F "file=@meeting_2h.m4a" \
  -F "return_timestamps=true" \
  -F "include_words=true" \
  -F "context=Qwen3,vLLM,季度复盘"

# 轮询进度与结果（含 segments 句级时间戳与 words 字词级时间戳）
curl http://<服务器IP>:8000/v1/asr/tasks/<task_id> \
  -H "Authorization: Bearer sk-asr-change-me-in-production"

# 直接导出 SRT 或 VTT 字幕
curl "http://<服务器IP>:8000/v1/asr/tasks/<task_id>/subtitle?format=srt" \
  -H "Authorization: Bearer sk-asr-change-me-in-production" -o meeting.srt
```
也可直接使用仓库自带客户端脚本：
```bash
python scripts/test_client.py meeting_2h.m4a \
  --base-url http://<服务器IP>:8000 \
  --api-key sk-asr-change-me-in-production \
  --save-srt meeting.srt
```

### 2. 短/中录音同步转写（默认上限 600 秒）
```bash
curl -X POST http://<服务器IP>:8000/v1/asr/transcribe \
  -H "Authorization: Bearer sk-asr-change-me-in-production" \
  -F "file=@sample.wav" \
  -F "return_timestamps=true" \
  -F "response_format=json"
```

### 3. OpenAI SDK 直接兼容调用
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://<服务器IP>:8000/v1",
    api_key="sk-asr-change-me-in-production",
)

with open("sample.wav", "rb") as f:
    res = client.audio.transcriptions.create(
        model="qwen3-asr-1.7b",
        file=f,
        response_format="verbose_json",
        timestamp_granularities=["segment", "word"],
    )

print(res.text)
print(res.segments)
```

---

## 本地无 GPU 调试与自动化测试

内置 `ASR_BACKEND=mock` 模式，可在开发机无显卡环境下完整跑通双 Worker 进程派发、切片、时间戳聚合与全部 API 测试：

```bash
pip install -r requirements-dev.txt
pytest -v
```
