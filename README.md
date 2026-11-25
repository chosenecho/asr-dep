# qwen3-asr-server

基于 **FastAPI + `qwen-asr[vllm]` + `Qwen3-ForcedAligner-0.6B` + Silero VAD (ONNX)** 的高并发长录音生产级语音识别 API 服务。

面向 **单卡 / 多卡 GPU（RTX 4080 SUPER 32GB / 双路 RTX 4090 24GB 等）**、**多并发小时级长录音离线转写** 与 **字词级/句级时间戳及 SRT/VTT 字幕直出** 深度优化；在单张 RTX 4080 SUPER 32GB 上处理高密度真实中文/英文多并发长录音（开启全量 ForcedAligner 词级对齐）实测稳态聚合吞吐约 **`120×–160×` 实时**（视语速密度、语种及是否开启对齐而定，1 小时真实录音约需 22–30 秒）。

## 文档导航

- [API 参考文档](docs/API.md)：全部接口说明、请求/响应示例、OpenAI 兼容层、Webhook 回调与错误码
- [部署与运维指南](docs/DEPLOYMENT.md)：软硬件依赖、完整环境变量表、Docker/Nginx/HTTPS、systemd 裸机部署、显存调优与排障
- [系统架构设计](docs/ARCHITECTURE.md)：多卡数据并行原理、长录音零 OOM 流式链路、时间戳对齐机制与项目目录结构

## 核心架构特性

- **多卡真并行（每卡独立 Worker 数据并行）**：每张 GPU 启动 1 个独立 `spawn` 进程（支持单卡 `GPU_IDS=0` 或多卡 `GPU_IDS=0,1`），各自常驻 `Qwen3-ASR-1.7B`（vLLM）+ `Qwen3-ForcedAligner-0.6B`。
- **响度自适应 + 无 GIL 批量 ONNX Silero VAD 无损切分**：
  - 仅对候选切分点（默认 120s ±15s）做临时响度归一化（自适应极小声/极大声录音，不改动原始送入 ASR 的波形）并用纯 NumPy 批量 ONNX Runtime C++ 推理定位真实停顿中点；
  - 100% 保留原始音频帧（零丢字、零重叠、满足 ForcedAligner 180s 上限），多并发零 Python GIL 争抢。
- **高并发长录音深度压榨（公平轮转 + 跨任务合批 + 后台预取）**：
  - **公平轮转调度器**：多条长录音按小切片粒度（`CHUNKS_PER_DISPATCH`）交错派发，消除队头阻塞，并限制并发 `ffmpeg` 转码数（`MAX_CONCURRENT_DECODES`）；
  - **Worker 跨任务动态合批**：后台预取线程自动将多个并发任务的切片与零散尾批合并为最高 `MAX_INFERENCE_BATCH_SIZE`（如 `24`）的大 Batch 一次送入 vLLM（原生保留各自 `language` 与 `context`）；
  - **ForcedAligner 显存解耦 + 预取梅尔频谱**：对齐子批大小（`ALIGNER_BATCH_SIZE`）与 vLLM 批大小解耦防止长音频峰值 OOM；预取线程在 GPU 推理上一批时并行提前计算 Whisper 128 维梅尔特征，对齐阶段零等待复用；
  - **内存盘临时音频区**：支持 `TEMP_DIR=/dev/shm/qwen3_asr_tmp` 消除临时 WAV 磁盘 I/O 延迟，任务完成后自动清理。
- **双粒度时间戳 + 字幕直出**：
  - `words`：来自 `Qwen3-ForcedAligner-0.6B` 的字/词级时间戳；
  - `segments`：自动将带标点的 ASR 文本与词级时间戳对齐，重建可读的句/子句级时间戳；
  - 内置 **SRT / WebVTT** 字幕导出接口；单片静音或不支持语种时自动降级，绝不拖垮整段长录音任务。
- **对外 API 完备**：
  - 长录音异步任务 API（`POST /v1/asr/tasks` + 进度轮询 + Webhook 回调 + SQLite WAL 持久化）；
  - 短/中音频同步 API（`POST /v1/asr/transcribe`，支持直接返回 `json` / `text` / `srt` / `vtt`）；
  - OpenAI Whisper 兼容接口（`POST /v1/audio/transcriptions`，支持 `verbose_json` / `srt` / `vtt`）；
  - API Key 鉴权（`Authorization: Bearer` 或 `X-API-Key`）、Webhook SSRF 防护、Nginx 大文件反向代理配置。

## 实际性能参考（单卡 RTX 4080 SUPER 32GB，全量 ForcedAligner 词/句对齐）

参考配置：`GPU_MEMORY_UTILIZATION=0.58`、`MAX_INFERENCE_BATCH_SIZE=24`、`ALIGNER_BATCH_SIZE=6`、`CHUNKS_PER_DISPATCH=6`、`CHUNK_STRATEGY=vad`、`TEMP_DIR=/dev/shm/qwen3_asr_tmp`

| 场景说明 | 典型聚合吞吐（带全量对齐） | GPU 负载与显存 |
| :--- | :---: | :--- |
| **真实高密度中文/多语种多并发长录音**（如 ~300 字/分钟播客/视频/会议，4–8 并发） | **约 `120×–160×` 实时**（1 小时 ≈ 22–30s） | 算力/显存带宽可达 `100%`，显存峰值 `~29.5–30.2 GB / 32.8 GB` |
| **稀疏/低语速音频或关闭时间戳（`return_timestamps=false`）** | 可进一步提升至 **`180×–250×+` 实时** | 随输出 Token 密度下降线性加快 |

> **说明**：自回归 ASR 解码耗时与单位时间输出 Token 数强相关，因此高语速中文内容（~300 字/分）的实际倍速会明显低于低语速或静音占比高的音频；多并发下通过跨任务动态合批与后台预取可稳定打满 GPU 算力。

## 显存规划建议

| 显卡规格 | 推荐 `.env` 关键参数 | 说明 |
| :--- | :--- | :--- |
| **RTX 4080 SUPER / 32GB** | `GPU_MEMORY_UTILIZATION=0.58`<br>`MAX_INFERENCE_BATCH_SIZE=24`<br>`ALIGNER_BATCH_SIZE=6` | vLLM 预留 ~19GB，余量给 ForcedAligner 与激活峰值，8 并发稳定无 OOM |
| **RTX 4090 / 24GB（单卡或双卡）** | `GPU_MEMORY_UTILIZATION=0.62`<br>`MAX_INFERENCE_BATCH_SIZE=16`<br>`ALIGNER_BATCH_SIZE=6` | 双卡设 `GPU_IDS=0,1` 即可线性翻倍吞吐 |

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

---

## License

本项目采用 [MIT License](LICENSE) 开源。模型权重与推理引擎请分别遵循 Qwen3-ASR 及相关上游项目的许可协议。
