# 部署与运维指南

本文档面向在 **双路 RTX 4090 (2 × 24GB)** Linux 服务器上部署与运维 Qwen3-ASR-1.7B 长录音转写服务的场景，涵盖环境准备、Docker / 裸机部署、参数调优与故障排查。

---

## 1. 硬件与软件前提

### 硬件建议
- **GPU**：2 × NVIDIA RTX 4090 (24GB)（也支持单卡或多卡，通过 `GPU_IDS` 调整）
- **内存**：建议 ≥ 32 GB RAM（长音频切片按需磁盘 Seek 读取，内存占用很低，但双进程加载模型时需留足系统内存）
- **磁盘**：建议预留 ≥ 30 GB 可用空间（模型权重约 5–6 GB + Docker 镜像 + 并发转码临时 WAV 空间）

### 软件依赖
- Linux（Ubuntu 22.04 / 24.04 推荐）
- NVIDIA Driver（支持 CUDA 12.x）
- Docker 部署需要：Docker Engine ≥ 24 + Docker Compose v2 + `nvidia-container-toolkit`
- 裸机部署需要：Python 3.12、`ffmpeg`、`libsndfile1`

验证宿主机 GPU 容器可用性：

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

---

## 2. 下载模型权重

建议先将权重下载到本地 `./models` 目录再挂载进容器，避免容器重启时重复拉取。

```bash
# 国内网络推荐使用 ModelScope
pip install -U modelscope
python scripts/download_models.py --source modelscope --output-dir ./models

# 海外环境可使用 HuggingFace
pip install -U huggingface_hub
python scripts/download_models.py --source hf --output-dir ./models
```

若业务**完全不需要时间戳**，可加 `--skip-aligner` 仅下载 ASR 模型，并在 `.env` 中将 `ALIGNER_MODEL_PATH=` 置空。

---

## 3. 环境变量完整说明

复制模板并修改：

```bash
cp .env.example .env
```

| 变量名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `HOST` | `0.0.0.0` | FastAPI 监听地址 |
| `PORT` | `8000` | 对外暴露端口（`docker-compose.yml` 映射到宿主机） |
| `LOG_LEVEL` | `INFO` | 日志级别：`DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `API_KEYS` | `""`（示例文件含占位符） | 允许访问的 API Key 列表，逗号分隔；**生产环境务必改为高强度随机密钥** |
| `ASR_MODEL_PATH` | `/models/Qwen3-ASR-1.7B` | `Qwen3-ASR-1.7B` 本地路径或仓库 ID |
| `ALIGNER_MODEL_PATH` | `/models/Qwen3-ForcedAligner-0.6B` | `Qwen3-ForcedAligner-0.6B` 路径；留空则禁用强制对齐 |
| `SERVED_MODEL_NAME` | `qwen3-asr-1.7b` | `/v1/models` 与转写结果中返回的模型名 |
| `ASR_BACKEND` | `vllm` | 推理后端：`vllm`（生产推荐）、`transformers`、`mock`（无卡测试） |
| `GPU_IDS` | `0,1` | 分配的物理 GPU ID，逗号分隔；每张卡启动 1 个独立 Worker 进程 |
| `GPU_MEMORY_UTILIZATION` | `0.72` | 单卡 vLLM 显存占用上限（`0.2 ~ 0.95`）；双模型同卡在 4090 上建议 `0.70 ~ 0.74` |
| `MAX_INFERENCE_BATCH_SIZE` | `8` | 单卡单次推理批大小（`1 ~ 128`） |
| `MAX_NEW_TOKENS` | `2048` | 单切片最大生成 Token 数（`128 ~ 8192`），2048 足以覆盖 120–150s 语速较快的音频 |
| `CHUNKS_PER_DISPATCH` | `4` | 每次派发给单个 GPU Worker 的切片数（`1 ~ 32`） |
| `USE_FLASH_ATTENTION` | `false` | 是否为 ForcedAligner / Transformers 启用 `flash_attention_2`（官方镜像支持开启） |
| `WORKER_STARTUP_TIMEOUT_SEC` | `300` | 启动时等待所有 GPU Worker 加载模型的超时时间（秒） |
| `CHUNK_TARGET_SEC` | `120` | 长音频目标切片时长（`10 ~ 165` 秒，必须低于 ForcedAligner 180s 硬上限） |
| `CHUNK_SEARCH_EXPAND_SEC` | `5` | 在目标切分点前后搜索静音谷底的半窗（`0.5 ~ 15` 秒） |
| `MAX_UPLOAD_SIZE_MB` | `500` | 单文件最大上传体积（MB，`1 ~ 4096`）；若使用 Nginx 需同步修改 `client_max_body_size` |
| `MAX_SYNC_AUDIO_SEC` | `600` | 同步接口允许的最大音频时长（秒，最低 30） |
| `MAX_CONCURRENT_JOBS` | `16` | 后台异步任务最大同时执行数（超出则在队列中等待） |
| `DATA_DIR` | `./data`（容器内 `/app/data`） | SQLite 数据库（`tasks.db`）与临时音频目录（`tmp_audio/`） |
| `TASK_RETENTION_HOURS` | `72` | 已完成/失败任务记录保留时长（小时），任务收尾时自动清理过期记录 |
| `WEBHOOK_TIMEOUT_SEC` | `15` | Webhook 回调 HTTP 超时时间（秒） |

---

## 4. Docker Compose 部署（推荐）

### 4.1 仅启动 API 服务

```bash
docker compose up -d --build
docker compose logs -f qwen3-asr-api
```

看到两张卡均输出 `Worker ready` 后，执行健康检查：

```bash
curl http://127.0.0.1:8000/health
```

### 4.2 启用内置 Nginx 反向代理（对外发布推荐）

内置的 `deploy/nginx.conf` 已配置：
- `client_max_body_size 512M`（支持大音频上传）
- `proxy_request_buffering off`（流式直传 FastAPI，避免 Nginx 二次落盘）
- `limit_req` 基础限流（`10r/s`，`burst=20`）
- `proxy_read_timeout 600s`（兼容同步中长音频请求）

启动命令：

```bash
docker compose --profile proxy up -d --build
```

**开启 HTTPS（TLS）**：
1. 将证书放到 `./deploy/certs/fullchain.pem` 与 `./deploy/certs/privkey.pem`；
2. 在 `docker-compose.yml` 中取消 `nginx` 服务的 `443:443` 端口和 `./deploy/certs` 挂载注释；
3. 在 `deploy/nginx.conf` 中取消底部 `443 ssl http2` server 块注释并修改 `server_name`；
4. 执行 `docker compose --profile proxy up -d --force-recreate nginx`。

---

## 5. 宿主机裸机部署（systemd）

适合不方便使用 Docker 的内网服务器：

```bash
# 1. 创建环境并安装依赖
conda create -n qwen3-asr python=3.12 -y
conda activate qwen3-asr
pip install -U "qwen-asr[vllm]"
pip install -r requirements.txt
sudo apt update && sudo apt install -y ffmpeg libsndfile1

# 2. 配置 .env（注意把模型路径改为宿主机实际路径，例如 ./models/Qwen3-ASR-1.7B）
cp .env.example .env
```

示例 `/etc/systemd/system/qwen3-asr.service`：

```ini
[Unit]
Description=Qwen3-ASR-1.7B Dual-GPU API Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/asr-dep
Environment="VLLM_WORKER_MULTIPROC_METHOD=spawn"
ExecStart=/opt/miniconda3/envs/qwen3-asr/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --proxy-headers
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

> **重要**：`uvicorn` 的 `--workers` 必须保持为 **`1`**。多卡并行由应用内部的 `DualGPUDispatcher` 按 `GPU_IDS` 派生独立子进程完成；若开启多个 uvicorn worker，会导致每张卡被重复加载模型并瞬间 OOM。

---

## 6. 显存与性能调优建议

### 单张 RTX 4090 (24GB) 显存预算
- `Qwen3-ASR-1.7B`（vLLM BF16 + KV Cache）：`GPU_MEMORY_UTILIZATION=0.72` ≈ **17.3 GB**
- `Qwen3-ForcedAligner-0.6B`（BF16 + 激活显存）：约 **2.5 – 3.5 GB**
- CUDA 上下文与峰值缓冲余量：约 **3.2 – 4.2 GB**

### 常见场景调参
- **官方镜像加速**：`qwenllm/qwen3-asr:latest` 已内置 FlashAttention 2，将 `USE_FLASH_ATTENTION=true` 可显著降低 ForcedAligner 延迟。
- **追求单段超长录音极限速度**：保持 `CHUNKS_PER_DISPATCH=2~4`，让两张 4090 更细粒度地分摊同一段录音的切片批次；同时可把 `MAX_INFERENCE_BATCH_SIZE` 调至 `12~16`（需用实际音频压测观察 `nvidia-smi` 显存峰值）。
- **单卡 / 四卡部署**：
  - 单卡：`GPU_IDS=0`，并在 `docker-compose.yml` 中把 `device_ids` 改为 `["0"]`；
  - 四卡：`GPU_IDS=0,1,2,3`，`device_ids` 改为 `["0", "1", "2", "3"]`。
- **不需要时间戳（纯文本高吞吐）**：置空 `ALIGNER_MODEL_PATH`，并将 `GPU_MEMORY_UTILIZATION` 提高到 `0.88`。

---

## 7. 运维与故障排查

| 现象 | 排查与处理方式 |
| :--- | :--- |
| 启动超时（`Timed out waiting for GPU workers`） | 首次拉取/编译或磁盘 IO 较慢时可调大 `WORKER_STARTUP_TIMEOUT_SEC=600`，并用 `docker compose logs -f` 查看具体加载阶段。 |
| ForcedAligner 报 CUDA OOM | 降低 `GPU_MEMORY_UTILIZATION`（如 `0.68`）或调小 `CHUNKS_PER_DISPATCH` / `MAX_INFERENCE_BATCH_SIZE`。 |
| Nginx 返回 `413 Request Entity Too Large` | 确认 `deploy/nginx.conf` 中的 `client_max_body_size` 与 `.env` 中的 `MAX_UPLOAD_SIZE_MB` 一致且足够大。 |
| Nginx 返回 `504 Gateway Time-out` | 长音频请使用异步接口 `POST /v1/asr/tasks`，不要走同步接口。 |
| 服务意外重启后原运行中任务卡死？ | 服务启动时 `TaskStore.mark_interrupted_tasks_failed()` 会自动把重启前未完成的 `queued`/`decoding`/`processing` 任务标记为 `failed`，调用方可安全重试。 |
| 磁盘占用增长 | 音频临时文件在每次请求/任务结束后通过 `finally` 自动删除；SQLite 任务记录按 `TASK_RETENTION_HOURS` 自动清理。如曾异常强制 `kill -9`，可定期检查 `./data/tmp_audio/` 是否残留孤立文件。 |
