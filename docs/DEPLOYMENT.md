# 部署与运维指南

本文档面向在 Linux GPU 服务器（单卡 32GB RTX 4080 SUPER / 5090、单/双路 24GB RTX 4090、或云端容器实例如 AutoDL）上部署与运维 `qwen3-asr-server` 的场景，涵盖环境准备、Docker / 裸机 / 云端数据盘部署、完整环境变量表、高并发调优与故障排查。

---

## 1. 硬件与软件前提

### 硬件建议
- **GPU**：单卡或多卡 NVIDIA GPU（24 GB+ 显存最佳，如 RTX 4080 SUPER 32GB、RTX 4090 24GB、A100/L40S），通过 `GPU_IDS` 指定卡号（如 `0` 或 `0,1`）
- **CPU / 内存**：建议 ≥ 8 vCPU、≥ 32 GB RAM；若开启 `TEMP_DIR=/dev/shm/qwen3_asr_tmp` 内存盘加速，建议 `/dev/shm` ≥ 8 GB
- **磁盘**：建议预留 ≥ 30 GB 可用空间（模型权重约 6.2 GB + Python/Docker 环境 + 并发临时文件空间）

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

建议先将权重下载到本地 `./models` 目录再挂载进容器或直接指向本地路径，避免服务重启时重复拉取。

```bash
# 国内网络推荐使用 ModelScope（实测远快于 HF 镜像）
pip install -U modelscope
python scripts/download_models.py --source modelscope --output-dir ./models

# 海外环境可使用 HuggingFace
pip install -U huggingface_hub
python scripts/download_models.py --source hf --output-dir ./models
```

下载完成后目录结构：
```text
models/
├── Qwen3-ASR-1.7B/              # ~4.4 GB
└── Qwen3-ForcedAligner-0.6B/    # ~1.8 GB
```

若业务**完全不需要时间戳**，可加 `--skip-aligner` 仅下载 ASR 模型，并在 `.env` 中将 `ALIGNER_MODEL_PATH=` 置空。

---

## 3. 环境变量完整说明

复制模板并修改：

```bash
cp .env.example .env
```

### 3.1 基础与鉴权
| 变量名 | 代码默认值 | 说明 |
| :--- | :--- | :--- |
| `HOST` | `0.0.0.0` | FastAPI 监听地址 |
| `PORT` | `8000` | 服务监听端口（AutoDL 自定义服务可设为 `6006`） |
| `LOG_LEVEL` | `INFO` | 日志级别：`DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `API_KEYS` | `""` | 允许访问的 API Key 列表（英文逗号分隔）；留空关闭鉴权，**生产环境务必改为高强度随机密钥** |

### 3.2 模型与推理引擎
| 变量名 | 代码默认值 | 说明 |
| :--- | :--- | :--- |
| `ASR_MODEL_PATH` | `Qwen/Qwen3-ASR-1.7B` | `Qwen3-ASR-1.7B` 本地路径或模型仓库 ID |
| `ALIGNER_MODEL_PATH` | `Qwen/Qwen3-ForcedAligner-0.6B` | `Qwen3-ForcedAligner-0.6B` 路径；留空则禁用强制对齐 |
| `SERVED_MODEL_NAME` | `qwen3-asr-1.7b` | `/v1/models` 与转写结果中返回的模型名 |
| `ASR_BACKEND` | `vllm` | 推理后端：`vllm`（生产推荐）、`transformers`、`mock`（无卡测试） |
| `GPU_IDS` | `0,1` | 分配的物理 GPU ID（逗号分隔，每卡独立启动 1 个 Worker 进程；单卡设 `0`） |
| `GPU_MEMORY_UTILIZATION` | `0.72`（32GB 推荐 `0.58`，24GB 推荐 `0.62`） | vLLM 显存预留比例（`0.2 ~ 0.95`）；需为 ForcedAligner 与激活峰值预留余量 |
| `MAX_INFERENCE_BATCH_SIZE` | `16`（32GB 推荐 `24`） | 单个 GPU Worker 跨任务合批送入 vLLM 的最大切片数（`1 ~ 128`） |
| `ALIGNER_BATCH_SIZE` | `6` | ForcedAligner 单次 GPU 前向子批大小（`1 ~ 64`），与 vLLM 批大小解耦防 OOM |
| `MAX_NEW_TOKENS` | `2048` | 单切片最大生成 Token 数（`128 ~ 8192`），2048 足以覆盖 120–150s 语速较快的音频 |
| `CHUNKS_PER_DISPATCH` | `6` | 公平轮转调度单次从单个任务派发的切片粒度（`1 ~ 32`） |
| `BATCH_COALESCE_WAIT_MS` | `15`（高并发长录音推荐 `40`） | 单卡 Worker 跨任务合批最大等待窗口（毫秒，`0 ~ 500`） |
| `USE_FLASH_ATTENTION` | `false` | 是否为 ForcedAligner / Transformers 显式启用 `flash_attention_2`（PyTorch 2.x 默认 SDPA 已走 Flash 内核） |
| `WORKER_STARTUP_TIMEOUT_SEC` | `300` | 启动时等待所有 GPU Worker 加载模型的超时时间（秒） |

### 3.3 长录音切分、并发控制与持久化
| 变量名 | 代码默认值 | 说明 |
| :--- | :--- | :--- |
| `CHUNK_STRATEGY` | `vad` | 长音频边界切分策略：`vad`（响度自适应批量 ONNX Silero VAD，推荐）\| `energy`（纯低能量谷底） |
| `CHUNK_TARGET_SEC` | `120` | 长音频目标切片时长（`10 ~ 165` 秒，需满足 `target + 2*expand ≤ 170s < 180s`） |
| `CHUNK_SEARCH_EXPAND_SEC` | `15` | 在目标切分点前后搜索停顿/静音的半窗（`0.5 ~ 25` 秒，±15s 即 30s 窗口） |
| `VAD_THRESHOLD` | `0.5` | Silero VAD 语音概率阈值（`0.1 ~ 0.9`） |
| `VAD_MIN_SILENCE_MS` | `150` | Silero VAD 优选切分停顿的最短时长（毫秒，`50 ~ 2000`） |
| `VAD_MAX_WORKERS` | `6` | 批量 ONNX Silero VAD 并行 CPU 线程数（`1 ~ 32`） |
| `MAX_CONCURRENT_DECODES` | `4` | 最大并发 `ffmpeg` 音频转码进程数（`1 ~ 32`，每进程 `-threads 2`） |
| `MAX_UPLOAD_SIZE_MB` | `500` | 单文件最大上传体积（MB，`1 ~ 4096`）；使用 Nginx 时需同步调大 `client_max_body_size` |
| `MAX_SYNC_AUDIO_SEC` | `600` | 同步接口允许的最大音频时长（秒，最低 `30`） |
| `MAX_CONCURRENT_JOBS` | `16` | 后台异步任务最大同时执行数（超出则在队列中等待） |
| `DATA_DIR` | `./data`（容器内 `/app/data`） | SQLite 任务库（`tasks.db`）与默认临时音频目录 |
| `TEMP_DIR` | `""` | 可选：覆盖临时 WAV 目录；Linux 大内存环境推荐设为 `/dev/shm/qwen3_asr_tmp` 走内存盘零磁盘 I/O |
| `TASK_RETENTION_HOURS` | `72` | 已完成/失败任务记录保留时长（小时），任务收尾时自动清理过期记录 |
| `WEBHOOK_TIMEOUT_SEC` | `15` | Webhook 回调 HTTP 超时时间（秒） |

---

## 4. Docker Compose 部署（推荐）

### 4.1 仅启动 API 服务

```bash
docker compose up -d --build
docker compose logs -f qwen3-asr-api
```

> 若使用单卡服务器，请在 `docker-compose.yml` 中把 `device_ids` 改为 `["0"]` 并在 `.env` 设 `GPU_IDS=0`。`docker-compose.yml` 已默认配置 `shm_size: "8gb"`，可在 `.env` 直接开启 `TEMP_DIR=/dev/shm/qwen3_asr_tmp`。

看到所有 GPU 均输出 `Worker ready` 后，执行健康检查：

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
Description=Qwen3-ASR-1.7B High-Throughput API Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/qwen3-asr-server
Environment="VLLM_WORKER_MULTIPROC_METHOD=spawn"
ExecStart=/opt/miniconda3/envs/qwen3-asr/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --proxy-headers
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

> **重要**：`uvicorn` 的 `--workers` 必须保持为 **`1`**。多卡并行由应用内部的 `DualGPUDispatcher` 按 `GPU_IDS` 派生独立子进程完成；若开启多个 uvicorn worker，会导致每张卡被重复加载模型并瞬间 OOM。

---

## 6. 云端 GPU 容器 / AutoDL 数据盘部署参考

对于系统盘较小（如 30 GB）但带独立数据盘（如 `/root/autodl-tmp`）且无 systemd 的云容器，建议将代码、虚拟环境、模型权重与缓存全部放到数据盘，并用 `tmux` 托管守护进程：

```bash
# 1. 将代码与 venv 建在数据盘，使用国内镜像快速安装
cd /root/autodl-tmp
git clone https://github.com/chosenecho/qwen3-asr-server.git
uv venv /root/autodl-tmp/venv --python 3.12
UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ uv pip install --python /root/autodl-tmp/venv/bin/python "qwen-asr[vllm]" -r qwen3-asr-server/requirements.txt modelscope

# 2. 通过 ModelScope 下载模型到数据盘
/root/autodl-tmp/venv/bin/python qwen3-asr-server/scripts/download_models.py --source modelscope --output-dir /root/autodl-tmp/qwen3-asr-server/models

# 3. 配置 .env（端口 6006 兼容 AutoDL 自定义服务，临时音频走 /dev/shm）
#    PORT=6006
#    GPU_IDS=0
#    GPU_MEMORY_UTILIZATION=0.58
#    MAX_INFERENCE_BATCH_SIZE=24
#    ALIGNER_BATCH_SIZE=6
#    CHUNKS_PER_DISPATCH=6
#    BATCH_COALESCE_WAIT_MS=40
#    TEMP_DIR=/dev/shm/qwen3_asr_tmp

# 4. 使用 tmux 后台运行（断开 SSH 不中断）
tmux new-session -d -s qwen3-asr "cd /root/autodl-tmp/qwen3-asr-server && \
  export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_CACHE_ROOT=/root/autodl-tmp/.cache/vllm HF_HOME=/root/autodl-tmp/.cache/huggingface && \
  /root/autodl-tmp/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 6006 --workers 1 --proxy-headers 2>&1 | tee -a logs/server.log"
```

本地通过 SSH 端口转发安全访问：

```bash
ssh -N -L 127.0.0.1:6006:127.0.0.1:6006 -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 <ssh-host>
```

---

## 7. 显存与高并发调优预设

| 硬件与业务场景 | 推荐参数组合 | 预期表现 |
| :--- | :--- | :--- |
| **单卡 32GB（RTX 4080 SUPER / 5090）** 长录音高并发（带词级对齐） | `GPU_IDS=0`<br>`GPU_MEMORY_UTILIZATION=0.58`<br>`MAX_INFERENCE_BATCH_SIZE=24`<br>`ALIGNER_BATCH_SIZE=6`<br>`CHUNKS_PER_DISPATCH=6`<br>`BATCH_COALESCE_WAIT_MS=40`<br>`TEMP_DIR=/dev/shm/qwen3_asr_tmp` | 8 并发 × 20m 实测 **`285.3×` 聚合 / `~319×` 稳态**，显存峰值稳定 `~30.2 GB / 32.8 GB` |
| **双卡 24GB（2 × RTX 4090）** 长录音高并发（带词级对齐） | `GPU_IDS=0,1`<br>`GPU_MEMORY_UTILIZATION=0.62`<br>`MAX_INFERENCE_BATCH_SIZE=16`<br>`ALIGNER_BATCH_SIZE=6`<br>`CHUNKS_PER_DISPATCH=4`<br>`BATCH_COALESCE_WAIT_MS=15` | 每张 4090 为 vLLM 留 ~14.9 GB、ForcedAligner 留 ~9.1 GB，双卡线性扩展 |
| **纯文本极速转写（不需要时间戳）** | 置空 `ALIGNER_MODEL_PATH=`<br>`GPU_MEMORY_UTILIZATION=0.88`<br>`MAX_INFERENCE_BATCH_SIZE=32` | 省去 ForcedAligner 显存与计算，吞吐再提升 ~20% |

---

## 8. 运维与故障排查

| 现象 | 排查与处理方式 |
| :--- | :--- |
| 启动超时（`Timed out waiting for GPU workers`） | 首次加载或磁盘 IO 较慢时可调大 `WORKER_STARTUP_TIMEOUT_SEC=600`，查看 `logs/server.log` 或 `docker compose logs -f` 确认卡在哪个阶段。 |
| ForcedAligner 报 CUDA OOM | 优先调小 `ALIGNER_BATCH_SIZE`（如 `6 -> 4`）或适度降低 `GPU_MEMORY_UTILIZATION`（如 `0.58 -> 0.54`），无需降低 `MAX_INFERENCE_BATCH_SIZE`。 |
| 日志提示 `Silero VAD unavailable` | 当前环境未安装 `silero-vad` 或 `onnxruntime`，服务已自动降级为低能量切分；执行 `pip install silero-vad onnxruntime` 后重启即可恢复 VAD 无损切分。 |
| Nginx 返回 `413 Request Entity Too Large` | 确认 `deploy/nginx.conf` 中的 `client_max_body_size` 与 `.env` 中的 `MAX_UPLOAD_SIZE_MB` 一致且足够大。 |
| Nginx 返回 `504 Gateway Time-out` | 长音频请使用异步接口 `POST /v1/asr/tasks`，不要走同步接口。 |
| 服务意外重启后原运行中任务卡死？ | 服务启动时 `TaskStore.mark_interrupted_tasks_failed()` 会自动把重启前未完成的 `queued`/`decoding`/`processing` 任务标记为 `failed`，调用方可安全重试。 |
| 临时目录或 `/dev/shm` 空间占用 | 每次请求/任务结束后均在 `finally` 中自动删除临时文件；若曾发生 `kill -9` 强杀，可重启前清空 `TEMP_DIR`（如 `rm -rf /dev/shm/qwen3_asr_tmp/*`）。 |
