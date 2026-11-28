# 部署与运维指南

本文档面向在 Linux GPU 服务器（单卡 32GB RTX 4080 SUPER / 5090、单/双路 24GB RTX 4090）、云端容器实例（如 AutoDL）以及 **Modal Serverless GPU（NVIDIA L4 24GB，闲时缩容到 0）** 上部署与运维 `qwen3-asr-server` 的场景，涵盖环境准备、Docker / 裸机 / 云端数据盘 / Modal 部署、完整环境变量表、冷启动优化、高并发调优与故障排查。

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
| `VLLM_MAX_MODEL_LEN` | `None` | 可选：覆盖 vLLM `max_model_len`（`2048 ~ 65536`）；如 24GB L4/4090 上设 `16384` 可收缩最小 KV Cache 占用，给 ForcedAligner 留出安全显存余量 |
| `VLLM_ENFORCE_EAGER` | `false` | 设为 `true` 时关闭 vLLM `torch.compile` 与 CUDA Graph 捕获；适合未开启 GPU 快照时的快速冷启动 |
| `VLLM_CUDAGRAPH_MODE` | `""` | 可选：覆盖 vLLM `compilation_config.cudagraph_mode`（如 `FULL_DECODE_ONLY`、`PIECEWISE`、`FULL`、`NONE`） |
| `VLLM_CUDAGRAPH_CAPTURE_SIZES` | `""` | 可选：英文逗号分隔的 CUDA Graph 录制 Batch Size 列表（如 `1,2,4,8,16`），用于裁剪捕获耗时与 GPU 快照体积 |
| `VLLM_COMPILATION_MODE` | `None` | 可选：覆盖 vLLM `compilation_config.mode`（`0=NONE`、`1=STOCK_TORCH_COMPILE`、`2=DYNAMO_TRACE_ONCE`、`3=VLLM_COMPILE`） |
| `WARMUP_ON_STARTUP` | `false` | 设为 `true` 时在 Worker 报告就绪前自动跑一遍轻量级 ASR + ForcedAligner + Silero VAD 预热，提前完成 CUDA / Triton / ONNX 首次编译 |

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
| `MAX_CONCURRENT_JOBS` | `16` | 同步、OpenAI、异步共用的活动作业数 |
| `MAX_QUEUED_JOBS` | `32` | 共用等待作业名额；容量满立即返回 503 |
| `MAX_CONCURRENT_UPLOADS` | `4` | multipart 接收和解析的并发上限 |
| `MAX_TEMP_STORAGE_MB` | `8192` | 应用接管源文件与预留 WAV 的总预算（MB） |
| `MAX_DECODED_AUDIO_MB` | `1024` | 单次解码 WAV 输出上限（MB） |
| `MAX_ASYNC_AUDIO_SEC` | `28800` | 异步音频时长上限（秒，8 小时） |
| `DECODE_TIMEOUT_SEC` | `600` | ffmpeg 执行期限（秒） |
| `BATCH_TIMEOUT_SEC` | `300` | 批次领取后到完成的期限（秒，包含预取和推理） |
| `BATCH_QUEUE_TIMEOUT_SEC` | `600` | 批次派发后等待领取的期限（秒） |
| `JOB_TIMEOUT_SEC` | `7200` | 暂存、等待、解码、切分和推理共用的总期限（秒） |
| `WORKER_RESTART_LIMIT` | `3` | 每卡运行期故障重建次数；0 为关闭重建 |
| `SHUTDOWN_GRACE_SEC` | `30` | 取消时等待读者确认的宽限期，之后强制回收相关 Worker |
| `ENABLE_ASYNC_TASKS` | `true` | 是否接受异步任务提交；Modal 模板固定 false |
| `DATA_DIR` | `./data`（容器内 `/app/data`） | SQLite 任务库（`tasks.db`）与默认临时音频目录 |
| `TEMP_DIR` | `""` | 可选：覆盖临时 WAV 目录；Linux 大内存环境推荐设为 `/dev/shm/qwen3_asr_tmp` 走内存盘零磁盘 I/O |
| `TASK_RETENTION_HOURS` | `72` | 已完成/失败任务记录保留时长（小时），任务收尾时自动清理过期记录 |
| `WEBHOOK_TIMEOUT_SEC` | `15` | Webhook 回调 HTTP 超时时间（秒） |

---

### 3.4 配额、迁移与关闭约定

保持 `uvicorn --workers 1`：作业、上传和存储预算是单 API 进程内的统一限制，多个主进程不会共享配额。`CHUNKS_PER_DISPATCH ≤ MAX_INFERENCE_BATCH_SIZE`、`CHUNK_TARGET_SEC + 2*CHUNK_SEARCH_EXPAND_SEC ≤ 170`、总作业期限至少覆盖任一种批次期限，重复 GPU ID 在启动配置校验时直接报错。

`TEMP_DIR` 的应用预算与 Starlette multipart 临时目录分别规划；后者通常取系统 `TMPDIR`，最坏需要额外考虑 `MAX_CONCURRENT_UPLOADS × MAX_UPLOAD_SIZE_MB` 的暂存空间。使用 `/dev/shm` 时应为两处预算和模型/CPU 缓存保留内存余量。Nginx 模板的 512 MB 请求上限覆盖默认 500 MB 文件及表单开销；修改文件上限时也同步调整反向代理。

升级前备份 SQLite 数据目录。启动自动执行事务迁移到 `user_version=1`，保留原任务表，仅增加三个字段；迁移失败回滚，高于当前支持的版本会中止启动。回滚旧版本前先恢复升级前备份。未结束任务在重启后标记 `SERVER_RESTARTED`，不续跑；仅清理应用生成的 `job_<uuid>` 目录，用户其他文件保留。

关闭时先停止准入、取消上传暂存和待提交事务，再等待在途读者；取消宽限结束会强制回收关联 GPU Worker，因此同卡其他在途批次也可能失败。Docker 模板使用 90 秒停止宽限，需按 Worker 数、读者最慢退出时间调整，给应用完成清算留出时间。`CLEANUP_FAILED` 时保留音频与配额并使健康检查降级，待完整 Worker 回收后再删除。

## 4. Docker Compose 部署（推荐）

Compose 将后端端口绑定到宿主机 `127.0.0.1`，外部访问通过可选 Nginx profile；容器内部仍监听 `0.0.0.0:8000`。GPU reservation 暴露已安装的所有 GPU，再由 `GPU_IDS` 选择 Worker 使用的卡号；单卡宿主机配置 `GPU_IDS=0` 即可。需要设备隔离时，在部署环境同时收窄容器 GPU 可见集合并核对卡号映射。

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

## 7. Modal Serverless GPU 部署（闲时缩容到 0）

仓库内置了生产级 Serverless 部署脚本 [`deploy/modal_app.py`](../deploy/modal_app.py)，默认在单张 **NVIDIA L4 (24GB)** 上以 **原版 BF16 权重**（`Qwen3-ASR-1.7B` + `Qwen3-ForcedAligner-0.6B`）全功能运行，闲时自动缩容到 `0`，按实际请求秒级计费。

### 7.1 前置准备与一键部署

```bash
# 1. 安装并登录 Modal CLI（建议使用 modal >= 1.6）
pip install -U modal
modal setup

# 2. 创建业务 API Key Secret（应用内校验 Authorization: Bearer / X-API-Key）
#    默认读取名为 qwen-api-key 的 Secret（兼容键名：API_KEYS / QWEN_API_KEY / SGLANG_API_KEY / API_KEY）
#    如需使用自定义 Secret 名称，可在部署前设置环境变量：export MODAL_API_SECRET_NAME=my-asr-secret
modal secret create qwen-api-key API_KEYS=sk-your-strong-secret-key

# 3. 一次性预下载 BF16 权重到同区域持久化 Volume `qwen3-asr-weights`（约 6.2 GB）
modal run deploy/modal_app.py::download_models

# 4. 部署 Serverless HTTPS 服务
modal deploy deploy/modal_app.py
```

部署完成后，Modal 会输出形如 `https://<your-workspace>--qwen3-asr-server-serve.modal.run` 的 HTTPS 访问地址。

### 7.2 Modal 边缘网关代理鉴权（`requires_proxy_auth=True`）

`deploy/modal_app.py` 默认开启了 Modal 原生边缘代理鉴权：

```python
REQUIRES_PROXY_AUTH = True

@modal.web_server(
    port=SERVER_PORT,
    startup_timeout=300,
    label=ENDPOINT_LABEL,
    requires_proxy_auth=REQUIRES_PROXY_AUTH,
)
```

- **核心收益（防刷防唤醒）**：缺失或错误的 `Modal-Key` / `Modal-Secret` 请求会在 Modal 边缘网关层直接被拦截并返回 `401 Unauthorized`，**完全不会唤醒 GPU 容器，产生 `$0` 算力费用**。
- **如何获取 Token**：在 Modal 控制台进入 **Settings → Proxy Auth Tokens**，创建一对 `Modal-Key`（`wk-...`）与 `Modal-Secret`（`ws-...`）。
- **调用方式**（注意：开启代理鉴权后，包括 `GET /health` 在内的所有路径均需携带 `Modal-Key` 和 `Modal-Secret`）：

```bash
# 健康检查（边缘代理层鉴权）
curl "https://<your-workspace>--qwen3-asr-server-serve.modal.run/health" \
  -H "Modal-Key: wk-..." \
  -H "Modal-Secret: ws-..."

# 同步转写（边缘代理鉴权 + 应用层 API Key 双重保护）
curl -X POST "https://<your-workspace>--qwen3-asr-server-serve.modal.run/v1/asr/transcribe" \
  -H "Modal-Key: wk-..." \
  -H "Modal-Secret: ws-..." \
  -H "Authorization: Bearer sk-your-strong-secret-key" \
  -F "file=@sample.wav" \
  -F "return_timestamps=true" \
  -F "include_words=true"
```

使用 OpenAI Python SDK 时，可通过 `default_headers` 透传边缘鉴权头：

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<your-workspace>--qwen3-asr-server-serve.modal.run/v1",
    api_key="sk-your-strong-secret-key",
    default_headers={
        "Modal-Key": "wk-...",
        "Modal-Secret": "ws-...",
    },
)
```

> **第三方客户端兼容性说明**：如果使用的标准 OpenAI 桌面/Web 客户端（如部分未提供自定义 Header 入口的客户端）无法附加 `Modal-Key` 和 `Modal-Secret`，可将 `deploy/modal_app.py` 顶部的 `REQUIRES_PROXY_AUTH` 改为 `False` 并重新执行 `modal deploy deploy/modal_app.py`。关闭后仍由服务内置的 `API_KEYS` 保护所有 `/v1/*` 接口，但未授权请求会先唤醒容器再返回 `401`。

### 7.3 冷启动优化策略与实测数据（NVIDIA L4 24GB，BF16）

`deploy/modal_app.py` 综合了 6 项面向 Serverless GPU 的冷启动优化：

1. **CPU + GPU 显存快照（核心）**：启用 `enable_memory_snapshot=True` + `experimental_options={"enable_gpu_snapshot": True}`，并在 `@modal.enter(snap=True)` 中完成模型加载、CUDA 初始化、KV Cache 分配与预热；后续从 0 缩容再唤醒时直接反序列化显存快照，跳过读盘与初始化。
2. **快照前全链路预热**：Worker 启动阶段开启 `WARMUP_ON_STARTUP=true` 预热 `Qwen3-ASR-1.7B`、`Qwen3-ForcedAligner-0.6B` 与多线程 Silero VAD ONNX；随后 `deploy/modal_app.py` 再向本地 `127.0.0.1:8000/v1/asr/transcribe` 发起一次真实 WAV 端到端 HTTP 预热请求，逼迫所有 CUDA / Triton / FlashInfer / ffmpeg 路径在拍快照前完成初始化，消除恢复后首包编译卡顿。
3. **持久化 JIT 编译缓存 Volume（`/cache`）**：挂载独立 Volume `qwen3-asr-cache` 到 `/cache`，将 `VLLM_CACHE_ROOT`、`TRITON_CACHE_DIR`、`TORCHINDUCTOR_CACHE_DIR`、`FLASHINFER_WORKSPACE_BASE`、`CUDA_CACHE_PATH` 全部持久化并在预热完成后同步 `cache_vol.commit()`，即使换镜像或快照失效也能大幅缩短兜底冷启时间。
4. **裁剪版 Decode-Only CUDA Graph**：开启快照时使用 `VLLM_CUDAGRAPH_MODE=FULL_DECODE_ONLY` 与 `VLLM_CUDAGRAPH_CAPTURE_SIZES=1,2,4,8,16`（仅录制 5 张小/中批 Decode 图，实测捕获仅耗时 ~2s、显存 ~0.04 GiB），兼顾快照体积与推理速度；若关闭快照（`ENABLE_GPU_SNAPSHOT = False`），自动降级为 `VLLM_ENFORCE_EAGER=true`。
5. **原版 BF16 单卡权重 + 进程内 EngineCore**：原版 BF16 权重（约 6.2 GB）预存于同区域 Volume `qwen3-asr-weights`（`/models`），启动零公网拉取；同时设置 `VLLM_ENABLE_V1_MULTIPROCESSING=0` 在独立 GPU Worker 内直接运行 vLLM V1 EngineCore，减少跨进程 IPC 与快照复杂度。
6. **短窗口保活（`SCALEDOWN_WINDOW_SECONDS = 60`）**：连续同步调用在 60 秒窗口内复用热容器，空闲后自动缩容到 0。

在 NVIDIA L4 (24GB) 上使用原版 BF16 权重 + 全量 ForcedAligner 的实测耗时对比：

| 阶段 / 场景 | 实测耗时 | 说明 |
| :--- | :---: | :--- |
| 未优化冷启（无快照、无 `/cache` 缓存） | `~129.8s` | 每次重新加载权重、编译算子并录制完整图 |
| 兜底冷启（快照失效，命中 `/cache` Volume） | `~86.7s` | 复用持久化 JIT 编译缓存，跳过算子重编译 |
| **开启 GPU 显存快照从 0 唤醒恢复** | **`~22.2s`** | **相比基线提速约 `5.8×`**，直接恢复已预热显存状态 |
| 快照恢复后首条真实音频推理（含 37 词对齐） | **`~1.6s`** | 零首次 JIT 卡顿，直接命中热内核与 Decode CUDA Graph |

> **Serverless 能力约定**：Modal 模板固定 `ENABLE_ASYNC_TASKS=false`，提供同步接口 `POST /v1/asr/transcribe` 与 `POST /v1/audio/transcriptions`。异步提交在解析上传前返回 `503 / ASYNC_TASKS_DISABLED`。容器 `/tmp` 内的 SQLite 与音频是临时状态；可靠异步需要外部持久任务状态和后台执行机制，留待独立扩展。

---

## 8. 显存与高并发调优预设

| 硬件与业务场景 | 推荐参数组合 | 预期表现 |
| :--- | :--- | :--- |
| **单卡 32GB（RTX 4080 SUPER / 5090）** 长录音高并发（带词级对齐） | `GPU_IDS=0`<br>`GPU_MEMORY_UTILIZATION=0.58`<br>`MAX_INFERENCE_BATCH_SIZE=24`<br>`ALIGNER_BATCH_SIZE=6`<br>`CHUNKS_PER_DISPATCH=6`<br>`BATCH_COALESCE_WAIT_MS=40`<br>`TEMP_DIR=/dev/shm/qwen3_asr_tmp` | 真实高密度长录音多并发约 **`120×–160×` 实时**，GPU 算力/带宽可跑满，显存峰值稳定 `~29.5–30.2 GB / 32.8 GB` |
| **双卡 24GB（2 × RTX 4090）** 长录音高并发（带词级对齐） | `GPU_IDS=0,1`<br>`GPU_MEMORY_UTILIZATION=0.62`<br>`MAX_INFERENCE_BATCH_SIZE=16`<br>`ALIGNER_BATCH_SIZE=6`<br>`CHUNKS_PER_DISPATCH=4`<br>`BATCH_COALESCE_WAIT_MS=15` | 每张 4090 为 vLLM 留 ~14.9 GB、ForcedAligner 留 ~9.1 GB，双卡线性扩展 |
| **单卡 24GB（NVIDIA L4 / Modal Serverless）** 闲时缩容到 0（带词级对齐） | `GPU_IDS=0`<br>`GPU_MEMORY_UTILIZATION=0.58`<br>`VLLM_MAX_MODEL_LEN=16384`<br>`MAX_INFERENCE_BATCH_SIZE=16`<br>`ALIGNER_BATCH_SIZE=6`<br>`VLLM_CUDAGRAPH_MODE=FULL_DECODE_ONLY`<br>`VLLM_CUDAGRAPH_CAPTURE_SIZES=1,2,4,8,16`<br>`WARMUP_ON_STARTUP=true` | 适配 L4 22.5 GiB 可用显存，配合 GPU 快照从 0 恢复约 `22.2s`，全功能 BF16 推理 |
| **纯文本极速转写（不需要时间戳）** | 置空 `ALIGNER_MODEL_PATH=`<br>`GPU_MEMORY_UTILIZATION=0.88`<br>`MAX_INFERENCE_BATCH_SIZE=32` | 省去 ForcedAligner 显存与计算，吞吐再提升 ~20% |

---

## 9. 运维与故障排查

| 现象 | 排查与处理方式 |
| :--- | :--- |
| 启动超时（`Timed out waiting for GPU workers`） | 首次加载或磁盘 IO 较慢时可调大 `WORKER_STARTUP_TIMEOUT_SEC=600`，查看 `logs/server.log` 或 `docker compose logs -f` 确认卡在哪个阶段。 |
| ForcedAligner 报 CUDA OOM | 优先调小 `ALIGNER_BATCH_SIZE`（如 `6 -> 4`）或适度降低 `GPU_MEMORY_UTILIZATION`（如 `0.58 -> 0.54`），无需降低 `MAX_INFERENCE_BATCH_SIZE`。 |
| 日志提示 `Silero VAD unavailable` | 当前环境未安装 `silero-vad` 或 `onnxruntime`，服务已自动降级为低能量切分；执行 `pip install silero-vad onnxruntime` 后重启即可恢复 VAD 无损切分。 |
| Nginx 返回 `413 Request Entity Too Large` | 确认 `deploy/nginx.conf` 中的 `client_max_body_size` 与 `.env` 中的 `MAX_UPLOAD_SIZE_MB` 一致且足够大。 |
| Nginx 返回 `504 Gateway Time-out` | 长音频请使用异步接口 `POST /v1/asr/tasks`，不要走同步接口。 |
| 服务意外重启后原运行中任务卡死？ | 服务启动时 `TaskStore.mark_interrupted_tasks_failed()` 会自动把重启前未完成的 `queued`/`decoding`/`processing` 任务标记为 `failed`，调用方可安全重试。 |
| 临时目录或 `/dev/shm` 空间占用 | 每次请求/任务结束后均在 `finally` 中自动删除临时文件；若曾发生 `kill -9` 强杀，可重启前清空 `TEMP_DIR`（如 `rm -rf /dev/shm/qwen3_asr_tmp/*`）。 |
