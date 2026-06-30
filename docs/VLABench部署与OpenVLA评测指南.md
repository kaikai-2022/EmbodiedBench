# VLABench 服务端部署与 OpenVLA 评测指南

> 适用对象：在无显示器 (headless) 的 Linux 服务器上部署 VLABench 评测框架，并使用 OpenVLA-7B 模型对 VLABench 任务进行批量评测。
> 参考环境：Linux 6.8.0-63-generic · miniconda · 4×RTX 4090 (24GB) · CUDA 12.9 · Driver 575.64.03。

---

## 1. 总体流程一览

```
┌─────────────────────────────────────────────────────────────┐
│  1. 硬件 & 系统准备                                          │
│  2. 安装 miniconda & 创建独立 conda 环境 (vlabench_openvla)   │
│  3. 克隆 VLABench 仓库并以可编辑模式安装                      │
│  4. 下载 VLABench 资产 (Mesh/URDF/纹理) + 标准评测 episode    │
│  5. 下载 OpenVLA-7B 模型权重 (~15GB)                         │
│  6. 注入 norm_stats 配置文件 (覆盖 5 个原生 primitive 任务)   │
│  7. 设置无头渲染环境变量 (MUJOCO_GL=egl)                      │
│  8. 单条 smoke test 验证                                     │
│  9. 批量评测守护脚本 (auto_run_batch.py) 跑全量               │
│ 10. 收集 metrics.json / detail_info.json 汇总结果             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 硬件与系统要求

| 组件 | 最低 | 推荐 |
|------|------|------|
| GPU | 单卡 16GB (int8/4bit 量化) | 单卡 24GB (RTX 4090) |
| 显存 | OpenVLA-7B 4bit ≈ 6GB，fp16 ≈ 14GB | 24GB 留余量给 MuJoCo+渲染 |
| CPU/内存 | 8 核 / 32GB | 16 核 / 64GB |
| 磁盘 | 模型 15GB + 资产 ~5GB + 轨迹 (可选) | SSD 1TB+ |
| OS | Ubuntu 20.04+ | Ubuntu 22.04 |
| 驱动 | NVIDIA Driver ≥ 535，CUDA ≥ 12.1 | 575+ / CUDA 12.9 |
| 网络 | 拉取 HF 模型 (~15GB) | hf-mirror.com 镜像可显著加速 |

> 多 GPU：本文档默认单卡 `CUDA_VISIBLE_DEVICES=0`。多卡并行可参考 `sh/evaluation/example_multi_gpu_eval.sh`，但 `device_map=auto` 需谨慎处理。

---

## 3. 安装 miniconda

```bash
# 如已安装可跳过
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
$HOME/miniconda/bin/conda init bash
source ~/.bashrc
```

---

## 4. 创建隔离的 conda 环境

> 关键点：OpenVLA 对 `transformers==4.40.1` 和 `tokenizers==0.19.1` 敏感；VLABench 又依赖 `mujoco==3.2.2` / `dm_control==1.0.22`。两者版本不兼容，必须分环境安装。本文档只保留 OpenVLA 评测这一路。

```bash
conda create -n vlabench_openvla python=3.10 -y
conda activate vlabench_openvla
```

### 4.1 PyTorch (CUDA 12.1)

```bash
pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
```

### 4.2 OpenVLA 推理依赖

```bash
pip install transformers==4.40.0 tokenizers==0.19.1 \
    accelerate==0.30.1 bitsandbytes==0.43.1 peft==0.11.1 \
    huggingface_hub timm
```

> 严格锁版本：日志中已观察到 `Expected transformers==4.40.1 and tokenizers==0.19.1` 的 warning。4bit 加载虽可工作但有 inference 回归风险。

### 4.3 VLABench 核心依赖

```bash
pip install mujoco==3.2.2 mujoco-mjx==3.2.2 dm_control==1.0.22 \
    gymnasium==0.29.1 opencv-python mediapy==1.2.0 \
    open3d==0.18.0 numpy==1.25.0 h5py==3.11.0 scipy==1.14.0 \
    tensorflow-datasets==4.9.2 imageio gdown notebook
```

### 4.4 克隆并安装 VLABench

```bash
cd /ssd/liuzirui
git clone https://github.com/OpenMOSS/VLABench.git
cd VLABench
pip install -e .
```

---

## 5. 下载 VLABench 资产与评测配置

### 5.1 资产 (Mesh / 纹理 / 场景)

`scripts/download_assets.py` 会拉取官方 Hugging Face 资产并解压到 `VLABench/assets/`：

```bash
cd /ssd/liuzirui/VLABench
export HF_ENDPOINT="https://hf-mirror.com"     # 国内镜像
python scripts/download_assets.py
```

### 5.2 标准评测 episode (5 个 track)

评测脚本要求 `VLABENCH_ROOT/configs/evaluation/tracks/track_*.json` 已存在。仓库内已自带：

```
VLABench/configs/evaluation/tracks/
├── track_1_in_distribution.json      # 域内任务（最常用）
├── track_2_cross_category.json       # 类别级泛化
├── track_3_common_sense.json         # 常识推理
├── track_4_semantic_instruction.json # 语义指令
└── track_6_unseen_texture.json       # 未见纹理
```

每条 track 是 JSON 字典，键为任务名、值为该任务的随机种子列表 — 决定 episode 的初始状态。

---

## 6. 下载 OpenVLA-7B 模型

### 6.1 用 huggingface-cli (推荐)

```bash
export HF_ENDPOINT="https://hf-mirror.com"
huggingface-cli download openvla/openvla-7b \
    --local-dir /ssd/liuzirui/models/openvla-7b \
    --local-dir-use-symlinks False
```

> 约 15GB，三个 safetensors 分片 + config/tokenizer/自定义 Python 模块 (`modeling_prismatic.py` / `configuration_prismatic.py` / `processing_prismatic.py`)。注意必须把 `modeling_prismatic.py` 等自定义代码一并下载（`trust_remote_code=True` 时 `transformers` 依赖这些文件）。

### 6.2 模型目录结构（最终）

```
/ssd/liuzirui/models/openvla-7b/
├── config.json                            # 关键 norm_stats 注入点
├── added_tokens.json
├── special_tokens_map.json
├── tokenizer.json / tokenizer.model / tokenizer_config.json
├── generation_config.json
├── preprocessor_config.json / processor_config.json
├── modeling_prismatic.py                  # 自定义模型代码
├── configuration_prismatic.py
├── processing_prismatic.py
├── model-00001-of-00003.safetensors      # 6.9 GB
├── model-00002-of-00003.safetensors      # 7.0 GB
├── model-00003-of-00003.safetensors      # 1.2 GB
└── model.safetensors.index.json
```

### 6.3 注入 VLABench norm_stats（关键）

OpenVLA 在反归一化动作时需要每个任务的 `norm_stats`（action mean/std/min/max）。VLABench 提供了覆盖后的 config：

```bash
# VLABench 仓库自带
/ssd/liuzirui/VLABench/VLABench/configs/model/openvla_config.json
```

`evaluate_policy.py` 在初始化 policy 时会**自动**把这份 config 的内容**覆盖**到 `--model_ckpt` 目录的 `config.json`（仅当 `--model_ckpt` 指向本地目录时）。所以模型目录里 `config.json` 会被改写 — 备份一下：

```bash
cp /ssd/liuzirui/models/openvla-7b/config.json{,.bak}
```

---

## 7. 环境变量（headless 渲染）

服务器没有 X11，MuJoCo 必须用 EGL 后端：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_ENDPOINT="https://hf-mirror.com"     # 国内必加
export TRANSFORMERS_OFFLINE=0                  # 首次拉模型=0；之后设 1 走本地
export VLABENCH_ROOT="/ssd/liuzirui/VLABench/VLABench"
export PYTHONPATH="/ssd/liuzirui/VLABench"
export CUDA_VISIBLE_DEVICES=0
```

> 服务器若没装 `libegl1-mesa` / `libgl1-mesa-glx` 等 EGL 依赖，需 `apt install libegl1-mesa libgl1-mesa-glx libgles2-mesa`。

---

## 8. 评测脚本

### 8.1 单条 smoke test

仓库内已带 `run_openvla_eval.sh`，跑 1 episode 用于冒烟测试：

```bash
cd /ssd/liuzirui/VLABench
bash run_openvla_eval.sh
```

命令等价于：

```bash
/ssd/liuzirui/miniconda/envs/vlabench_openvla/bin/python \
    scripts/evaluate_policy.py \
    --policy openvla \
    --model_ckpt /ssd/liuzirui/models/openvla-7b \
    --tasks select_fruit \
    --n-episode 1 \
    --load_in_4bit \
    --device_map auto \
    --save-dir ./openvla_eval_results \
    --metrics success_rate intention_score progress_score
```

`run_openvla_eval.sh` 默认走 `--tasks select_fruit`，但更常用的入口是 `--eval-track track_1_in_distribution`（从 JSON 解析任务列表）。

### 8.2 主要 CLI 参数（evaluate_policy.py）

| 参数 | 含义 | 备注 |
|------|------|------|
| `--policy` | `openvla` / `openpi` / `gr00t` / `random` | 本文 OpenVLA |
| `--model_ckpt` | 本地模型目录 | 必须含 `modeling_prismatic.py` |
| `--lora_ckpt` | LoRA 权重目录 | zero-shot 评测不传 |
| `--eval-track` | 5 个 track 名 | 优先使用 |
| `--tasks` | 自定义任务名列表 | 与 `--eval-track` 互斥倾向 |
| `--n-episode` | 每任务 episode 数 | 建议 5~50 |
| `--load_in_4bit` / `--load_in_8bit` | bitsandbytes 量化 | 4bit 显存约 6GB |
| `--device_map` | `auto` / `cuda:0` / `cpu` | 多卡建议固定单卡 |
| `--save-dir` | 结果根目录 | 自动按 track 分子目录 |
| `--metrics` | `success_rate` / `intention_score` / `progress_score` | 可多选 |
| `--visulization` | 渲染视频 | headless 也可，只是不弹窗 |

### 8.3 批量评测守护脚本（`auto_run_batch.py`）

20 个 episode 跑完约 6+ 小时，单次长跑易被 EGL 偶发崩溃打断。本仓库自带分批守护：

```bash
cd /ssd/liuzirui/VLABench
nohup python auto_run_batch.py > vlabench_batch.log 2>&1 &
```

脚本特性：
- **分 4 批 × 5 episode**：每批约 2 小时，批间 `gc.collect()` + `torch.cuda.empty_cache()` + 5 秒 sleep。
- **白名单 5 个原生任务**：`select_toy` / `select_fruit` / `select_painting` / `select_mahjong` / `select_poker`。OpenVLA-7B 的 norm_stats 只覆盖这 5 个，跑其它任务（如 `insert_flower` / `add_condiment` / `select_book`）会被反归一化钳到 sub-cm 动作，零成功率。
- **单卡约束**：`CUDA_VISIBLE_DEVICES=0`，避免 `device_map=auto` 多卡分片。
- **离线模式**：批内 `TRANSFORMERS_OFFLINE=1`，确保只读本地模型。
- **3 小时超时**：单批 10800s 上限。

日志：`/ssd/liuzirui/vlabench_batch_eval.log`
完成标记：`/ssd/liuzirui/vlabench_eval_done.flag`（如配置）

---

## 9. 结果输出

```
openvla_eval_results/
└── track_1_in_distribution/
    ├── openvla/evaluation_result.json       # 顶层汇总
    └── openvla/<task_name>/
        ├── metrics.json                     # success_rate / intention_score / progress_score
        ├── detail_info.json                 # 每个 episode 的判定
        └── videos/ (optional, --visulization)
```

`auto_run_batch.py` 跑完后会自动 `rglob metrics.json / detail_info.json` 汇总，输出每个任务的 success/total 与平均分。

---

## 10. 已知踩坑

1. **`config.json` 被覆盖** — `evaluate_policy.py` 启动时把 VLABench 的 `openvla_config.json` 写入 `--model_ckpt/config.json`。**必须备份原文件**，否则需要重新下载。
2. **`torch.cuda.OutOfMemoryError` 随机发生** — EGL 渲染在长 episode 序列里偶发累积。`auto_run_batch.py` 已通过分批 + `empty_cache()` 缓解。
3. **`transformers`/`tokenizers` 版本警告** — `Expected transformers==4.40.1 and tokenizers==0.19.1 but got 4.40.0 and 0.19.1`。0.0.1 的差异会偶发 `pixel_values` 维度不匹配（`torch.split` 报错）。可强升 `pip install transformers==4.40.1` 消除。
4. **HEADLESS_X 渲染崩溃** — 必须装 EGL 后端依赖；否则 `MUJOCO_GL=egl` 启动时直接 segfault。
5. **HF 国内拉不动** — `export HF_ENDPOINT=https://hf-mirror.com` 是必需项。
6. **`HUGGING_FACE_HUB_TOKEN`** — 公开模型不需要；若以后评测 finetune 的私有 LoRA 需配。

---

## 11. 一键部署速查

```bash
# 0. 环境
export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=0
conda activate vlabench_openvla

# 1. 拉仓库 & 装
cd /ssd/liuzirui
git clone https://github.com/OpenMOSS/VLABench.git
cd VLABench
pip install -e .

# 2. 资产 + 模型（首次）
python scripts/download_assets.py
huggingface-cli download openvla/openvla-7b \
    --local-dir /ssd/liuzirui/models/openvla-7b --local-dir-use-symlinks False
cp /ssd/liuzirui/models/openvla-7b/config.json{,.bak}

# 3. 无头环境
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export VLABENCH_ROOT="$(pwd)/VLABench" PYTHONPATH="$(pwd)"

# 4. smoke
bash run_openvla_eval.sh

# 5. 全量批量
nohup python auto_run_batch.py > vlabench_batch.log 2>&1 &
tail -f vlabench_batch_eval.log
```

---

## 12. 参考命令 — 评测 OpenPI / Gr00t

文档中暂未在当前服务器跑通，但入口已就绪：

```bash
# openpi: 需起 server (docker + uv venv) 再连 client
python scripts/evaluate_policy.py --policy openpi --host localhost --port 5555 \
    --eval-track track_1_in_distribution --n-episode 20

# gr00t 同上端口协议
python scripts/evaluate_policy.py --policy gr00t --host localhost --port 5555 \
    --eval-track track_1_in_distribution --n-episode 20
```

OpenPI 需先在 `third_party/openpi` 启动 `serve_policy.py` 并加载 LoRA checkpoint（如 `VLABench/pi0-base-vlabench-lora/99999`）。
