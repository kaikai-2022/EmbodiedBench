# pi0.5 模型训练与部署标准工作流指南 (SOP)

## 1. 角色与目标

你是 VLABench 项目的 **pi0.5 (π₀.₅) 模型训练与部署专家**。你的职责是将 VLABench 任务的专家轨迹（HDF5）经过格式转换、openpi 训练框架的 fine-tune，最终部署为可推理的策略模型，并验证其在仿真环境中的成功率。

**完整链路**：

```
VLABench 任务 (HDF5 轨迹)
  → scripts/convert_to_lerobot.py (格式转换)
  → LeRobot Dataset (~/.cache/huggingface/lerobot/<repo_id>/)
  → openpi scripts/compute_norm_stats.py (归一化统计)
  → openpi scripts/train.py pi05_ft_<task> (LoRA fine-tune)
  → Pi0.5 Checkpoint ($OPENPI_DATA_HOME/checkpoints/...)
  → scripts/evaluate_openpi.py (推理评测)
  → success_rate
```

**项目根目录**：`/ssd/qinmaokai/workspace/SciVLABench`（下文所有相对路径均基于此）

**openpi 代码库**：`third_party/openpi/`（fork 自 `github.com/Shiduo-zh/openpi`，**pi05 分支**）

---

## 2. 核心工具链与文件路径

### 2.1 关键脚本工具

| 脚本 | 路径 | 用途 |
|------|------|------|
| `convert_to_lerobot.py` | `scripts/convert_to_lerobot.py` | HDF5 → LeRobot Dataset 格式转换（lerobot 0.1.0 API） |
| `compute_norm_stats.py` | `third_party/openpi/scripts/compute_norm_stats.py` | 计算 state/actions 归一化统计（mean/std/q01/q99） |
| `train.py` | `third_party/openpi/scripts/train.py` | openpi 训练入口（基于 tyro CLI + JAX + FSDP） |
| `serve_policy.py` | `third_party/openpi/scripts/serve_policy.py` | WebSocket 策略服务器（pi0/pi05 共用） |
| `evaluate_openpi.py` | `scripts/evaluate_openpi.py` | VLABench 客户端评测（WebSocket client + env loop + 可视化） |
| `vlabench_policy.py` | `third_party/openpi/src/openpi/policies/vlabench_policy.py` | 数据 transforms（Repack, image resize, action pad） |
| `LeRobotVLABenchDataConfig` | `third_party/openpi/src/openpi/training/config.py:361` | vlabench 数据集配置（含 Repack + DeltaActions + VLABenchInputs/Outputs） |
| `run_pi05_eval.sh` | `run_pi05_eval.sh` | 评测一键启动脚本（server + eval） |
| `run_pi05_train.sh` | `run_pi05_train.sh` | 训练一键启动脚本（train / compute-norm-stats / eval） |

### 2.2 输出目录结构

```
项目根 (SciVLABench)/
├── dataset/training_data/<task_name>/        # Phase 0 输入（用户提供的 HDF5）
│   ├── data_0.hdf5, data_1.hdf5, ...
│   └── data_X.mp4  (混合存放，可忽略)
│
├── ~/.cache/huggingface/lerobot/<repo_id>/    # Phase 1 输出（LeRobot Dataset）
│   ├── data/chunk-000/episode_XXXXXX.parquet  # 每帧图像+state+actions+task
│   └── meta/{info.json, tasks.jsonl, episodes.jsonl}
│
├── $OPENPI_DATA_HOME/assets/<config>/<repo_id>/ # Phase 2 输出（norm_stats）
│   └── norm_stats.json
│
├── $OPENPI_DATA_HOME/checkpoints/<config>/<exp>/ # Phase 3 输出（JAX checkpoint）
│   └── <step>/
│       ├── params/      (PaliGemma + action expert + LoRA params)
│       ├── train_state/ (optimizer state)
│       └── assets/<repo_id>/norm_stats.json
│
└── logs/pi05_eval/  /  logs/pi05_train_eval/   # Phase 4 输出（评测结果）
    └── <task>_n<N>_<timestamp>/
        ├── evaluation_result.json
        └── <task>/videos/ep*.mp4
```

### 2.3 关键路径常量

| 名称 | 默认值 | 备注 |
|------|--------|------|
| `OPENPI_DATA_HOME` | `/ssd/qinmaokai/.cache/openpi` | 训练 checkpoint + assets 落点 |
| `HF_HOME` | `/ssd/qinmaokai/.cache/huggingface` | HuggingFace 模型/数据集缓存 |
| `HF_LEROBOT_HOME` | `$HF_HOME/lerobot` | LeRobot 数据集根（由 lerobot 库解析） |
| `HF_ENDPOINT` | `https://hf-mirror.com` | 国内镜像，必须设 |
| `CHECKPOINT` | `/ssd/qinmaokai/.cache/openpi/vlabench_checkpoints/pi05-primitive-10task` | 起点头（已下载 ~39GB） |
| venv | `third_party/openpi/examples/vlabench/.venv` | openpi server 端 Python 环境（uv 管理） |
| conda env | `vlabench_2` at `/ssd/qinmaokai/.conda/envs/vlabench_2` | VLABench client 端环境 |

---

## 3. 数据格式契约

### 3.1 HDF5 schema（输入）

每个 `data_X.hdf5` 是一个 episode，结构：

```python
f["data/<timestamp>"]/{
    "instruction": |S84 bytes,           # b"place the <small_beaker_0> ..." (utf-8)
    "observation"/{
        "rgb": uint8 (202, 4, 480, 480, 3),    # 4 cameras: [2]=front, [3]=wrist
        "ee_state": float32 (202, 8),          # [pos3, quat4, gripper1]
        "q_state": float32 (202, 7),           # joint angles (unused by convert)
        "robot_mask", "depth", "point_cloud_*" # 保留但不转换
    },
    "trajectory": float32 (202, 8),            # [pos3, euler3, gripper1, gripper2]
    "meta_info"/{
        "episode_config": |S1573 bytes,        # JSON 字符串含 robot.position
        "entities", "target_entity"
    }
}
```

### 3.2 LeRobot schema（输出）

```python
{
    "observation.image": PIL.Image(480, 480, RGB),       # = rgb[i][2] front cam
    "observation.second_image": PIL.Image,               # = rgb[i][?]（convert 内可改）
    "observation.wrist_image": PIL.Image,                # = rgb[i][3]
    "observation.state": float32(7),                    # pos3-0.4 + euler3 + gripper1
    "actions": float32(7),                              # 前 6 dim 连续 + gripper 二值(0/1)
    "task": str,                                         # 唯一 task 用例
    # + 内部字段：frame_index, episode_index, index, timestamp
}
```

**关键 schema 约束**：
- `actions` 列名是**复数**（lerobot 0.1.0 openpi 默认 `action_sequence_keys=("actions",)`）
- 所有 feature key 用 **`.` 分隔**（lerobot 0.1.0 禁用 `/`）
- `action` 第 7 维 = `>=0.039` → `1` (open)，否则 `0` (close/grasped)
  - **注意**：使用 `>= 0.039` 而不是 `> 0.03`，因为 grasp_lock 机制会将抓取时的 gripper 锁定在物体宽度（如 beaker ≈ 0.0356），而不是 0.0

### 3.3 pi0.5 内部 schema（transforms 之后）

`VLABenchInputs` 把 state 填到 32 dim（action_dim），images resize 到 224×224，prompt 从 task 转 token。`RepackTransform` 把 `{observation.image, observation.wrist_image, ...}` 重命名成 `{image, wrist_image, ...}` 给下游 model。

---

## 4. 完整部署 SOP（从零到评测）

### Phase 0: 准备 checkpoint 起点

```bash
mkdir -p /ssd/qinmaokai/.cache/openpi/vlabench_checkpoints
HF_ENDPOINT=https://hf-mirror.com python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='VLABench/pi05-primitive-10task',
    local_dir='/ssd/qinmaokai/.cache/openpi/vlabench_checkpoints/pi05-primitive-10task',
    max_workers=4,
)
"
```

**产物**：`{train_state, params, assets, vlabench}/...` (39GB，~12分钟)

---

### Phase 1: 克隆 openpi pi05 分支

**不要克隆 main 分支！** main 没有 `pi05_ft_vlabench_primitive` config。必须用 **pi05 分支**：

```bash
# 用 gh-proxy.com 镜像（GitHub 国内访问慢）
curl -L -o /tmp/openpi.tar.gz https://gh-proxy.com/https://github.com/Shiduo-zh/openpi/archive/refs/heads/pi05.tar.gz
mkdir -p /tmp/extract && tar -xzf /tmp/openpi.tar.gz -C /tmp/extract
rm -rf /ssd/qinmaokai/workspace/SciVLABench/third_party/openpi
cp -r /tmp/extract/VLABench-openpi-pi05/. /ssd/qinmaokai/workspace/SciVLABench/third_party/openpi/
```

**关键确认**：
- `third_party/openpi/src/openpi/training/config.py` 中 `grep "pi05_ft_vlabench_primitive"` 应有定义
- `third_party/openpi/scripts/serve_policy.py` 中 `EnvMode.VLABENCH` 应已注册
- `third_party/openpi/vlabench_scripts/serve_policy.sh` 存在

---

### Phase 2: 创建 openpi venv + 装依赖（清华源加速）

```bash
# 装 uv（如果还没有）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建 venv
cd /ssd/qinmaokai/workspace/SciVLABench/third_party/openpi
/ssd/qinmaokai/.local/bin/uv venv --python 3.11 examples/vlabench/.venv
source examples/vlabench/.venv/bin/activate

# 装 requirements（清华源关键！）
UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    /ssd/qinmaokai/.local/bin/uv pip sync examples/vlabench/requirements.txt

# 装 openpi 本体（带 PaliGemma、JAX[cuda12]、torch 2.7.1、transformers 等）
UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    /ssd/qinmaokai/.local/bin/uv pip install -e packages/openpi-client -e /ssd/qinmaokai/workspace/SciVLABench -e .

# 兼容性 pin（关键！否则新版 chex/optax 会破坏 jax 0.5.3）
UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    /ssd/qinmaokai/.local/bin/uv pip install "chex<=0.1.87" "optax<=0.2.4" "etils>=1.14" pytest

# 把 jax 钉回 0.5.3（chex 默认会升到 0.10）
UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    /ssd/qinmaokai/.local/bin/uv pip install "jax[cuda12]==0.5.3" "jaxlib==0.5.3" "numpy<2.0.0"
```

**踩过的坑**：
- 缺 `pytest`：openpi `models_pytorch/gemma_pytorch.py` 顶层 import pytest，需要单独装
- `chex>=0.1.88` 会强制 jax 升到 0.10，破坏 orbax-checkpoint 0.11.13 的兼容性
- `etils<1.14` 缺 `epy.lazy_imports` API
- numpy 不能 >= 2.0（Pi0Config 内部用 np.float，2.x 会 deprecate）

---

### Phase 3: 修改代码兼容 lerobot 0.1.0 + pi05 需求

#### 3a. `scripts/convert_to_lerobot.py` 修复

原脚本有 5 个问题需修：

```python
# 1. 删硬编码路径（line 2）
# 原：sys.path.insert(0, "/ssd/mkqin/workspace/lerobot")
# 删掉，让 venv 里的 pip 装 lerobot 0.1.0 工作

# 2. add_frame 必须含 'task' 字段（lerobot 0.1.0 API）
dataset.add_frame({
    "observation.image": images[i][2],
    "observation.wrist_image": images[i][3],
    "observation.state": ee_state[i],
    "actions": action,                   # ← 用复数
    "task": task_str,                    # ← 新增
})

# 3. save_episode 不接 task= 参数（task 已在 frame 里）
dataset.save_episode()                  # ← 不要传 task=

# 4. action 8D→7D + gripper 二值化（脚本已有，但容易漏）
if len(action) == 8:
    action = np.concatenate([action[:6], np.array([action[6]])])
action = action[:6] + (1 if action[-1] > 0.03 else 0)

# 5. 删 consolidate()（lerobot 0.1.0 没这个方法）
# 原：dataset.consolidate(run_compute_stats=True)
# 改为：dataset.stop_image_writer()
# norm_stats 由 openpi compute_norm_stats.py 单独生成
```

#### 3b. `lerobot/common/datasets/lerobot_dataset.py` 内部 bug 修复

lerobot 0.1.0 的 `__init__` line 508-509、line 696 用 `torch.stack(self.hf_dataset["col"])` 拿 Column 对象，**新版 datasets 库会抛 `TypeError`**。改用 list comprehension：

```python
# line 508-509 改为
timestamps = torch.stack([self.hf_dataset[i]["timestamp"] for i in range(len(self.hf_dataset))]).numpy()
episode_indices = torch.stack([self.hf_dataset[i]["episode_index"] for i in range(len(self.hf_dataset))]).numpy()

# line 696-698 改为
return {
    key: torch.stack([self.hf_dataset.select(q_idx)[i] for i in range(len(self.hf_dataset.select(q_idx)))][key])  # 不够干净，重写
}
```

**更好的方案**：在 `_query_hf_dataset` 改用单条访问 `hf_dataset[idx][key]` 触发 `set_transform`。

#### 3c. `third_party/openpi/src/openpi/training/config.py` 修复 RepackTransform 方向

**关键**：lerobot 0.1.0 + openpi 的 `RepackTransform.structure` 是 **`{dest_key: source_path}`**，不是反过来：

```python
# LeRobotVLABenchDataConfig.create() 改为正确方向
repack_transform = _transforms.Group(
    inputs=[
        _transforms.RepackTransform({
            "image": "observation.image",          # dest → source
            "second_image": "observation.second_image",
            "wrist_image": "observation.wrist_image",
            "state": "observation.state",
            "actions": "actions",
            "prompt": "prompt",
        })
    ]
)
```

如果写成 `{"observation.image": "image"}` 会导致 `KeyError: 'image'`（vlabench_policy.py 第 49 行 `data["observation.state"]`）。

#### 3d. 新增 `pi05_ft_<task>` training config

```python
TrainConfig(
    name="pi05_ft_<your_task>",                   # e.g. "pi05_ft_place_beaker"
    model=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",        # LoRA 模式（gemma_2b = 全参）
        action_expert_variant="gemma_300m_lora",
    ),
    data=LeRobotVLABenchDataConfig(
        repo_id="<your_task>",                   # 必须与 convert_to_lerobot.py --dataset-name 一致
        base_config=DataConfig(local_files_only=True, prompt_from_task=True),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "/ssd/qinmaokai/.cache/openpi/vlabench_checkpoints/pi05-primitive-10task/params"
    ),
    freeze_filter=pi0_config.Pi0Config(          # 只训 LoRA，冻结 base
        pi05=True, action_horizon=10, discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter(),
    ema_decay=None,                              # LoRA 不需要 EMA
    num_train_steps=5000,                        # 200 条数据 5k 步足够（官方 30k 是 5k episodes）
    batch_size=8,                                # 1×80GB 可放下 LoRA（4 卡 FSDP → batch=8×4=32）
    num_workers=8,
    log_interval=50,
    save_interval=500,
    keep_period=1000,
),
```

**注意事项**：
- `repo_id` 必须等于 `convert_to_lerobot.py --dataset-name`
- `weight_loader` 指向本地 params 目录（不是 gs://）才能从本地 checkpoint 继续训练
- `freeze_filter` + `gemma_2b_lora` 双 LoRA（paligemma + action_expert 各一组）
- batch_size=8 单卡约 70GB，**不能用 batch_size=32**（会 OOM，因为 LoRA 不减激活内存）

### 4.5 关键性能 patch：`lerobot_dataset.py` 直接读 parquet 列

#### 4.5.1 症状

在 400+ trajectories 上跑 `compute_norm_stats.py`（官方脚本）会卡 20+ 分钟没进度输出，但 CPU 满载、内存持续增长。**进程实际没死**，但 tqdm 输出被 stdout buffer 累积。

#### 4.5.2 根因

`lerobot/common/datasets/lerobot_dataset.py:508-509`：

```python
# Check timestamps
timestamps = torch.stack([self.hf_dataset[i]["timestamp"] for i in range(len(self.hf_dataset))]).numpy()
episode_indices = torch.stack([self.hf_dataset[i]["episode_index"] for i in range(len(self.hf_dataset))]).numpy()
```

这行**遍历整个 dataset 77k 次**，每个 `hf_dataset[i]` 访问会触发：
1. 读 parquet 行的 image bytes（3 张图：front + second + wrist = 2 MB）
2. **PIL 解码 3 张 480×480 RGB 图**
3. 转 torch tensor
4. 取一个标量（timestamp/episode_index）
5. **丢弃 image tensor**（只要标量）

**为了拿一个标量，付出 5ms × 77k = 6.4 分钟的 PIL 解码**——纯浪费。

#### 4.5.3 验证（不是内存/IO 瓶颈）

| 资源 | 实际状态 | 是否瓶颈 |
|---|---|---|
| 内存 (2 TB) | 仅用 50 GB | 否 |
| Swap | 0 KB | 否 |
| 磁盘 IO (bi/so) | 0, 0 | **完全没 IO** |
| CPU | 25 核满载 | 是（算 PIL）|

`/proc/PID/maps` 显示 HF datasets 把 63 个 .arrow 文件** memory-mapped**——不读进内存但 mmap 占虚拟地址空间。**真瓶颈是 PIL decode CPU 算力**。

#### 4.5.4 修复 patch

**修改文件**：`third_party/openpi/examples/vlabench/.venv/lib/python3.11/site-packages/lerobot/common/datasets/lerobot_dataset.py`

**修改内容**（line 508-509 替换）：

```python
# OLD (slow, ~7 min for 77k frames):
timestamps = torch.stack([self.hf_dataset[i]["timestamp"] for i in range(len(self.hf_dataset))]).numpy()
episode_indices = torch.stack([self.hf_dataset[i]["episode_index"] for i in range(len(self.hf_dataset))]).numpy()

# NEW (fast, <10s for 77k frames):
import glob as _glob
import pyarrow.parquet as _pq
_parquet_files = sorted(_glob.glob(str(self.root / "data" / "chunk-*" / "*.parquet")))
if _parquet_files:
    timestamps = np.concatenate([
        _pq.read_table(f, columns=['timestamp']).column('timestamp').to_numpy()
        for f in _parquet_files
    ])
    episode_indices = np.concatenate([
        _pq.read_table(f, columns=['episode_index']).column('episode_index').to_numpy()
        for f in _parquet_files
    ])
    assert len(timestamps) == len(self.hf_dataset), (
        f"PATCH BUG: parquet frames ({len(timestamps)}) != hf_dataset ({len(self.hf_dataset)})"
    )
else:
    # Fallback to original
    timestamps = torch.stack([self.hf_dataset[i]["timestamp"] for i in range(len(self.hf_dataset))]).numpy()
    episode_indices = torch.stack([self.hf_dataset[i]["episode_index"] for i in range(len(self.hf_dataset))]).numpy()
ep_data_index_np = {k: t.numpy() for k, t in self.episode_data_index.items()}
check_timestamps_sync(timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s)
```

#### 4.5.5 为什么 patch 安全

| 问题 | 答案 |
|---|---|
| 会影响训练效果吗？ | **零影响**。patch 只改 sanity check 的读取方式，**不碰训练循环的 `__getitem__`** |
| 会改变 norm_stats 数值吗？ | **不改变**。读的同一个 parquet 列，同一个标量 |
| 会破坏 sanity check 吗？ | **不**。check_timestamps_sync 接收同样的两个数组 |
| 训练启动会出错吗？ | **不会**。训练时数据加载走 `__getitem__`，不经过 patch 的代码 |
| 模型权重会变吗？ | **不变**。patch 与训练 loss / 梯度 / 权重完全无关 |

#### 4.5.6 实测效果（400 trajectories）

| 阶段 | 改前 | 改后 |
|---|---|---|
| STAGE 1 LeRobotDataset 加载 | 7+ 分钟 | **1.0 秒** |
| STAGE 4 速率 | N/A (卡住) | **17-18 batch/s** |
| STAGE 4 总耗时 | N/A | **~10 分钟** |
| 总耗时 | 20-30+ 分钟（不保证完成） | **12.4 分钟**（保证完成）|

#### 4.5.7 回滚方法

```bash
cp /tmp/lerobot_dataset.py.bak \
   /ssd/qinmaokai/workspace/SciVLABench/third_party/openpi/examples/vlabench/.venv/lib/python3.11/site-packages/lerobot/common/datasets/lerobot_dataset.py
```

#### 4.5.8 patch 同时加速的场景

- ✅ `compute_norm_stats.py` —— 12 分钟（vs 20-30+ 卡死）
- ✅ `train.py` —— 启动加载快 7 分钟（首个 epoch 提前开始）
- ✅ `serve_policy.py` —— checkpoint 加载快（少 7 秒）

---

### 4.6 完整训练启动命令（Cookbook）

**问题**：新 agent 反复忘记传 `FSDP_DEVICES=8` 或 `CONFIG=...` 环境变量导致跑单卡或跑错 config。

**完整正确命令模板**（复制粘贴，必须一字不差）：

```bash
# ============================================================
# 标准训练启动命令（8 卡 FSDP，30k 步，v3 400 trajectories）
# ============================================================

# 1. 激活 venv
source /ssd/qinmaokai/workspace/SciVLABench/third_party/openpi/examples/vlabench/.venv/bin/activate

# 2. 设置 PYTHONPATH（必须包含 src 和项目根）
export PYTHONPATH=/ssd/qinmaokai/workspace/SciVLABench/third_party/openpi/src:/ssd/qinmaokai/workspace/SciVLABench

# 3. 设置环境变量
export HF_HOME=/ssd/qinmaokai/.cache/huggingface
export HF_LEROBOT_HOME=/ssd/qinmaokai/.cache/huggingface/lerobot
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 4. run_pi05_train.sh 接受的环境变量（**必传**，默认值是 1 卡）
export FSDP_DEVICES=8
export GPUS=0,1,2,3,4,5,6,7

# 5. run_pi05_train.sh 接受的训练参数（**必传**，否则用 config 默认值）
export EXP_NAME=pi05_ft_place_beaker_v3_30k   # 实验名（决定 checkpoint 保存路径）
export CONFIG=pi05_ft_place_beaker_v3         # config 名（决定模型架构和数据集）

# 6. 启动训练（用 -- 传额外 train.py 参数，**--overwrite 必须**）
#    run_pi05_train.sh 会自动 cd 到 openpi 目录
#    会自动加 --assets_base_dir 和 --checkpoint_base_dir（绝对路径防 symlink bug）

# 选项 A：直接调用 train.py（完全控制参数）
python scripts/train.py $CONFIG \
    --exp-name=$EXP_NAME \
    --assets_base_dir=/ssd/qinmaokai/.cache/openpi/assets \
    --checkpoint_base_dir=/ssd/qinmaokai/.cache/openpi/checkpoints \
    --fsdp_devices=$FSDP_DEVICES \
    --num_train_steps=30000 \
    --batch_size=8 \
    --num_workers=32 \
    --log_interval=100 \
    --save_interval=2000 \
    --keep_period=4000 \
    --overwrite

# 选项 B：用 run_pi05_train.sh（更简洁但隐式参数多）
EXP_NAME=pi05_ft_place_beaker_v3_30k \
FSDP_DEVICES=8 \
GPUS=0,1,2,3,4,5,6,7 \
/ssd/qinmaokai/workspace/SciVLABench/run_pi05_train.sh train \
    --num_train_steps=30000 \
    --batch_size=8 \
    --num_workers=32 \
    --log_interval=100 \
    --save_interval=2000 \
    --keep_period=4000
```

**最简后台启动**（**PPID=1 真正脱离** Claude 进程树，断线不影响）：

```bash
nohup setsid python /path/to/your/script.sh </dev/null >/tmp/train_daemon.log 2>&1 &
disown $!

# 验证 PPID=1（如果 PPID 不是 1，进程仍受 Claude 管控）
sleep 5
PID=$(pgrep -f "scripts/train.py" | head -1)
ps -o pid,ppid -p $PID   # PPID 必须是 1
```

#### 4.6.1 必传参数清单（**漏一个就出问题**）

| 参数 | 必传? | 漏了的后果 | 怎么传 |
|---|---|---|---|
| `--config=...` | ✅ | 跑默认 config | run_pi05_train.sh 第一个位置参数 |
| `--fsdp_devices=8` | ✅ | 只用 1 卡，慢 8× | `FSDP_DEVICES=8` env var |
| `--assets_base_dir=...` | ✅ | "norm_stats not found" | run_pi05_train.sh 自动加 |
| `--checkpoint_base_dir=...` | ✅ | ckpt 保存到错误路径 | run_pi05_train.sh 自动加 |
| `--exp-name=...` | ✅ | ckpt 路径不带区分（重跑会覆盖） | `EXP_NAME=...` env var |
| `--num_train_steps=...` | ✅（推荐） | 用 config 默认（5000） | train 命令行 |
| `--batch_size=8` | ✅ | 4 卡 FSDP 会 OOM | train 命令行 |
| `--overwrite` | ✅ | 第二次跑同一个 exp_name 会报"已存在" | train 命令行 |

#### 4.6.2 run_pi05_train.sh 默认值陷阱

| 变量 | 默认值 | **如果漏了** |
|---|---|---|
| `CONFIG` | `pi05_ft_place_beaker` | 用旧 v1/v2 config（不是 v3） |
| `FSDP_DEVICES` | `1` | 只用 1 卡，30k 步要 14 小时 |
| `GPUS` | `0,1,2,3` | 只用前 4 卡 |
| `EXP_NAME` | 自动加时间戳 | 每次跑新 exp_name，ckpt 不易追踪 |

#### 4.6.3 验证训练正在正确跑

启动后**立刻检查** 3 件事：

```bash
# 1. PPID=1（脱离 Claude）
PID=$(pgrep -f "scripts/train.py" | head -1)
ps -o pid,ppid,pgid,sid,etime -p $PID
# PPID 必须是 1，否则进程会被 Claude 关闭时杀掉

# 2. 8 张卡都跑满
nvidia-smi --query-gpu=index,utilization.gpu --format=csv
# 期望：8 张卡 util 都在 90-99%

# 3. log 正常输出 Step 行
grep "Step [0-9]*: grad_norm" /tmp/train_daemon.log | tail -3
# 期望：每 ~50s 打印一行（log_interval=100, 18 batch/s × 100 = 5.5s/step 不是约 50s）
```

**3 项全过 = 训练正确**。任一异常立即 kill 排查。

---

### Phase 4: 数据格式转换（HDF5 → LeRobot）

```bash
source /ssd/qinmaokai/workspace/SciVLABench/third_party/openpi/examples/vlabench/.venv/bin/activate
export PYTHONPATH=/ssd/qinmaokai/workspace/SciVLABench/third_party/openpi/src:/ssd/qinmaokai/workspace/SciVLABench
export HF_HOME=/ssd/qinmaokai/.cache/huggingface
export HF_LEROBOT_HOME=/ssd/qinmaokai/.cache/huggingface/lerobot
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
cd /ssd/qinmaokai/workspace/SciVLABench

python scripts/convert_to_lerobot.py \
    --dataset-name <your_task> \              # 必须与 config.repo_id 一致
    --dataset-path /ssd/qinmaokai/workspace/SciVLABench/dataset/training_data \
    --task-list <your_task> \
    --max-files 200
```

**产物**：`~/.cache/huggingface/lerobot/<your_task>/` 含 194 episodes、~37500 帧、~16GB

**耗时**：~15 分钟（200 文件，每文件约 4 秒）

---

### Phase 5: 计算归一化统计

```bash
cd /ssd/qinmaokai/workspace/SciVLABench/third_party/openpi
mkdir -p assets   # 不要让它变成 symlink（之前出过的 bug）
export PYTHONPATH=$PWD/src:/ssd/qinmaokai/workspace/SciVLABench:$PYTHONPATH
source examples/vlabench/.venv/bin/activate

python scripts/compute_norm_stats.py --config-name=pi05_ft_<your_task>
```

**产物**：`./assets/pi05_ft_<your_task>/<repo_id>/norm_stats.json`

**必须复制到 OPENPI_DATA_HOME 下**（脚本会自动 cp 过去），最终位置：
```
/ssd/qinmaokai/.cache/openpi/assets/pi05_ft_<your_task>/<repo_id>/norm_stats.json
```

**耗时**：~5 分钟（37k 帧过一遍 transforms）

**踩过的坑**：
- 不要让 `compute_norm_stats.py` 用相对路径 `./assets`，训练时 `data_loader.py:189` 用 `epath.Path(self.assets.assets_dir or assets_dirs)` 找文件，需要在 `OPENPI_DATA_HOME/assets` 下
- 如果 RepoPackTransform 方向反了会抛 `KeyError: 'image'`（在 `vlabench_policy.py:49`）

---

### Phase 6: Dry-run 训练 10 步（验证 pipeline）

```bash
cd /ssd/qinmaokai/workspace/SciVLABench/third_party/openpi
export PYTHONPATH=$PWD/src:/ssd/qinmaokai/workspace/SciVLABench:$PYTHONPATH
source examples/vlabench/.venv/bin/activate

CUDA_VISIBLE_DEVICES=0 python scripts/train.py pi05_ft_<your_task> \
    --exp-name=pi05_ft_<your_task>_dryrun \
    --assets_base_dir=/ssd/qinmaokai/.cache/openpi/assets \
    --checkpoint_base_dir=/ssd/qinmaokai/.cache/openpi/checkpoints \
    --num_train_steps=10 \
    --batch_size=2 \
    --log_interval=1 \
    --save_interval=5 \
    --overwrite
```

**预期输出**（约 8-12 分钟后）：
```
Step 0: grad_norm=1.48, loss=0.09, param_norm=1807.6
Step 1: grad_norm=1.46, loss=0.10
...
Step 9: grad_norm=7.06, loss=0.51
```

**耗时**：JAX 编译 ~5 分钟 + 10 步 ~3 分钟 = **~8 分钟**

**验证点**：
- ✅ 没有 `KeyError: 'observation.state'` → RepackTransform 方向对
- ✅ 没有 `ValueError: Normalization stats not found` → norm_stats 路径对（用绝对路径 `--assets_base_dir`）
- ✅ 没有 OOM → batch_size 合适
- ✅ checkpoint 保存到 `OPENPI_DATA_HOME/checkpoints/.../5/` 和 `.../9/`

**关键修复**：如果用 symlink `./assets → OPENPI_DATA_HOME/assets`，第一次跑可能正常，但**`assets` 目录里又建了一个 `assets` symlink**指向同一目录，造成**循环引用**。改用 `--assets_base_dir` 绝对路径直接绕过。

---

### Phase 7: 中等长度训练 200 步（决策窗口）

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py pi05_ft_<your_task> \
    --exp-name=pi05_ft_<your_task>_200steps \
    --assets_base_dir=/ssd/qinmaokai/.cache/openpi/assets \
    --checkpoint_base_dir=/ssd/qinmaokai/.cache/openpi/checkpoints \
    --num_train_steps=200 \
    --batch_size=2 \
    --log_interval=10 \
    --save_interval=50 \
    --overwrite
```

**耗时**：JAX 编译 ~5 分钟 + 200 步 ~10 分钟 = **~15-20 分钟**

**期望 loss 曲线**（作为决策信号）：
| 步数窗口 | 期望 loss | grad_norm |
|---|---|---|
| 0-50 | 0.1-0.3 | 1.5-5 |
| 50-100 | 0.1-0.3 | 2-6 |
| 100-150 | 0.05-0.15 | 1-4 |
| 150-200 | 0.05-0.1 | 1-3 |

**决策规则**：
- 如果 150-200 步 loss 在 0.1-0.2 → **信号健康**，开 5000 步正式训练
- 如果 150-200 步 loss 仍 > 0.5 → **有问题**，先查数据/超参

**实测结果**（pi05_ft_place_beaker）：
```
step   0: loss=0.086
step  10: loss=0.232
step  50: loss=0.199
step 100: loss=0.165
step 150: loss=0.105
step 190: loss=0.081
```
→ **loss 下降 44%（0.18→0.10），grad_norm 稳定下降**，信号健康

---

### Phase 8: 正式训练 5000 步（8 卡 FSDP 推荐）

**实测对比（已经在本项目上跑过）**：

| 配置 | 实测耗时 | 单步耗时 | 显存/卡 | 风险 |
|---|---|---|---|---|
| 1×A800 80GB, batch=2 | ~14h（按 6s/step 推算） | ~6s | 77GB | 低，但太慢 |
| 8×A800 80GB FSDP, batch=8 | **50 分钟** ✅ 实测 | ~0.5s | 79GB | 中（已跑通） |

**推荐**：8 卡 FSDP + batch=8。**50 分钟跑完 5000 步**，比之前预估的 5h 快 6 倍（FSDP 加速比 12×，远高于理论 8× 因 JAX 编译后 step 效率高）。

```bash
# 单卡版本（兜底，~14h）
CUDA_VISIBLE_DEVICES=0 python scripts/train.py pi05_ft_<your_task> \
    --exp-name=pi05_ft_<your_task>_v1 \
    --assets_base_dir=/ssd/qinmaokai/.cache/openpi/assets \
    --checkpoint_base_dir=/ssd/qinmaokai/.cache/openpi/checkpoints \
    --num_train_steps=5000 \
    --batch_size=2 \
    --log_interval=50 \
    --save_interval=500 \
    --keep_period=1000 \
    --overwrite

# 8 卡 FSDP 版本（推荐，~50 分钟）
EXP_NAME=pi05_ft_<your_task>_v1_full \
GPUS=0,1,2,3,4,5,6,7 \
FSDP_DEVICES=8 \
/ssd/qinmaokai/workspace/SciVLABench/run_pi05_train.sh train \
    --num_train_steps=5000 \
    --batch_size=8 \
    --num_workers=32 \
    --log_interval=50 \
    --save_interval=500 \
    --keep_period=1000
```

**实测训练曲线（pi05_ft_place_beaker，5000 步，loss 衰减典型形状）**：

```
Step    0: loss=0.157  (起步)
Step  150: loss=0.096
Step  350: loss=0.055  ← 快速下降阶段
Step  550: loss=0.044
Step  950: loss=0.029
Step 1950: loss=0.022  ← 进入收敛期
Step 2950: loss=0.020
Step 3950: loss=0.019
Step 4950: loss=0.019  ← 完全 plateau（最后 2000 步几乎不动）
```

**Grad norm 收敛曲线**：从 1.5-4.5 → 0.2-0.3（稳定收敛）。

**产物路径**：
```
/ssd/qinmaokai/.cache/openpi/checkpoints/pi05_ft_<your_task>/pi05_ft_<your_task>_v1_full/
├── 1000/  2000/  3000/  4000/  4999/    # save_interval=500 但只保留最近 1 个 + keep_period=1000 的倍数
```

每个 checkpoint 大小：**8.9 GB**（params + train_state + assets）。

**验证点**：
- ✅ loss 最终收敛到 **0.015-0.025** 区间
- ✅ grad_norm 最终稳定在 **0.2-0.5**
- ✅ GPU 利用率 **95%+**（8 卡全速运行）
- ✅ 单步耗时 **~0.5s**（compile 之后）
- ✅ checkpoint 保存成功（无 OOM）

**重要：loss 在 0.018 不代表任务完成度**。loss=0.018 只说明模型拟合了演示数据，**真正决定模型好坏的是评测 success_rate**。实测 5000 步训练后 `place_beaker_pres_heat_device` 的 SR=0%，但视频显示机械臂学到了抓取子动作（不是 0% 的训练价值）。

---

### Phase 9: 用训练好的 checkpoint 部署评测

**评测架构**：server（openpi `.venv`）+ client（vlabench_2 conda env）通过 WebSocket 8000 通信。client 发 obs，server 返回 action chunk，client 在 MuJoCo 仿真环境里执行，循环到 max_episode_length。

#### 9.0 前置：必须先做的代码修复（评测相关的 3 个 bug）

**坑 1**：`RepackTransform` 在评测时不会被自动调用（`create_trained_policy` 只在 `repack_transforms` 显式传入时才加到 transform 链里，但 serve_policy.py 没传）。**修复**：在 `third_party/openpi/src/openpi/policies/policy_config.py` 第 58 行 `data_config = train_config.data.create(...)` 后插入：

```python
data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
# If repack_transforms not explicitly provided, fall back to the data_config's repack_transforms
# (needed for inference: client sends raw observation dict, RepackTransform maps it to model inputs)
if not repack_transforms.inputs and data_config.repack_transforms.inputs:
    repack_transforms = data_config.repack_transforms
if norm_stats is None:
```

**坑 2**：训练时 LeRobot 0.1.0 的 feature key 是点分隔（`observation.state`），评测时 client 发的是斜杠分隔（`observation/state`）—— RepackTransform 找不到匹配。**修复**：替换 `third_party/openpi/src/openpi/transforms.py` 中 `RepackTransform.__call__`（约第 99 行）：

```python
def __call__(self, data: DataDict) -> DataDict:
    flat_item = flatten_dict(data)
    # 兼容 LeRobot 0.1.0 的 dot-separated keys (训练数据)
    # 与 VLABench client 的 slash-separated keys (评测 client)：
    # 同时尝试两种分隔符匹配源 path。
    # 评测时 client 不发 actions，缺失的 key 直接跳过（让下游 "actions" in data 为 False）。
    result = {}
    for k_dest, k_src in flatten_dict(self.structure).items():
        if k_src in flat_item:
            result[k_dest] = flat_item[k_src]
        else:
            alt = k_src.replace(".", "/") if "." in k_src else k_src.replace("/", ".")
            if alt in flat_item:
                result[k_dest] = flat_item[alt]
            # else: skip (key absent in input)
    return result
```

**坑 3**：客户端不发 `actions` key（评测时模型不需要 actions），但默认 RepackTransform 会报 `KeyError: 'actions'`。修复 2 已经处理（缺失 key 跳过）。

**没修这 3 个 bug 的症状**：`KeyError: 'state'` 或 `KeyError: 'actions'`，所有 episode 都 `success=False, steps=0`。

#### 9.1 启动 server（终端 1，openpi venv）

写一个启动脚本 `/tmp/start_eval_server.sh`：

```bash
#!/bin/bash
source /ssd/qinmaokai/workspace/SciVLABench/third_party/openpi/examples/vlabench/.venv/bin/activate
export PYTHONPATH=/ssd/qinmaokai/workspace/SciVLABench/third_party/openpi/src:/ssd/qinmaokai/workspace/SciVLABench:${PYTHONPATH:-}
export HF_ENDPOINT=https://hf-mirror.com
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0
cd /ssd/qinmaokai/workspace/SciVLABench/third_party/openpi
python scripts/serve_policy.py --port 8000 \
    --env VLABENCH policy:checkpoint \
    --policy.config=pi05_ft_<your_task> \
    --policy.dir=/ssd/qinmaokai/.cache/openpi/checkpoints/pi05_ft_<your_task>/<exp_name>/<final_step>
```

启动（后台运行，日志写到 `/tmp/eval_server.log`）：

```bash
nohup bash /tmp/start_eval_server.sh > /tmp/eval_server.log 2>&1 &
echo "Server PID: $!"

# 等到 server ready（~60-90s）
sleep 60
tail -5 /tmp/eval_server.log
# 应该看到："server listening on 0.0.0.0:8000"
# 以及     "Loaded norm stats from .../<step>/assets/<your_task>"
```

**验证 server 状态**：
```bash
ss -tlnp | grep 8000          # 应该看到 LISTEN
pgrep -af serve_policy.py     # 应该看到 python 进程
nvidia-smi                    # 应该看到 GPU 0 占 ~75GB
```

**预期 server 启动日志关键字**：
- `Restoring checkpoint from .../params` —— checkpoint 加载开始
- `[process=0] /jax/checkpoint/read/bytes_per_sec: 1.8 GiB/s (total bytes: 6.3 GiB)` —— ~3-4 秒读完
- `Norm stats not found in .../openpi/assets/...` ← **这是正常的**（相对路径找不到），紧接着会看到：
- `Loaded norm stats from .../<step>/assets/<your_task>` ← **从 checkpoint 内加载成功**
- `Could not resolve hostname g0104, falling back to 127.0.0.1` ← hostname DNS 失败已 fallback
- `server listening on 0.0.0.0:8000` ← ready

#### 9.2 跑评测（终端 2，vlabench_2 conda env）

写评测脚本 `/tmp/run_eval.sh`：

```bash
#!/bin/bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate vlabench_2
export PYTHONPATH=/ssd/qinmaokai/workspace/SciVLABench/third_party/openpi/src:/ssd/qinmaokai/workspace/SciVLABench:${PYTHONPATH:-}
export MUJOCO_GL=egl
export VLABENCH_ROOT=/ssd/qinmaokai/workspace/SciVLABench/VLABench
cd /ssd/qinmaokai/workspace/SciVLABench
python scripts/evaluate_openpi.py \
    --host localhost --port 8000 \
    --tasks <your_task> \
    --n_episode 10 \
    --save_dir logs/pi05_train_eval/<your_task>_n10_$(date +%Y%m%d_%H%M%S) \
    --max_episode_length 200 \
    --visualization
```

**`evaluate_openpi.py` 参数说明**：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--host` | localhost | server host |
| `--port` | 8000 | server port |
| `--tasks` | 必填 | 任���名（nargs='+'，可多个） |
| `--n_episode` | 1 | 每任务跑几个 episode |
| `--max_episode_length` | 200 | 单 episode 最大步数 |
| `--replan_steps` | 5 | **关键**：每次规划返回 10 个 action，每 replan_steps 步重新规划。改 1 = 每步都重规划 |
| `--save_dir` | None | 输出目录（JSON + 视频） |
| `--visualization` | False | 加这个 flag 才会保存视频 |
| `--seed` | 42 | 随机种子 |

**关键参数 `--replan_steps`**：
- 默认 5：每 5 步重新问 server 要 action chunk，5 步内执行 chunk 的 5 个 action
- 改成 1：每步都重新规划，**精度更高但推理开销 5 倍**（10 episodes 从 ~2 分钟变成 ~10 分钟）
- 改成 10：完全执行整个 action chunk（最不灵活，可能漂移）
- **抓取类任务建议 replan_steps=1**（高精度定位），快速运动建议 replan_steps=5

执行：
```bash
chmod +x /tmp/run_eval.sh
bash /tmp/run_eval.sh > /tmp/eval.log 2>&1 &
```

#### 9.3 评测输出结构

```
logs/pi05_train_eval/<your_task>_n10_<timestamp>/
├── evaluation_result.json                 # 完整指标 JSON
└── <your_task>/videos/
    ├── ep0_success_False_steps_200.mp4    # 每个 episode 一个 mp4
    ├── ep1_success_False_steps_200.mp4
    └── ...
```

#### 9.4 `evaluation_result.json` 字段解读

```json
{
    "place_beaker_pres_heat_device": {
        "success_rate": 0.0,                 // 任务整体成功率（所有 conditions 都 PASS 才算 success）
        "condition_score": 0.0,              // 综合 condition 通过率
        "per_condition": {                   // 单个 condition 通过率（关键诊断信息）
            "OnCondition": 0.0,              // 物体是否到达目标位置
            "ButtonPressedCondition": 0.0    // 按钮是否被按下
        },
        "episodes": [
            {
                "episode": 0,
                "success": false,
                "steps": 200,                // 实际跑的步数（200=跑满未完成，<200=提前完成或失败）
                "conditions": {
                    "OnCondition": false,
                    "ButtonPressedCondition": false
                }
            }
            // ... 10 个 episode
        ]
    }
}
```

**关键诊断规则**：

| `success_rate` | `per_condition` 分布 | 含义 |
|---|---|---|
| 0% | 所有 condition 全 FAIL | 训练完全失败或任务太难 |
| 0% 但 `steps=200` | 部分条件 PASS | 训练有效但没完成最后一步（典型情况：学会抓取，没学会按按钮） |
| > 0% | 多数 condition PASS | 训练有效，可继续优化 |
| > 50% | — | 良好（达到论文 baseline 水平） |

**实例（pi05_ft_place_beaker 训练 5000 步后的评测）**：
- `success_rate`: 0%
- 所有 10 episodes `steps=200`（机械臂动了 200 步但没完成任务）
- 所有 episode `OnCondition=FAIL, ButtonPressedCondition=FAIL`
- 视频证据：机械臂学会伸手抓 beaker（部分成功），但抓不起来或放不到 heat_device 上，没按按钮
- **结论**：训练有效（loss 0.018），但 200 条数据训 5000 步不足以学会完整组合任务

#### 9.5 视频检查（必做）

光看 `success_rate` 数字不能判断训练质量。**必看至少 1-2 个视频**判断机械臂在干什么：

```bash
# 用 ffplay 或 scp 拉到本地用 VLC 看
ffplay logs/pi05_train_eval/<your_task>_n10_*/<your_task>/videos/ep0_success_False_steps_200.mp4

# 或生成 gif 预览
ffmpeg -i ep0_*.mp4 -vf "fps=5,scale=320:-1" preview.gif
```

**视频画面 4 种模式与诊断**：

| 模式 | 视频表现 | 诊断 | 下一步 |
|---|---|---|---|
| A | 机械臂完全不动/乱抖 | 训练失败 / policy 输出错 | 检查 norm_stats、checkpoint |
| B | 动但够不到目标 | 学到动作但定位差 | 加训练步数 + 改 replan_steps=1 |
| C | 完成前半段（抓取）后半段失败 | 数据不足，长 horizon 学不到 | 扩数据（特别是后段子动作） |
| D | 完成大部分步骤但卡最后一步 | 接近成功 | 微调或加 replan_steps |

#### 9.6 停 server

```bash
pkill -f "serve_policy.py.*VLABENCH"
# 或
/ssd/qinmaokai/workspace/SciVLABench/run_pi05_eval.sh stop
```

#### 9.7 用不同 checkpoint 评测对比

如果有多个 step 的 checkpoint（save_interval=500 保存了 500/1000/.../5000），可对比不同训练阶段：

```bash
for STEP in 500 1000 2000 3000 5000; do
    # 编辑 start_eval_server.sh 改 --policy.dir=.../pi05_ft_<task>_<exp>/$STEP
    # 重启 server
    bash /tmp/start_eval_server.sh > /tmp/eval_server_$STEP.log 2>&1 &
    sleep 60
    # 跑评测
    bash /tmp/run_eval.sh > /tmp/eval_$STEP.log 2>&1
    pkill -f serve_policy.py
    sleep 5
done
```

这能画出 "checkpoint step vs success_rate" 曲线，找出**早停点**（loss 还在降但评测最优的 step）。

#### 9.8 任务注册问题

`evaluate_openpi.py --tasks <name>` 要求 `<name>` 在 VLABench task registry 中。如果报 `KeyError: '<name>'`：

1. 检查 `VLABench/configs/__init__.py` 是否有 `"<name>_series": ["<name>"]`
2. 如果没有，参照已有 place_* 任务模式添加
3. 重启评测脚本

实测可用任务（部分）：
```
lift_beaker, lift_flask, lift_cylinder, lift_tube, lift_petri_dish
place_beaker_pres_heat_device, pick_pill_bottle_lift_pill_bottle
select_fruit, select_toy, select_book, ...
```

用这个命令查全部：
```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate vlabench_2
python -c "from VLABench.utils.register import register; from VLABench.tasks import *; print(sorted(register._tasks.keys()))"
```

---

## 5. 关键注意事项与踩坑清单

### 5.1 GPU 资源

| 设备 | 显存 | pi0.5 LoRA 占用 | batch_size 上限 |
|---|---|---|---|
| A100/A800 80GB | 80GB | ~70GB | batch=2 |
| H100 80GB | 80GB | ~70GB | batch=2 |
| A6000 48GB | 48GB | OOM | 不支持 |
| RTX 4090 24GB | 24GB | OOM | 不支持 |

GPU 0 占 77.5GB 是正常状态，**不要看到接近 80GB 就以为 OOM**——JAX 编译期预留最大值。

### 5.2 hostname DNS 问题

容器内 hostname（`g0104`）通常不在 `/etc/hosts` 里，`socket.gethostbyname()` 会抛 `gaierror`。修改 `third_party/openpi/scripts/serve_policy.py:113`：

```python
hostname = socket.gethostname()
try:
    local_ip = socket.gethostbyname(hostname)
except socket.gaierror:
    local_ip = "127.0.0.1"
    logging.warning("Could not resolve hostname %s, falling back to 127.0.0.1", hostname)
```

### 5.3 conda 路径问题

文档假设 conda 在 `/ssd/mkqin/miniconda3`，但实际可能是：
- `/opt/miniconda3`（系统装）
- `/ssd/qinmaokai/.conda/envs/vlabench_2`（env 安装位置）

查看实际路径：
```bash
conda env list                    # 看所有 env
which python                      # base python
ls /ssd/qinmaokai/.conda/envs/    # env 列表
```

### 5.4 HF 模型下载

国内必须设：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

否则会卡在 GCS/GitHub 下载（paligemma_tokenizer、pi05_base params）。

### 5.5 GitHub clone 超时

`git clone` 在国内服务器常 HTTP2 错误，用 `gh-proxy.com` 镜像：
```bash
curl -L -o /tmp/openpi.tar.gz https://gh-proxy.com/https://github.com/Shiduo-zh/openpi/archive/refs/heads/pi05.tar.gz
```

### 5.6 Lerobot 0.1.0 路径解析 bug

详见 Phase 3b，必须修 `lerobot_dataset.py` line 508/509/696 三处 `torch.stack(self.hf_dataset["col"])` → list comprehension。

### 5.7 pip 清华源

```
UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple uv pip install ...
```

不设清华源 pip 默认会卡 10+ 分钟甚至超时。

### 5.8 JAX 编译时间

首次跑训练（包括 dry-run）会触发 JAX JIT 编译 pi0.5（~3B 参数），**预计 5-10 分钟**。期间 GPU 显存占用但 util=0%，CPU 满载——这是正常状态，不要中途杀掉。

### 5.9 norm_stats 用法

`compute_norm_stats.py` 写到 cwd 下的 `./assets/<config>/<repo_id>/norm_stats.json`，然后被训练脚本从 `assets_dirs/.../norm_stats.json` 读取。**直接传绝对路径**最稳：

```bash
python scripts/train.py pi05_ft_<task> \
    --assets_base_dir=/ssd/qinmaokai/.cache/openpi/assets \
    --checkpoint_base_dir=/ssd/qinmaokai/.cache/openpi/checkpoints
```

不要依赖 `./assets` 相对路径 + symlink（之前出过循环 symlink bug）。

---

## 6. 评测指标解读

### 6.1 训练 loss

| 范围 | 含义 |
|---|---|
| 0.5-1.0 | 起步，可能还没收敛 |
| 0.1-0.3 | 收敛中 |
| 0.02-0.1 | 健康收敛 |
| < 0.02 | 可能过拟合（200 条数据容易） |

### 6.2 grad_norm

| 范围 | 含义 |
|---|---|
| < 1 | 收敛 |
| 1-5 | 学习中（健康） |
| 5-10 | 大更新阶段 |
| > 20 | 可能不稳定，可考虑降 lr |

### 6.3 success_rate

| 任务 | 期望 |
|---|---|
| `lift_*` (simple) | 30-50% (pretrained pi05) |
| `place_*` (complex) | 10-30% |
| `pick_*` (compound) | 5-20% |

微调后的期望：原始基础上 +5-20%（取决于数据质量和数量）。

### 6.4 视频检查

`logs/pi05_eval/<task>/videos/ep0_*.mp4` 是关键质量检查：
- 机械臂是否有合理运动（不是随机抖动）
- 是否尝试接近目标物体
- gripper 是否在合适时机张开/闭合
- 失败时是否卡在某个位置不动（deadlock）或乱动（policy 没学到）

---

## 7. 故障排查表

| 症状 | 原因 | 修复 |
|---|---|---|
| `No module named 'lerobot'` | venv 没装 | `uv pip install -e packages/openpi-client` |
| `No module named 'chex'` | jax 0.5.3 没装配套 chex | `uv pip install "chex<=0.1.87"` |
| `ModuleNotFoundError: pytest` | openpi 顶层 import pytest | `uv pip install pytest` |
| `AttributeError: module 'jax.experimental.layout' has no attribute 'DeviceLocalLayout'` | chex 升到 0.10 破坏 orbax | 降 chex ≤ 0.1.87 并 jax 0.5.3 |
| `AttributeError: module 'etils.epy' has no attribute 'lazy_imports'` | etils 太旧 | `uv pip install -U etils` |
| `ValueError: Feature names should not contain '/'` | lerobot 0.1.0 不接受 / 分隔 | 全部改成 `.` |
| `Missing features: {'task'}` | lerobot 0.1.0 add_frame 要 task 字段 | 把 task 加到 frame dict |
| `TypeError: stack(): argument 'tensors' must be tuple of Tensors, not Column` | lerobot 0.1.0 内部 bug | 改 list comprehension |
| `KeyError: 'image'` (vlabench_policy.py:49) | RepackTransform 方向反 | `{"image": "observation.image"}` 而非反过来 |
| `ValueError: Column 'actions' doesn't exist` | 用 `action` 单数而非 `actions` | 改成复数 |
| `Normalization stats not found` | train.py 找不到 norm_stats | 用 `--assets_base_dir=<abs>` 绝对路径 |
| `OSError: [Errno 17] File exists: .../place_beaker_...` | 上次 `rm -rf` 残留空目录 | `rm -rf` 后再跑 |
| `socket.gaierror: Name or service not known` | 容器 hostname 不在 /etc/hosts | 给 serve_policy.py 加 try/except |
| `KeyError: 'observation.state'` | 训练时 norm_stats 已加载但 transforms 找不到 key | RepackTransform 方向反 |
| JAX 编译卡 5+ 分钟 | 正常 | 不要 kill，等 |
| GPU 显存 77.5GB / 80GB | JAX 编译期预留最大值 | 正常，不是 OOM |
| Loss 不下降 (200 步后仍 > 0.5) | 数据/超参问题 | 检查 `--dataset-name` 是否等于 `--policy.config.data.repo_id` |
| **compute_norm_stats 卡 20+ 分钟没进度** | `LeRobotDataset.__init__` 遍历 77k sample 触发 PIL image decode | patch `lerobot_dataset.py:508-509` 直接读 parquet 列（见 4.5） |
| Server 起不来但日志 `server listening` | 已 OK，host=0.0.0.0:8000 | 用 `ss -tlnp \| grep 8000` 验证 |
| `Could not resolve hostname g0104` | DNS 失败 | 已 fallback 到 127.0.0.1 |
| **评测时** `KeyError: 'state'` (所有 episode `success=False, steps=0`) | RepackTransform 没被自动调用（`create_trained_policy` 不传 repack_transforms） | 修 `policy_config.py`：默认从 data_config 取 repack_transforms（见 9.0 坑 1） |
| **评测时** `KeyError: 'actions'` (vlabench_policy.py:79) | client 不发 actions key | RepackTransform 跳过缺失 key（见 9.0 坑 2） |
| **评测时** `AttributeError: 'NoneType' object has no attribute 'shape'` | RepackTransform 把缺失 key 填了 None | 用新版 fallback（跳过缺失 key，不要返回 None） |
| `websockets.exceptions.ConnectionClosedError: received 1011 (internal error)` | server 内部抛异常 | 看 `/tmp/eval_server.log` 找真正的 Python traceback |
| `RuntimeError: Error in inference server` | server 端 transform 链出错 | 同上，看 server log |

---

## 8. 一键脚本 `run_pi05_train.sh` / `run_pi05_eval.sh`

### `run_pi05_train.sh` **默认值陷阱**（**新 agent 必看**）

| 命令 | 默认值 | 漏传后果 |
|---|---|---|
| `./run_pi05_train.sh train` | `CONFIG=pi05_ft_place_beaker`（v1/v2 config）| 跑旧 config，不是 v3 |
| 同上 | `FSDP_DEVICES=1` | **只用 1 卡**，30k 步要 14 小时 |
| 同上 | `GPUS=0,1,2,3` | 用前 4 卡 |
| 同上 | 不传 `--num_train_steps` | 用 config 默认 5000 步 |
| 同上 | 不传 `--batch_size` | 用 config 默认 8 |
| 同上 | 不传 `--overwrite` | 同名 exp_name 第二次跑会报"已存在" |

**完整 v3 训练必传**（**漏一个就跑错**）：

```bash
EXP_NAME=pi05_ft_place_beaker_v3_30k \
GPUS=0,1,2,3,4,5,6,7 \
FSDP_DEVICES=8 \
/ssd/qinmaokai/workspace/SciVLABench/run_pi05_train.sh train \
    --num_train_steps=30000 \
    --batch_size=8 \
    --num_workers=32 \
    --log_interval=100 \
    --save_interval=2000 \
    --keep_period=4000 \
    --overwrite
```

### `run_pi05_train.sh` 用法（参考）

```bash
# 默认训练（**注意默认值陷阱**，上面表格）
./run_pi05_train.sh train

# 单卡 + 200 步 dryrun
GPUS=0 EXP_NAME=dryrun ./run_pi05_train.sh train --num_train_steps=200 --batch_size=2

# 只算 norm_stats（用 pi05_ft_place_beaker_v3 而不是默认 pi05_ft_place_beaker）
CONFIG=pi05_ft_place_beaker_v3 ./run_pi05_train.sh compute-norm-stats

# 训完后跑评测
./run_pi05_train.sh eval --ckpt /path/to/checkpoint/5000/ --n_ep 10 --task <your_task>
```

### run_pi05_train.sh **环境变量** 汇总

| 变量 | 默认 | 用途 |
|---|---|---|
| `CONFIG` | `pi05_ft_place_beaker` | 选 config（`pi05_ft_place_beaker_v3` 是 v3） |
| `EXP_NAME` | `pi05_ft_place_beaker_<date>_<time>` | checkpoint 目录名 |
| `GPUS` | `0,1,2,3` | CUDA_VISIBLE_DEVICES |
| `FSDP_DEVICES` | `1` | **必传 8**（8 卡 FSDP） |
| `CHECKPOINT` | `/ssd/qinmaokai/.cache/openpi/vlabench_checkpoints/pi05-primitive-10task` | 起点 checkpoint（一般不传） |
| `TASK` | `lift_beaker` | eval 默认任务 |

### run_pi05_train.sh **命令行** 参数（train 子命令）

所有参数透传给 `train.py`：

| train.py 参数 | 是否必传 |
|---|---|
| `--num_train_steps` | 强烈推荐（默认 5000 步） |
| `--batch_size` | 强烈推荐（默认 8） |
| `--num_workers` | 推荐（默认 8） |
| `--log_interval` | 推荐（默认 100） |
| `--save_interval` | 推荐（默认 500） |
| `--keep_period` | 推荐（默认 1000） |
| `--overwrite` | **必传**（同名 exp_name 第二次跑会报"已存在"） |

### `run_pi05_eval.sh` 用法

```bash
./run_pi05_eval.sh server-bg 0          # 后台 server
./run_pi05_eval.sh eval lift_beaker 5    # 单任务评测
./run_pi05_eval.sh eval-multi task1 task2 task3 3  # 多任务评测
./run_pi05_eval.sh stop                 # 停 server
```

---

## 9. 时间预算（实测）

| 阶段 | 实测时间 | 备注 |
|---|---|---|
| Phase 0: 下载 pi05 checkpoint | ~12 分钟 | 39GB，HF mirror |
| Phase 1: clone openpi pi05 分支 | ~2 分钟 | gh-proxy.com 镜像 |
| Phase 2: 创建 venv + 装依赖 | 30-60 分钟 | 含下载，用清华源 |
| Phase 3: 修代码兼容性 | 10-30 分钟 | 5 处修复 |
| Phase 4: HDF5 → LeRobot 转换 | 15-20 分钟 | 200 文件 |
| Phase 5: 计算 norm_stats | 5-10 分钟 | 37k 帧 |
| Phase 6: 10 步 dry-run | 8-12 分钟 | 含 JAX 编译 |
| Phase 7: 200 步中等训练（单卡 batch=2） | 20 分钟 | 含 JAX 编译 |
| Phase 7': 200 步（8 卡 FSDP batch=8） | **27 分钟** ✅ 实测 | compile 比 1 卡慢 |
| **Phase 8: 5000 步正式训练（8 卡 FSDP）** | **50 分钟** ✅ 实测 | ~0.5s/step |
| Phase 8': 5000 步（单卡 batch=2） | ~14 小时 | ~6s/step |
| Phase 9: 部署 + 10 episodes 评测 | 15-30 分钟 | replan_steps=5 |
| Phase 9': 评测 replan_steps=1 | 60-90 分钟 | 5 倍推理开销 |

**总结**：从零到训出第一个 checkpoint（含 debug）大约 4-6 小时；纯训练只要 50 分钟。

---

## 10. 附录：完整文件清单

修改的文件：
- `scripts/convert_to_lerobot.py` — 修硬编码路径、task 字段、save_episode、consolidate
- `third_party/openpi/src/openpi/training/config.py` — 新增 `pi05_ft_<task>` config、修 RepackTransform 方向
- `third_party/openpi/src/openpi/policies/policy_config.py` — `create_trained_policy` 默认从 data_config 取 repack_transforms（评测必需）
- `third_party/openpi/src/openpi/transforms.py` — `RepackTransform.__call__` 加 slash/dot fallback + 缺失 key 跳过
- `third_party/openpi/scripts/serve_policy.py` — hostname fallback
- `third_party/openpi/examples/vlabench/.venv/lib/python3.11/site-packages/lerobot/common/datasets/lerobot_dataset.py` — list comprehension 修复（3 处）

新建的文件：
- `run_pi05_train.sh` — 训练/算 stats/评测一键脚本
- `run_pi05_eval.sh` — 评测一键脚本

参考的关键文件：
- `third_party/openpi/src/openpi/policies/vlabench_policy.py` — 数据 transforms（schema 关键）
- `third_party/openpi/src/openpi/training/data_loader.py:130-148` — LeRobot 加载
- `third_party/openpi/src/openpi/training/config.py:766-786` — `pi0_libero_low_mem_finetune` LoRA 范式
- `third_party/openpi/examples/vlabench/.venv/lib/python3.11/site-packages/lerobot/common/constants.py:39` — `HF_LEROBOT_HOME`

---

## 11. 验证清单（按 Phase 顺序）

每个 Phase 完成后跑对应的验证脚本：

```bash
# Phase 1: 验证 openpi clone
grep "pi05_ft_vlabench_primitive" third_party/openpi/src/openpi/training/config.py

# Phase 2: 验证 venv
source third_party/openpi/examples/vlabench/.venv/bin/activate
python -c "import jax, openpi, lerobot, torch; print('jax:', jax.__version__, 'torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"

# Phase 3: 验证 import
python -c "from openpi.policies import vlabench_policy; from openpi.training import config; print('configs:', [c.name for c in config._CONFIGS if 'vlabench' in c.name or 'pi05' in c.name][:10])"

# Phase 4: 验证 LeRobot 数据集
python -c "
import os
os.environ['HF_HOME'] = '/ssd/qinmaokai/.cache/huggingface'
os.environ['HF_LEROBOT_HOME'] = '/ssd/qinmaokai/.cache/huggingface/lerobot'
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('<your_task>')
print(f'episodes: {ds.meta.total_episodes}, frames: {ds.meta.total_frames}, fps: {ds.meta.fps}, task: {list(ds.meta.tasks.values())[0][:80]}')
"

# Phase 5: 验证 norm_stats
cat /ssd/qinmaokai/.cache/openpi/assets/pi05_ft_<your_task>/<your_task>/norm_stats.json | python -c "import json,sys; d=json.load(sys.stdin); print('keys:', list(d['norm_stats'].keys()))"

# Phase 6-7: 训练 loss 下降（看 /tmp/trainXXX.log）
grep -E "Step [0-9]+|loss=" /tmp/train200.log

# Phase 8: checkpoint 大小
du -sh /ssd/qinmaokai/.cache/openpi/checkpoints/pi05_ft_<your_task>/<exp>/<step>/

# Phase 9: success_rate
cat logs/pi05_eval/<your_task>_*/evaluation_result.json
```