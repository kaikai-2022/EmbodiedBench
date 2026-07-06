# ACT 模型训练与部署标准工作流指南 (SOP)

## 1. 角色与目标

你是 VLABench 项目的 **ACT 模型训练与部署专家**。你的职责是将一个 VLABench 任务（从注册到已有任务列表中选取）转换为专家轨迹数据，再经过格式转换、多卡训练，最终部署为可推理的策略模型，并验证其在仿真环境中的成功率。

**完整链路**：

```
VLABench 任务
  → scripts/trajectory_generation.py (专家轨迹采集)
  → HDF5 文件 (dataset/hdf5_trajectory/<task_name>/)
  → scripts/convert_to_lerobot.py (格式转换)
  → LeRobot Dataset (~/.cache/huggingface/lerobot/<repo_id>/)
  → lerobot/scripts/train.py (ACT 多卡训练)
  → ACT Checkpoint (models/act_<task>_<N>ep_<date>/checkpoints/)
  → scripts/evaluate_policy.py (推理评测)
  → success_rate
```

**项目根目录**：`/ssd/mkqin/workspace/VLABench`（下文所有相对路径均基于此）

**LeRobot 代码库**：`/ssd/mkqin/workspace/lerobot`（ACT 训练框架所在）

---

## 2. 核心工具链与文件路径

### 2.1 脚本工具

| 脚本 | 路径 | 用途 |
|------|------|------|
| **trajectory_generation.py** | `scripts/trajectory_generation.py` | 执行专家策略，输出 HDF5 轨迹文件 |
| **convert_to_lerobot.py** | `scripts/convert_to_lerobot.py` | HDF5 → LeRobot Dataset 格式转换 |
| **train.py** | `lerobot/lerobot/scripts/train.py` | 多 GPU DDP 训练 ACT 策略 |
| **evaluate_policy.py** | `scripts/evaluate_policy.py` | 在 VLABench 仿真环境中评测策略 |
| **act.py** (wrapper) | `VLABench/evaluation/model/policy/act.py` | ACT 推理时序控制与坐标变换 |

### 2.2 输出目录结构

```
VLABench/
├── dataset/hdf5_trajectory/<task_name>/      # Phase 1 输出
│   ├── data_000.hdf5
│   ├── data_001.hdf5
│   └── ...
├── ~/.cache/huggingface/lerobot/<repo_id>/  # Phase 2 输出（LeRobot Dataset）
│   ├── meta/
│   │   ├── info.json
│   │   ├── stats.safetensors
│   │   └── tasks.jsonl
│   └── data/...
└── models/act_<task>_<N>ep_<date>/          # Phase 3 输出（训练模型）
    ├── checkpoints/
    │   ├── 001000/pretrained_model/
    │   │   ├── config.json      ← ⚠️ 缺 "type": "act"，需手动补
    │   │   ├── model.safetensors
    │   │   └── train_config.json
    │   ├── 002000/...
    │   └── last → <latest>      # 软链接指向最新 checkpoint
    └── train.log
```

### 2.3 关键环境变量

| 变量 | 取值 | 用于阶段 | 说明 |
|------|------|---------|------|
| `MUJOCO_GL` | `egl` | Phase 1 | GPU 加速 OpenGL 渲染，数据采集用 |
| `MUJOCO_GL` | `osmesa` | Phase 4 | CPU 软件渲染，评估用（避免 X11/EGL 依赖） |
| `PYOPENGL_PLATFORM` | `egl` | Phase 1 | OpenGL 后端选择 |
| `LOCAL_RANK` / `WORLD_SIZE` | torchrun 设置 | Phase 3 | DDP 分布式训练，由 torchrun 自动注入 |

---

## 3. 标准操作步骤

### 场景 A: Phase 1 — 生成专家轨迹数据 (HDF5)

适用于：为指定 VLABench 任务生成专家演示轨迹，输出为 HDF5 文件。

**前置条件**：
- 目标任务已注册在 `VLABench.configs.name2config` 或 `VLABench.utils.register`
- 任务类实现了 `get_expert_skill_sequence()` 方法（定义专家行为）
- 机器人类型默认为 `franka`

#### Step 1: 验证任务可加载

```python
import sys
sys.path.insert(0, "/ssd/mkqin/workspace/VLABench")
from VLABench.envs import load_env

env = load_env("<task_name>", robot="franka")
skill_seq = env.get_expert_skill_sequence()
print(f"Expert skills: {[s.func.__name__ for s in skill_seq]}")
```

> ✅ 成功打印技能列表则任务可用；报错则任务未注册。

#### Step 2: 生成轨迹

```bash
cd /ssd/mkqin/workspace/VLABench

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    python scripts/trajectory_generation.py \
    --task-name <task_name> \
    --n-sample <N> \
    --start-id 0 \
    --save-dir /ssd/mkqin/workspace/VLABench/dataset/hdf5_trajectory \
    --record-video \
    --robot franka
```

**参数说明**：
- `--task-name`：任务名称，如 `lift_flask`、`place_beaker`、`lift_beaker`
- `--n-sample`：生成多少条轨迹（推荐 ≥ 50，100 条效果更稳定）
- `--start-id`：起始文件编号（多机并行时错开）
- `--save-dir`：HDF5 输出根目录（实际保存在 `<save-dir>/<task_name>/`）
- `--record-video`：同时录制仿真回放视频（.mp4），便于人工检查质量

**实战数据**：

| 任务 | Episodes | 帧数/Episode | 采集耗时 |
|------|---------|-------------|---------|
| lift_flask | 100 | 51–84（均值 59.5）| ~15 分钟 |
| place_beaker | 100 | 82–125（均值 103.8）| ~25 分钟 |

> ⚠️ 数据采集速度取决于 episode 长度和仿真帧率。`--record-video` 会显著拖慢速度。

#### Step 3: 验证 HDF5 数据结构

```python
import h5py
import numpy as np

path = "dataset/hdf5_trajectory/<task_name>/data_000.hdf5"
with h5py.File(path, "r") as f:
    timestamps = list(f["data"].keys())
    ts = timestamps[0]
    print(f"Episodes: {len(timestamps)}")
    print(f"HDF5 schema keys: {list(f['data'][ts].keys())}")

    obs = f[f"data/{ts}/observation"]
    traj = f[f"data/{ts}/trajectory"]
    meta = f[f"data/{ts}/meta_info"]

    rgb_shape = obs["rgb"].shape
    ee_shape = obs["ee_state"].shape
    traj_shape = traj.shape
    robot_pos = eval(meta["episode_config"][()].item().decode())["robot"]["position"]

    print(f"  rgb: {rgb_shape}         # (T, 4, 480, 480, 3) uint8, 4 cameras")
    print(f"  ee_state: {ee_shape}   # (T, 8) float32 [pos3, quat4, gripper1]")
    print(f"  trajectory: {traj_shape} # (T, 8) float32 [pos3, euler3, gripper1, gripper2]")
    print(f"  robot.position: {robot_pos}")
    print(f"  instruction: {np.asarray(f[f'data/{ts}/instruction'][()][0]).item().decode()}")
```

**HDF5 Schema 各字段说明**：

| 字段 | Shape | 类型 | 说明 |
|------|-------|------|------|
| `observation/rgb` | `(T, 4, 480, 480, 3)` | uint8 | 4 个相机视角的 RGB 图像 |
| `observation/ee_state` | `(T, 8)` | float32 | 末端执行器状态 `[pos3, quat4, gripper1]` |
| `trajectory` | `(T, 8)` | float32 | VLABench 格式动作 `[pos3, euler3, gripper1, gripper2]` |
| `instruction` | `(1,)` | bytes | 任务语言描述 |
| `meta_info/episode_config` | JSON string | dict | 仿真配置，含 `robot.position` |

#### Step 4: 数据质量检查清单

- [ ] 文件数量 = `n_sample`
- [ ] 每条 episode 长度在合理范围（50–130 帧）
- [ ] 视频回放显示成功完成（机械臂执行了目标动作）
- [ ] gripper 状态在合适时机开合（张开接近物体 → 闭合抓取 → 张开释放）
- [ ] `robot.position` 一致（所有 episode 应相同，默认 `[0, -0.4, 0.78]`）

---

### 场景 B: Phase 2 — HDF5 → LeRobot 格式转换

适用于：将 Phase 1 生成的 HDF5 轨迹转换为 LeRobot Dataset 格式，供 ACT 训练使用。

#### Step 1: 执行转换

```bash
cd /ssd/mkqin/workspace/VLABench

python scripts/convert_to_lerobot.py \
    --dataset-name <repo_id> \
    --dataset-path /ssd/mkqin/workspace/VLABench/dataset/hdf5_trajectory \
    --max-files 500 \
    --task-list <task_name>
```

**参数说明**：
- `--dataset-name`：LeRobot Dataset 名称（会作为 HF cache 子目录名），格式建议 `vlabench_<task>_<N>ep`
- `--dataset-path`：HDF5 根目录（脚本会搜索子目录中所有 `.hdf5` 文件）
- `--task-list`：要转换的任务名列表（可多个，如 `lift_flask place_beaker`）
- `--max-files`：每个任务最多转换多少个 HDF5 文件（500 表示"全部"）

> ⚠️ **关键陷阱**：LeRobot Dataset 输出目录 `~/.cache/huggingface/lerobot/<repo_id>/` **不能预先存在**，否则报错 `FileExistsError`。转换前请先执行：
> ```bash
> rm -rf ~/.cache/huggingface/lerobot/<repo_id>
> ```

#### Step 2: 转换内部逻辑（关键）

`convert_to_lerobot.py` 执行以下不可跳过的变换：

**① 坐标系变换 — world frame → robot-relative frame**

```python
# 读取 episode_config 中的 robot 位置
robot_frame_pos = np.array(episode_config["robot"]["position"])  # 默认 [0, -0.4, 0.78]

# ee_state: [pos3, quat4, gripper1] → 先转为 euler
ee_euler = quat2euler(ee_quat)  # scipy Rotation

# 坐标变换：减去 robot frame 偏移
ee_pos -= robot_frame_pos

# 合并：[pos3_rel, euler3, gripper1] = (7,)
ee_state = np.concatenate([ee_pos, ee_euler, gripper], axis=1)
```

> ⚠️ 这一步至关重要！如果不做变换，训练出的模型输出的动作会带有一个固定偏移，导致推理时机械臂飞走。

**② Action 格式 — VLABench 8D → LeRobot 7D**

| 格式 | Shape | 字段 |
|------|-------|------|
| VLABench | `(T, 8)` | `[pos3, euler3, gripper1, gripper2]` |
| LeRobot | `(T, 7)` | `[pos3, euler3, gripper1]` |

转换时取 `action[:6]` + `action[6]`（第一个 gripper 值），丢弃第二个 gripper。

**③ Gripper 二值化**

```python
if action[-1] > 0.03:
    action = np.concatenate([action[:6], np.array([1])])  # 张开
else:
    action = np.concatenate([action[:6], np.array([0])])  # 闭合
```

**④ 相机选择**

| VLABench index | 内容 | LeRobot feature |
|---------------|------|----------------|
| `images[i][2]` | 前视相机 (front) | `observation.image` |
| `images[i][3]` | 腕部相机 (wrist) | `observation.wrist_image` |

#### Step 3: 验证 LeRobot Dataset

```python
import json
import os

cache_dir = os.path.expanduser("~/.cache/huggingface/lerobot/<repo_id>")

with open(f"{cache_dir}/meta/info.json") as f:
    info = json.load(f)
    print(f"Total episodes: {info['total_episodes']}")
    print(f"Total frames: {info['total_frames']}")
    print(f"Features: {list(info['features'].keys())}")

with open(f"{cache_dir}/meta/tasks.jsonl") as f:
    for line in f:
        print(f"  task: {line.strip()}")
```

**预期输出**：
```
Total episodes: 200
Total frames: 16323
Features: ['observation.image', 'observation.wrist_image', 'observation.state', 'action']
  task: {"task": "Lift the <flask_0>."}
  task: {"task": "Pick the <small_beaker_0> and place it on <square_mat_0>."}
```

---

### 场景 C: Phase 3 — 多 GPU DDP 训练 ACT 模型

适用于：用 LeRobot Dataset 训练 ACT (Action Chunking Transformers) 策略，输出可部署的 checkpoint。

#### Step 1: 启动 DDP 多卡训练

```bash
cd /ssd/mkqin/workspace/lerobot

OUTPUT_DIR="/ssd/mkqin/workspace/VLABench/models/act_<task>_<N>ep_$(date +%Y%m%d_%H%M%S)"

torchrun --standalone --nproc_per_node=4 lerobot/scripts/train.py \
    --policy.type=act \
    --dataset.repo_id=<repo_id> \
    --dataset.local_files_only=true \
    --dataset.image_transforms.enable=false \
    --dataset.use_imagenet_stats=true \
    --output_dir="$OUTPUT_DIR" \
    --batch_size=2 \
    --num_workers=2 \
    --n_epochs=50 \
    --save_freq=10000 \
    --eval_freq=10000 \
    --log_freq=100 \
    --device=cuda \
    --seed=1000
```

> ⚠️ **关键**：不要预先创建 `--output_dir`，lerobot 会自动创建。若目录已存在会报错 `FileExistsError`。

#### Step 2: 关键参数说明

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--nproc_per_node` | 4 | GPU 数量（使用 4 张 RTX 4090） |
| `--batch_size` | 2 | 每 GPU batch size，全局 batch = 4 × 2 = 8 |
| `--num_workers` | 2 | 每 GPU 数据加载线程数 |
| `--n_epochs` | 50 | 训练轮数（见下方实战数据） |
| `--save_freq` | 10000 | 每多少步保存 checkpoint |
| `--eval_freq` | 10000 | 每多少步运行一次评估 |
| `--local_files_only` | true | **必须设为 true**，否则会尝试访问 HuggingFace remote |
| `--image_transforms.enable` | false | 禁用额外数据增强 |
| `--use_imagenet_stats` | true | 使用 ImageNet 预训练归一化参数 |
| `--seed` | 1000 | 随机种子，保证可复现性 |

#### Step 3: 训练时长与步数预估

**计算公式**：

```
steps_per_epoch = total_frames / global_batch_size
total_steps = n_epochs × steps_per_epoch
```

**以 200 episodes (16,323 帧) 为例**：

| 参数 | 值 |
|------|-----|
| 总帧数 | 16,323 |
| 全局 batch | 8 (4 GPU × 2) |
| 每 epoch 步数 | 2,040 |
| 50 epochs 总步数 | 102,000 |

**实战数据（4×RTX 4090）**：

| 训练量 | 步数 | 耗时 | Success Rate |
|--------|------|------|-------------|
| Smoke test | 50 | ~10 分钟 | 0% |
| 5 epochs | 10,000 | ~4.5 小时 | ~0% |
| **50 epochs** | **102,000** | **~50 小时** | **66–100%** |

> ⚠️ 50 epochs 是经验门槛。10k 步以下模型几乎不收敛；50 epochs 可稳定达到 66–100%（lift_beaker 任务）。

#### Step 4: 监控训练进度

**实时查看 loss**：
```bash
tail -f /tmp/train_<task>.log | grep "step:"
```

**训练日志格式**：
```
INFO step:37K smpl:295K ep:4K epch:18.09 loss:0.059 grdn:10.300 lr:1.0e-05 updt_s:0.624 data_s:0.000
```

**各字段含义**：

| 字段 | 说明 |
|------|------|
| `step` | 当前训练步数 |
| `smpl` | 已见样本数（应 ≈ step × batch_size × world_size） |
| `epch` | 当前 epoch 数（修复 world_size bug 后才准确） |
| `loss` | 训练 loss，应稳定下降 |
| `grdn` | 梯度范数，5–20 正常，>50 梯度爆炸 |
| `lr` | 学习率 |
| `updt_s` | 每次更新耗时（秒） |
| `data_s` | 数据加载耗时，>1 表示数据瓶颈 |

**Checkpoint 保存时机**：

| Step | Epoch | 预计时间 | 目录 |
|------|-------|---------|------|
| 010000 | 5 | ~4.5h | `checkpoints/010000/` |
| 020000 | 10 | ~9h | `checkpoints/020000/` |
| ... | ... | ... | ... |
| 050000 | 25 | ~22h | `checkpoints/050000/` |
| 100000 | 49 | ~45h | `checkpoints/100000/` |
| 102000 | 50 | ~50h | `checkpoints/102000/` |

#### Step 5: 收敛判断标准

| 指标 | 收敛标准 | 异常信号 |
|------|---------|---------|
| train_loss | 从 ~30–100 降至 < 0.5 | NaN / Inf / loss 不降 |
| eval_loss | 与 train_loss 同步下降 | eval_loss 远高于 train_loss |
| grad_norm | 稳定在 5–20 | > 50（梯度爆炸）|
| data_s | < 1 | > 1（数据加载瓶颈）|

> ⚠️ **已知问题**：多卡 DDP 训练时 `EpisodeAwareSampler` 有 bug，导致 loss 异常低（0.01–0.1）但推理失败。单卡训练 loss 行为更可靠（36→9 正常范围）。若 loss 低于 0.01，请用单卡验证。

---

### 场景 D: Phase 4 — 评估训练好的模型

适用于：将 Phase 3 保存的 checkpoint 加载到 VLABench 仿真环境，运行指定任务评测成功率。

#### Step 1: 修复 config.json（⚠️ 必须先做）

LeRobot 训练时将 `"type": "act"` 存在 `train_config.json`，但 `pretrained_model/config.json` **缺此字段**。直接用 `evaluate_policy.py` 会报错或加载 RandomPolicy。

**Python 一键修复**：
```python
import json, os

ckpt_dir = "models/act_<task>_<N>ep_<date>/checkpoints/010000/pretrained_model"
config_path = os.path.join(ckpt_dir, "config.json")

with open(config_path) as f:
    cfg = json.load(f)

cfg["type"] = "act"  # 添加缺失字段

with open(config_path, "w") as f:
    json.dump(cfg, f, indent=4)

print(f"✓ 已修复: {config_path}")
```

**Bash 一行版**：
```bash
python -c "
import json
p = 'models/act_<task>_<N>ep_<date>/checkpoints/010000/pretrained_model/config.json'
c = json.load(open(p))
c['type'] = 'act'
json.dump(c, open(p,'w'), indent=4)
print('done')
"
```

#### Step 2: 启动评估

```bash
cd /ssd/mkqin/workspace/VLABench

MUJOCO_GL=osmesa python scripts/evaluate_policy.py \
    --tasks <task_name> \
    --n-episode 10 \
    --policy act \
    --model_ckpt /ssd/mkqin/workspace/VLABench/models/act_<task>_<N>ep_<date>/checkpoints/010000/pretrained_model \
    --replanstep 4 \
    --save-dir logs/act_<task>_eval \
    --visulization \
    --metrics success_rate intention_score progress_score
```

#### Step 3: 关键参数说明

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--policy` | `act` | 选择 ACT 策略 |
| `--model_ckpt` | `.../pretrained_model/` | **必须是 `pretrained_model/` 目录**，不是 `checkpoints/010000/` |
| `--tasks` | `<task_name>` | 评测任务名（可多个，如 `lift_beaker place_beaker`） |
| `--n-episode` | 10–50 | 每个任务评测多少个 episode |
| `--replanstep` | 4 | 每多少步重新查询策略（类似 OpenPI） |
| `--visulization` | （flag）| 录制仿真视频保存到 `videos/` 目录 |
| `--save-dir` | `logs/act_<task>_eval` | 评测结果输出目录 |

> ⚠️ `MUJOCO_GL=osmesa` 是关键！不用 egl/X11，避免 `gladLoadGL error`。

#### Step 4: 评估输出结构

```
logs/act_<task>_eval/
├── metrics.json              # 全局指标 {"task": {"success_rate": 0.X}}
├── act/
│   └── evaluation_result.json  # 同上
└── <task_name>/
    ├── detail_info.json        # 每个 episode 详情
    └── videos/
        ├── 0_success_True_progress_1.00.mp4
        ├── 1_success_False_progress_0.33.mp4
        └── ...
```

#### Step 5: 解析评测结果

```python
import json

with open("logs/act_<task>_eval/metrics.json") as f:
    results = json.load(f)
    for task, metrics in results.items():
        sr = metrics["success_rate"]
        n = int(round(sr * 10))  # 假设 n_episode=10
        print(f"{task}: {sr:.1%} ({n}/10)")

# detail_info.json 结构示例
with open("logs/act_<task>_eval/<task_name>/detail_info.json") as f:
    episodes = json.load(f)
    for ep in episodes:
        status = "✅" if ep["success"] else "❌"
        print(f"  {status} Episode {ep.get('episode_idx')}: "
              f"progress={ep['progress_score']:.2f}, "
              f"steps={ep['consumed_step']}")
```

**实战评测结果**：

| 任务 | 训练数据 | Episodes | 成功率 | 说明 |
|------|---------|---------|--------|------|
| lift_beaker | 未在训练集 | 3 | **100% (3/3)** | 跨任务泛化 |
| place_beaker | 在训练集 | 3 | **33.3% (1/3)** | 多步任务较难 |

> 💡 lift_beaker 未在训练数据中但达到 100%，说明模型学到了可泛化的抓取策略。

---

## 4. 关键避坑指南与隐性知识

### 4.1 config.json 缺失 "type": "act" 字段 ⚠️

- **❌ 错误现象**：`evaluate_policy.py` 加载时报 `ParsingError: Expected a dict with a 'type' key` 或默认加载 RandomPolicy（eval 显示 0%）
- **✅ 解决**：评估前在 `config.json` 顶部手动添加 `"type": "act",`
- **原因**：LeRobot 训练时 `type` 字段存在 `train_config.json`，但 `pretrained_model/config.json` 生成时未复制

### 4.2 MUJOCO_GL 环境变量

- **❌ 错误**：`MUJOCO_GL=egl` 在 SSH 无 X11 环境报错 `gladLoadGL error`
- **✅ 解决**：
  - Phase 1（数据采集）用 `MUJOCO_GL=egl`（需要 GPU 加速）
  - Phase 4（评估推理）用 `MUJOCO_GL=osmesa`（纯 CPU 软件渲染，无依赖）

### 4.3 坐标系变换 (world ↔ robot-relative)

| 方向 | 代码位置 | 操作 |
|------|---------|------|
| 训练时（world→robot） | `convert_to_lerobot.py` | `ee_pos -= robot_frame_pos` |
| 推理时（robot→world） | `act.py` `select_action()` | `target_pos = action[:3] + robot_frame` |

- 默认 `robot_frame_pos = [0, -0.4, 0.78]`
- 若变换不一致，推理时机械臂会飞走（动作偏移 ~40cm）

### 4.4 Action 格式转换

| 步骤 | 格式 | Shape | 说明 |
|------|------|-------|------|
| HDF5 原始 | VLABench 8D | `(T, 8)` | `[pos3, euler3, gripper1, gripper2]` |
| 转换后 | LeRobot 7D | `(T, 7)` | `[pos3, euler3, gripper1]` |

- 转换取 `action[:6]` + `action[6]`（第一个 gripper 值）
- Gripper 二值化阈值 `0.03`：`> 0.03` → 张开(1)，否则闭合(0)

### 4.5 相机索引约定

- VLABench 4 相机：`[0, 1, 2, 3]`
- ACT 使用：`[2]` = front (前视) + `[3]` = wrist (腕部)
- `convert_to_lerobot.py` 中：`images[i][2]` → `observation.image`，`images[i][3]` → `observation.wrist_image`

### 4.6 LeRobot Dataset 目录不可覆盖

- **❌ 错误**：`LeRobotDataset.create(repo_id=<已有名称>)` 报错 `FileExistsError`
- **✅ 解决**：转换前先 `rm -rf ~/.cache/huggingface/lerobot/<repo_id>`

### 4.7 训练量与成功率的关系

> **必须跑满 50 epochs** 才能稳定达到 66–100% 成功率。

| 训练量 | 步数 | Loss | Success Rate |
|--------|------|------|-------------|
| 50 steps | 50 | ~9 | 0% |
| 5 epochs | 10,000 | ~0.1–4 | ~0–10% |
| 50 epochs | 102,000 | ~0.01–0.1 | **66–100%** |

> ⚠️ DDP 多卡时 loss 可能异常低（0.01–0.1）但推理失败。此时需用单卡重新验证。

### 4.8 CLI 布尔值格式（Draccus parser）

- **❌ 错误**：`--dataset.local_files_only=True`（Python 风格）
- **✅ 正确**：`--dataset.local_files_only=true`（Draccus YAML 风格）

### 4.9 LeRobot 输出目录禁止预创建

- **❌ 错误**：先 `mkdir -p "$OUTPUT_DIR"` 再启动训练
- **✅ 正确**：lerobot 会自动创建；若目录已存在则报错 `FileExistsError`

### 4.10 常见错误速查表

| 错误 | 原因 | 解决 |
|------|------|------|
| `RepositoryNotFoundError: 401 Unauthorized` | 没设 `local_files_only=true` | 添加 `--dataset.local_files_only=true` |
| `ParsingError: Expected a dict with a 'type' key` | config.json 缺 type | 手动加 `"type": "act"` |
| `gladLoadGL error` | X11/EGL 不可用 | 改用 `MUJOCO_GL=osmesa` |
| `FileExistsError: output_dir already exists` | 预先 mkdir 了 | 不要预先创建输出目录 |
| 推理时机械臂飞走 | 坐标系变换不一致 | 确认 convert 和 eval 使用相同 robot_frame |
| loss = NaN | 梯度爆炸 | 降低 lr 或检查数据格式 |
| data_s > 1 | 数据加载瓶颈 | 增加 `num_workers` |
| 评估 0% 但 loss 正常 | config.json type 缺失 | 加 `"type": "act"` |
| `Couldn't parse 'False'` | CLI 用 Python bool | 改小写 `false` |

---

## 5. 快速参考：完整命令序列

以下脚本可一键复制运行，完成 Phase 1–4 全流程。

**使用方式**：替换顶部的 `TASK_NAME`、`N_EPISODES` 变量，然后粘贴到终端执行。

```bash
#!/bin/bash
set -e
# ============================================================
# ACT 模型训练全流程 — 参数配置
# ============================================================
TASK_NAME="lift_beaker"                        # 任务名称
N_EPISODES=50                                  # 轨迹数量（推荐 ≥ 50）
REPO_ID="vlabench_${TASK_NAME}_${N_EPISODES}ep"  # LeRobot Dataset ID
DATE=$(date +%Y%m%d_%H%M%S)                    # 时间戳
MODEL_NAME="act_${TASK_NAME}_${N_EPISODES}ep_${DATE}"  # 模型输出目录名
CKPT_STEP="010000"                             # 评测用的 checkpoint 步数

VLABENCH_ROOT="/ssd/mkqin/workspace/VLABench"
LEROBOT_ROOT="/ssd/mkqin/workspace/lerobot"
HDF5_DIR="${VLABENCH_ROOT}/dataset/hdf5_trajectory"
MODEL_DIR="${VLABENCH_ROOT}/models/${MODEL_NAME}"
CKPT_DIR="${MODEL_DIR}/checkpoints/${CKPT_STEP}/pretrained_model"
LEROBOT_CACHE="${HOME}/.cache/huggingface/lerobot/${REPO_ID}"

# ============================================================
# Phase 1: 生成专家轨迹数据 (HDF5)
# ============================================================
echo "=== Phase 1: 生成 ${N_EPISODES} 条轨迹 ==="
cd ${VLABENCH_ROOT}

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    python scripts/trajectory_generation.py \
    --task-name ${TASK_NAME} \
    --n-sample ${N_EPISODES} \
    --start-id 0 \
    --save-dir ${HDF5_DIR} \
    --record-video \
    --robot franka

echo "  ✓ 轨迹已保存至 ${HDF5_DIR}/${TASK_NAME}/"
ls ${HDF5_DIR}/${TASK_NAME}/*.hdf5 | wc -l

# ============================================================
# Phase 2: HDF5 → LeRobot Dataset 转换
# ============================================================
echo "=== Phase 2: 转换格式 ==="
rm -rf ${LEROBOT_CACHE}  # ⚠️ 必须先删除，否则报错 FileExistsError

python scripts/convert_to_lerobot.py \
    --dataset-name ${REPO_ID} \
    --dataset-path ${HDF5_DIR} \
    --max-files ${N_EPISODES} \
    --task-list ${TASK_NAME}

echo "  ✓ LeRobot Dataset 已保存至 ${LEROBOT_CACHE}"

# ============================================================
# Phase 3: 多 GPU DDP 训练 ACT 模型
# ============================================================
echo "=== Phase 3: 启动 4-GPU 训练 ==="
cd ${LEROBOT_ROOT}

torchrun --standalone --nproc_per_node=4 lerobot/scripts/train.py \
    --policy.type=act \
    --dataset.repo_id=${REPO_ID} \
    --dataset.local_files_only=true \
    --dataset.image_transforms.enable=false \
    --dataset.use_imagenet_stats=true \
    --output_dir=${MODEL_DIR} \
    --batch_size=2 \
    --num_workers=2 \
    --n_epochs=50 \
    --save_freq=10000 \
    --eval_freq=10000 \
    --log_freq=100 \
    --device=cuda \
    --seed=1000

echo "  ✓ 训练完成，模型在 ${MODEL_DIR}"

# ============================================================
# Phase 4: 评估训练好的模型
# ============================================================
echo "=== Phase 4: 评估 ==="
cd ${VLABENCH_ROOT}

# ⚠️ 关键：修复 config.json 缺失的 "type": "act" 字段
python -c "
import json, os
config_path = '${CKPT_DIR}/config.json'
with open(config_path) as f:
    cfg = json.load(f)
cfg['type'] = 'act'
with open(config_path, 'w') as f:
    json.dump(cfg, f, indent=4)
print('✓ config.json 已修复')
"

# 启动评估
MUJOCO_GL=osmesa python scripts/evaluate_policy.py \
    --tasks ${TASK_NAME} \
    --n-episode 10 \
    --policy act \
    --model_ckpt ${CKPT_DIR} \
    --replanstep 4 \
    --save-dir logs/act_${TASK_NAME}_eval_${DATE} \
    --visulization \
    --metrics success_rate intention_score progress_score

# 显示结果
cat logs/act_${TASK_NAME}_eval_${DATE}/metrics.json
echo ""
echo "✓ 全部流程完成！"
echo "  模型目录: ${MODEL_DIR}"
echo "  评测结果: logs/act_${TASK_NAME}_eval_${DATE}/"
echo "  视频回放: logs/act_${TASK_NAME}_eval_${DATE}/${TASK_NAME}/videos/"
```

---

## 附录 A: VLABench 任务注册结构（参考）

新增自定义任务时，需在以下位置注册：

| 文件 | 路径 | 内容 |
|------|------|------|
| ConfigManager | `VLABench/tasks/autogen_tasks/primitive/<task_name>_series.py` | `load_objects()`、`get_instruction()`、`get_condition_config()` |
| Task 类 | 同上文件 | `get_expert_skill_sequence()` 返回 `partial(SkillLib.<skill>, ...)` |
| 注册装饰器 | 同上文件 | `@register.add_config_manager("<task_name>")` + `@register.add_task("<task_name>")` |
| constant.py | `VLABench/configs/constant.py` | `name2class_xml` 中注册模型 XML |

**参考示例**：`VLABench/tasks/autogen_tasks/primitive/lift_beaker_series.py`

## 附录 B: LeRobot ACT 模型架构参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `n_obs_steps` | 1 | 输入观测步数 |
| `chunk_size` | 100 | 动作序列分块大小 |
| `n_action_steps` | 100 | 输出动作序列长度 |
| `vision_backbone` | resnet18 | 视觉编码器（ImageNet 预训练） |
| `dim_model` | 512 | Transformer 隐层维度 |
| `n_heads` | 8 | 注意力头数 |
| `n_encoder_layers` | 4 | ViT encoder 层数 |
| `n_decoder_layers` | 1 | Action decoder 层数 |
| `use_vae` | True | 使用 VAE 潜变量 |
| `latent_dim` | 32 | VAE 潜变量维度 |
| 参数量 | ~51.6M | 总可训练参数 |
