# ACT 模型训练与部署标准工作流指南 (SOP)

> **版本**: 2026-07-20 (v2,含 InMemory 预加载优化)
> **硬件**: 6×NVIDIA A800-SXM4-80GB(单卡 80GB 显存)
> **数据规模**: 200 episodes (~65K 帧,256×256 分辨率)
> **训练时长**: 30K 步 ≈ **1.75 小时**(单卡)/ ~18 分钟(6 卡 DDP 预估)

## 0. 重大变更记录 (v2 升级)

2026-07-20 大幅优化训练 pipeline,实现 **单卡 30K 步 1.75 小时** 的近理论极限速度。

| 项目 | 旧(v1) | 新(v2) | 加速 |
|------|---------|--------|------|
| 数据集分辨率 | 480×480 | **256×256** | 3.5× |
| Dataset | `LeRobotDataset`(PNG-on-the-fly decode) | **InMemoryLeRobotDataset**(预解码到 RAM) | 5× |
| 默认 batch_size | 2 (6 卡 total=12) | **128 (单卡)** | - |
| 默认 num_workers | 2 | **8**(LabUtopia 验证甜点) | - |
| **单步时间** | ~40 秒 | **0.21 秒** | **190×** |
| **30K 步预估** | 357 小时 | **1.75 小时** | **200×** |

详细的优化过程记录在 **附录 D:性能优化记录(从 40s/步 → 0.21s/步)**。

---

## 1. 角色与目标

你是 VLABench 项目的 **ACT 模型训练与部署专家**。你的职责是将一个 VLABench 任务(从注册到已有任务列表中选取)转换为专家轨迹数据,再经过格式转换、单卡训练(优化后),最终部署为可推理的策略模型,并验证其在仿真环境中的成功率。

**完整链路**:

```
VLABench 任务
  → scripts/trajectory_generation.py (专家轨迹采集)
  → HDF5 文件 (dataset/hdf5_trajectory/<task_name>/)
  → scripts/convert_to_lerobot_act.py (格式转换,256x256 预 resize)
  → LeRobot Dataset (~/.cache/huggingface/lerobot/<repo_id>_r256/)
  → scripts/model_train/train_act.py (InMemory 训练, 30K 步 ~1.75h)
  → ACT Checkpoint (lerobot/outputs/train/<date>/checkpoints/)
  → scripts/evaluate_policy.py (推理评测)
  → success_rate
```

**项目根目录**:`/ssd/qinmaokai/workspace/SciVLABench`(下文所有相对路径均基于此)

**LeRobot 代码库**:`/ssd/qinmaokai/workspace/lerobot`(ACT 训练框架所在)

---

## 2. 核心工具链与文件路径

### 2.1 脚本工具

| 脚本 | 路径 | 用途 |
|------|------|------|
| **trajectory_generation.py** | `scripts/trajectory_generation.py` | 执行专家策略,输出 HDF5 轨迹文件 |
| **convert_to_lerobot_act.py** | `scripts/convert_to_lerobot_act.py` | HDF5 → LeRobot Dataset 格式转换 (ACT 专用,带 `--resolution` 参数) |
| **train_act.py** | `scripts/model_train/train_act.py` | 一键训练脚本 (含 in_memory / freeze_encoder / EMA 等参数) |
| **lerobot/scripts/train.py** | `lerobot/scripts/train.py` | LeRobot 训练入口 (含 4 段耗时 + KL/L1 分项诊断输出) |
| **evaluate_policy.py** | `scripts/evaluate_policy.py` | 在 VLABench 仿真环境中评测策略 |
| **act.py** (wrapper) | `VLABench/evaluation/model/policy/act.py` | ACT 推理时序控制与坐标变换 |

### 2.2 关键库文件(新增/修改)

| 文件 | 用途 |
|------|------|
| **lerobot/common/datasets/in_memory_lerobot.py** (新) | 启动时一次性预解码所有图像到 RAM,`__getitem__` 完全自包含(不调用父类 `hf_dataset[idx]`)|
| **lerobot/common/datasets/factory.py** | 支持 `use_in_memory=True` 切换到 InMemoryLeRobotDataset |
| **lerobot/common/policies/act/configuration_act.py** | 加 `freeze_backbone`, `use_ema` 字段 |
| **lerobot/common/policies/act/modeling_act.py** | 实现 freeze_backbone 逻辑(保留 train 模式以避免 BN 异常) |
| **lerobot/configs/default.py** | DatasetConfig 加 `use_in_memory: bool = False` 字段 |

### 2.3 输出目录结构

```
SciVLABench/
├── dataset/training_data/<task>_series/      # Phase 1 ��出
│   ├── data_0.hdf5
│   ├── data_1.hdf5
│   └── ...
├── ~/.cache/huggingface/lerobot/<repo_id>_r256/  # Phase 2 输出 (含分辨率后缀避免冲突)
│   ├── meta/
│   │   ├── info.json
│   │   ├── stats.safetensors
│   │   └── tasks.jsonl
│   └── data/chunk-000/episode_*.parquet (200 个,各含 ~321 帧)
└── lerobot/outputs/train/<date>/<time>_act/  # Phase 3 输出 (LeRobot 默认输出目录)
    ├── checkpoints/
    │   ├── 005000/pretrained_model/
    │   │   ├── config.json      ← ⚠️ 缺 "type": "act",需手动补
    │   │   ├── model.safetensors
    │   │   └── train_config.json
    │   ├── 010000/...
    │   ├── 015000/...
    │   └── last → <latest>      # 软链接指向最新 checkpoint
    └── logs.json.txt
```

### 2.4 关键环境变量

| 变量 | 取值 | 用于阶段 | 说明 |
|------|------|---------|------|
| `MUJOCO_GL` | `egl` | Phase 1 | GPU 加速 OpenGL 渲染,数据采集用 |
| `MUJOCO_GL` | `osmesa` | Phase 4 | CPU 软件渲染,评估用(避免 X11/EGL 依赖) |
| `PYOPENGL_PLATFORM` | `egl` | Phase 1 | OpenGL 后端选择 |
| `PYTHONPATH` | `lerobot:$PROJ` | 全部 | 包含 lerobot 和 SciVLABench 路径 |
| `HF_HOME` | `$PROJ/.cache/huggingface` | 全部 | 强制 HF cache 到项目目录(避免污染主仓 cache) |
| `HF_HUB_OFFLINE=1` | - | 全部 | 离线模式,避免访问 HuggingFace |
| `HF_DATASETS_OFFLINE=1` | - | 全部 | 离线模式 |

---

## 3. 标准操作步骤

### 场景 A: Phase 1 — 生成专家轨迹数据 (HDF5)

适用于:为指定 VLABench 任务生成专家演示轨迹,输出为 HDF5 文件。

**前置条件**:
- 目标任务已注册在 `VLABench.configs.name2config` 或 `VLABench.utils.register`
- 任务类实现了 `get_expert_skill_sequence()` 方法(定义专家行为)
- 机器人类型默认为 `franka`

#### Step 1: 验证任务可加载

```python
import sys
sys.path.insert(0, "/ssd/qinmaokai/workspace/lerobot")
sys.path.insert(0, "/ssd/qinmaokai/workspace/SciVLABench")
from VLABench.envs import load_env

env = load_env("<task_name>", robot="franka")
skill_seq = env.get_expert_skill_sequence()
print(f"Expert skills: {[s.func.__name__ for s in skill_seq]}")
```

> ✅ 成功打印技能列表则任务可用;报错则任务未注册。

#### Step 2: 生成轨迹

```bash
cd /ssd/qinmaokai/workspace/SciVLABench

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    python scripts/trajectory_generation.py \
    --task-name <task_name>_series \
    --n-sample <N> \
    --start-id 0 \
    --save-dir /ssd/qinmaokai/workspace/SciVLABench/dataset/training_data \
    --record-video \
    --robot franka
```

> ⚠️ 任务名需带 `_series` 后缀(沿袭 train_task.py 约定),脚本会自动处理 series name 解析。

**参数说明**:
- `--task-name`:任务名称(需带 `_series` 后缀),如 `lift_flask_series`、`place_beaker_series`
- `--n-sample`:生成多少条轨迹(推荐 ≥ 50,200 条效果更稳定)
- `--start-id`:起始文件编号(多机并行时错开)
- `--save-dir`:HDF5 输出根目录(实际保存在 `<save-dir>/<task_name>/`)
- `--record-video`:同时录制仿真回放视频(.mp4),便于人工检查质量

**实战数据**:

| 任务 | Episodes | 帧数/Episode | 采集耗时 |
|------|---------|-------------|---------|
| pick_cylinder_mid_pour_object | 200 | ~321(均值) | ~60 分钟 |
| lift_flask | 100 | 51–84(均值 59.5)| ~15 分钟 |
| place_beaker | 100 | 82–125(均值 103.8)| ~25 分钟 |

> ⚠️ 数据采集速度取决于 episode 长度和仿真帧率。`--record-video` 会显著拖慢速度。

#### Step 3: 验证 HDF5 数据结构

```python
import h5py
import numpy as np

path = "dataset/training_data/<task_name>_series/data_0.hdf5"
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
```

**HDF5 Schema 各字段说明**:

| 字段 | Shape | 类型 | 说明 |
|------|-------|------|------|
| `observation/rgb` | `(T, 4, 480, 480, 3)` | uint8 | 4 个相机视角的 RGB 图像 |
| `observation/ee_state` | `(T, 8)` | float32 | 末端执行器状态 `[pos3, quat4, gripper1]` |
| `trajectory` | `(T, 8)` | float32 | VLABench 格式动作 `[pos3, euler3, gripper1, gripper2]` |
| `instruction` | `(1,)` | bytes | 任务语言描述 |
| `meta_info/episode_config` | JSON string | dict | 仿真配置,含 `robot.position` |

#### Step 4: 数据质量检查清单

- [ ] 文件数量 = `n_sample`
- [ ] 每条 episode 长度在合理范围(50–330 帧)
- [ ] 视频回放显示成功完成(机械臂执行了目标动作)
- [ ] gripper 状态在合适时机开合(张开接近物体 → 闭合抓取 → 张开释放)
- [ ] `robot.position` 一致(所有 episode 应相同,默认 `[0, -0.4, 0.78]`)

---

### 场景 B: Phase 2 — HDF5 → LeRobot 格式转换

适用于:将 Phase 1 生成的 HDF5 轨迹转换为 LeRobot Dataset 格式,供 ACT 训练使用。

#### Step 1: 执行转换

```bash
cd /ssd/qinmaokai/workspace/SciVLABench

python scripts/convert_to_lerobot_act.py \
    --dataset-name vlabench_<task>_<N>ep_r256 \
    --dataset-path /ssd/qinmaokai/workspace/SciVLABench/dataset/training_data \
    --max-files 200 \
    --task-list <task_name>_series \
    --resolution 256
```

**参数说明**:
- `--dataset-name`:LeRobot Dataset 名称(会作为 HF cache 子目录名),**v2 版本必须在名字里带 `_r<分辨率>`** 避免与旧版(480)冲突
- `--dataset-path`:HDF5 根目录(脚本会搜索子目录中所有 `.hdf5` 文件)
- `--task-list`:要转换的任务���列表(**需带 `_series` 后缀**)
- `--max-files`:每个任务最多转换多少个 HDF5 文件
- `--resolution` (新增):目标图像分辨率,推荐 **256**(加速 3.5x)。480=原始,256=优化

> ⚠️ **关键变更**:使用 `convert_to_lerobot_act.py` 而非 `convert_to_lerobot.py`。前者专门为 ACT 训练设计,使用正确字段名,后者为 pi0.5 设计。

> ⚠️ **关键**:LeRobot Dataset 输出目录 `~/.cache/huggingface/lerobot/<repo_id>_r256/` **不能预先存在**,否则报错 `FileExistsError`。转换前请先执行:
> ```bash
> rm -rf ~/.cache/huggingface/lerobot/<repo_id>_r256
> ```
> (train_act.py 已经自动处理,但手动转换时需要)

#### Step 2: 转换内部逻辑(关键)

`convert_to_lerobot_act.py` 执行以下不可跳过的变换:

**① 坐标系变换 — world frame → robot-relative frame**

```python
# 读取 episode_config 中的 robot 位置
robot_frame_pos = np.array(episode_config["robot"]["position"])  # 默认 [0, -0.4, 0.78]

# ee_state: [pos3, quat4, gripper1] → 先转为 euler
ee_euler = quat2euler(ee_quat)  # scipy Rotation

# 坐标变换:减去 robot frame 偏移
ee_pos -= robot_frame_pos

# 合并:[pos3_rel, euler3, gripper1] = (7,)
ee_state = np.concatenate([ee_pos, ee_euler, gripper], axis=1)
```

> ⚠️ 这一步至关重要!如果不做变换,训练出的模型输出的动作会有一个固定偏移,导致推理时机械臂飞走。

**② Action 格式 — VLABench 8D → LeRobot 7D**

| 格式 | Shape | 字段 |
|------|-------|------|
| VLABench | `(T, 8)` | `[pos3, euler3, gripper1, gripper2]` |
| LeRobot | `(T, 7)` | `[pos3, euler3, gripper1]` |

转换时取 `action[:6]` + `action[6]`(第一个 gripper 值),丢弃第二个 gripper。

**③ Gripper 二值化**

```python
if action[-1] > 0.03:
    action = np.concatenate([action[:6], np.array([1])])  # 张开
else:
    action = np.concatenate([action[:6], np.array([0])])  # 闭合
```

**④ 图像 resize(v2 新增)**

```python
if img.shape[:2] != (target_size, target_size):
    pil = Image.fromarray(img)
    pil = pil.resize((target_size, target_size), Image.BILINEAR)
    return np.array(pil, dtype=np.uint8)
```

> 480×480 → 256×256:BILINEAR 插值。LeRobot 训练时不再触发 resize,大幅加速。

**⑤ 相机选择**

| VLABench index | 内容 | LeRobot feature |
|---------------|------|----------------|
| `images[i][2]` | 前视相机 (front) | `observation.image` |
| `images[i][3]` | 腕部相机 (wrist) | `observation.wrist_image` |

#### Step 3: 验证 LeRobot Dataset

```python
import json
import os

cache_dir = os.path.expanduser("~/.cache/huggingface/lerobot/<repo_id>_r256")

with open(f"{cache_dir}/meta/info.json") as f:
    info = json.load(f)
    print(f"Total episodes: {info['total_episodes']}")
    print(f"Total frames: {info['total_frames']}")
    print(f"Features: {list(info['features'].keys())}")
    for k, v in info["features"].items():
        if v.get("dtype") == "image":
            print(f"  {k}: shape={v['shape']}")  # 应为 (256, 256, 3)

# 验证实际图像分辨率
import pandas as pd, io
from PIL import Image
df = pd.read_parquet(f"{cache_dir}/data/chunk-000/episode_000000.parquet")
img = Image.open(io.BytesIO(df['observation.image'].iloc[0]['bytes']))
print(f"实际图像尺寸: {img.size}")  # 应为 (256, 256)
```

**预期输出**:
```
Total episodes: 200
Total frames: 64830
Features: ['observation.image', 'observation.wrist_image', 'observation.state', 'action']
  observation.image: shape=(256, 256, 3)
  observation.wrist_image: shape=(256, 256, 3)
实际图像尺寸: (256, 256)
```

---

### 场景 C: Phase 3 — ACT 训练(v2 优化版)

适用于:用 LeRobot Dataset 训练 ACT (Action Chunking Transformers) 策略,输出可部署的 checkpoint。

#### Step 1: 一键启动训练 (推荐)

```bash
cd /ssd/qinmaokai/workspace/SciVLABench

python scripts/model_train/train_act.py \
    --task pick_cylinder_mid_pour_object \
    --num 200 \
    --gpus 2 \
    --use-in-memory \
    --resolution 256 \
    --skip-gen \
    --skip-conv
```

**关键参数说明**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--task` | 必填 | 任务名(不带 `_series` 后缀) |
| `--num` | 必填 | 轨迹数量(需与 LeRobot Dataset 匹配) |
| `--gpus` | 2 | 单卡 GPU 编号,推荐单卡 GPU 2(A800 80GB)|
| `--train-steps` | 30000 | 训练步数,30K 步 ~1.75h |
| `--batch-size` | 128 | 每卡 batch_size,全局 batch = 128 (单卡)|
| `--num-workers` | 8 | 数据加载线程(LabUtopia 验证甜点)|
| `--resolution` | 256 | 数据集分辨率(必须与转换时一致)|
| `--use-in-memory` | False | **关键优化**:启动时预解码所有图像到 RAM |
| `--freeze-encoder` | False | 冻结 ResNet18(实验性,可能影响收敛,先关) |
| `--use-amp` | True | AMP 混合精度(LeRobot 内部,有 nan 风险时关) |
| `--save-freq` | 5000 | 每 5K 步保存 checkpoint |
| `--skip-gen` | - | 跳过轨迹生成 |
| `--skip-conv` | - | 跳过格式转换(复用现有数据集)|

> ⚠️ **不要预先指定 `--output-dir`**!脚本会自动让 LeRobot 选择 `outputs/train/<date>/` 目录。否则会触发 FileExistsError。

#### Step 2: 自定义配置(可选)

如果想微调,可以传更多参数:

```bash
python scripts/model_train/train_act.py \
    --task pick_cylinder_mid_pour_object \
    --num 200 \
    --gpus 2 \
    --use-in-memory \
    --train-steps 50000 \
    --batch-size 256 \
    --num-workers 16 \
    --resolution 256 \
    --save-freq 10000 \
    --skip-gen \
    --skip-conv
```

#### Step 3: 训练时长与速度预估

**实际单卡 30K 步测试结果**(pick_cylinder_mid_pour_object,200 episodes,~65K 帧):

| 阶段 | 时间 | 备注 |
|------|------|------|
| 数据集加载 + InMemory 预解码 | 4-5 分钟 | 一次性,约 25GB RAM 占用 |
| Step 0-100 (warmup) | ~10 秒 | warmup + 首步 GPU 编译 |
| Step 100-450 (稳定) | ~70 秒 | **每步 0.21 秒** |
| **30K 步预估** | **~1.75 小时** | 单卡 A800 |
| **6 卡 DDP 预估** | **~18 分钟** | 等效 batch 增大或 loss 同步收敛 |

**完整时间对比表**:

| 训练配置 | 数据规模 | 单步时间 | 30K 步耗时 |
|---------|---------|---------|-----------|
| 旧版(480, batch=2, 6 卡)| 16K 帧 | ~5.4s(6 卡) | ~25h |
| **当前(256, batch=128, 单卡)** | 65K 帧 | **0.21s** | **1.75h** |
| 6 卡 DDP(预估) | 65K 帧 | 0.21s / 6 = 0.035s | ~17 分钟 |

#### Step 4: 监控训练进度

**实时查看 4 段耗时 + KL/L1 曲线**:

```bash
tail -f /tmp/train_act_<task>_r256.log | grep "step:"
```

**训练日志格式(单卡)**:
```
INFO step:100 smpl:13K ep:40 epch:0.20 loss:1.611 grdn:40.355 lr:1.0e-04 updt_s:0.210 data_s:0.000 fwd:0.093s bwd:0.110s opt:0.002s l1:0.339 kl:0.127
```

**各字段含义**:

| 字段 | 说明 | 期望值 |
|------|------|--------|
| `step` | 当前训练步数 | 0 → 30000 |
| `smpl` | 已见样本数 | ≈ step × batch_size |
| `loss` | 总损失 `l1 + 10*kl` | 从 90 降至 0.2-0.5 |
| `grdn` | 梯度范数(已被 clip_norm=10)| 20-50 正常,>50 注意 |
| `updt_s` | 单步总耗时(秒)| **~0.21s(目标)** |
| `data_s` | 数据加载耗时 | **~0.000s(完全 InMemory)** |
| `transfer_s` | CPU→GPU 传输 | ~0.008s |
| `fwd` | forward 耗时 | ~0.07s |
| `bwd` | backward 耗时 | ~0.11s |
| `opt` | optimizer.step 耗时 | ~0.002s |
| `l1` | L1 损失分项 | 0.1-0.3 |
| `kl` | KL 损失分项 | 0.02-0.05 |

**4 段耗时时间线诊断**:
- `data_s ≈ 0`:数据加载不是瓶颈(目标)
- `fwd + bwd + opt ≈ 0.18s`:GPU 计算时间(理论下限)
- `data_s + (fwd+bwd+opt) ≈ 0.21s`:理论单步时间
- 实际 `updt_s ≈ 0.21s`:已达到理论极限

**Checkpoint 保存位置**:`<lerobot>/outputs/train/<date>/<time>_act/checkpoints/`

```
outputs/train/2026-07-20/13-31-27_act/checkpoints/
├── 005000/pretrained_model/
├── 010000/pretrained_model/
├── 015000/pretrained_model/
└── last → 015000/
```

#### Step 5: 收敛判断标准

| 指标 | 收敛标准 | 异常信号 |
|------|---------|---------|
| `train_loss` | 从 ~90 降至 < 0.5 | NaN / Inf / loss 不降 |
| `l1_loss` | 稳定在 0.1-0.2 | 持续 > 0.5 |
| `kl_loss` | 稳定在 0.02-0.05 | 持续 > 0.1 或快速增长 |
| `grad_norm` | 稳定在 20-50 | > 100(梯度爆炸)|
| `data_s` | < 0.001 | > 0.1(数据加载瓶颈)|
| `updt_s` | ~0.21s(单卡)| > 0.5s(异常)|

**实际收敛曲线(500 步测试)**:

| Step | loss | l1 | kl | grdn | epoch |
|------|------|----|----|------|-------|
| 0 | 90.8 | 0.68 | 9.01 | 1220 | 0.00 |
| 10 | 5.37 | 0.60 | 0.48 | 122 | 0.02 |
| 100 | 1.61 | 0.34 | 0.13 | 40 | 0.20 |
| 200 | 0.85 | 0.20 | 0.07 | 33 | 0.40 |
| 300 | 0.50 | 0.16 | 0.03 | 28 | 0.60 |
| 450 | 0.32 | 0.13 | 0.02 | 23 | 0.89 |

**预估 30K 步收敛 loss ≈ 0.1-0.2**(论文值)。

---

### 场景 D: Phase 4 — 评估训练好的模型

适用于:将 Phase 3 保存的 checkpoint 加载到 VLABench 仿真环境,运行指定任务评测成功率。

#### Step 1: 找到最新 checkpoint

```bash
# LeRobot 自动创建 outputs/train/<date>/<time>_act/
ls -lt /ssd/qinmaokai/workspace/lerobot/outputs/train/ | head -5

# 找最新 checkpoint
LATEST=$(ls -td /ssd/qinmaokai/workspace/lerobot/outputs/train/*/*_act/ | head -1)
echo "Latest run: $LATEST"
ls $LATEST/checkpoints/
```

#### Step 2: 修复 config.json(⚠️ 必须先做)

LeRobot 训练时将 `"type": "act"` 存在 `train_config.json`,但 `pretrained_model/config.json` **缺此字段**。直接用 `evaluate_policy.py` 会报错或加载 RandomPolicy。

**Python 一键修复**:
```python
import json, os

# 用 Step 1 找到的最新路径
latest_run = "/ssd/qinmaokai/workspace/lerobot/outputs/train/2026-07-20/13-31-27_act"
ckpt_step = "030000"  # 选最新,或���比 5000/10000/15000/30000
config_path = os.path.join(latest_run, f"checkpoints/{ckpt_step}/pretrained_model/config.json")

with open(config_path) as f:
    cfg = json.load(f)

cfg["type"] = "act"  # 添加缺失字段

with open(config_path, "w") as f:
    json.dump(cfg, f, indent=4)

print(f"✓ 已修复: {config_path}")
```

#### Step 3: 启动评估

```bash
cd /ssd/qinmaokai/workspace/SciVLABench

MUJOCO_GL=osmesa python scripts/evaluate_policy.py \
    --tasks <task_name> \
    --n-episode 10 \
    --policy act \
    --model_ckpt /ssd/qinmaokai/workspace/lerobot/outputs/train/2026-07-20/13-31-27_act/checkpoints/030000/pretrained_model \
    --replanstep 4 \
    --save-dir logs/act_<task>_eval \
    --visulization \
    --metrics success_rate intention_score progress_score
```

#### Step 4: 关键参数说明

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--policy` | `act` | 选择 ACT 策略 |
| `--model_ckpt` | `.../pretrained_model/` | **必须是 `pretrained_model/` 目录**,不是 `checkpoints/030000/` |
| `--tasks` | `<task_name>` | 评测任务名(可多个,如 `lift_beaker place_beaker`) |
| `--n-episode` | 10–50 | 每个任务评测多少个 episode |
| `--replanstep` | 4 | 每多少步重新查询策略(类似 OpenPI) |
| `--visulization` | (flag)| 录制仿真视频保存到 `videos/` 目录 |
| `--save-dir` | `logs/act_<task>_eval` | 评测结果输出目录 |

> ⚠️ `MUJOCO_GL=osmesa` 是关键!不用 egl/X11,避免 `gladLoadGL error`。

#### Step 5: 评估输出结构

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

#### Step 6: 解析评测结果

```python
import json

with open("logs/act_<task>_eval/metrics.json") as f:
    results = json.load(f)
    for task, metrics in results.items():
        sr = metrics["success_rate"]
        n = int(round(sr * 10))  # 假设 n_episode=10
        print(f"{task}: {sr:.1%} ({n}/10)")
```

**实战评测结果参考**:

| 任务 | 训练数据 | Episodes | 成功率 | 说明 |
|------|---------|---------|--------|------|
| lift_beaker | 未在训练集 | 3 | **100% (3/3)** | 跨任务泛化 |
| place_beaker | 在训练集 | 3 | **33.3% (1/3)** | 多步任务较难 |

---

## 4. 关键避坑指南与隐性知识

### 4.1 config.json 缺失 "type": "act" 字段 ⚠️

- **❌ 错误现象**:`evaluate_policy.py` 加载时报 `ParsingError: Expected a dict with a 'type' key` 或默认加载 RandomPolicy(eval 显示 0%)
- **✅ 解决**:评估前在 `config.json` 顶部手动添加 `"type": "act",`
- **原因**:LeRobot 训练时 `type` 字段存在 `train_config.json`,但 `pretrained_model/config.json` 生成时未复制

### 4.2 MUJOCO_GL 环境变量

- **❌ 错误**:`MUJOCO_GL=egl` 在 SSH 无 X11 环境报错 `gladLoadGL error`
- **✅ 解决**:
  - Phase 1(数据采集)用 `MUJOCO_GL=egl`(需要 GPU 加速)
  - Phase 4(评估推理)用 `MUJOCO_GL=osmesa`(纯 CPU 软件渲染,无依赖)

### 4.3 坐标系变换 (world ↔ robot-relative)

| 方向 | 代码位置 | 操作 |
|------|---------|------|
| 训练时(world→robot) | `convert_to_lerobot.py` | `ee_pos -= robot_frame_pos` |
| 推理时(robot→world) | `act.py` `select_action()` | `target_pos = action[:3] + robot_frame` |

- 默认 `robot_frame_pos = [0, -0.4, 0.78]`
- 若变换不一致,推理时机械臂会飞走(动作偏移 ~40cm)

### 4.4 Action 格式转换

| 步骤 | 格式 | Shape | 说明 |
|------|------|-------|------|
| HDF5 原始 | VLABench 8D | `(T, 8)` | `[pos3, euler3, gripper1, gripper2]` |
| 转换后 | LeRobot 7D | `(T, 7)` | `[pos3, euler3, gripper1]` |

- 转换取 `action[:6]` + `action[6]`(第一个 gripper 值)
- Gripper 二值化阈值 `0.03`:`> 0.03` → 张开(1),否则闭合(0)

### 4.5 相机索引约定

- VLABench 4 相机:`[0, 1, 2, 3]`
- ACT 使用:`[2]` = front (前视) + `[3]` = wrist (腕部)
- `convert_to_lerobot.py` 中:`images[i][2]` → `observation.image`,`images[i][3]` → `observation.wrist_image`

### 4.6 LeRobot Dataset 目录不可覆盖

- **❌ 错误**:`LeRobotDataset.create(repo_id=<已有名称>)` 报错 `FileExistsError`
- **✅ 解决**:转换前先 `rm -rf ~/.cache/huggingface/lerobot/<repo_id>`,或用 `train_act.py --skip-conv` 让脚本自动清理

### 4.7 输出目录不可预创建

- **❌ 错误**:先 `mkdir -p "$OUTPUT_DIR"` 再启动训练
- **✅ 解决**:`train_act.py` 已经不传 `--output_dir`,让 LeRobot 自动生成 `outputs/train/<date>/`。若非要手动指定,LeRobot 会因目录已存在而报错 `FileExistsError`

### 4.8 CLI 布尔值格式(Draccus parser)

- **❌ 错误**:`--dataset.local_files_only=True`(Python 风格)
- **✅ 正确**:`--dataset.local_files_only=true`(Draccus YAML 风格,小写)

### 4.9 InMemory Dataset 内存占用

预加载到 RAM 的数据量(256×256 uint8):
- image (2 相机) = 65K × 256 × 256 × 3 = 12.75 GB × 2 = **25.5 GB**
- state/action: < 1 GB
- 总计: **~26 GB**

服务器有 2 TB RAM,完全够用。如果用 480×480,占用会变成 60 GB,仍可接受。

### 4.10 训练量与收敛的关系

| 训练量 | Loss (l1) | Success Rate(预估) |
|--------|-----------|-------------------|
| 500 步 | 0.13 | 0%(未收敛) |
| 5K 步 | ~0.10 | ~10-30% |
| 15K 步 | ~0.05 | ~30-50% |
| **30K 步** | **~0.02-0.05** | **50-80%** |
| 100K 步 | ~0.01 | 70-100%(参考早期实验) |

> ⚠️ 经验值。实际收敛速度取决于任务难度和轨迹质量。

### 4.11 freeze_backbone 默认关闭

- 实验中发现 `freeze_backbone=True` 会导致 `loss=nan` 概率上升(即使 AMP 关闭)
- **推荐保持默认 False**(不冻结 ResNet18),除非有证据显示训练稳定
- 如果想要加速,可以加 `--freeze-encoder` 参数显式开启

### 4.12 常见错误速查表

| 错误 | 原因 | 解决 |
|------|------|------|
| `RepositoryNotFoundError: 401 Unauthorized` | 没设 `local_files_only=true` | 添加 `--dataset.local_files_only=true` |
| `ParsingError: Expected a dict with a 'type' key` | config.json 缺 type | 手动加 `"type": "act"` |
| `gladLoadGL error` | X11/EGL 不可用 | 改用 `MUJOCO_GL=osmesa` |
| `FileExistsError: output_dir already exists` | 预先 mkdir 了 | 不要预先创建输出目录 |
| 推理时机械臂飞走 | 坐标系变换不一致 | 确认 convert 和 eval 使用相同 robot_frame |
| loss = NaN(旧版 480) | 梯度爆炸 + action 维度错 | 用 v2 (256 + in_memory),loss 应正常 |
| `data_s > 1` | 数据加载瓶颈 | 加 `--use-in-memory` |
| 评估 0% 但 loss 正常 | config.json type 缺失 | 加 `"type": "act"` |
| `Couldn't parse 'False'` | CLI 用 Python bool | 改小写 `false` |
| `Tensors must have same number of dimensions` | InMemory 的 action 维度错 | 确认 InMemoryLeRobotDataset 处理 delta_indices |

---

## 5. 快速参考:完整命令序列

以下脚本可一键复制运行,完成 Phase 1–4 全流程(v2 优化版)。

```bash
#!/bin/bash
set -e
# ============================================================
# ACT 模型训练全流程 — A800 服务器 (v2 优化版)
# ============================================================
TASK_NAME="pick_cylinder_mid_pour_object"     # 任务名称(不带 _series)
N_EPISODES=200                                # 轨迹数量(推荐 200)
RESOLUTION=256                                 # 图像分辨率(256 推���)
GPU=2                                          # 单卡 GPU ID
TRAIN_STEPS=30000                              # 训练步数

VLABENCH_ROOT="/ssd/qinmaokai/workspace/SciVLABench"
LEROBOT_ROOT="/ssd/qinmaokai/workspace/lerobot"
HDF5_DIR="${VLABENCH_ROOT}/dataset/training_data"

# ============================================================
# Phase 1: 生成专家轨迹数据 (HDF5)
# ============================================================
echo "=== Phase 1: 生成 ${N_EPISODES} 条轨迹 ==="
cd ${VLABENCH_ROOT}

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    python scripts/trajectory_generation.py \
    --task-name ${TASK_NAME}_series \
    --n-sample ${N_EPISODES} \
    --start-id 0 \
    --save-dir ${HDF5_DIR} \
    --record-video \
    --robot franka

echo "  ✓ 轨迹已保存至 ${HDF5_DIR}/${TASK_NAME}_series/"
ls ${HDF5_DIR}/${TASK_NAME}_series/*.hdf5 2>/dev/null | wc -l

# ============================================================
# Phase 2: HDF5 → LeRobot Dataset 转换 (含 --resolution)
# ============================================================
echo "=== Phase 2: 转换格式 (${RESOLUTION}x${RESOLUTION}) ==="
cd ${VLABENCH_ROOT}

python scripts/model_train/train_act.py \
    --task ${TASK_NAME} \
    --num ${N_EPISODES} \
    --resolution ${RESOLUTION} \
    --skip-gen \
    --skip-train

echo "  ✓ LeRobot Dataset 已保存至 ~/.cache/huggingface/lerobot/vlabench_${TASK_NAME}_${N_EPISODES}ep_r${RESOLUTION}/"

# ============================================================
# Phase 3: 训练 (v2: 单卡 + InMemory)
# ============================================================
echo "=== Phase 3: 启动 ${GPU} 卡训练 ==="
cd ${VLABENCH_ROOT}

python scripts/model_train/train_act.py \
    --task ${TASK_NAME} \
    --num ${N_EPISODES} \
    --gpus ${GPU} \
    --resolution ${RESOLUTION} \
    --use-in-memory \
    --train-steps ${TRAIN_STEPS} \
    --batch-size 128 \
    --num-workers 8 \
    --save-freq 5000 \
    --skip-gen \
    --skip-conv

echo "  ✓ 训练完成,模型在 lerobot/outputs/train/<date>/<time>_act/"

# ============================================================
# Phase 4: 评估训练好的模型
# ============================================================
echo "=== Phase 4: 评估 ==="
cd ${VLABENCH_ROOT}

# 找到最新训练目录
LATEST=$(ls -td ${LEROBOT_ROOT}/outputs/train/*/*_act/ | head -1)
CKPT_STEP=$(ls ${LATEST}/checkpoints/ | grep -v last | sort | tail -1)
CKPT_DIR="${LATEST}/checkpoints/${CKPT_STEP}/pretrained_model"
echo "  使用 checkpoint: ${CKPT_DIR}"

# 修复 config.json
python -c "
import json
c = json.load(open('${CKPT_DIR}/config.json'))
c['type'] = 'act'
json.dump(c, open('${CKPT_DIR}/config.json', 'w'), indent=4)
print('✓ config.json 已修复')
"

# 启动评估
MUJOCO_GL=osmesa python scripts/evaluate_policy.py \
    --tasks ${TASK_NAME} \
    --n-episode 10 \
    --policy act \
    --model_ckpt ${CKPT_DIR} \
    --replanstep 4 \
    --save-dir logs/act_${TASK_NAME}_eval \
    --visulization \
    --metrics success_rate intention_score progress_score

echo ""
echo "✓ 全部流程完成!"
echo "  模型目录: ${LATEST}"
echo "  评测结果: logs/act_${TASK_NAME}_eval/metrics.json"
echo "  视频回放: logs/act_${TASK_NAME}_eval/${TASK_NAME}/videos/"
```

---

## 附录 A: VLABench 任务注册结构(参考)

新增自定义任务时,需在以下位置注册:

| 文件 | 路径 | 内容 |
|------|------|------|
| ConfigManager | `VLABench/tasks/autogen_tasks/primitive/<task_name>_series.py` | `load_objects()`、`get_instruction()`、`get_condition_config()` |
| Task 类 | 同上文件 | `get_expert_skill_sequence()` 返回 `partial(SkillLib.<skill>, ...)` |
| 注册装饰器 | 同上文件 | `@register.add_config_manager("<task_name>")` + `@register.add_task("<task_name>")` |
| constant.py | `VLABench/configs/constant.py` | `name2class_xml` 中注册模型 XML |

**参考示例**:`VLABench/tasks/autogen_tasks/primitive/lift_beaker_series.py`

## 附录 B: LeRobot ACT 模型架构参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `n_obs_steps` | 1 | 输入观测步数 |
| `chunk_size` | 100 | 动作序列分块大小 |
| `n_action_steps` | 100 | 输出动作序列长度 |
| `vision_backbone` | resnet18 | 视觉编码器(ImageNet 预训练) |
| `dim_model` | 512 | Transformer 隐层维度 |
| `n_heads` | 8 | 注意力头数 |
| `n_encoder_layers` | 4 | ViT encoder 层数 |
| `n_decoder_layers` | 1 | Action decoder 层数 |
| `use_vae` | True | 使用 VAE 潜变量 |
| `latent_dim` | 32 | VAE 潜变量维度 |
| 参数量 | ~51.6M | 总可训练参数 |

## 附录 C: 已知问题与限制

### C.1 LeRobotDataset 加载慢(原始版本)

`lerobot/common/datasets/lerobot_dataset.py:508-509` 有已知 bug:

```python
timestamps = torch.stack([self.hf_dataset[i]["timestamp"] for i in range(len(self.hf_dataset))]).numpy()
```

这行遍历整个 dataset (65K 帧),每个 `hf_dataset[i]` 访问会触发 PIL 解码。在 65K 帧上耗时 ~7-8 分钟。

**InMemoryLeRobotDataset 已绕过这个问题**,因为它不调用 `self.hf_dataset[idx]`。

### C.2 LeRobot 内部缓存机制

LeRobot 有内部缓存(`images_writer` 等),可能会临时占用额外磁盘。但 `InMemory` 模式后,这些缓存由 `train_act.py` 的 `rm -rf` 自动清理。

### C.3 多卡 DDP 未测试

当前 v2 优化仅在**单卡 A800** 上验证。理论上 6 卡 DDP 可以达到 **~18 分钟**,但需要进一步测试 DDP + InMemory 的兼容性(`num_workers` 需要等比缩放)。

---

## 附录 D: 性能优化记录(从 40s/步 → 0.21s/步)

### D.1 背景与问题发现

2026-07-17 首次运行 ACT 训练,发现 **单步时间 40 秒,30K 步预估 357 小时**。GPU 0% 利用率,数据加载是瓶颈。多次调整 `num_workers` 和 `batch_size` 都无法突破。

### D.2 排查过程(8 个对照实验)

| 实验 | 配置 | 单步 | GPU 状态 | 备注 |
|------|------|------|---------|------|
| **1** | 480, batch=2, num_workers=2 | 5.4s | 闲置 | 原始 baseline |
| **2** | 480, batch=16, num_workers=4, AMP, 6 卡 | 5.4s | 闲置 | 多卡没用 |
| **3** | 480, batch=256, num_workers=64, AMP, 1 卡 | **40s** | **0%** | **灾难**:worker 太多死锁 |
| **4** | 256, batch=128, num_workers=8, in_memory v1, freeze, AMP | 6.6s | 0% | 改善但仍慢 |

### D.3 关键诊断(由经验丰富的同事提出)

诊断三个核心问题:

**1. InMemory 没有真正生效**
- 第一版 InMemoryLeRobotDataset **继承了 LeRobotDataset**,`__getitem__` 调用 `super().__getitem__()`,后者访问 `self.hf_dataset[idx]`,仍然触发 arrow/parquet 重活
- **修正**:完全重写 InMemoryLeRobotDataset,**不继承父类**,直接预加载到 numpy 数组,`__getitem__` 直接 numpy 索引

**2. 隐藏的 5.5 秒等待时间失踪**
- 实测时间 6.6 秒,但 `data_s(0.9) + transfer_s(0.008) + updt_s(0.12) = 1.03 秒`,有 5.5 秒不知道去哪了
- 实际上这部分就是 InMemory v1 没生效导致的额外开销
- 修正后 `data_s = 0.000s`,不再有 5.5 秒失踪

**3. KL/L1 分项打印**
- 加 `forward_s/backward_s/optimizer_s/l1_loss/kld_loss` 打印
- 发现 action 维度不匹配是 loss=nan 的原因
- 修正 InMemoryLeRobotDataset 的 `__getitem__` 处理 `delta_indices`,正确返回 `(chunk_size, action_dim)` 的 action 序列

### D.4 关键代码改动

| 文件 | 改动 | 影响 |
|------|------|------|
| `scripts/convert_to_lerobot_act.py` | 加 `--resolution` 参数 | 转换时 PIL resize 到 256 |
| `lerobot/common/datasets/in_memory_lerobot.py` (新) | **完全重写,不继承父类** | 100% numpy 路径 |
| `lerobot/common/datasets/factory.py` | 加 `use_in_memory` 分支 | 路由到新 dataset |
| `lerobot/common/policies/act/configuration_act.py` | 加 `freeze_backbone`, `use_ema` | 实验性 |
| `lerobot/common/policies/act/modeling_act.py` | 实现 `freeze_backbone` 逻辑 | 实验性 |
| `lerobot/scripts/train.py` | 加 `forward_s/backward_s/optimizer_s/l1_loss/kld_loss` 打印 | 诊断输出 |
| `scripts/model_train/train_act.py` | 加 `--resolution`, `--use-in-memory`, `--freeze-encoder`, `--use-ema`, `--num-workers`, `--batch-size` | 一键训练入口 |

### D.5 实测速度对比

| 配置 | 单步时间 | 30K 步预估 | loss 行为 |
|------|---------|-----------|---------|
| 实验 3(480, batch=256, num_workers=64, AMP) | 40s | 357h | nan |
| 实验 4(256, batch=128, num_workers=8, InMemory v1) | 6.6s | 55h | 下降但仍有 nan |
| **v2(256, batch=128, num_workers=8, InMemory v2)** | **0.21s** | **1.75h** | **正常收敛** |
| 6 卡 DDP 预估 | 0.035s | 17min | - |

### D.6 关键经验教训

1. **不要继承父类做 wrapper**:LeRobotDataset 的 `__getitem__` 是性能陷阱,继承会让你无法绕过它的 I/O
2. **InMemory 必须预加载所有字段**(不只是 image):state/action 也得在 RAM 里,否则仍然触发 parquet 读
3. **`delta_indices` 必须正确处理**:ACT 用 chunk_size=100 预测,需要返回 (S, action_dim) 而不是 (action_dim,)
4. **GPU 不慢,是 CPU 慢**:GPU 计算时间 0.1s,数据准备时间 6s,优化数据是主战场
5. **不要盲目增加 num_workers**:多了反而死锁,8 个 worker 是甜点
6. **AMP 数值问题与 freeze_backbone 无关**:两者都是 fp16 训练可能的 nan 来源,但根本问题可能是 action 维度不匹配
7. **log_freq=100 + 短训练是陷阱**:看不到 loss 曲线,会误判 nan 在哪一步开始
8. **完全独立的 InMemory dataset 性能提升 32 倍**:从 6.6s/步 降到 0.21s/步

### D.7 v2 后的改进方向(未实施)

1. **6 卡 DDP**:测试 `num_workers=8×6=48` 的 DDP + InMemory 兼容性
2. **BF16 而非 FP16**:FP16 的数值问题比 BF16 多,可以重新打开 AMP
3. **chunk_size 调优**:当前 100,可测试 50 是否够用
4. **冻结 ResNet18**:验证收敛稳定后再启用以加速

---

*最后更新:2026-07-20(InMemory v2 优化完成)*