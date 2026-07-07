# DP 模型训练与部署标准工作流指南 (SOP)

## 1. 概述

本指南描述如何利用 VLABench 的专家轨迹数据，通过 LeRobot 框架训练 Diffusion Policy (DP) 模型，并在 VLABench 中完成评测。

**核心优势**：DP 与 ACT 共用同一套训练框架（LeRobot），只需切换 `--policy.type` 参数。

**完整链路**：

```
VLABench 任务
  → scripts/trajectory_generation.py (专家轨迹采集)
  → HDF5 文件 (dataset/hdf5_trajectory/<task_name>/)
  → scripts/convert_to_lerobot.py (格式转换)
  → LeRobot Dataset (~/.cache/huggingface/lerobot/<repo_id>/)
  → lerobot/scripts/train.py --policy.type=diffusion (DP 训练)
  → DP Checkpoint (models/diffusion_<task>_<N>ep_<date>/checkpoints/)
  → scripts/evaluate_policy.py --policy dp (推理评测)
  → success_rate
```

**项目根目录**：
- VLABench：`/ssd/mkqin/workspace/VLABench`
- LeRobot：`/ssd/mkqin/workspace/lerobot`

---

## 2. DP vs ACT 训练对比

| 项目 | ACT | DP |
|------|-----|-----|
| **Policy Type** | `--policy.type=act` | `--policy.type=diffusion` |
| **模型架构** | Transformer (CVAE) | 1D UNet + DDPM |
| **数据格式** | 完全相同 | 完全相同 |
| **转换脚本** | `convert_to_lerobot.py` | `convert_to_lerobot.py` |
| **训练框架** | LeRobot `train.py` | LeRobot `train.py` |
| **LeRobot 实现** | `lerobot/common/policies/act/` | `lerobot/common/policies/diffusion/` |
| **输入格式** | `observation.image`, `observation.wrist_image`, `observation.state`, `action` | 完全相同 |
| **推理速度** | 快 (~2ms) | 慢 (~100ms, diffusion 采样) |
| **推理接口** | `select_action()` | `select_action()` |

---

## 3. 标准操作步骤

### 场景 A: Phase 1 — 生成专家轨迹数据 (HDF5)

此步骤与 ACT 训练完全相同，已有轨迹数据可跳过。

```bash
cd /ssd/mkqin/workspace/VLABench

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    python scripts/trajectory_generation.py \
    --task-name <task_name> \
    --n-sample 50 \
    --start-id 0 \
    --save-dir /ssd/mkqin/workspace/VLABench/dataset/hdf5_trajectory \
    --record-video \
    --robot franka
```

### 场景 B: Phase 2 — HDF5 → LeRobot 格式转换

此步骤与 ACT 训练完全相同。

```bash
cd /ssd/mkqin/workspace/VLABench

python scripts/convert_to_lerobot.py \
    --dataset-name <repo_id> \
    --dataset-path /ssd/mkqin/workspace/VLABench/dataset/hdf5_trajectory \
    --max-files 500 \
    --task-list <task_name>
```

**注意**：DP 和 ACT 共用同一份 LeRobot Dataset，无需额外转换。

### 场景 C: Phase 3 — 多 GPU DDP 训练 DP 模型

#### Step 1: 启动 DDP 多卡训练

```bash
cd /ssd/mkqin/workspace/lerobot

OUTPUT_DIR="/ssd/mkqin/workspace/VLABench/models/diffusion_<task>_<N>ep_$(date +%Y%m%d_%H%M%S)"

torchrun --standalone --nproc_per_node=4 lerobot/scripts/train.py \
    --policy.type=diffusion \
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

#### Step 2: 关键参数说明

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--policy.type` | `diffusion` | **关键区别**：选择 DP 而非 ACT |
| `--nproc_per_node` | 4 | GPU 数量 |
| `--batch_size` | 2 | 每 GPU batch size |
| `--num_workers` | 2 | 数据加载线程数 |
| `--n_epochs` | 50 | 训练轮数 |
| `--local_files_only` | true | 必须设为 true |
| `--image_transforms.enable` | false | 禁用数据增强 |

#### Step 3: DP 模型关键超参数（LeRobot 默认值）

| 参数 | LeRobot 默认值 | 说明 |
|------|----------------|------|
| `n_obs_steps` | 2 | 观测历史步数 |
| `horizon` | 16 | 动作预测序列长度 |
| `n_action_steps` | 8 | 单次输出动作步数 |
| `num_train_timesteps` | 100 | DDPM 训练步数 |
| `num_inference_steps` | 100 | DDPM 推理步数 |
| `vision_backbone` | resnet18 | 视觉编码器 |
| `down_dims` | (512,1024,2048) | UNet 下采样维度 |
| `diffusion_step_embed_dim` | 128 | 时间嵌入维度 |
| `crop_shape` | (84,84) | 默认裁剪，VLABench 设 None |

### 场景 D: Phase 4 — 评估训练好的模型

#### Step 1: 修复 config.json（⚠️ 必须先做）

```python
import json

ckpt_dir = "models/diffusion_<task>_<N>ep_<date>/checkpoints/<step>/pretrained_model"
config_path = os.path.join(ckpt_dir, "config.json")

with open(config_path) as f:
    cfg = json.load(f)

cfg["type"] = "diffusion"  # 添加缺失字段

with open(config_path, "w") as f:
    json.dump(cfg, f, indent=4)

print(f"✓ 已修复: {config_path}")
```

#### Step 2: 启动评估

```bash
cd /ssd/mkqin/workspace/VLABench

MUJOCO_GL=osmesa python scripts/evaluate_policy.py \
    --tasks <task_name> \
    --n-episode 10 \
    --policy dp \
    --model_ckpt /ssd/mkqin/workspace/VLABench/models/diffusion_<task>_<N>ep_<date>/checkpoints/<step>/pretrained_model \
    --replanstep 4 \
    --save-dir logs/dp_<task>_eval \
    --visulization \
    --metrics success_rate intention_score progress_score
```

---

## 4. 快速参考：完整命令序列

```bash
#!/bin/bash
set -e
# ============================================================
# DP 模型训练全流程 — 参数配置
# ============================================================
TASK_NAME="lift_small_beaker"                     # 任务名称
N_EPISODES=50                                      # 轨迹数量
REPO_ID="vlabench_${TASK_NAME}_${N_EPISODES}ep"    # LeRobot Dataset ID
DATE=$(date +%Y%m%d_%H%M%S)                        # 时间戳
MODEL_NAME="diffusion_${TASK_NAME}_${N_EPISODES}ep_${DATE}"  # 模型输出目录名
CKPT_STEP="010000"                                 # 评测用的 checkpoint 步数

VLABENCH_ROOT="/ssd/mkqin/workspace/VLABench"
LEROBOT_ROOT="/ssd/mkqin/workspace/lerobot"
HDF5_DIR="${VLABENCH_ROOT}/dataset/hdf5_trajectory"
MODEL_DIR="${VLABENCH_ROOT}/models/${MODEL_NAME}"
CKPT_DIR="${MODEL_DIR}/checkpoints/${CKPT_STEP}/pretrained_model"
LEROBOT_CACHE="${HOME}/.cache/huggingface/lerobot/${REPO_ID}"

# ============================================================
# Phase 1: 生成专家轨迹数据 (HDF5) - 如果没有轨迹数据
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

# ============================================================
# Phase 2: HDF5 → LeRobot Dataset 转换
# ============================================================
echo "=== Phase 2: 转换格式 ==="
rm -rf ${LEROBOT_CACHE}

python scripts/convert_to_lerobot.py \
    --dataset-name ${REPO_ID} \
    --dataset-path ${HDF5_DIR} \
    --max-files ${N_EPISODES} \
    --task-list ${TASK_NAME}

# ============================================================
# Phase 3: 多 GPU DDP 训练 DP 模型
# ============================================================
echo "=== Phase 3: 启动 4-GPU 训练 ==="
cd ${LEROBOT_ROOT}

torchrun --standalone --nproc_per_node=4 lerobot/scripts/train.py \
    --policy.type=diffusion \
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

# ============================================================
# Phase 4: 评估训练好的模型
# ============================================================
echo "=== Phase 4: 评估 ==="
cd ${VLABENCH_ROOT}

# ⚠️ 关键：修复 config.json 缺失的 "type": "diffusion" 字段
python -c "
import json, os
config_path = '${CKPT_DIR}/config.json'
with open(config_path) as f:
    cfg = json.load(f)
cfg['type'] = 'diffusion'
with open(config_path, 'w') as f:
    json.dump(cfg, f, indent=4)
print('✓ config.json 已修复')
"

# 启动评估
MUJOCO_GL=osmesa python scripts/evaluate_policy.py \
    --tasks ${TASK_NAME} \
    --n-episode 10 \
    --policy dp \
    --model_ckpt ${CKPT_DIR} \
    --replanstep 4 \
    --save-dir logs/dp_${TASK_NAME}_eval_${DATE} \
    --visulization \
    --metrics success_rate intention_score progress_score

echo "✓ 全部流程完成！"
```

---

## 5. 常见问题与解决

### 5.1 DP 推理速度慢

**原因**：DP 使用 DDPM 采样，需要 100 步去噪

**解决**：
- 减少 `num_inference_steps`（如设为 10）可大幅提速
- 但会降低动作质量

### 5.2 Zero-shot 测试报错

**原因**：LeRobot 需要完整的统计量（mean, std, min, max）

**解决**：已实现 dummy_stats，包含所有必需字段

### 5.3 训练 loss 不收敛

**检查项**：
- 数据格式是否正确
- 归一化参数是否正确加载
- 学习率是否合适

---

## 6. LeRobot DP 实现文件

| 文件 | 说明 |
|------|------|
| `lerobot/common/policies/diffusion/configuration_diffusion.py` | DiffusionConfig 配置类 |
| `lerobot/common/policies/diffusion/modeling_diffusion.py` | DiffusionPolicy 实现 |
| `lerobot/common/policies/factory.py` | 策略工厂（支持 diffusion） |
| `lerobot/scripts/train.py` | 训练入口（通用） |

---

*文档生成时间: 2026-07-07*
*集成完成时间: 2026-07-07*
