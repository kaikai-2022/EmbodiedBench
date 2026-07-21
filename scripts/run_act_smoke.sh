#!/bin/bash
# ACT 模型冒烟测试：1 条轨迹 → LeRobot 转换 → 单卡 500 步训练
#
# 用法: bash scripts/run_act_smoke.sh
#
# 关键设计:
#   - 单 GPU (CUDA_VISIBLE_DEVICES=0)
#   - 1 条专家轨迹
#   - --offline.steps=500 (用步数，不用 epoch)
#   - --eval_freq=1000000 跳过 eval（数据太少）
#   - 大规模训练再切 --nproc_per_node=6 DDP + 50 epochs

set -e

# ============== 配置 ==============
TASK=pick_cylinder_mid_pour_object
SERIES="${TASK}_series"
REPO_ID="vlabench_${TASK}_1ep_smoke"
DATE=$(date +%Y%m%d_%H%M%S)
N_SAMPLES=1
TRAIN_STEPS=500

VLABENCH_ROOT=/ssd/qinmaokai/workspace/SciVLABench
LEROBOT_ROOT=/ssd/qinmaokai/workspace/lerobot
HDF5_DIR="${VLABENCH_ROOT}/dataset/training_data/${SERIES}"
LEROBOT_CACHE_DIR="${VLABENCH_ROOT}/.cache/huggingface/lerobot"
LEROBOT_CACHE="${LEROBOT_CACHE_DIR}/${REPO_ID}"
MODEL_DIR="${VLABENCH_ROOT}/models/act_${TASK}_smoke_${DATE}"
TRAIN_LOG=/tmp/train_${TASK}_smoke.log

echo "=========================================="
echo "  ACT 模型冒烟测试"
echo "  任务: ${TASK}"
echo "  REPO_ID: ${REPO_ID}"
echo "  输出: ${MODEL_DIR}"
echo "=========================================="

# ============== 准备环境 ==============
source /ssd/qinmaokai/miniconda3_new/etc/profile.d/conda.sh
conda activate vlabench_2
export PYTHONPATH="${LEROBOT_ROOT}:${VLABENCH_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${VLABENCH_ROOT}/.cache/huggingface"
export HF_LEROBOT_HOME="${LEROBOT_CACHE_DIR}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p "${LEROBOT_CACHE_DIR}" "${VLABENCH_ROOT}/models" "${VLABENCH_ROOT}/dataset/training_data"

# ============== Phase 1: 采集 1 条轨迹 ==============
echo ""
echo "=== Phase 1: 采集 1 条轨迹 ==="
cd "${VLABENCH_ROOT}"
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    python scripts/trajectory_generation.py \
    --task-name "${SERIES}" \
    --n-sample ${N_SAMPLES} \
    --start-id 0 \
    --save-dir "${VLABENCH_ROOT}/dataset/training_data" \
    --record-video \
    --robot franka

echo ""
echo "  生成的轨迹文件:"
ls -la "${HDF5_DIR}/" 2>&1 || echo "  ✗ 没有生成 HDF5 文件（任务可能失败）"
HDF5_COUNT=$(find "${HDF5_DIR}" -name "*.hdf5" 2>/dev/null | wc -l)
echo "  HDF5 文件数: ${HDF5_COUNT}"

# ============== Phase 2: 转 LeRobot 格式 ==============
echo ""
echo "=== Phase 2: HDF5 → LeRobot ==="
echo "  ⚠️  先删除旧 cache: ${LEROBOT_CACHE}"
rm -rf "${LEROBOT_CACHE}"

cd "${VLABENCH_ROOT}"
python scripts/convert_to_lerobot_act.py \
    --dataset-name "${REPO_ID}" \
    --dataset-path "${VLABENCH_ROOT}/dataset/training_data" \
    --max-files ${N_SAMPLES} \
    --task-list "${SERIES}"

echo ""
echo "  LeRobot dataset: ${LEROBOT_CACHE}"
ls "${LEROBOT_CACHE}/" 2>&1 | head -10
ls "${LEROBOT_CACHE}/meta/" 2>&1 | head -5

# ============== Phase 3: 单卡训练 ==============
echo ""
echo "=== Phase 3: 单卡 ${TRAIN_STEPS} 步训练 (GPU 0) ==="
cd "${LEROBOT_ROOT}"

CUDA_VISIBLE_DEVICES=0 \
    python lerobot/scripts/train.py \
    --policy.type=act \
    --dataset.repo_id="${REPO_ID}" \
    --dataset.local_files_only=true \
    --dataset.image_transforms.enable=false \
    --dataset.use_imagenet_stats=true \
    --output_dir="${MODEL_DIR}" \
    --batch_size=2 \
    --num_workers=2 \
    --offline.steps=${TRAIN_STEPS} \
    --save_freq=${TRAIN_STEPS} \
    --eval_freq=1000000 \
    --log_freq=20 \
    --device=cuda \
    --seed=1000 \
    > "${TRAIN_LOG}" 2>&1

echo ""
echo "=== ✓ 冒烟测试完成 ==="
echo "  模型目录: ${MODEL_DIR}"
echo "  训练日志: ${TRAIN_LOG}"
echo ""
echo "  查看 checkpoints:"
ls -la "${MODEL_DIR}/checkpoints/" 2>&1 || echo "  (无)"
echo ""
echo "  Loss 摘要:"
grep -E "loss:" "${TRAIN_LOG}" 2>&1 | tail -5 || echo "  (未找到 loss 行)"
