#!/bin/bash
# 在已部署的 pi0.5 (pi05) 上继续 fine-tune，用于单任务（place_beaker_pres_heat_device）。
#
# 用法:
#   ./run_pi05_train.sh                                              # 默认配置 + 4 卡 FSDP
#   ./run_pi05_train.sh --fsdp_devices=1                            # 单卡
#   ./run_pi05_train.sh --num_train_steps=3000 --batch_size=4       # 改超参
#   ./run_pi05_train.sh compute-norm-stats                          # 只算 norm_stats
#   ./run_pi05_train.sh eval --ckpt checkpoints/.../29999           # 训完跑评测
#
# 训练产物:
#   - $OPENPI_DATA_HOME/checkpoints/<config>/<exp_name>/<step>/     JAX checkpoint
#   - $OPENPI_DATA_HOME/assets/<config>/<repo_id>/norm_stats.json  norm_stats
set -e

CONFIG=${CONFIG:-pi05_ft_place_beaker}
EXP_NAME=${EXP_NAME:-${CONFIG}_$(date +%Y%m%d_%H%M%S)}
GPUS=${GPUS:-0,1,2,3}
FSDP_DEVICES=${FSDP_DEVICES:-1}   # 8 卡 FSDP 用 FSDP_DEVICES=8 GPUS=0,1,2,3,4,5,6,7

OPENPI_DIR=/ssd/qinmaokai/workspace/SciVLABench/third_party/openpi
VLABENCH_DIR=/ssd/qinmaokai/workspace/SciVLABench
SERVER_VENV=${OPENPI_DIR}/examples/vlabench/.venv

# 数据/模型缓存根目录
export OPENPI_DATA_HOME=/ssd/qinmaokai/.cache/openpi
export HF_HOME=/ssd/qinmaokai/.cache/huggingface
export HF_LEROBOT_HOME=/ssd/qinmaokai/.cache/huggingface/lerobot
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl
export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export CUDA_VISIBLE_DEVICES=${GPUS}

# norm_stats 和 checkpoint 落到 OPENPI_DATA_HOME 下
ASSETS_BASE="${OPENPI_DATA_HOME}/assets"
CKPT_BASE="${OPENPI_DATA_HOME}/checkpoints"
mkdir -p "${ASSETS_BASE}" "${CKPT_BASE}"

# 激活 venv
# shellcheck disable=SC1091
source "${SERVER_VENV}/bin/activate"
export PYTHONPATH=${OPENPI_DIR}/src:${VLABENCH_DIR}:${PYTHONPATH:-}

cd "${OPENPI_DIR}"

case "${1:-train}" in
    compute-norm-stats)
        echo "[norm_stats] computing for config=${CONFIG}"
        # compute_norm_stats.py 写到 cwd 下的 ./assets/<config>/<repo_id>/，
        # 跑完后挪到 OPENPI_DATA_HOME 下。
        python scripts/compute_norm_stats.py --config-name="${CONFIG}"
        mkdir -p "${ASSETS_BASE}/${CONFIG}"
        cp -r "./assets/${CONFIG}/"* "${ASSETS_BASE}/${CONFIG}/" 2>/dev/null || true
        rm -rf "./assets/${CONFIG}" 2>/dev/null || true
        echo "[norm_stats] saved to ${ASSETS_BASE}/${CONFIG}/"
        ;;

    train)
        # 1. 先算 norm_stats（必须先做，否则 train 启动时会找不到 norm_stats.json）
        REPO_ID=$(python -c "from openpi.training import config; print(config.get_config('${CONFIG}').data.repo_id)")
        if [ ! -f "${ASSETS_BASE}/${CONFIG}/${REPO_ID}/norm_stats.json" ]; then
            echo "[train] norm_stats missing, computing first..."
            python scripts/compute_norm_stats.py --config-name="${CONFIG}"
            mkdir -p "${ASSETS_BASE}/${CONFIG}"
            cp -r "./assets/${CONFIG}/"* "${ASSETS_BASE}/${CONFIG}/" 2>/dev/null || true
            rm -rf "./assets/${CONFIG}" 2>/dev/null || true
        else
            echo "[train] norm_stats already exists, skipping computation"
        fi

        # 2. 启动训练
        shift || true
        echo "[train] config=${CONFIG}, exp_name=${EXP_NAME}, gpus=${GPUS}, fsdp_devices=${FSDP_DEVICES}"
        # train.py 通过 --assets_base_dir / --checkpoint_base_dir 接收绝对路径，
        # 这样不用靠 ./assets 相对路径，避免 symlink 解析问题。
        cd "${OPENPI_DIR}"
        python scripts/train.py "${CONFIG}" \
            --exp-name="${EXP_NAME}" \
            --assets_base_dir="${ASSETS_BASE}" \
            --checkpoint_base_dir="${CKPT_BASE}" \
            --fsdp_devices="${FSDP_DEVICES}" \
            --overwrite \
            "$@"
        ;;

    eval)
        # 训练完后评测：用新 ckpt 启动 server，跑 evaluate_openpi.py
        shift || true
        CKPT=""
        N_EP=5
        TASK=place_beaker_pres_heat_device
        while [ $# -gt 0 ]; do
            case "$1" in
                --ckpt) CKPT="$2"; shift 2 ;;
                --n_ep) N_EP="$2"; shift 2 ;;
                --task) TASK="$2"; shift 2 ;;
                *) echo "Unknown arg: $1"; exit 1 ;;
            esac
        done
        if [ -z "${CKPT}" ]; then
            # 用最新一个 step
            CKPT=$(ls -d "${CKPT_BASE}/${CONFIG}/"*/*/ 2>/dev/null | sort -V | tail -1)
            if [ -z "${CKPT}" ]; then
                echo "No checkpoint found under ${CKPT_BASE}/${CONFIG}/"; exit 1
            fi
            echo "[eval] using latest checkpoint: ${CKPT}"
        fi

        # 启动 server（后台）
        export PYTHONPATH=${OPENPI_DIR}/src:${VLABENCH_DIR}:${PYTHONPATH:-}
        nohup python scripts/serve_policy.py --port 8000 \
            --env VLABENCH policy:checkpoint \
            --policy.config="${CONFIG}" \
            --policy.dir="${CKPT}" \
            > /tmp/pi05_server.log 2>&1 &
        SERVER_PID=$!
        trap "kill ${SERVER_PID} 2>/dev/null" EXIT
        echo "[eval] server pid=${SERVER_PID}, waiting for 'listening on 0.0.0.0:8000'..."
        for i in $(seq 1 60); do
            if grep -q "listening on 0.0.0.0:8000" /tmp/pi05_server.log 2>/dev/null; then
                echo "[eval] server ready"
                break
            fi
            sleep 2
        done

        # 跑 client 评测
        # shellcheck disable=SC1091
        source /opt/miniconda3/etc/profile.d/conda.sh
        conda activate vlabench_2
        export PYTHONPATH=${OPENPI_DIR}/src:${VLABENCH_DIR}:${PYTHONPATH:-}
        SAVE_DIR=${SAVE_DIR:-${VLABENCH_DIR}/logs/pi05_train_eval/$(date +%Y%m%d_%H%M%S)}
        cd "${VLABENCH_DIR}"
        python scripts/evaluate_openpi.py \
            --host localhost --port 8000 \
            --tasks "${TASK}" \
            --n_episode "${N_EP}" \
            --save_dir "${SAVE_DIR}" \
            --visualization
        echo "[eval] results: ${SAVE_DIR}"
        ;;

    *)
        cat <<EOF
Usage: $0 {train|compute-norm-stats|eval} [args]
  train [--fsdp_devices=N --num_train_steps=N --batch_size=N ...]
  compute-norm-stats
  eval --ckpt /path/to/checkpoint [--n_ep N] [--task TASK]
EOF
        exit 1
        ;;
esac