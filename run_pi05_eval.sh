#!/bin/bash
# 一键启动 pi0.5 (pi05) 评测：用 Shiduo-zh/openpi `pi05` 分支的 serve_policy.py 作 server，
# VLABench 的 vlabench_2 conda env 作 client，端口 8000。
#
# 用法:
#   ./run_pi05_eval.sh server [gpu_id]                              # 前台启动 server（默认 GPU 0）
#   ./run_pi05_eval.sh server-bg [gpu_id]                           # 后台启动 server，日志写到 /tmp/pi05_server.log
#   ./run_pi05_eval.sh eval <task_name> [n_episode]                 # 跑评测（默认 lift_beaker, 1 episode）
#   ./run_pi05_eval.sh eval-multi task1 task2 ... [n_episode]       # 评测多个任务
#   ./run_pi05_eval.sh stop                                         # 停掉后台 server
#
# 示例:
#   ./run_pi05_eval.sh server-bg
#   tail -f /tmp/pi05_server.log                  # 等 "server listening on 0.0.0.0:8000"
#   ./run_pi05_eval.sh eval lift_beaker 1
#   ./run_pi05_eval.sh eval-multi lift_beaker lift_flask pick_glass_stirring_rod_insert_glass_stirring_rod 3
#   ./run_pi05_eval.sh stop
set -e

PORT=8000
DEFAULT_TASK=lift_beaker
DEFAULT_N_EPISODE=1
CHECKPOINT=/ssd/qinmaokai/.cache/openpi/vlabench_checkpoints/pi05-primitive-10task
SAVE_DIR=/ssd/qinmaokai/workspace/SciVLABench/logs/pi05_eval

OPENPI_DIR=/ssd/qinmaokai/workspace/SciVLABench/third_party/openpi
VLABENCH_DIR=/ssd/qinmaokai/workspace/SciVLABench
SERVER_VENV=${OPENPI_DIR}/examples/vlabench/.venv
SERVER_LOG=/tmp/pi05_server.log
SERVER_PID_FILE=/tmp/pi05_server.pid

start_server() {
    local gpu_id="${1:-0}"
    # shellcheck disable=SC1091
    source "${SERVER_VENV}/bin/activate"
    export PYTHONPATH=${OPENPI_DIR}/src:${VLABENCH_DIR}:${PYTHONPATH:-}
    export HF_ENDPOINT=https://hf-mirror.com
    export MUJOCO_GL=egl
    export CUDA_VISIBLE_DEVICES=$gpu_id

    # 不依赖 OPENPI_DATA_HOME/HF_HOME：Fine-tuning 时由 --policy.dir 自动覆盖
    unset OPENPI_DATA_HOME HF_HOME HF_LEROBOT_HOME
    unset HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE HF_HUB_OFFLINE

    cd "${OPENPI_DIR}"
    exec python scripts/serve_policy.py \
        --port "${PORT}" \
        --env VLABENCH policy:checkpoint \
        --policy.config=pi05_ft_vlabench_primitive \
        --policy.dir="${CHECKPOINT}"
}

start_server_bg() {
    local gpu_id="${1:-0}"
    nohup "$0" server "${gpu_id}" > "${SERVER_LOG}" 2>&1 &
    echo $! > "${SERVER_PID_FILE}"
    echo "Server started in background. PID=$(cat ${SERVER_PID_FILE})"
    echo "Log: ${SERVER_LOG}"
    echo "Wait for 'server listening on 0.0.0.0:${PORT}' then run: $0 eval ${DEFAULT_TASK}"
}

run_eval() {
    local tasks=("$@")
    local n_ep="${tasks[-1]}"
    if [[ "$n_ep" =~ ^[0-9]+$ ]]; then
        unset 'tasks[-1]'
    else
        n_ep=${DEFAULT_N_EPISODE}
    fi
    if [ ${#tasks[@]} -eq 0 ]; then
        tasks=("${DEFAULT_TASK}")
    fi

    # shellcheck disable=SC1091
    source /opt/miniconda3/etc/profile.d/conda.sh
    conda activate vlabench_2
    export PYTHONPATH=${OPENPI_DIR}/src:${VLABENCH_DIR}:${PYTHONPATH:-}
    export HF_ENDPOINT=https://hf-mirror.com
    export MUJOCO_GL=egl

    local save_dir="${SAVE_DIR}/${tasks[0]}_n${n_ep}_$(date +%Y%m%d_%H%M%S)"
    cd "${VLABENCH_DIR}"
    python scripts/evaluate_openpi.py \
        --host localhost \
        --port "${PORT}" \
        --tasks "${tasks[@]}" \
        --n_episode "${n_ep}" \
        --save_dir "${save_dir}" \
        --visualization
    echo ""
    echo "Results saved to: ${save_dir}"
    echo "Videos: ${save_dir}/${tasks[0]}/videos/"
}

stop_server() {
    if [ -f "${SERVER_PID_FILE}" ]; then
        local pid
        pid=$(cat "${SERVER_PID_FILE}")
        if kill "${pid}" 2>/dev/null; then
            echo "Stopped server PID=${pid}"
        else
            echo "Server PID=${pid} not running"
        fi
        rm -f "${SERVER_PID_FILE}"
    fi
    pkill -f "serve_policy.py.*VLABENCH" 2>/dev/null && echo "Killed leftover serve_policy process" || true
}

case "${1:-help}" in
    server)
        shift
        start_server "$@"
        ;;
    server-bg)
        shift
        start_server_bg "$@"
        ;;
    eval)
        shift
        run_eval "$@"
        ;;
    eval-multi)
        shift
        run_eval "$@"
        ;;
    stop)
        stop_server
        ;;
    *)
        cat <<EOF
Usage: $0 {server|server-bg|eval|eval-multi|stop} [args]
  server [gpu_id]                       Foreground server (Ctrl-C to stop)
  server-bg [gpu_id]                    Background server, log -> ${SERVER_LOG}
  eval <task_name> [n_episode]          Run eval on one task (default task=${DEFAULT_TASK}, n=${DEFAULT_N_EPISODE})
  eval-multi task1 task2 ... [n]        Run eval on multiple tasks
  stop                                  Stop the background server

Common tasks: lift_beaker, lift_flask, pick_tube_lift_tube,
              pick_glass_stirring_rod_insert_glass_stirring_rod,
              select_fruit, select_toy
EOF
        exit 1
        ;;
esac