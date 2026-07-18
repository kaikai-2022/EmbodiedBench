#!/bin/bash
#
# 多卡并行轨迹生成脚本
#
# 用法:
#   bash scripts/model_train/generate_trajectories.sh --task pick_flask_on_the_mat --num 200 --gpus 0,1,2,3,4,5,6,7
#   bash scripts/model_train/generate_trajectories.sh --task pick_flask_on_the_mat --num 200 --gpus 0,2,3,5,6
#
# 参数说明:
#   --task     任务名（不带 _series 后缀），脚本自动拼接
#   --num      目标轨迹数量
#   --gpus     GPU ID 列表，逗号分隔，默认 0,1,2,3,4,5,6,7
#   --samples  每张卡每次生成的尝试次数，默认 10
#
# 特点:
#   - 8/多卡并行，各卡分配独立 start_id 区间，互不冲突
#   - 自动跳过已有轨迹，继续生成直到达到目标
#   - grasplock 已禁用（trajectory_generation.py 内置）
#   - 自动生成视频 (--record-video)
#

set -e

# 默认值
GPU_LIST="0,1,2,3,4,5,6,7"
SAMPLES_PER_GPU=40
MAX_EPISODE=1000

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK_NAME="$2"
            shift 2
            ;;
        --num)
            NUM_TRAJECTORIES="$2"
            shift 2
            ;;
        --gpus)
            GPU_LIST="$2"
            shift 2
            ;;
        --samples)
            SAMPLES_PER_GPU="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 检查必需参数
if [ -z "$TASK_NAME" ] || [ -z "$NUM_TRAJECTORIES" ]; then
    echo "用法: $0 --task <task_name> --num <num_trajectories> [--gpus 0,1,2,...] [--samples 10]"
    exit 1
fi

SERIES_NAME="${TASK_NAME}_series"
PROJECT_ROOT="/ssd/qinmaokai/workspace/SciVLABench"
# SAVE_DIR 只到 series 这一层：trajectory_generation.py 内部会拼上 task_name
# (=SERIES_NAME)，最终落到 ${SERIES_NAME}/。convert_to_lerobot.py 也是按
# series_name 作为 task-list 直接 os.walk 这一层，与下面 TASK_DIR 对齐。
SAVE_DIR="$PROJECT_ROOT/dataset/training_data/${SERIES_NAME}"
LOG_DIR="$PROJECT_ROOT/logs/traj_gen/${SERIES_NAME}"
TASK_DIR="$SAVE_DIR"

# 解析 GPU 列表为数组
IFS=',' read -ra GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}

echo "=========================================="
echo "多卡并行轨迹生成"
echo "=========================================="
echo "  任务名称:    $TASK_NAME"
echo "  Series 名:   $SERIES_NAME"
echo "  目标数量:    $NUM_TRAJECTORIES"
echo "  GPU 列表:    ${GPUS[*]} (共 ${NUM_GPUS} 张)"
echo "  每卡尝试:    $SAMPLES_PER_GPU 次"
echo "  输出目录:    $SAVE_DIR"
echo "=========================================="

mkdir -p "$SAVE_DIR" "$LOG_DIR"

# 获取已有轨迹数量（只统计 success 的 HDF5，排除失败残留）
get_existing_count() {
    local count
    if [ -d "$TASK_DIR" ]; then
        count=$(find "$TASK_DIR" -maxdepth 1 -name "*.hdf5" 2>/dev/null | wc -l)
    else
        count=0
    fi
    echo "$count"
}

# 获取已分配的最大 index（用于确定 start_id 起点）
get_max_index() {
    local max_idx
    if [ -d "$TASK_DIR" ]; then
        max_idx=$(find "$TASK_DIR" -maxdepth 1 -name "data_*.hdf5" 2>/dev/null | \
                  sed 's/.*data_\([0-9]*\)\.hdf5/\1/' | sort -n | tail -1)
    fi
    echo "${max_idx:-0}"
}

# 主循环
ITER=0
while true; do
    ITER=$((ITER + 1))
    CURRENT=$(get_existing_count)
    REMAINING=$((NUM_TRAJECTORIES - CURRENT))

    echo ""
    echo "[$(date '+%H:%M:%S')] 第 ${ITER} 轮检查: 当前 ${CURRENT}/${NUM_TRAJECTORIES} 条, 剩余 ${REMAINING} 条"

    if [ $REMAINING -le 0 ]; then
        echo "已达到目标数量 ${NUM_TRAJECTORIES}!"
        break
    fi

    # 计算每张卡分多少
    # 策略：每次每张卡分配 min(samples_per_gpu, remaining/gpus_left) 次尝试
    # 保证不超额，同时尽量均衡
    remaining_attempts=$((REMAINING + NUM_GPUS - 1))  # 向上取整
    samples_each=$((remaining_attempts / NUM_GPUS))
    [ $samples_each -gt $SAMPLES_PER_GPU ] && samples_each=$SAMPLES_PER_GPU
    [ $samples_each -lt 1 ] && samples_each=1

    echo "本轮: 每卡生成 ${samples_each} 条, 从 index $(get_max_index) 之后继续"

    # 启动所有 GPU
    pids=()
    for gpu in "${GPUS[@]}"; do
        # 每张卡的 start_id 错开，避免冲突
        gpu_offset=$((gpu * SAMPLES_PER_GPU))
        # 用已有最大 index + 1 作为全局基准，保证接续
        base_index=$(get_max_index)
        # 避免重叠：GPU i 的 start = base + i * samples_each
        # 但这样所有 GPU 都会从同一个 base 开始，还是会冲突
        # 正确做法：每个 GPU 用不重叠的 index 区间
        # 改为：GPU i 从 (已有最大+1) + i*samples_each 开始
        start_id=$((base_index + 1 + gpu * samples_each))

        echo "  启动 GPU $gpu: start_id=$start_id, samples=${samples_each}"

        CUDA_VISIBLE_DEVICES=$gpu \
            conda run -n vlabench_2 env \
                MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
                python "$PROJECT_ROOT/scripts/trajectory_generation.py" \
                    --task-name "$SERIES_NAME" \
                    --n-sample "$samples_each" \
                    --start-id "$start_id" \
                    --save-dir "$SAVE_DIR" \
                    --record-video \
                    --max-episode "$MAX_EPISODE" \
                    --robot franka \
                    > "$LOG_DIR/gpu${gpu}_iter${ITER}.log" 2>&1 &
        pids+=($!)
    done

    echo "  等待所有 GPU 完成 (samples=${samples_each} × ${NUM_GPUS} 卡)..."

    # 等待所有子进程
    failed=0
    for i in "${!pids[@]}"; do
        pid=${pids[$i]}
        gpu=${GPUS[$i]}
        if ! wait $pid; then
            echo "  GPU $gpu 进程退出码异常 (PID=$pid)"
            failed=$((failed + 1))
        else
            echo "  GPU $gpu 完成 (PID=$pid)"
        fi
    done

    if [ $failed -eq $NUM_GPUS ]; then
        echo "所有 GPU 都失败了，退出"
        exit 1
    fi

    # 短暂等待让文件系统同步
    sleep 2
done

# 最终统计
FINAL=$(get_existing_count)
FINAL_MP4=$(find "$TASK_DIR" -maxdepth 1 -name "*success_True*.mp4" 2>/dev/null | wc -l)
FAILED_MP4=$(find "$TASK_DIR" -maxdepth 1 -name "*success_False*.mp4" 2>/dev/null | wc -l)
DISK=$(du -sh "$TASK_DIR" 2>/dev/null | cut -f1)

echo ""
echo "=========================================="
echo "生成完成!"
echo "=========================================="
echo "  成功轨迹:  $FINAL 条"
echo "  成功 MP4:  $FINAL_MP4"
echo "  失败 MP4:  $FAILED_MP4"
echo "  磁盘占用:  $DISK"
echo "  输出路径:  $TASK_DIR"
echo "=========================================="
