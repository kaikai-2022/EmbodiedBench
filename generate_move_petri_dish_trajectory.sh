#!/bin/bash
# 生成 move_petri_dish 任务的轨迹数据

# 激活 conda 环境
source /ssd/mkqin/miniconda3/etc/profile.d/conda.sh
conda activate vlabench_2

# 设置环境变量
export MUJOCO_GL=osmesa
export VLABENCH_ROOT=/ssd/mkqin/workspace/VLABench/VLABench

# 进入项目根目录
cd /ssd/mkqin/workspace/VLABench

# 创建临时输出目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TEMP_DIR="./temp_trajectory_output_${TIMESTAMP}"
mkdir -p "$TEMP_DIR"

# 运行轨迹生成脚本（仅1个样本用于快速测试）
echo "正在生成 move_petri_dish 任务的轨迹数据（仅1个样本用于快速测试）..."
echo "输出目录: $TEMP_DIR"
python scripts/trajectory_generation.py \
    --task-name move_petri_dish \
    --n-sample 1 \
    --save-dir "$TEMP_DIR" \
    --early-stop \
    2>&1 | tee move_petri_dish_generation.log

# 检查是否生成成功
if [ -d "$TEMP_DIR/move_petri_dish" ]; then
    echo ""
    echo "============================"
    echo "✓ 轨迹生成成功!"
    echo "============================"
    echo "保存位置: $TEMP_DIR/move_petri_dish"
    ls -la "$TEMP_DIR/move_petri_dish"

    # 如果生成了 HDF5 文件，说明任务成功
    if ls "$TEMP_DIR/move_petri_dish"/*.hdf5 1> /dev/null 2>&1; then
        echo ""
        echo "成功生成的样本："
        ls -lh "$TEMP_DIR/move_petri_dish"/*.hdf5
    fi

    # 如果生成了视频，显示视频文件
    if ls "$TEMP_DIR/move_petri_dish"/*.mp4 1> /dev/null 2>&1; then
        echo ""
        echo "生成的视频："
        ls -lh "$TEMP_DIR/move_petri_dish"/*.mp4
    fi
else
    echo ""
    echo "============================"
    echo "✗ 轨迹生成失败"
    echo "============================"
    echo "查看日志: move_petri_dish_generation.log"
    echo "输出目录: $TEMP_DIR"
fi
