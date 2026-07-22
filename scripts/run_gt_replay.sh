#!/bin/bash
# GT Replay 批量测试脚本
# 用法: ./run_gt_replay.sh /path/to/hdf5/dir [num_samples]

HDF5_DIR=${1:-"/ssd/qinmaokai/workspace/SciVLABench/dataset/gt_test/lift_beaker"}
NUM_SAMPLES=${2:-5}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLABENCH_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "GT Replay Batch Test"
echo "Directory: $HDF5_DIR"
echo "Samples: $NUM_SAMPLES"
echo "============================================"

# 检查目录
if [ ! -d "$HDF5_DIR" ]; then
    echo "Error: Directory not found: $HDF5_DIR"
    echo "Please run trajectory generation first:"
    echo "  python scripts/trajectory_generation.py --task-name lift_beaker --n-sample $NUM_SAMPLES"
    exit 1
fi

# 统计 HDF5 文件
HDF5_FILES=($(ls -1 "$HDF5_DIR"/*.hdf5 2>/dev/null | head -$NUM_SAMPLES))
NUM_FILES=${#HDF5_FILES[@]}

if [ $NUM_FILES -eq 0 ]; then
    echo "Error: No HDF5 files found in $HDF5_DIR"
    exit 1
fi

echo "Found $NUM_FILES HDF5 files"
echo ""

# 运行测试
SUCCESS_COUNT=0
FAIL_COUNT=0

for hdf5_file in "${HDF5_FILES[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "Testing: $(basename $hdf5_file)"
    echo "----------------------------------------"

    python "$SCRIPT_DIR/gt_replay_test.py" --hdf5-path "$hdf5_file" --vis --save-dir "$VLABENCH_ROOT/gt_replay_results"

    if [ $? -eq 0 ]; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo "[PASS] $hdf5_file"
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "[FAIL] $hdf5_file"
    fi
done

echo ""
echo "============================================"
echo "GT Replay Test Summary"
echo "============================================"
echo "Total: $NUM_FILES"
echo "Success: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"
echo "Success Rate: $((SUCCESS_COUNT * 100 / NUM_FILES))%"
echo ""
echo "Videos saved to: $VLABENCH_ROOT/gt_replay_results"
