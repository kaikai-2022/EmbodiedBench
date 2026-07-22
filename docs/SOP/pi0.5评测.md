# Pi0.5 评测 SOP

## 概述

本 SOP 说明如何使用训练好的 Pi0.5 checkpoint 对特定任务进行评测，验证模型性能。

---

## 前置条件

### 1. 环境准备

确保以下环境可用：

- **OpenPI 服务器环境**: `/ssd/qinmaokai/workspace/SciVLABench/third_party/openpi/examples/vlabench/.venv`
- **VLABench 客户端环境**: `vlabench_2` conda 环境

### 2. 文件准备

- **Checkpoint**: 训练好的模型权重
- **任务文件**: 已注册的任务（`VLABench/tasks/autogen_tasks/primitive/*_series.py`）
- **Config 注册**: 任务已在 `VLABench/configs/__init__.py` 的 `name2config` 中注册

---

## 评测流程

### 步骤 1: 启动 OpenPI Policy Server

在**终端 1**中启动服务器：

```bash
# 设置环境变量
cd /ssd/qinmaokai/workspace/SciVLABench/third_party/openpi
export PYTHONPATH=/ssd/qinmaokai/workspace/SciVLABench/third_party/openpi/src:/ssd/qinmaokai/workspace/SciVLABench:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=7  # 选择有足够显存的 GPU

# 启动服务器（后台运行）
nohup /ssd/qinmaokai/workspace/SciVLABench/third_party/openpi/examples/vlabench/.venv/bin/python \
    scripts/serve_policy.py --port 8000 policy:checkpoint \
    --policy.config=<CONFIG_NAME> \
    --policy.dir=<CHECKPOINT_PATH> \
    > /tmp/pi05_server.log 2>&1 &

# 等待服务器就绪
sleep 15
tail -20 /tmp/pi05_server.log
```

**预期输出**：看到 `server listening on 0.0.0.0:8000` 即表示启动成功。

**参数说明**：
- `<CONFIG_NAME>`: 配置名称（如 `pi05_ft_place_flask_on_the_mat`）
- `<CHECKPOINT_PATH>`: checkpoint 路径（如 `/ssd/qinmaokai/workspace/SciVLABench/third_party/openpi/checkpoints/pi0.5_ft_place_flask/pi0.5_ft_place_flask_100K/60000`）

---

### 步骤 2: 运行评测客户端

在**终端 2**中运行评测：

```bash
# 激活客户端环境
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate vlabench_2

# 设置环境变量
cd /ssd/qinmaokai/workspace/SciVLABench
export PYTHONPATH=/ssd/qinmaokai/workspace/SciVLABench/third_party/openpi/src:/ssd/qinmaokai/workspace/SciVLABench:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export MUJOCO_GL=egl

# 运行评测
python scripts/evaluate_openpi.py \
    --host 127.0.0.1 \
    --port 8000 \
    --tasks <TASK_NAME> \
    --n_episode 3 \
    --save_dir logs/pi05_eval \
    --visualization
```

**参数说明**：
- `--host`: 服务器地址（默认 localhost，如果服务器 hostname 解析问题则用 127.0.0.1）
- `--port`: 服务器端口（默认 8000）
- `--tasks`: 任务名称（如 `place_flask_on_the_mat`）
- `--n_episode`: 每个 task 运行的 episode 数（建议 3-10）
- `--save_dir`: 结果和视频保存目录
- `--visualization`: 启用视频录制（flag，加上即启用）

---

### 步骤 3: 查看结果

评测完成后，结果保存在 `--save_dir` 指定的目录：

```
logs/pi05_eval/
├── evaluation_result.json      # 汇总结果
└── <TASK_NAME>/
    └── videos/
        ├── ep0_success_<True/False>_steps_<N>.mp4
        ├── ep1_success_<True/False>_steps_<N>.mp4
        └── ep2_success_<True/False>_steps_<N>.mp4
```

**结果文件说明**：

`evaluation_result.json` 示例：
```json
{
  "place_flask_on_the_mat": {
    "success_rate": 0.0,
    "condition_score": 0.0,
    "per_condition": {
      "IsGraspedCondition": 0.0,
      "OnCondition": 0.0
    },
    "episodes": [
      {
        "episode": 0,
        "success": false,
        "steps": 200,
        "conditions": {
          "IsGraspedCondition": false,
          "OnCondition": false
        }
      }
    ]
  }
}
```

**关键指标**：
- `success_rate`: 任务完成成功率（0-100%）
- `condition_score`: 条件通过率平均值（0-100%）
- `per_condition`: 各条件的具体通过率

---

## 常见任务配置

### 任务 1: `place_flask_on_the_mat`

```bash
# 终端 1: 启动 server
cd /ssd/qinmaokai/workspace/SciVLABench/third_party/openpi
export PYTHONPATH=/ssd/qinmaokai/workspace/SciVLABench/third_party/openpi/src:/ssd/qinmaokai/workspace/SciVLABench:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=7

nohup /ssd/qinmaokai/workspace/SciVLABench/third_party/openpi/examples/vlabench/.venv/bin/python \
    scripts/serve_policy.py --port 8000 policy:checkpoint \
    --policy.config=pi05_ft_place_flask_on_the_mat \
    --policy.dir=/ssd/qinmaokai/workspace/SciVLABench/third_party/openpi/checkpoints/pi0.5_ft_place_flask/pi0.5_ft_place_flask_100K/60000 \
    > /tmp/pi05_server.log 2>&1 &

# 终端 2: 运行评测
conda activate vlabench_2
cd /ssd/qinmaokai/workspace/SciVLABench
export PYTHONPATH=/ssd/qinmaokai/workspace/SciVLABench/third_party/openpi/src:/ssd/qinmaokai/workspace/SciVLABench:$PYTHONPATH

python scripts/evaluate_openpi.py \
    --host 127.0.0.1 --port 8000 \
    --tasks place_flask_on_the_mat \
    --n_episode 3 \
    --save_dir logs/pi05_place_flask_eval \
    --visualization
```

### 任务 2: `lift_beaker`

```bash
# 终端 1: 启动 server
--policy.config=pi05_ft_lift_beaker \
--policy.dir=/path/to/lift_beaker/checkpoint

# 终端 2: 运行评测
--tasks lift_beaker
```

---

## 故障排除

### 问题 1: 连接被拒绝

**症状**: `ConnectionRefusedError: [Errno 111] Connection refused`

**解决**:
1. 确认服务器已启动：`ps aux | grep serve_policy`
2. 检查服务器日志：`tail -50 /tmp/pi05_server.log`
3. 确认看到 "server listening on 0.0.0.0:8000"
4. 尝试使用 `--host 127.0.0.1` 而不是 `localhost`

### 问题 2: Config not found

**症状**: `ValueError: Config 'xxx' not found. Did you mean 'yyy'?`

**解决**:
1. 查找正确的 config name（错误信息会提示）
2. 检查 checkpoint 路径是否正确
3. 确认 config 在 openpi 训练配置中存在

### 问题 3: 任务未注册

**症状**: `TypeError: 'NoneType' object is not iterable`

**解决**:
1. 确认任务在 `VLABench/configs/__init__.py` 的 `name2config` 中有映射
2. 确认任务文件存在于 `VLABench/tasks/autogen_tasks/primitive/`

### 问题 4: GPU 显存不足

**症状**: `jaxlib.xla_extension.XlaRuntimeError`

**解决**:
1. 检查 GPU 显存：`nvidia-smi`
2. 切换到空闲 GPU：`export CUDA_VISIBLE_DEVICES=<GPU_ID>`
3. 或者终止其他占用 GPU 的进程

### 问题 5: 评测结果全为 0%

**可能原因**:
1. **数据问题**: 使用了修复前的 `convert_to_lerobot.py`（gripper 阈值 0.03 导致 lock 宽度误判为张开）
2. **训练不充分**: checkpoint 训练步数不够
3. **任务太难**: 模型泛化能力不足

**排查**:
1. 查看 `evaluation_result.json` 中各条件的通过情况
2. 观看视频确认具体失败位置
3. 检查训练数据是否使用了正确的 `convert_to_lerobot.py`

---

## 快速参考

### 常用命令

```bash
# 启动 server 模板
cd /ssd/qinmaokai/workspace/SciVLABench/third_party/openpi
export PYTHONPATH=/ssd/qinmaokai/workspace/SciVLABench/third_party/openpi/src:/ssd/qinmaokai/workspace/SciVLABENCH:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=7

nohup /ssd/qinmaokai/workspace/SciVLABench/third_party/openpi/examples/vlabench/.venv/bin/python \
    scripts/serve_policy.py --port 8000 policy:checkpoint \
    --policy.config=<CONFIG> \
    --policy.dir=<CHECKPOINT> \
    > /tmp/pi05_server.log 2>&1 &

# 运行评测模板
conda activate vlabench_2
cd /ssd/qinmaokai/workspace/SciVLABench
export PYTHONPATH=/ssd/qinmaokai/workspace/SciVLABench/third_party/openpi/src:/ssd/qinmaokai/workspace/SciVLABENCH:$PYTHONPATH

python scripts/evaluate_openpi.py \
    --host 127.0.0.1 --port 8000 \
    --tasks <TASK> --n_episode 3 \
    --save_dir logs/<NAME> \
    --visualization

# 检查 server 状态
tail -f /tmp/pi05_server.log

# 终止 server
pkill -f "serve_policy.py"
```

---

## 相关文件

| 文件 | 用途 |
|------|------|
| `scripts/evaluate_openpi.py` | 评测客户端 |
| `third_party/openpi/scripts/serve_policy.py` | 服务器 |
| `VLABench/tasks/autogen_tasks/primitive/*_series.py` | 任务定义 |
| `VLABench/configs/__init__.py` | 任务注册 |

---

## 维护者

VLABench Team

**最后更新**: 2026-07-17
