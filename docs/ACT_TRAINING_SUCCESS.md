# ACT 模型训练文档

## 概述

本文档记录了在 VLABench 上使用 LeRobot 框架训练 ACT (Action Chunking Transformers) 模型的全过程，包括数据转换、训练配置、问题排查和解决方案。

**训练成果：**
- 成功完成 1000 步训练
- Loss 从 93.314 下降到约 2.3
- 训练耗时：约 54 分钟
- 模型保存在 `models/act_lift_small_beaker_50ep_smoke_20260615_053401/`

---

## 1. 数据准备

### 1.1 数据集信息

- **任务**: `lift_small_beaker` (小烧杯搬运)
- **轨迹数**: 50 条
- **总帧数**: 3407 帧
- **平均每条轨迹**: 约 68 帧

### 1.2 数据转换脚本

**文件**: `/ssd/mkqin/workspace/VLABench/scripts/convert_to_lerobot.py`

**关键修改** - 特征键名必须加 `observation.` 前缀：

```python
# features 字典定义
features={
    "observation.image":       {"dtype": "image", "shape": (480,480,3), "names":["height","width","channels"]},
    "observation.wrist_image": {"dtype": "image", "shape": (480,480,3), "names":["height","width","channels"]},
    "observation.state":       {"dtype": "float", "shape": (7,),       "names":["state"]},
    "action":                  {"dtype": "float", "shape": (7,),       "names":["action"]},
}

# add_frame 调用
dataset.add_frame({
    "observation.image":       images[i][2],   # 前视相机
    "observation.wrist_image": images[i][3],   # 腕部相机
    "observation.state":       ee_state[i],
    "action":                  action,
})
```

### 1.3 生成数据集

```bash
cd /ssd/mkqin/workspace/VLABench

# 备份旧数据集（如果存在）
if [ -d "/ssd/mkqin/.cache/huggingface/lerobot/vlabench_lift_small_beaker_50ep" ]; then
    mv /ssd/mkqin/.cache/huggingface/lerobot/vlabench_lift_small_beaker_50ep \
       /ssd/mkqin/.cache/huggingface/lerobot/vlabench_lift_small_beaker_50ep.broken_$(date +%Y%m%d_%H%M%S)
fi

# 运行转换
python3 scripts/convert_to_lerobot.py \
    --dataset-name vlabench_lift_small_beaker_50ep \
    --dataset-path /ssd/mkqin/workspace/VLABench/datasets/lift_small_beaker_raw \
    --max-files 50 \
    --task-list lift_small_beaker
```

### 1.4 验证数据集

```python
import sys
sys.path.insert(0, '/ssd/mkqin/workspace/lerobot')
from lerobot.common.datasets.utils import dataset_to_policy_features
import json

# 检查数据集特征
info = json.load(open('/ssd/mkqin/.cache/huggingface/lerobot/vlabench_lift_small_beaker_50ep/meta/info.json'))
features = dataset_to_policy_features(info['features'])

for k, v in features.items():
    print(f"{k}: {v.type}, {v.shape}")

# 预期输出应包含:
# observation.image: VISUAL, (3, 480, 480)
# observation.wrist_image: VISUAL, (3, 480, 480)
# observation.state: STATE, (7,)
# action: ACTION, (7,)
```

---

## 2. 训练配置

### 2.1 关键配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `--policy.type` | `act` | ACT 策略 |
| `--dataset.repo_id` | `vlabench_lift_small_beaker_50ep` | 数据集 ID |
| `--dataset.local_files_only` | `true` | **必须设为 true**，否则会尝试访问远程仓库 |
| `--dataset.image_transforms.enable` | `false` | 禁用图像增强 |
| `--dataset.use_imagenet_stats` | `true` | 使用 ImageNet 统计进行归一化 |
| `--batch_size` | `2` | 批次大小（根据显存调整） |
| `--num_workers` | `0` | 数据加载线程数 |
| `--offline.steps` | `1000` | 训练步数 |
| `--save_freq` | `500` | checkpoint 保存频率 |
| `--device` | `cuda` | 训练设备 |

### 2.2 完整训练命令

```bash
cd /ssd/mkqin/workspace/lerobot

OUTPUT_DIR="/ssd/mkqin/workspace/VLABench/models/act_lift_small_beaker_50ep_smoke_$(date +%Y%m%d_%H%M%S)"

python3 lerobot/scripts/train.py \
    --policy.type=act \
    --dataset.repo_id=vlabench_lift_small_beaker_50ep \
    --dataset.local_files_only=true \
    --dataset.image_transforms.enable=false \
    --dataset.use_imagenet_stats=true \
    --output_dir="$OUTPUT_DIR" \
    --batch_size=2 \
    --num_workers=0 \
    --offline.steps=1000 \
    --save_freq=500 \
    --eval_freq=1000 \
    --log_freq=100 \
    --device=cuda \
    --seed=1000
```

---

## 3. 调试脚本

如果训练遇到问题，可以使用带调试信息的脚本：

### 3.1 调试版本训练脚本

**文件**: `/ssd/mkqin/workspace/lerobot/lerobot/scripts/train_debug.py`

**使用方法**:

```bash
cd /ssd/mkqin/workspace/lerobot

python3 lerobot/scripts/train_debug.py \
    --policy.type=act \
    --dataset.repo_id=vlabench_lift_small_beaker_50ep \
    --dataset.local_files_only=true \
    --dataset.image_transforms.enable=false \
    --dataset.use_imagenet_stats=true \
    --output_dir="/ssd/mkqin/workspace/VLABench/models/act_debug_$(date +%Y%m%d_%H%M%S)" \
    --batch_size=2 \
    --num_workers=0 \
    --offline.steps=10 \
    --save_freq=500 \
    --eval_freq=1000 \
    --log_freq=1 \
    --device=cuda \
    --seed=1000
```

### 3.2 独立调试脚本

**文件**: `/tmp/debug_training_with_logging.py`

用于独立测试各个组件（数据集、DataLoader、策略、前向传播）：

```bash
python3 /tmp/debug_training_with_logging.py
```

---

## 4. 问题排查

### 4.1 常见错误及解决方案

#### 错误 1: `RepositoryNotFoundError` / `401 Unauthorized`

**错误信息**:
```
huggingface_hub.errors.RepositoryNotFoundError: 401 Client Error
Repository Not Found for url: https://hf-mirror.com/api/datasets/vlabench_lift_small_beaker_50ep/refs
```

**原因**: `local_files_only` 未设置或设置为 `false`

**解决方案**:
```bash
--dataset.local_files_only=true
```

#### 错误 2: `KeyError: 'observation.state'`

**错误信息**:
```
KeyError: 'observation.state'
```

**原因**: 数据集特征键名缺少 `observation.` 前缀

**解决方案**: 重新生成数据集，使用正确的键名：
- `observation.image` (不是 `image`)
- `observation.wrist_image` (不是 `wrist_image`)
- `observation.state` (不是 `state`)
- `action` (保持不变)

#### 错误 3: `Couldn't parse 'False' into a bool`

**错误信息**:
```
draccus.utils.DecodingError: `dataset.image_transforms.enable`: Couldn't parse 'False' into a bool
```

**原因**: CLI 参数使用了 Python 的 `True/False` 而非 YAML 的 `true/false`

**解决方案**: 使用小写布尔值：
```bash
--dataset.image_transforms.enable=false  # 不是 False
--dataset.local_files_only=true          # 不是 True
```

#### 错误 4: 训练卡在 step 0，GPU 使用率为 0

**症状**: 训练进程运行但每步耗时很长，GPU 利用率为 0%

**排查方法**:
1. 检查训练脚本是否有调试日志输出
2. 使用 `nvidia-smi` 检查 GPU 使用情况
3. 检查数据加载是否正常

**解决方案**: 使用带调试信息的 `train_debug.py`，添加日志：

```python
# 在训练循环中添加
logging.info(f"Step {step}: Getting batch from dataloader...")
batch = next(dl_iter)
logging.info(f"Step {step}: Batch received, keys={list(batch.keys())}")

logging.info(f"Step {step}: Calling update_policy...")
train_info = update_policy(policy, batch, ...)
logging.info(f"Step {step}: update_policy completed")
```

### 4.2 OOM 应急处理

如果遇到显存不足，按顺序尝试：

1. **启用自动混合精度**:
   ```bash
   --use_amp=true
   ```

2. **减小批次大小**:
   ```bash
   --batch_size=4
   ```

3. **继续减小**:
   ```bash
   --batch_size=2 --num_workers=2
   ```

---

## 5. 训练结果

### 5.1 Loss 收敛曲线

| Step | Loss | 说明 |
|------|------|------|
| 0 | 93.314 | 初始 loss |
| 100 | 5.151 | 下降 94.5% |
| 200 | 3.196 | - |
| 300 | 3.057 | - |
| 400 | 2.716 | - |
| 500 | 3.018 | checkpoint 保存 |
| 600 | 1.904 | - |
| 700 | 1.584 | 最低点 |
| 800 | 2.853 | - |
| 900 | 2.299 | - |
| 1000 | - | 训练完成 |

### 5.2 Checkpoint 结构

```
models/act_lift_small_beaker_50ep_smoke_20260615_053401/
├── checkpoints/
│   ├── 000500/
│   │   ├── pretrained_model/
│   │   │   ├── config.json
│   │   │   ├── model.safetensors (206MB)
│   │   │   └── train_config.json
│   │   └── training_state.pth
│   ├── 001000/
│   │   └── ...
│   └── last -> 001000 (符号链接)
```

### 5.3 训练统计

- **总步数**: 1000
- **批次大小**: 2
- **设备**: CUDA (单卡)
- **每步耗时**: 约 3.1 秒
- **总耗时**: 约 54 分钟
- **模型参数**: 52M

---

## 6. 使用训练好的模型

### 6.1 加载模型

```python
from lerobot.common.policies.factory import make_policy
from lerobot.configs.train import TrainPipelineConfig

# 加载配置
cfg = TrainPipelineConfig.from_pretrained(
    "/ssd/mkqin/workspace/VLABench/models/act_lift_small_beaker_50ep_smoke_20260615_053401/checkpoints/last/pretrained_model"
)

# 创建策略
policy = make_policy(
    cfg=cfg.policy,
    device='cuda',
    ds_meta=None,  # 或传入数据集 meta
)
```

### 6.2 推理示例

```python
# 准备输入
batch = {
    "observation.image": images,        # (B, 3, H, W)
    "observation.wrist_image": wrist,   # (B, 3, H, W)
    "observation.state": state,         # (B, 7)
}

# 前向传播
policy.eval()
with torch.no_grad():
    action = policy(batch)  # (B, action_dim)
```

---

## 7. 注意事项

1. **数据集键名格式**: LeRobot 的 `dataset_to_policy_features` 只识别以 `observation.` 开头的键作为状态特征
2. **本地文件模式**: 始终设置 `--dataset.local_files_only=true` 避免网络问题
3. **CLI 布尔值**: 使用 YAML 格式的 `true/false` 而非 Python 的 `True/False`
4. **数据加载器**: `num_workers=0` 最稳定，避免多进程问题
5. **Chunk Size**: ACT 默认 `chunk_size=100`，对于短轨迹会被 padding

---

## 8. 相关文件

| 文件路径 | 说明 |
|----------|------|
| `/ssd/mkqin/workspace/VLABench/scripts/convert_to_lerobot.py` | 数据转换脚本 |
| `/ssd/mkqin/workspace/lerobot/lerobot/scripts/train.py` | 训练脚本 |
| `/ssd/mkqin/workspace/lerobot/lerobot/scripts/train_debug.py` | 带调试信息的训练脚本 |
| `/ssd/mkqin/workspace/lerobot/lerobot/common/datasets/utils.py` | `dataset_to_policy_features` 函数 |
| `/ssd/mkqin/workspace/lerobot/lerobot/common/policies/act/modeling_act.py` | ACT 模型实现 |
| `/ssd/mkqin/.cache/huggingface/lerobot/vlabench_lift_small_beaker_50ep/` | 生成的数据集 |

---

*文档生成时间: 2026-06-15*
*训练完成时间: 2026-06-15 06:28:53*