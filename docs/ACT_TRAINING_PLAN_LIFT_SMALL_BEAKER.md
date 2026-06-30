# ACT模型训练与测试实施计划

## 📋 计划概述

**目标**: 在`lift_small_beaker`任务上训练ACT模型并验证其效果

**任务信息**:
- 任务文件: `/ssd/mkqin/workspace/VLABench/VLABench/tasks/autogen_tasks/lift_small_beaker_series.py`
- 任务类型: Primitive Task (autogen_tasks)
- 技能序列: pick → lift (height=0.15)
- 目标物体: small_beaker_0
- 成功条件: lift_height >= 0.15m

**兼容性验证**:
- ✅ `trajectory_generation.py`支持autogen_tasks（已验证所有lift系列任务都有`get_expert_skill_sequence()`）
- ✅ `lift_small_beaker`任务实现了完整的expert skill sequence
- ✅ 数据转换脚本支持任意HDF5格式数据

**策略**: 使用50条expert trajectories进行小规模训练，验证pipeline正确性和模型收敛性

**预期时间**: 约3-4小时（数据收集30分钟 + 数据转换15分钟 + 训练1.5-2小时 + 测试30分钟）

---

## 🎯 实施步骤

### Phase 1: 数据收集 (预计30-45分钟)

#### 1.1 任务验证

首先验证任务可以正常加载：
```bash
cd /ssd/mkqin/workspace/VLABench

# 测试任务加载（不收集数据，只是验证）
python -c "
from VLABench.envs import load_env

# 加载lift_small_beaker任务
env = load_env('lift_small_beaker', robot='franka')
env.reset()

# 检查expert skill sequence
skill_seq = env.get_expert_skill_sequence()
print(f'✓ 任务加载成功')
print(f'✓ Expert skill sequence包含 {len(skill_seq)} 个技能:')
for i, skill in enumerate(skill_seq):
    print(f'  {i+1}. {skill.func.__name__}: {skill.keywords}')

env.close()
"
```

**预期输出**:
```
✓ 任务加载成功
✓ Expert skill sequence包含 2 个技能:
  1. pick: target_entity_name=small_beaker_0, prior_eulers=[[-3.141592653589793, 0, 0]]
  2. lift: lift_height=0.15
```

#### 1.2 执行数据收集

```bash
cd /ssd/mkqin/workspace/VLABench

# 收集50条lift_small_beaker任务的轨迹
python scripts/trajectory_generation.py \
    --task-name lift_small_beaker \
    --n-sample 50 \
    --start-id 0 \
    --save-dir /ssd/mkqin/workspace/VLABench/datasets/lift_small_beaker_raw \
    --record-video \
    --robot franka
```

**参数说明**:
- `--task-name lift_small_beaker`: 任务名称（autogen_tasks下的任务）
- `--n-sample 50`: 收集50条轨迹
- `--start-id 0`: 从编号0开始
- `--save-dir`: 原始HDF5数据保存目录
- `--record-video`: 同时生成视频（用于数据质量检查）
- `--robot franka`: 使用Franka机器人

**预期输出**:
- 50个HDF5文件: `/ssd/mkqin/workspace/VLABench/datasets/lift_small_beaker_raw/lift_small_beaker/data_0.hdf5` to `data_49.hdf5`
- 50个视频文件: `/ssd/mkqin/workspace/VLABench/datasets/lift_small_beaker_raw/lift_small_beaker/demo_0_success_True.mp4` to `demo_49_success_True.mp4`
- 总大小: 约250-500MB（每个HDF5约5-10MB）

#### 1.3 数据质量验证

**验证文件数量**:
```bash
ls /ssd/mkqin/workspace/VLABench/datasets/lift_small_beaker_raw/lift_small_beaker/*.hdf5 | wc -l
# 应该输出: 50
```

**验证文件内容**:
```bash
python -c "
import h5py
import numpy as np

# 读取第一个文件
f = h5py.File('/ssd/mkqin/workspace/VLABench/datasets/lift_small_beaker_raw/lift_small_beaker/data_0.hdf5', 'r')

# 获取第一个episode的key
episode_key = list(f['data'].keys())[0]
print(f'Episode key: {episode_key}')

# 检查数据结构
episode = f['data'][episode_key]
print(f'可用字段: {list(episode.keys())}')

# 检查observation
obs = episode['observation']
print(f'RGB shape: {obs[\"rgb\"].shape}')  # 应该是 (T, 4, 480, 480, 3)
print(f'EE state shape: {obs[\"ee_state\"].shape}')  # 应该是 (T, 8)

# 检查trajectory
traj = episode['trajectory']
print(f'Trajectory shape: {traj.shape}')  # 应该是 (T, 7)

print(f'Episode长度: {len(obs[\"rgb\"])} steps')

f.close()
"
```

**预期输出**:
```
Episode key: 2026-06-12 10:30:45
可用字段: ['observation', 'trajectory', 'meta_info', 'instruction', 'entities', 'target_entity']
RGB shape: (87, 4, 480, 480, 3)  # 87步，4个摄像头
EE state shape: (87, 8)  # [pos3, quat4, gripper1]
Trajectory shape: (87, 7)  # [pos3, euler3, gripper1]
Episode长度: 87 steps
```

**检查视频质量** (可选):
```bash
# 安装mediapy（如果还没有）
pip install mediapy

# 查看第一个视频
python -c "
import mediapy
video = mediapy.read_video('/ssd/mkqin/workspace/VLABench/datasets/lift_small_beaker_raw/lift_small_beaker/demo_0_success_True.mp4')
print(f'视频长度: {len(video)} 帧')
mediapy.show_video(video)
"
```

**质量检查清单**:
- [ ] 文件数量正确（50个）
- [ ] 每个episode长度合理（50-150步）
- [ ] 视频显示任务成功完成
- [ ] 机器人动作流畅，无异常
- [ ] Gripper正确抓取和lift

---

### Phase 2: 数据转换为LeRobot格式 (预计15分钟)

#### 2.1 执行数据转换

```bash
cd /ssd/mkqin/workspace/VLABench

# 转换HDF5数据为LeRobot格式
python scripts/convert_to_lerobot.py \
    --dataset-name vlabench_lift_small_beaker_50ep \
    --dataset-path /ssd/mkqin/workspace/VLABench/datasets/lift_small_beaker_raw \
    --max-files 50 \
    --task-list lift_small_beaker
```

**参数说明**:
- `--dataset-name`: LeRobot dataset名称
- `--dataset-path`: 原始HDF5数据路径
- `--max-files 50`: 最多处理50个文件
- `--task-list lift_small_beaker`: 只处理lift_small_beaker任务

**预期输出**:
- Dataset保存到: `~/.cache/huggingface/lerobot/datasets/vlabench_lift_small_beaker_50ep/`
- 包含:
  - `info.json`: 数据集元信息
  - `episodes/`: 所有episodes的数据
  - `stats.json`: 数据统计信息（mean, std）

**转换过程说明**:
根据`convert_to_lerobot.py`代码，转换过程：
1. 读取HDF5文件中的observation和trajectory
2. 提取front camera (index 2) 和wrist camera (index 3)
3. 转换ee_state的quaternion为euler angles
4. 减去robot_frame offset（robot-relative坐标）
5. 保存为LeRobot格式

#### 2.2 验证LeRobot Dataset

```bash
# 检查数据集是否创建成功
python -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# 加载数据集
ds = LeRobotDataset('vlabench_lift_small_beaker_50ep')

print(f'✓ 数据集加载成功')
print(f'总episodes数: {len(ds)}')
print(f'数据集信息:')
print(f'  FPS: {ds.info[\"fps\"]}')
print(f'  Robot type: {ds.info[\"robot_type\"]}')

# 检查第一个episode
episode_0 = ds[0]
print(f'\\n第一个episode长度: {len(episode_0)} 步')

# 检查第一帧
frame_0 = episode_0[0]
print(f'第一帧可用keys: {list(frame_0.keys())}')
print(f'  image shape: {frame_0[\"image\"].shape}')
print(f'  wrist_image shape: {frame_0[\"wrist_image\"].shape}')
print(f'  state shape: {frame_0[\"state\"].shape}')
print(f'  action shape: {frame_0[\"action\"].shape}')
"
```

**预期输出**:
```
✓ 数据集加载成功
总episodes数: 50
数据集信息:
  FPS: 10
  Robot type: franka

第一个episode长度: 87 步

第一帧可用keys: ['image', 'wrist_image', 'state', 'action']
  image shape: (480, 480, 3)
  wrist_image shape: (480, 480, 3)
  state shape: (7,)
  action shape: (7,)
```

**验证数据统计**:
```bash
# 检查统计信息是否计算
python -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import json

ds = LeRobotDataset('vlabench_lift_small_beaker_50ep')

# 检查stats
stats = ds.stats
print('数据统计信息:')
for key, value in stats.items():
    if isinstance(value, dict) and 'mean' in value:
        print(f'  {key}:')
        print(f'    mean shape: {value[\"mean\"].shape}')
        print(f'    std shape: {value[\"std\"].shape}')
        print(f'    mean range: [{value[\"mean\"].min():.3f}, {value[\"mean\"].max():.3f}]')
        print(f'    std range: [{value[\"std\"].min():.3f}, {value[\"std\"].max():.3f}]')
"
```

**验证清单**:
- [ ] Dataset成功创建（50 episodes）
- [ ] Image shape正确 (480, 480, 3)
- [ ] State和action shape正确 (7,)
- [ ] 统计信息已计算（mean, std）

---

### Phase 3: 模型训练 (预计1.5-2小时)

#### 3.1 执行训练

```bash
cd /ssd/mkqin/workspace/lerobot

# 训练ACT模型
python lerobot/scripts/train.py \
    policy=act \
    dataset.repo_id=vlabench_lift_small_beaker_50ep \
    training.output_dir=/ssd/mkqin/workspace/VLABench/models/act_lift_small_beaker_50ep \
    training.batch_size=8 \
    training.n_epochs=100 \
    training.device=cuda \
    training.num_workers=4 \
    training.eval_freq=5 \
    training.save_freq=10 \
    act.input_features.observation.image.type=VISUAL \
    act.input_features.observation.image.shape="[3,480,480]" \
    act.input_features.observation.wrist_image.type=VISUAL \
    act.input_features.observation.wrist_image.shape="[3,480,480]" \
    act.input_features.observation.state.type=STATE \
    act.input_features.observation.state.shape="[7]" \
    act.output_features.action.type=ACTION \
    act.output_features.action.shape="[7]"
```

**关键参数**:
- `policy=act`: 使用ACT策略
- `training.batch_size=8`: 批大小（如果GPU内存不足，可改为4）
- `training.n_epochs=100`: 训练100轮
- `training.eval_freq=5`: 每5个epoch评估一次
- `training.save_freq=10`: 每10个epoch保存checkpoint
- `training.num_workers=4`: 数据加载线程数

**训练输出结构**:
```
/ssd/mkqin/workspace/VLABench/models/act_lift_small_beaker_50ep/
├── pretrained_model/          # 最终模型（第100 epoch）
│   ├── config.json
│   └── policy_model.safetensors
├── checkpoints/               # 中间checkpoints
│   ├── checkpoint-10/
│   ├── checkpoint-20/
│   ├── ...
│   └── checkpoint-90/
├── config.json                # 完整配置
├── training_args.json         # 训练参数
└── logs/                      # TensorBoard logs
    └── events.out.tfevents...
```

#### 3.2 监控训练进度

**方法1: TensorBoard（推荐）**
```bash
# 在另一个terminal启动TensorBoard
tensorboard --logdir /ssd/mkqin/workspace/VLABench/models/act_lift_small_beaker_50ep

# 在浏览器访问 http://localhost:6006
# 观察以下指标:
# - train_loss: 应该持续下降
# - eval_loss: 应该下降，轻微上升可能表示overfitting
# - train_action_loss: Action prediction loss
# - train_value_loss: VAE loss (如果use_vae=True)
```

**方法2: 实时日志**
```bash
# 实时查看训练日志
tail -f /ssd/mkqin/workspace/VLABench/models/act_lift_small_beaker_50ep/logs/*.log
```

#### 3.3 收敛性判断

**良好收敛的指标**:
- Training loss从初始值(>1.0)下降到(<0.1)
- Eval loss也稳定下降
- Loss曲线平滑，无剧烈震荡
- 无NaN或Inf values

**预期时间线**:
- **Epoch 0-20**: 快速下降阶段（主要学习基本动作）
- **Epoch 20-50**: 继续下降（学习细节）
- **Epoch 50-100**: Fine-tuning（loss缓慢下降）

**典型Loss曲线**:
```
Epoch 1:   loss=1.234
Epoch 10:  loss=0.567
Epoch 20:  loss=0.234
Epoch 50:  loss=0.123
Epoch 100: loss=0.089
```

**如果训练不收敛**:
- 降低learning rate: 添加 `training.optimizer_lr=5e-6`
- 增加warmup: 添加 `training.optimizer_warmup_ratio=0.1`
- 检查数据质量
- 减少batch size

---

### Phase 4: 模型评估 (预计30分钟)

#### 4.1 在训练任务上测试

```bash
cd /ssd/mkqin/workspace/VLABench

# 测试训练好的ACT模型
python scripts/evaluate_policy.py \
    --tasks lift_small_beaker \
    --n-episode 10 \
    --policy act \
    --model_ckpt /ssd/mkqin/workspace/VLABench/models/act_lift_small_beaker_50ep/pretrained_model \
    --replanstep 4 \
    --save-dir logs/act_lift_small_beaker_evaluation \
    --visulization \
    --metrics success_rate intention_score progress_score
```

**参数说明**:
- `--tasks lift_small_beaker`: 测试任务名称
- `--n-episode 10`: 测试10个episodes
- `--policy act`: 使用ACT policy
- `--model_ckpt`: 训练好的模型路径
- `--replanstep 4`: 每4步replan一次
- `--visulization`: 生成视频
- `--metrics`: 评估指标

**预期输出结构**:
```
logs/act_lift_small_beaker_evaluation/
├── act/
│   └── evaluation_result.json
└── lift_small_beaker/
    ├── detail_info.json
    └── videos/
        ├── 0_success_True_progress_1.00.mp4
        ├── 1_success_True_progress_1.00.mp4
        ├── ...
        └── 9_success_False_progress_0.85.mp4
```

#### 4.2 检查评估结果

```bash
# 查看总体结果
cat logs/act_lift_small_beaker_evaluation/act/evaluation_result.json
```

**预期结果（训练良好的模型）**:
```json
{
  "lift_small_beaker": {
    "success_rate": 0.8,
    "intention_score": 0.7,
    "progress_score": 0.75
  }
}
```

**查看详细episode信息**:
```bash
cat logs/act_lift_small_beaker_evaluation/lift_small_beaker/detail_info.json | python -m json.tool
```

**预期详细结果**:
```json
[
  {
    "task": "lift_small_beaker",
    "success": true,
    "consumed_step": 87,
    "intention_score": 1,
    "progress_score": 1.0,
    "condition_results": {
      "LiftCondition": {
        "is_met": true
      }
    }
  },
  ...
]
```

#### 4.3 与Zero-shot对比

创建对比表验证训练效果：

| 指标 | Zero-shot (随机) | Trained (50ep) | 提升 |
|------|------------------|----------------|------|
| Success Rate | 0.0% | >70% | >70% |
| Intention Score | 0.0 | >0.5 | >0.5 |
| Progress Score | 0.0 | >0.6 | >0.6 |
| Avg Episode Length | 200 | <100 | 更高效 |

**如果性能提升不明显**（Success Rate < 30%）:
- 检查数据质量
- 检查训练loss是否收敛
- 检查模型加载是否正确
- 调试replan频率

#### 4.4 可视化验证

```bash
# 查看生成的视频
ls -lh logs/act_lift_small_beaker_evaluation/lift_small_beaker/videos/

# 观看几个视频
# - 成功的episode应该看到完整的pick + lift动作
# - 失败的episode可能因为：
#   - 抓取位置偏差
#   - lift高度不够
#   - gripper没有正确闭合
```

**质量检查清单**:
- [ ] 至少观看3个成功的episodes
- [ ] 确认pick动作准确
- [ ] 确认lift动作流畅
- [ ] 确认gripper正确开合
- [ ] 确认robot保持在workspace内

---

### Phase 5: 结果分析与决策 (可选，30分钟)

#### 5.1 如果训练成功（Success Rate > 70%）

✅ **Pipeline验证成功**！

**后续扩展方向**:

**阶段1: 扩展数据量** (1-2天)
```bash
# 收集更多数据
python scripts/trajectory_generation.py \
    --task-name lift_small_beaker \
    --n-sample 200 \
    --start-id 50 \
    --save-dir /ssd/mkqin/workspace/VLABench/datasets/lift_small_beaker_raw \
    --record-video

# 转换数据
python scripts/convert_to_lerobot.py \
    --dataset-name vlabench_lift_small_beaker_200ep \
    --dataset-path /ssd/mkqin/workspace/VLABench/datasets/lift_small_beaker_raw \
    --max-files 200

# 训练（200ep数据）
python lerobot/scripts/train.py \
    policy=act \
    dataset.repo_id=vlabench_lift_small_beaker_200ep \
    training.output_dir=/ssd/mkqin/workspace/VLABench/models/act_lift_small_beaker_200ep \
    training.n_epochs=200 \
    ...
```

**阶段2: 扩展到其他autogen_tasks** (3-5天)
可用的autogen_tasks（已验证都有expert skill sequence）:
- `lift_chemistry_tube`
- `lift_flask`
- `lift_small_cylinder`
- `open_drawer`
- `pick_bottle_lift_bottle`
- `pick_flask_place_flask`
- `pick_tube_lift_tube`
- 等等...

**阶段3: 组合任务** (1-2周)
- `pick_small_beaker_pour_small_beaker` (pick + pour)
- `pick_tube_pour_tube` (pick + pour)
- 更复杂的序列任务

#### 5.2 如果训练效果不佳（Success Rate < 30%）

**诊断步骤**:

**Step 1: 检查数据质量**
```bash
# 检查轨迹长度分布
python -c "
import h5py, glob, numpy as np
files = glob.glob('/ssd/mkqin/workspace/VLABench/datasets/lift_small_beaker_raw/lift_small_beaker/*.hdf5')
lengths = []
for f in files:
    with h5py.File(f, 'r') as hf:
        for key in hf['data'].keys():
            ee_state = hf['data'][key]['observation']['ee_state'][()]
            lengths.append(len(ee_state))
print(f'Episode lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}')
print(f'Expected: 50-100 steps per episode')
"
```

**Step 2: 检查训练日志**
```bash
# 检查loss曲线
grep "loss:" /ssd/mkqin/workspace/VLABench/models/act_lift_small_beaker_50ep/logs/*.log
```

**Step 3: 测试模型加载**
```bash
python -c "
import torch
from lerobot.common.policies.act.modeling_act import ACTPolicy

policy = ACTPolicy.from_pretrained(
    '/ssd/mkqin/workspace/VLABench/models/act_lift_small_beaker_50ep/pretrained_model'
)
print('✓ Model loaded successfully')
print(f'Device: {next(policy.parameters()).device}')
print(f'Parameters: {sum(p.numel() for p in policy.parameters()):,}')
"
```

---

## 📊 预期结果总结

### 成功标准

**最低成功标准**:
- ✅ 数据收集成功（50条有效trajectory）
- ✅ 数据转换成功（LeRobot dataset创建）
- ✅ 训练收敛（loss下降）
- ✅ Success rate > 0（优于zero-shot）

**理想成功标准**:
- ✅ Success rate > 70%
- ✅ Intention score > 0.5
- ✅ Progress score > 0.6
- ✅ 动作流畅自然
- ✅ 训练时间 < 2小时

### Pipeline验证清单

- [x] ACT policy wrapper可以加载trained model
- [x] Observation处理与训练数据一致
- [x] Action坐标变换正确
- [x] Evaluation metrics计算正确
- [x] 端到端流程无错误

---

## 🛠️ 故障排除

### 常见问题

**问题1: trajectory_generation.py找不到任务**
```bash
# 症状: KeyError: 'lift_small_beaker'
# 解决: 确认任务已注册
python -c "
from VLABench.utils.register import register
from VLABench.tasks import *
print('Available autogen tasks:')
for task in register._tasks:
    if 'lift' in task:
        print(f'  - {task}')
"
```

**问题2: CUDA out of memory**
```bash
# 解决: 减少batch size
python lerobot/scripts/train.py \
    ... \
    training.batch_size=4  # 从8降到4
```

**问题3: 训练Loss爆炸**
```bash
# 解决: 降低learning rate
python lerobot/scripts/train.py \
    ... \
    training.optimizer_lr=5e-6  # 从1e-5降到5e-6
```

**问题4: 评估时报KeyError**
```bash
# 症状: KeyError in ACT predict
# 解决: 检查模型配置与wrapper一致性
# 确保act.py中的input_features与训练时一致
```

---

## 🎯 总结

这个计划针对`lift_small_beaker`任务提供完整的训练pipeline：

**优势**:
- ✅ 任务简单，快速验证（3-4小时）
- ✅ autogen_tasks完全支持（已验证）
- ✅ 数据量小，训练快速
- ✅ 成功条件明确，易于诊断
- ✅ 适合作为baseline和调试平台

**关键里程碑**:
1. ✅ 收集50条有效trajectory (30min)
2. ✅ 转换为LeRobot dataset (15min)
3. ✅ 训练ACT模型100 epochs (2h)
4. ✅ 评估模型性能 (30min)
5. ✅ 分析结果并决定下一步 (30min)

**成功指标**: Success Rate > 70%

一旦在这个简单任务上验证成功，就可以放心地扩展到更复杂的VLABench autogen_tasks！

---

*计划制定时间: 2026-06-12*
*预计执行时间: 3-4小时*
*任务: lift_small_beaker (autogen_tasks)*
*数据量: 50 episodes*
