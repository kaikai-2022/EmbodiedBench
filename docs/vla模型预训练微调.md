# VLABench 轨迹数据生成与 OpenVLA 微调完整流程

## 一、轨迹数据生成流程

### 1.1 入口脚本

**主入口**：`/ssd/mkqin/workspace/VLABench/scripts/trajectory_generation.py`

**核心调用链**：
```
trajectory_generation.py::main()
  → 解析命令行参数 (--task-name, --save-dir, --n-sample, --robot)
  → for i in range(n_sample):
      → generate_trajectory(args, i, logger)
          → load_env(task_name, robot, eval)  # 创建环境
          → env.reset()  # 重置并应用域随机化
          → env.save()  # 获取确定��配置
          → env.get_expert_skill_sequence()  # 获取技能序列
          → for skill in skill_seq:  # 执行每个技能
              → skill(env)  # 如 SkillLib.pick(env, target_entity_name="xxx")
          → 保存 HDF5 文件
```

### 1.2 任务定义

**关键文件**：
- `/ssd/mkqin/workspace/VLABench/VLABench/tasks/dm_task.py` - 任务基类 `LM4ManipBaseTask`
- `/ssd/mkqin/workspace/VLABench/VLABench/tasks/config_manager.py` - 配置管理器
- `/ssd/mkqin/workspace/VLABench/VLABench/tasks/condition.py` - 条件系统（20+ 条件类）

**任务结构**：
```python
class LM4ManipBaseTask(composer.Task):
    def __init__(self, ...):
        self.config_manager = BenchTaskConfigManager(...)  # 管理物体配置
        self.entities = {...}  # 场景中所有实体
        self.workspace = [-0.3, 0.3, -0.2, 0.2, 0.8, 1.5]  # 工作空间
        self.conditions = ConditionSet(...)  # 终止条件
```

**任务类型**：
- **原始任务**：`VLABench/tasks/hierarchical_tasks/primitive/`（如 pick_bottle_lift_bottle）
- **组合任务**：`VLABench/tasks/hierarchical_tasks/composite/`（如 set_dining_table, play_poker）

### 1.3 技能库（Skill Library）

**关键文件**：`/ssd/mkqin/workspace/VLABench/VLABench/utils/skill_lib.py`

**核心技能**（30+ 个原子技能）：
- `pick` - 抓取物体
- `place` - 放置到容器
- `lift` - 提升物体
- `open_door` / `close_door` - 开关门
- `pour` - 倾倒
- `press` - 按压按钮
- `push` / `pull` - 推拉操作
- `stir_entity_with_tool` - 搅拌
- 等等

**技能调用模式**（以 `pick` 为例）：
```python
def pick(env, target_entity_name, ...):
    # 1. 自动选择抓取点
    key_pos, prepare_pos, key_quat = find_keypoint_and_prepare_grasp(target_entity)

    # 2. 获取障碍物点云
    obstacle_pcd = env.get_obstacle_pcd()

    # 3. RRT 运动规划（两阶段）
    # 阶段1: 当前位置 → 准备点（物体上方）
    init2prepare_path = rrt_motion_planning(start_pos, prepare_pos, obstacle_pcd)
    # 阶段2: 准备点 → 抓取点（垂直下降）
    prepare2key_path = rrt_motion_planning(prepare_pos, key_pos, obstacle_pcd)

    # 4. 路径插值（SLERP 四元数）
    interpolated_path, interpolated_quat = interpolate_path(
        path, quats, target_velocity=0.05
    )

    # 5. 执行轨迹
    obs, waypoints, _, task_success = SkillLib.step_trajectory(
        env, interpolated_path, interpolated_quat, gripper_state
    )

    # 6. 关闭夹爪
    SkillLib.close_gripper(env)
    return observations, waypoints, stage_success, task_success
```

### 1.4 运动规划器（RRT + 路径平滑）

**RRT 算法**：
- **文件**：`/ssd/mkqin/workspace/VLABench/VLABench/algorithms/motion_planning/rrt.py`
- **函数**：`rrt_motion_planning(start_pos, end_pos, obstacle_pcd, ...)`
- **参数**：
  - `q=0.05` - 树的边长度
  - `r=0.02` - 碰撞检查的最小边长度
  - `max_samples=1024` - 最大采样数
  - `margin=2e-2` - 障碍物膨胀半径
  - `smooth_method` - 平滑方法（None/"bezier"/"polynomial"）

**路径平滑**：
- **Bezier 曲线**：`bezier_smoothing(path, smoothing_factor=0.1)`（使用 scipy B-样条拟合）
- **多项式插值**：`polynomial_smoothing(path, num_points=100)`（三次样条插值）

**四元数插值（SLERP）**：
- **文件**：`/ssd/mkqin/workspace/VLABench/VLABench/algorithms/utils.py`
- **函数**：`qauternion_slerp(start_quat, end_quat, t)` 和 `interpolate_path(positions, quaternions, target_velocity=0.05)`

### 1.5 指令增强（Instruction Augmentation）

**关键文件**：
- `/ssd/mkqin/workspace/VLABench/VLABench/tasks/dm_task.py` - `build_instruction()` 方法（第 464-486 行）
- `/ssd/mkqin/workspace/VLABench/VLABench/utils/gpt_utils.py` - `query_gpt4_v()` 函数
- `/ssd/mkqin/workspace/VLABench/VLABench/configs/prompt/prompt.json` - Prompt 模板

**实现流程**：
```python
# 1. GPT-4 调用（每个任务生成 10 条指令）
def build_instruction(self):
    prompt = f"{sys_prompt_0} {entity_names}. {sys_prompt_1} {target_entity}. {sys_prompt_2}"
    instructions = query_gpt4_v(prompt, model="gpt-4-turbo")
    instructions = re.findall(r'instruction:\s*"([^"]*)"', instructions)
    self.instructions = instructions  # 存储为列表

# 2. 按轨迹级别随机采样
def get_instruction(self):
    if isinstance(self.instructions, list):
        return random.sample(self.instructions, 1)[-1]  # 随机选择一条
    return self.instructions
```

**Prompt 模板**（来自 `prompt.json`）：
```json
{
  "instruction_0": "I am going to make some task instructions for a robot arm.",
  "instruction_1": "",
  "instruction_2": "The current task involves a dining scenario... Please provide ten tasks..."
}
```

### 1.6 域随机化（Domain Randomization）

**关键文件**：
- `/ssd/mkqin/workspace/VLABench/VLABench/tasks/components/entity.py` - `Entity.initialize_episode()`（第 63-90 行）
- `/ssd/mkqin/workspace/VLABench/VLABench/tasks/components/scene.py` - `Scene.initialize_episode()`

**随机化维度**：
1. **物体位置/方向**：
   ```python
   if randomness.get("pos"):
       new_xpos += randomness["pos"] * uniform(-1, 1)
   if randomness.get("quat"):
       new_xquat = quaternion_multiply(new_xquat, euler_to_quaternion(...))
   ```

2. **网格缩放**：
   ```python
   if randomness.get("scale"):
       scale_value = uniform(scale_range[0], scale_range[1])
       self.set_scale(physics, scale_value)
   ```

3. **场景布局**（网格采样）：
   ```python
   # dm_task.py 第 124-143 行
   sampled_points = grid_sample(workspace, ngrid=[10, 10], n_points, farthest_sample=True)
   # 使用 Farthest Point Sampling 确保均匀分布
   ```

4. **背景/物体纹理**：
   ```python
   # Scene.set_texture() - 修改地板材质
   # Entity.set_texture() - 抽象方法，由子类实现
   FLOOR_TEXTURE = [f"floor{i}" for i in range(13)] + ["floor_gray"]
   ```

5. **默认随机化参数**（来自 `config_manager.py`）：
   ```python
   DEFAULT_RABDOMNESS = dict(
       pos=[0.02, 0.02, 0],  # x, y, z 方向的位置扰动
       quat=[0, 0, 0.05],    # roll, pitch, yaw 的角度扰动
   )
   ```

### 1.7 数据保存格式

**关键文件**：`/ssd/mkqin/workspace/VLABench/VLABench/utils/data_utils.py`

**保存格式**：HDF5（使用 h5py，gzip 压缩）

**文件结构**：
```
data_{index}.hdf5
└── data/
    └── data_{timestamp}/
        ├── observation/
        │   ├── q_state           # 关节位置 (7,)
        │   ├── q_velocity        # 关节速度
        │   ├── q_acceleration    # 关节加速度
        │   ├── rgb              # RGB 图像 (n_cam, 480, 480, 3)
        │   ├── depth            # 深度图 (n_cam, 480, 480)
        │   ├── segmentation     # 语义分割
        │   ├── robot_mask       # 机器人掩码
        │   ├── instrinsic       # 相机内参
        │   ├── extrinsic        # 相机外参
        │   ├── point_cloud_points  # 点云坐标
        │   ├── point_cloud_colors  # 点云颜色
        │   └── ee_state         # 末端执行器状态 (8,)
        ├── meta_info/
        │   ├── entities         # 实体列表
        │   ├── target_entity    # 目标实体
        │   ├── episode_config   # 确定性配置（JSON，含完整指令列表）
        │   └── instruction      # 任务指令（随机采样的 1 条）
        └── trajectory           # 动作序列 (T, 8)
```

**动作维度**：`[x, y, z, roll, pitch, yaw, gripper_width_1, gripper_width_2]`（8 维）

**采样频率**：
- **物理时间步**：0.001s（来自 `task_config.json`）
- **控制时间步**：0.1s（`physics_timestep × 100`）
- **观测采样**：每个控制步采一次（10 Hz）

**坐标系转换**（保存到 HDF5 前）：
```python
robot_frame_waypoints = [
    waypoint - np.concatenate([robot_position, np.zeros(5)])
    for waypoint in waypoints
]
```

---

## 二、OpenVLA 微调流程

### 2.1 数据准备

**数据格式转换**：
1. **HDF5 → LeRobot**（推荐）：
   - **脚本**：`/ssd/mkqin/workspace/VLABench/scripts/convert_to_lerobot.py`
   - **函数**：`create_lerobot_dataset_from_hdf5()`
   - **输出**：LeRobot 标准格式（存储在 `HF_HOME/lerobot/`）

2. **HDF5 → RLDS**（备选）：
   - **脚本**：`/ssd/mkqin/workspace/VLABench/scripts/convert_to_rlds.py`
   - **函数**：`extract_step_data()` + `process_step()`
   - **输出**：TFDS 格式

**LeRobot 特征配置**（来自 `convert_to_lerobot.py`）：
```python
features = {
    "image": {           # 前置相机（index 2）
        "dtype": "image",
        "shape": (480, 480, 3),
    },
    "wrist_image": {     # 腕部相机（index 3）
        "dtype": "image",
        "shape": (480, 480, 3),
    },
    "state": {           # 末端执行器状态 (7 维)
        "dtype": "float",
        "shape": (7,),
    },
    "action": {          # 动作 (7 维)
        "dtype": "float",
        "shape": (7,),
    },
}
```

**关键转换逻辑**：
```python
# 1. 图像视角映射
"image": images[i][2],      # 前置相机
"wrist_image": images[i][3] # 腕部相机

# 2. 四元数 → 欧拉角
ee_euler = np.array([quat2euler(q) for q in ee_quat])

# 3. 坐标系转换
ee_pos -= robot_frame_pos  # 转到机器人基座坐标系

# 4. 夹爪二值化
if gripper_width > 0.03:
    gripper = 1  # 张开
else:
    gripper = 0  # 闭合
```

### 2.2 数据采样规模

**数据集统计**（来自 `models/openvla-7b-vlabench-primitive-lora/dataset_statistics.json`）：
```json
{
  "primitive": {
    "num_trajectories": 5000,
    "num_transitions": 575101,
    "action": {
      "mean": [...],  # 7 维均值
      "std": [...],   # 7 维标准差
      "min": [...],
      "max": [...]
    }
  }
}
```

**采样策略**：
- **基础任务**（primitive）：10 个任务（select_fruit, select_toy, select_poker, select_painting, select_mahjong, texas_holdem, etc.）
- **每个任务**：500 条轨迹（总计 5000 条）
- **论文提到的 "1600 条"**：可能是其他训练配置（100 条/任务 × 16 任务）

### 2.3 微调策略（LoRA）

**微调脚本**（不在 VLABench 仓库中，使用 OpenVLA 官方）：
- **官方脚本**：`https://github.com/openvla/openvla/blob/main/scripts/finetune.py`

**LoRA 超参数**（从模型路径提取）：
```
LORA_CKPT=".../openvla-7b+vlabench_primitive+b16+lr-0.0005+lora-r32+dropout-0.0"
```

| 参数 | 值 |
|------|-----|
| **LoRA Rank (r)** | 32 |
| **LoRA Dropout** | 0.0 |
| **Learning Rate** | 0.0005 |
| **Batch Size** | 16 |
| **优化器** | AdamW（推测） |
| **训练步数** | ~312 steps/epoch（5000/16） |

**LoRA 加载代码**（来自 `VLABench/evaluation/model/policy/openvla.py`）：
```python
from peft import PeftModel, PeftConfig

model = AutoModelForVision2Seq.from_pretrained(model_ckpt, ...)
if lora_ckpt is not None:
    peft_config = PeftConfig.from_pretrained(lora_ckpt)
    model = PeftModel.from_pretrained(model, lora_ckpt, config=peft_config)
```

**注意**：当前存储的 `openvla-7b-vlabench-primitive-lora/` 是**已合并的完整模型**（约 14.8 GB，4 个 safetensors 文件），LoRA 适配器已合并到基础模型中。

### 2.4 输入图像处理

**图像配置**（来自 `preprocessor_config.json`）：
```json
{
  "image_resize_strategy": "resize-naive",
  "input_sizes": [[3, 224, 224], [3, 224, 224]],  // 双视觉编码器
  "interpolations": ["bicubic", "bicubic"],
  "means": [
    [0.485, 0.456, 0.406],  // DINOv2 (ImageNet)
    [0.5, 0.5, 0.5]          // SigLIP
  ],
  "stds": [
    [0.229, 0.224, 0.225],
    [0.5, 0.5, 0.5]
  ]
}
```

**视觉编码器架构**（来自 `config.json`）：
```json
{
  "vision_backbone_id": "dinosiglip-vit-so-224px",
  "timm_model_ids": [
    "vit_large_patch14_reg4_dinov2.lvd142m",  // DINOv2 ViT-L/14
    "vit_so400m_patch14_siglip_224"            // SigLIP ViT-So400M/14
  ],
  "use_fused_vision_backbone": true
}
```

**图像处理流程**：
1. **相机选择**：
   ```python
   CAMERA_VIEW_INDEX = {
       "select_painting": 1,  # 第 2 个相机
       "select_chemistry_tube": 2,
       "texas_holdem": 2,
       # ...
   }
   cam_index = CAMERA_VIEW_INDEX.get(task_name, 2)  # 默认前置相机
   rgb = obs["rgb"][cam_index]
   ```

2. **预处理**：
   ```python
   # Resize 到 224×224
   # 双编码器归一化
   inputs = processor(prompt, Image.fromarray(rgb).convert("RGB"))
   ```

3. **Prompt 构建**：
   ```python
   def build_prompt(instruction):
       prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut: "
       prompt = prompt.replace("_seen", "").replace("_unseen", "")
       return prompt
   ```

### 2.5 动作表示与归一化

**动作维度**：7 维 `[x, y, z, roll, pitch, yaw, gripper]`

**归一化方法**（基于 `dataset_statistics.json`）：
```python
# 使用 q01 和 q99 分位数截断
action_clipped = np.clip(action, q01, q99)
# 使用 mean 和 std 标准化
action_normalized = (action_clipped - mean) / std
```

**动作分箱**（用于训练时的分类损失）：
```json
{
  "n_action_bins": 256  // 动作被离散化为 256 个 bin
}
```

**推理时的动作处理**：
```python
def predict(obs, unnorm_key=None):
    # 1. 模型预测 delta 动作
    delta_action = model.predict_action(**inputs, unnorm_key=unnorm_key)

    # 2. 转换为绝对位姿
    current_ee_state = obs["ee_state"]  # (8,) [pos(3), quat(4), gripper(1)]
    pos, quat = current_ee_state[:3], current_ee_state[3:7]
    euler = quaternion_to_euler(quat)

    # 3. 累加 delta
    target_pos = pos + delta_action[:3]
    target_euler = euler + delta_action[3:6]
    gripper_open = delta_action[-1]

    # 4. 转换为控制命令
    gripper_state = np.ones(2)*0.04 if gripper_open >= 0.1 else np.zeros(2)
    return target_pos, target_euler, gripper_state
```

### 2.6 训练超参数（部分已知）

**已知参数**（从路径和配置推断）：
| 参数 | 值 | 证据 |
|------|------|------|
| **Batch Size** | 16 | `+b16+` |
| **Learning Rate** | 0.0005 | `+lr-0.0005+` |
| **LoRA Rank** | 32 | `+lora-r32+` |
| **LoRA Dropout** | 0.0 | `+dropout-0.0` |
| **优化器** | AdamW | OpenVLA 默认 |
| **是否冻结视觉编码器** | 是 | OpenVLA 默认 |

**未知参数**（需查看 OpenVLA 官方 `finetune.py`）：
- **训练步数/Epochs**
- **Gradient Accumulation**
- **LR Scheduler**（余弦退火？）
- **Weight Decay**
- **Warmup Steps**

---

## 三、完整调用链总结

### 3.1 数据生成完整调用链

```
1. trajectory_generation.py::main()
   ↓
2. load_env(task_name, robot)  # envs/__init__.py
   ├─ 读取 robot_config.json, task_config.json
   ├─ register.load_robot(robot)  # 创建机器人
   ├─ register.load_task(task_name)  # 创建任务
   └─ LM4ManipDMEnv(task=task)  # 创建环境
   ↓
3. env.reset()
   ├─ task.initialize_episode(physics, random_state)
   │  ├─ ConfigManager.get_task_config()  # 生成实体配置
   │  ├─ grid_sample(workspace, ngrid)  # 场景布局随机化
   │  └─ Entity.initialize_episode()  # 每个实体应用域随机化
   ├─ task.build_instruction()  # GPT-4 生成 10 条指令
   └─ reset_camera_views()  # 重置相机
   ↓
4. env.get_expert_skill_sequence()
   └─ 返回技能序列 [partial(SkillLib.pick, ...), partial(SkillLib.lift, ...), ...]
   ↓
5. 执行每个技能 skill(env)
   ├─ SkillLib.pick(env, target_entity_name="xxx")
   │  ├─ find_keypoint_and_prepare_grasp()  # 自动选择抓取点
   │  ├─ env.get_obstacle_pcd()  # 获取障碍物点云
   │  ├─ rrt_motion_planning()  # RRT 搜索（两阶段）
   │  ├─ interpolate_path()  # SLERP 四元数插值
   │  ├─ SkillLib.step_trajectory()  # 执行轨迹
   │  │  ├─ for point, quat in zip(path, quats):
   │  │  │  ├─ robot.get_qpos_from_ee_pos()  # IK 求解
   │  │  │  ├─ env.step(action)  # 仿真一步
   │  │  │  └─ env.get_observation()  # 收集观测
   │  │  └─ return obs, waypoints
   │  └─ SkillLib.close_gripper(env)
   └─ 累积 observations 和 waypoints
   ↓
6. 检查任务条件
   └─ task.conditions.is_met(physics)
   ↓
7. 保存数据
   ├─ process_observations(observations)  # 合并观测
   ├─ 转换坐标系到机器人基座
   ├─ task.get_instruction()  # 随机采样 1 条指令
   └─ save_single_data()  # 保存 HDF5
```

### 3.2 微调完整调用链

```
阶段 A：数据准备
1. convert_to_lerobot.py
   ↓
2. create_lerobot_dataset_from_hdf5(hdf5_files)
   ├─ 读取 HDF5 文件
   ├─ 提取 images（front + wrist）、ee_state、actions、instruction
   ├─ 转换格式
   │  ├─ quat → euler
   │  ├─ 全局坐标 → 机器人坐标
   │  └─ 夹爪二值化
   └─ 保存 LeRobot 格式（存储在 HF_HOME/lerobot/）

阶段 B：OpenVLA 微调（使用官方脚本）
3. OpenVLA finetune.py（不在 VLABench 仓库）
   ├─ 加载 LeRobot 数据集
   ├─ 加载 OpenVLA-7B 基座
   ├─ 应用 LoRA（rank=32, dropout=0.0）
   ├─ 训练循环
   │  ├─ 图像预处理（224×224, DINOv2+SigLIP）
   │  ├─ 构建 Prompt: "In: What action should the robot take to {instruction}?\nOut: "
   │  ├─ 模型前向传播
   │  ├─ 计算损失（L1/L2 或 CE）
   │  └─ 反向传播 + 更新
   └─ 保存 LoRA 适配器（或合并完整模型）

阶段 C：评估
4. evaluate_policy.py
   ├─ 加载 OpenVLA 模型
   ├─ 加载任务配置（track_1_in_distribution.json）
   └─ 运行评估循环
```

---

## 四、关键代码文件清单

### 4.1 轨迹生成核心文件

| 文件路径 | 作用 |
|---------|------|
| `/ssd/mkqin/workspace/VLABench/scripts/trajectory_generation.py` | **数据生成入口** |
| `/ssd/mkqin/workspace/VLABench/VLABench/utils/skill_lib.py` | **技能库（30+ 技能）** |
| `/ssd/mkqin/workspace/VLABench/VLABench/algorithms/motion_planning/rrt.py` | **RRT 运动规划** |
| `/ssd/mkqin/workspace/VLABench/VLABench/algorithms/path_smoothing/bezier_curve.py` | Bezier 平滑 |
| `/ssd/mkqin/workspace/VLABench/VLABench/algorithms/utils.py` | **SLERP 四元数插值** |
| `/ssd/mkqin/workspace/VLABench/VLABench/tasks/dm_task.py` | 任务基类、指令生成 |
| `/ssd/mkqin/workspace/VLABench/VLABench/tasks/config_manager.py` | 配置管理器 |
| `/ssd/mkqin/workspace/VLABench/VLABench/tasks/condition.py` | 条件系统 |
| `/ssd/mkqin/workspace/VLABench/VLABench/envs/dm_env.py` | **仿真环境** |
| `/ssd/mkqin/workspace/VLABench/VLABench/robots/single_arm/franka.py` | Franka 机器人 |
| `/ssd/mkqin/workspace/VLABench/VLABench/utils/data_utils.py` | **数据保存（HDF5）** |

### 4.2 指令增强与域随机化文件

| 文件路径 | 作用 |
|---------|------|
| `/ssd/mkqin/workspace/VLABench/VLABench/utils/gpt_utils.py` | **GPT-4 API 调用** |
| `/ssd/mkqin/workspace/VLABench/VLABench/configs/prompt/prompt.json` | **Prompt 模板** |
| `/ssd/mkqin/workspace/VLABench/VLABench/tasks/components/entity.py` | **实体基类、域随机化** |
| `/ssd/mkqin/workspace/VLABench/VLABench/tasks/components/scene.py` | 场景随机化 |
| `/ssd/mkqin/workspace/VLABench/VLABench/tasks/config_manager.py` | 默认随机化参数 |

### 4.3 数据格式转换文件

| 文件路径 | 作用 |
|---------|------|
| `/ssd/mkqin/workspace/VLABench/scripts/convert_to_lerobot.py` | **HDF5 → LeRobot** |
| `/ssd/mkqin/workspace/VLABench/scripts/convert_to_rlds.py` | HDF5 → RLDS |
| `/ssd/mkqin/workspace/VLABench/VLABench/utils/rlds_builder.py` | RLDS 构建 |

### 4.4 OpenVLA 微调与评估文件

| 文件路径 | 作用 |
|---------|------|
| `/ssd/mkqin/workspace/VLABench/VLABench/evaluation/model/policy/openvla.py` | **OpenVLA 策略实现** |
| `/ssd/mkqin/workspace/VLABench/scripts/evaluate_policy.py` | 评估入口 |
| `/ssd/mkqin/workspace/VLABench/sh/evaluate_openvla.sh` | 批量评估脚本 |
| `/ssd/mkqin/workspace/VLABench/models/openvla-7b-vlabench-primitive-lora/config.json` | 模型配置 |
| `/ssd/mkqin/workspace/VLABench/models/openvla-7b-vlabench-primitive-lora/preprocessor_config.json` | 图像预处理 |
| `/ssd/mkqin/workspace/VLABench/models/openvla-7b-vlabench-primitive-lora/dataset_statistics.json` | **数据集统计** |

---

## 五、核心设计要点

### 5.1 轨迹生成核心要点

1. **模块化技能库**：30+ 原子技能，通过 `partial` 函数组合成复杂任务
2. **两阶段运动规划**：先到准备点，再垂直下降（避免侧面碰撞）
3. **多种路径平滑方法**：Bezier、多项式插值可选
4. **完整 Grasp Lock 机制**：避免抓取后物体滑动
5. **灵活的条件系统**：支持单条件和组合条件（20+ 条件类）
6. **指令多样性**：GPT-4 生成 10 条指令，每条轨迹随机采样 1 条
7. **丰富的域随机化**：位置、方向、缩放、纹理、布局、光照

### 5.2 OpenVLA 微调核心要点

1. **LoRA 高效微调**：rank=32, dropout=0.0, lr=0.0005, batch=16
2. **双视觉编码器**：DINOv2 + SigLIP 融合，都输出 224×224 特征
3. **单视角输入**：只使用一个相机（通常是前置相机）
4. **Delta 动作预测**：预测增量动作，推理时累加到当前状态
5. **动作分箱**：256 个 bin（用于训练时的分类损失）
6. **数据归一化**：使用 q01/q99 截断 + mean/std 标准化
7. **Prompt 模板**：`"In: What action should the robot take to {instruction}?\nOut: "`

---

## 六、使用示例

### 6.1 生成轨迹数据

```bash
# 单个任务
python scripts/trajectory_generation.py \
    --task-name select_poker \
    --save-dir /path/to/dataset \
    --n-sample 100 \
    --robot franka

# 批量任务（使用脚本）
sh dataset_generation.sh
```

### 6.2 转换数据格式

```bash
# 转换为 LeRobot（推荐）
python scripts/convert_to_lerobot.py \
    --dataset-name vlabench \
    --dataset-path /path/to/hdf5/data \
    --max-files 100

# 转换为 RLDS（备选）
python scripts/convert_to_rlds.py \
    --task select_poker,select_mahjong \
    --save_dir /path/to/dataset
```

### 6.3 评估策略

```bash
# 评估 OpenVLA
python scripts/evaluate_policy.py \
    --model-name openvla \
    --model-ckpt /path/to/openvla-7b-vlabench-primitive-lora \
    --config configs/evaluate/track_1_in_distribution.json \
    --save-dir /path/to/results

# 批量评估
bash sh/evaluate_openvla.sh
```

---

## 七、与论文对应关系验证

### 7.1 数据生成验证

| 论文描述 | 代码实现 | 证据 |
|---------|---------|------|
| **自动化轨迹生成** | `trajectory_generation.py` + `skill_lib.py` | 完整的自动化数据收集管线 |
| **环境点云、抓取点先验** | `env.get_obstacle_pcd()` + `find_keypoint_and_prepare_grasp()` | entity.py, skill_lib.py |
| **技能库（Pick、Place 等）** | `SkillLib` 类（30+ 技能） | skill_lib.py |
| **RRT 算法** | `rrt_motion_planning()` | rrt.py |
| **SLERP 四元数插值** | `qauternion_slerp()` + `interpolate_path()` | algorithms/utils.py |
| **Bezier 曲线平滑** | `bezier_smoothing()` | bezier_curve.py |
| **GPT-4 生成 10 条指令** | `build_instruction()` + `query_gpt4_v()` | dm_task.py |
| **域随机化**（位置/方向/缩放/纹理/光照） | `Entity.initialize_episode()` | entity.py |
| **资产库（163 类别，2164 物品）** | `name2class_xml` + `assets/obj/meshes/` | constant.py + assets 目录 |

### 7.2 OpenVLA 微调验证

| 论文描述 | 代码实现 | 证据 |
|---------|---------|------|
| **LoRA 微调** | 从路径名称推断：`lora-r32+dropout-0.0` | evaluate_openvla.sh |
| **1600 条轨迹**（论文）/ **5000 条**（代码） | `dataset_statistics.json` 显示 5000 条 | dataset_statistics.json |
| **每个任务 100 条**（论文推测） | 可能是其他训练配置 | 需查 OpenVLA 官方脚本 |
| **单视角 224×224** | `preprocessor_config.json`：`input_sizes: [[3,224,224], ...]` | preprocessor_config.json |
| **DINOv2 + SigLIP 双编码器** | `config.json`：`use_fused_vision_backbone: true` | config.json |

---

## 八、总结

### 8.1 轨迹数据生成流程总结

**核心思路**：通过模块化技能库 + RRT 运动规划 + 域随机化，自动生成高质量的机器人操作轨迹。

**关键技术**：
1. **技能组合**：将复杂任务分解为原子技能序列（如 `pick` → `lift` → `place`）
2. **运动规划**：使用 RRT 算法进行避障路径搜索，分为两阶段（准备点 → 目标点）
3. **路径平滑**：Bezier 曲线或多项式插值，SLERP 四元数插值
4. **指令增强**：GPT-4 生成 10 条多样化指令，每条轨迹随机采样
5. **域随机化**：位置、方向、缩放、纹理、布局、光照等多维度随机化
6. **数据保存**：HDF5 格式，包含图像、深度、点云、关节状态、动作、指令

### 8.2 OpenVLA 微调流程总结

**核心思路**：将 VLABench 数据转换为 LeRobot 格式，使用 LoRA 策略微调 OpenVLA-7B 基座模型。

**关键技术**：
1. **数据转换**：HDF5 → LeRobot（图像视角选择、坐标转换、夹爪二值化）
2. **LoRA 微调**：rank=32, dropout=0.0, lr=0.0005, batch=16
3. **双视觉编码器**：DINOv2 + SigLIP 融合，224×224 输入
4. **Delta 动作预测**：预测增量动作，推理时累加
5. **Prompt 模板**：`"In: What action should the robot take to {instruction}?\nOut: "`
6. **归一化**：q01/q99 截断 + mean/std 标准化

### 8.3 完整数据流

```
任务定义（JSON + Python 类）
  ↓
技能序列（partial 函数列表）
  ↓
RRT 运动规划 + SLERP 插值 + Bezier 平滑
  ↓
仿真执行（dm_control/MuJoCo）
  ↓
观测收集（RGB、深度、点云、关节状态）
  ↓
域随机化（位置/方向/缩放/纹理/布局）
  ↓
指令生成（GPT-4 生成 10 条，随机采样 1 条）
  ↓
HDF5 保存
  ↓
LeRobot 转换（图像/坐标/夹爪处理）
  ↓
OpenVLA 微调（LoRA, 224×224, 双编码器）
  ↓
策略评估（evaluate_policy.py）
```

---

## 九、需要进一步验证的点

1. **训练超参数细节**：需查看 OpenVLA 官方 `finetune.py` 获取：
   - 优化器类型（AdamW 参数）
   - 训练步数/Epochs
   - Gradient Accumulation
   - LR Scheduler
   - Weight Decay
   - Warmup Steps

2. **论文与代码的差异**：
   - 论文提到 "1600 条轨迹"，代码中是 5000 条
   - 可能是不同训练配置，需查证

3. **LoRA Alpha 值**：路径名称中没有体现，需查训练脚本

---

## Context

本文档梳理了 VLABench 项目中从轨迹数据生成到 OpenVLA 模型微调的完整流程，基于对项目代码的深入探索。目标是帮助理解：
1. 如何生成一个任务的轨迹数据（从入口脚本到 HDF5 保存的完整调用链）
2. 如何将这些轨迹数据用于 OpenVLA 模型的微调（数据转换、LoRA 策略、训练配置）

**用户需求**：结合论文和代码，梳理 VLABench 轨迹生成和 OpenVLA 微调的具体实现细节。
