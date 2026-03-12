# VLABench技能序列生成详细流程指南

## 概述

本文档详细记录了VLABench中从任务定义到生成完整标准答案技能序列的全过程，包括涉及的所有文件、代码逻辑和数据流转。

**核心理念**：VLABench的标准答案技能序列是通过程序员在代码中预定义的专家策略，在Mujoco物理模拟器中实际执行后生成的。

---

## 完整流程概览

```
1. 任务类定义 (Task Class Definition)
   ↓
2. 专家技能序列定义 (Expert Skill Sequence)
   ↓
3. 轨迹生成脚本执行 (Trajectory Generation)
   ↓
4. Mujoco物理模拟执行 (Physics Simulation)
   ↓
5. 数据收集与保存 (Data Collection)
   ↓
6. 标准答案生成 (Ground Truth Generation)
```

---

## 阶段一：任务类定义

### 1.1 任务基类结构

**文件位置**：`/ssd/mkqin/workspace/VLABench/VLABench/tasks/dm_task.py`

**关键代码**（第411-416行）：
```python
def get_expert_skill_sequence(self, physics):
    """
    Expert trajectory generation for the task.
    Notice that the success rate is not 100%.
    """
    logging.info(f"Task:{self.task_name} did not implement get_expert_skill_sequence method")
    return None
```

**说明**：
- 所有任务类继承自 `DMTask` 基类
- 每个具体任务必须实现 `get_expert_skill_sequence()` 方法
- 该方法返回一个技能序列列表

### 1.2 任务类层次结构

```
DMTask (基类)
├── PrimitiveTask (原始任务)
│   ├── SelectPokerTask
│   ├── AddCondimentTask
│   ├── InsertFlowerTask
│   └── ...
└── CompositeTask (复合任务)
    ├── ClusterBookTask
    ├── CookDishesTask
    └── ...
```

---

## 阶段二：专家技能序列定义

### 2.1 简单任务示例：Select Poker

**文件位置**：`/ssd/mkqin/workspace/VLABench/VLABench/tasks/hierarchical_tasks/primitive/select_poker_series.py`

**关键代码**（第163-168行）：
```python
def get_expert_skill_sequence(self, physics):
    skill_sequence = [
        partial(SkillLib.pick,
                target_entity_name=self.target_entity,
                prior_eulers=[[-np.pi, 0, 0]]),
        partial(SkillLib.lift),
    ]
    return skill_sequence
```

**技能序列解析**：
1. `pick(target_entity_name=扑克牌)` - 抓取目标扑克牌
2. `lift()` - 抬起扑克牌

### 2.2 中等复杂度任务示例：Add Condiment

**文件位置**：`/ssd/mkqin/workspace/VLABench/VLABench/tasks/hierarchical_tasks/primitive/add_condiment_series.py`

**关键代码**（第96-104行）：
```python
def get_expert_skill_sequence(self, physics):
    # 动态计算目标位置（容器上方0.2米）
    target_pos = np.array(self.entities[self.target_container].get_xpos(physics)) + np.array([0, 0, 0.2])

    skill_sequence = [
        partial(SkillLib.pick,
                target_entity_name=self.target_entity,
                prior_eulers=[[-np.pi/2, -np.pi/2, np.pi/2]]),
        partial(SkillLib.lift,
                lift_height=0.2,
                gripper_state=np.zeros(2)),
        partial(SkillLib.moveto,
                target_pos=target_pos,
                gripper_state=np.zeros(2)),
        partial(SkillLib.pour)
    ]
    return skill_sequence
```

**技能序列解析**：
1. `pick(调料瓶)` - 抓取调料瓶，指定抓取角度
2. `lift(高度=0.2米)` - 抬起0.2米，保持夹爪闭合
3. `moveto(目标位置)` - 移动到目标容器上方
4. `pour()` - 倾倒调料

**关键特点**：
- 使用 `physics` 参数动态计算目标位置
- 参数化设计，适应不同场景配置

### 2.3 复杂任务示例：Cluster Book

**文件位置**：`/ssd/mkqin/workspace/VLABench/VLABench/tasks/hierarchical_tasks/composite/base.py`

**关键代码**（第46-68行）：
```python
def get_expert_skill_sequence(self, physics, prior_eulers):
    # 获取两类物体
    cluster_entities_1 = self.config_manager.entities_to_load["cls_1"]
    cluster_entities_2 = self.config_manager.entities_to_load["cls_2"]

    # 获取两个目标容器
    container_1 = self.config_manager.container_to_load["target_container_1"]
    container_2 = self.config_manager.container_to_load["target_container_2"]

    skill_sequence = []

    # 为每一类物体生成技能序列
    for index, (cluster_entities, container) in enumerate(
        zip([cluster_entities_1, cluster_entities_2],
            [container_1, container_2])):

        for i, entity in enumerate(cluster_entities):
            # 抓取和抬起
            skill_sequence.extend([
                partial(SkillLib.pick,
                        target_entity_name=entity,
                        prior_eulers=prior_eulers),
                partial(SkillLib.lift,
                        gripper_state=np.zeros(2)),
            ])

            # 计算放置位置（错开排列）
            target_container = self.entities[container]
            target_place_point = np.array(target_container.get_place_point(physics)[-1])
            target_place_point[1] += 0.1 * (i - 0.5)  # Y轴偏移

            # 放置
            skill_sequence.append(
                partial(SkillLib.place,
                        target_container_name=container,
                        target_pos=target_place_point)
            )

    return skill_sequence
```

**技能序列解析**（假设有3本书分两类）：
1. `pick(书1)` → `lift()` → `place(容器1, 位置A)`
2. `pick(书2)` → `lift()` → `place(容器1, 位置B)`
3. `pick(书3)` → `lift()` → `place(容器2, 位置C)`

**关键特点**：
- 循环生成多个物体的操作序列
- 动态计算每个物体的放置位置，避免碰撞

---

## 阶段三：技能库（Skill Library）

### 3.1 技能库文件

**文件位置**：`/ssd/mkqin/workspace/VLABench/VLABench/utils/skill_lib.py`

### 3.2 可用技能列表

| 技能名称 | 功能描述 | 主要参数 |
|---------|---------|---------|
| `pick` | 抓取物体 | `target_entity_name`, `prior_eulers`（抓取角度） |
| `place` | 放置物体到容器 | `target_container_name`, `target_pos` |
| `lift` | 抬起物体 | `lift_height`, `gripper_state` |
| `moveto` | 移动到指定位置 | `target_pos`, `target_quat`, `gripper_state` |
| `pour` | 倾倒液体 | `target_delta_qpos`, `target_q_velocity` |
| `push` | 推动物体 | `target_pos`, `push_distance` |
| `pull` | 拉动物体 | `target_pos`, `pull_distance` |
| `open_door` | 打开门 | `target_container_name` |
| `close_door` | 关闭门 | `target_container_name` |
| `open_drawer` | 打开抽屉 | `target_container_name`, `drawer_id` |
| `press` | 按压按钮 | `target_pos`, `target_quat` |
| `flip` | 翻转物体 | `gripper_state` |
| `wait` | 等待 | `wait_time` |

### 3.3 技能执行返回值

每个技能执行后返回：
```python
obs, waypoint, stage_success, task_success = skill(env)
```

- `obs`: 观察数据（RGB图像、深度图、机器人状态等）
- `waypoint`: 机器人关节轨迹点
- `stage_success`: 当前技能是否执行成功
- `task_success`: 整个任务是否完成

---

## 阶段四：轨迹生成脚本执行

### 4.1 脚本文件

**文件位置**：`/ssd/mkqin/workspace/VLABench/scripts/trajectory_generation.py`

### 4.2 执行命令

```bash
python scripts/trajectory_generation.py \
    --task_name select_poker \
    --n_sample 100 \
    --save_dir ./dataset/training_data \
    --record_video
```

**参数说明**：
- `--task_name`: 任务名称
- `--n_sample`: 生成样本数量
- `--save_dir`: 数据保存目录
- `--record_video`: 是否录制视频
- `--early_stop`: 技能失败时是否提前终止
- `--debug`: 调试模式

### 4.3 核心生成函数

**关键代码**（第30-112行）：

```python
def generate_trajectory(args, index, logger):
    # 1. 初始化环境
    env = create_env(task_name=args.task_name,
                     obs_type=args.obs_type,
                     obs_camera_ids=args.obs_camera_ids,
                     task_variation=args.task_variation)

    # 2. 保存环境配置
    episode_config = env.save()

    # 3. 获取任务元信息
    target_entity = env.task.config_manager.target_entity
    instruction = env.task.get_instruction()
    meta_info = dict(
        target_entity=[target_entity],
        entities=list(env.task.entities.keys()),
        instruction=[instruction],
    )

    # 4. 获取专家技能序列（这里调用前面定义的方法）
    skill_seq = env.get_expert_skill_sequence()

    # 5. 执行技能序列
    observations, waypoints = [], []
    if skill_seq is not None:
        for skill in skill_seq:
            # 执行单个技能
            obs, waypoint, stage_success, task_success = skill(env)

            # 收集数据
            observations.extend(obs)
            waypoints.extend(waypoint)

            # 检查执行状态
            if args.early_stop and not stage_success:
                logger.warning(f"{skill} failed, early quit...")
                break

            if task_success:
                break

    # 6. 录制视频（可选）
    if args.record_video:
        frames = []
        for o in observations:
            # 将四个视角拼接成2x2网格
            frames.append(np.vstack([
                np.hstack(o["rgb"][:2]),
                np.hstack(o["rgb"][2:4])
            ]))
        mediapy.write_video(
            os.path.join(task_dir, f"demo_{index}_success_{task_success}.mp4"),
            frames, fps=10
        )

    # 7. 检查任务是否成功
    if not task_success:
        logger.warning("Task failed, skip saving data")
        return
    else:
        logger.info("Task success, saving data")

    # 8. 处理并保存数据
    data_to_save = process_observations(observations)

    # 转换机器人轨迹到机器人坐标系
    robot_position = env.robot.robot_config["position"]
    robot_frame_waypoints = [
        np.array(waypoint) - np.concatenate([robot_position, np.zeros(5)])
        for waypoint in waypoints
    ]

    # 组装数据
    data_to_save["trajectory"] = robot_frame_waypoints
    data_to_save["entities"] = meta_info["entities"]
    data_to_save["target_entity"] = meta_info["target_entity"]
    data_to_save["episode_config"] = json.dumps(episode_config)
    data_to_save["instruction"] = meta_info["instruction"]

    # 保存为HDF5文件
    save_single_data(data_to_save,
                     save_dir=task_dir,
                     filename=f"data_{index}.hdf5")

    env.close()
```

### 4.4 数据流转示意图

```
任务类.get_expert_skill_sequence(physics)
    ↓ 返回
[Skill1, Skill2, Skill3, ...]
    ↓ 遍历执行
for skill in skill_seq:
    ↓ 执行
    obs, waypoint, success, task_success = skill(env)
    ↓ 收集
    observations.extend(obs)
    waypoints.extend(waypoint)
    ↓ 保存
    save_single_data(data_to_save, ...)
```

---

## 阶段五：Mujoco物理模拟执行

### 5.1 环境类

**文件位置**：`/ssd/mkqin/workspace/VLABench/VLABench/envs/dm_env.py`

**关键代码**：
```python
class LM4ManipDMEnv(composer.Environment):
    def __init__(self, task, robot, obs_type, obs_camera_ids):
        super().__init__(task=task,
                         time_limit=float('inf'),
                         random_state=np.random.RandomState(42))
        self.robot = robot
        self.task = task
```

### 5.2 技能执行过程

每个技能内部会调用：

```python
# 伪代码示例
def pick(env, target_entity_name, prior_eulers):
    # 1. 规划运动路径（使用RRT算法）
    path = plan_path(current_pos, target_pos)

    # 2. 执行路径
    for waypoint in path:
        action = compute_action(waypoint)
        obs = env.step(action)  # Mujoco物理模拟
        observations.append(obs)

    # 3. 抓取动作
    env.gripper.close()

    # 4. 检查是否成功抓取
    success = check_grasped(env.physics, entity)

    return observations, waypoints, success, task_success
```

### 5.3 物理模拟步骤

```
env.step(action)
    ↓
physics.step()  # Mujoco物理引擎计算
    ↓
- 更新关节位置
- 计算碰撞
- 更新物体位置
- 计算接触力
    ↓
返回新的observation（图像、状态等）
```

---

## 阶段六：数据收集与保存

### 6.1 HDF5训练数据结构

**保存位置**：`{save_dir}/{task_name}/data_{index}.hdf5`

**数据内容**：
```python
{
    "rgb": [N, H, W, 3],              # RGB图像序列
    "depth": [N, H, W],               # 深度图序列
    "robot_state": [N, D],            # 机器人状态
    "trajectory": [N, 7],             # 机器人轨迹（位置+姿态）
    "entities": List[str],            # 场景中的实体列表
    "target_entity": List[str],       # 目标实体
    "instruction": List[str],         # 任务指令
    "episode_config": str,            # 环境配置JSON
}
```

### 6.2 VLM评测数据结构

**保存位置**：`dataset/vlm_evaluation_v1.0/{Dimension}/{TaskName}/example{N}/`

**目录结构**：
```
example0/
├── env_config/
│   └── env_config.json          # 环境配置
├── input/
│   ├── four_view.png            # 四视角拼接图
│   ├── instruction.json         # 任务指令
│   └── visual_prompt_*.png      # 分割可视化图
└── output/
    └── operation_sequence.json  # 标准答案技能序列
```

---

## 阶段七：标准答案生成

### 7.1 从执行数据提取技能序列

**关键逻辑**：

执行过程中，每个技能的名称和参数会被记录，最终生成简化的JSON格式：

**示例1：Select Poker任务**

```json
{
    "skill_sequence": [
        {
            "name": "pick",
            "params": {
                "target_entity_name": 7
            }
        },
        {
            "name": "lift",
            "params": {}
        }
    ]
}
```

**示例2：Add Condiment任务**

```json
{
    "skill_sequence": [
        {
            "name": "pick",
            "params": {
                "target_entity_name": 3
            }
        },
        {
            "name": "lift",
            "params": {}
        },
        {
            "name": "moveto",
            "params": {}
        },
        {
            "name": "pour",
            "params": {
                "target_container_name": 2
            }
        }
    ]
}
```

**示例3：Take Chemistry Experiment（复杂任务）**

```json
{
    "skill_sequence": [
        {"name": "pick", "params": {"target_entity_name": 3}},
        {"name": "pour", "params": {"target_container_name": 1}},
        {"name": "place", "params": {"target_container_name": 0}},
        {"name": "pick", "params": {"target_entity_name": 5}},
        {"name": "pour", "params": {"target_container_name": 1}},
        {"name": "place", "params": {"target_container_name": 0}},
        {"name": "pick", "params": {"target_entity_name": 5}},
        {"name": "pour", "params": {"target_container_name": 1}},
        {"name": "place", "params": {"target_container_name": 0}}
    ]
}
```

### 7.2 参数映射规则

- `target_entity_name`: 映射到 `env_config.json` 中 `components` 列表的索引
- `target_container_name`: 映射到 `env_config.json` 中容器的索引
- 其他参数：通常省略或简化

### 7.3 环境配置文件

**文件**：`env_config/env_config.json`

**示例结构**：
```json
{
    "scene": "tabletop",
    "robot": {
        "name": "franka_emika_panda",
        "position": [0.0, 0.0, 0.0]
    },
    "components": [
        {
            "type": "entity",
            "name": "apple_0",
            "position": [0.2, 0.1, 0.05],
            "class_id": 0
        },
        {
            "type": "container",
            "name": "bowl_0",
            "position": [0.3, -0.2, 0.05],
            "class_id": 1
        }
    ],
    "conditions": [
        {
            "type": "ContainCondition",
            "entity": "apple_0",
            "container": "bowl_0"
        }
    ],
    "instruction": "Put the red apple into the blue bowl"
}
```

---

## 涉及文件总览

### 核心代码文件

| 文件路径 | 功能 | 关键内容 |
|---------|------|---------|
| `VLABench/tasks/dm_task.py` | 任务基类 | `get_expert_skill_sequence()` 基类定义 |
| `VLABench/tasks/hierarchical_tasks/primitive/*.py` | 原始任务定义 | 各任务的专家技能序列实现 |
| `VLABench/tasks/hierarchical_tasks/composite/*.py` | 复合任务定义 | 复杂任务的技能序列生成逻辑 |
| `VLABench/utils/skill_lib.py` | 技能库 | 所有可用技能的实现 |
| `VLABench/envs/dm_env.py` | Mujoco环境包装 | 环境接口和执行逻辑 |
| `scripts/trajectory_generation.py` | 轨迹生成脚本 | 数据生成主流程 |

### 数据文件

| 文件路径 | 类型 | 内容 |
|---------|------|------|
| `{save_dir}/{task_name}/data_{N}.hdf5` | HDF5 | VLA模型训练数据 |
| `dataset/vlm_evaluation_v1.0/{Dim}/{Task}/example{N}/env_config/env_config.json` | JSON | 环境配置 |
| `dataset/vlm_evaluation_v1.0/{Dim}/{Task}/example{N}/input/instruction.json` | JSON | 任务指令 |
| `dataset/vlm_evaluation_v1.0/{Dim}/{Task}/example{N}/input/four_view.png` | PNG | 输入图像 |
| `dataset/vlm_evaluation_v1.0/{Dim}/{Task}/example{N}/output/operation_sequence.json` | JSON | 标准答案技能序列 |

### 配置文件

| 文件路径 | 功能 |
|---------|------|
| `VLABench/configs/evaluation/dim2task.json` | 评测维度到任务的映射 |
| `VLABench/configs/tasks/*.json` | 任务配置参数 |

---

## 实战示例：为新任务添加技能序列

### 场景：添加"Sort Fruit by Color"任务

#### 步骤1：创建任务类

**文件**：`VLABench/tasks/hierarchical_tasks/primitive/sort_fruit_series.py`

```python
from functools import partial
import numpy as np
from VLABench.tasks.dm_task import DMTask
from VLABench.utils.skill_lib import SkillLib

@register.add_task("sort_fruit_by_color")
class SortFruitByColorTask(DMTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        """
        任务：将红色水果放入红色碗，绿色水果放入绿色碗
        """
        red_fruits = self.config_manager.entities_to_load["red_fruits"]
        green_fruits = self.config_manager.entities_to_load["green_fruits"]
        red_bowl = self.config_manager.container_to_load["red_bowl"]
        green_bowl = self.config_manager.container_to_load["green_bowl"]

        skill_sequence = []

        # 处理红色水果
        for fruit in red_fruits:
            skill_sequence.extend([
                partial(SkillLib.pick,
                        target_entity_name=fruit,
                        prior_eulers=[[-np.pi, 0, 0]]),
                partial(SkillLib.lift,
                        gripper_state=np.zeros(2)),
                partial(SkillLib.place,
                        target_container_name=red_bowl)
            ])

        # 处理绿色水果
        for fruit in green_fruits:
            skill_sequence.extend([
                partial(SkillLib.pick,
                        target_entity_name=fruit,
                        prior_eulers=[[-np.pi, 0, 0]]),
                partial(SkillLib.lift,
                        gripper_state=np.zeros(2)),
                partial(SkillLib.place,
                        target_container_name=green_bowl)
            ])

        return skill_sequence
```

#### 步骤2：生成数据

```bash
python scripts/trajectory_generation.py \
    --task_name sort_fruit_by_color \
    --n_sample 100 \
    --save_dir ./dataset/vlm_evaluation_v1.0/Semantic/sort_fruit_by_color \
    --record_video \
    --early_stop
```

#### 步骤3：检查生成结果

查看生成的标准答案：
```bash
cat dataset/vlm_evaluation_v1.0/Semantic/sort_fruit_by_color/example0/output/operation_sequence.json
```

预期输出：
```json
{
    "skill_sequence": [
        {"name": "pick", "params": {"target_entity_name": 0}},
        {"name": "lift", "params": {}},
        {"name": "place", "params": {"target_container_name": 4}},
        {"name": "pick", "params": {"target_entity_name": 1}},
        {"name": "lift", "params": {}},
        {"name": "place", "params": {"target_container_name": 4}},
        {"name": "pick", "params": {"target_entity_name": 2}},
        {"name": "lift", "params": {}},
        {"name": "place", "params": {"target_container_name": 5}},
        {"name": "pick", "params": {"target_entity_name": 3}},
        {"name": "lift", "params": {}},
        {"name": "place", "params": {"target_container_name": 5}}
    ]
}
```

---

## 常见问题与解决方案

### Q1: 技能序列执行失败（success rate < 100%）

**原因**：
- 物理碰撞导致物体飞出
- 抓取角度不合适
- 运动规划失败

**解决方案**：
1. 调整 `prior_eulers` 参数
2. 增加 `lift_height` 避免碰撞
3. 检查物体初始位置是否合理
4. 使用 `--early_stop` 参数跳过失败样本

### Q2: 生成的技能序列与预期不符

**原因**：
- `get_expert_skill_sequence()` 实现有误
- 实体/容器名称映射错误

**解决方案**：
1. 在 `get_expert_skill_sequence()` 中添加日志输出
2. 检查 `config_manager` 中的实体配置
3. 使用 `--debug` 模式查看详细信息

### Q3: 如何调试技能执行过程

**方法**：
1. 使用 `--record_video` 参数生成可视化视频
2. 在技能库中添加 `print()` 语句
3. 查看Mujoco GUI实时渲染（设置 `render=True`）

---

## 评测使用

生成的标准答案用于VLM评测：

**评测脚本**：`VLABench/evaluation/evaluator/vlm.py`

**评测流程**：
```python
# 1. 加载标准答案
with open("output/operation_sequence.json") as f:
    ground_truth = json.load(f)

# 2. VLM生成预测
vlm_output = vlm_model.predict(image, instruction)

# 3. 对比评分
score = get_final_score(
    ground_truth["skill_sequence"],
    vlm_output["skill_sequence"],
    dependency=task_dependency
)
```

**评分维度**（详见 `VLABench/evaluation/utils.py`）：
- `skill_match_score` (40%): 技能名称匹配
- `entity_match_score` (40%): 实体识别准确度
- `skill_with_entity_match_score` (10%): 技能+实体联合匹配
- `exact_match_score` (10%): 精确匹配（考虑依赖关系）

---

## 总结

VLABench的技能序列生成是一个**代码定义 + 物理验证 + 数据收集**的三阶段流程：

1. **代码定义**：程序员在任务类中手写专家策略
2. **物理验证**：在Mujoco中实际执行，确保可行性
3. **数据收集**：生成训练数据和评测标准答案

这种设计既保证了数据的物理合理性，又避免了大规模人工标注的成本。

---

## 参考资源

- 论文：https://arxiv.org/abs/2412.18194
- 项目主页：https://vlabench.github.io/
- GitHub：https://github.com/OpenMOSS/VLABench
- 数据集：https://huggingface.co/datasets/VLABench/vlm_evaluation_v1.0

---

**文档版本**：v1.0
**最后更新**：2026-03-10
**作者**：VLABench Team
