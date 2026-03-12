# VLABench 操作序列扩展指南

## 📋 概述

本文档详细说明如何为 VLABench 添加新的机器人操作（skill），包括修改哪些文件、如何实现、以及测试方法。

---

## 🎯 快速开始

### 添加新操作的三个步骤

1. **在 `skill_lib.py` 中实现操作函数**
2. **在 `task_creator.py` 中更新 LLM prompt**
3. **（可选）在 `task_creator.py` 中添加评测条件**

---

## 📂 涉及的文件

| 文件路径 | 作用 | 是否必须修改 |
|---------|------|------------|
| `VLABench/utils/skill_lib.py` | 实现具体的操作逻辑 | ✅ 必须 |
| `scripts/vlabench_agent/nodes/task_creator.py` | 更新 Agent 的操作列表和 prompt | ✅ 必须 |
| `scripts/vlabench_agent/nodes/task_creator.py` | 添加评测条件生成逻辑 | ⚠️ 可选 |

---

## 🔧 详细步骤

### 步骤 1: 在 skill_lib.py 中实现操作

**文件位置**: `VLABench/utils/skill_lib.py`

#### 1.1 理解现有操作的结构

所有操作都是 `SkillLib` 类的静态方法，遵循统一的函数签名：

```python
@staticmethod
def operation_name(env, param1, param2, ..., **kwargs):
    """
    操作描述

    Args:
        env: LM4manipEnv object (必需)
        param1: 参数描述
        param2: 参数描述
        ...

    Returns:
        observations: list of observations
        waypoints: list of waypoints
        stage_success: bool, 该阶段是否成功
        task_success: bool, 任务是否完成
    """
    # 实现逻辑
    pass
```

#### 1.2 核心 API 说明

操作实现中常用的 API：

| API | 说明 | 示例 |
|-----|------|------|
| `env.robot.get_qpos(env.physics)` | 获取机械臂关节角度 (7个关节) | `qpos = np.array(env.robot.get_qpos(env.physics))` |
| `env.robot.get_end_effector_pos(env.physics)` | 获取末端执行器位置 (x, y, z) | `pos = env.robot.get_end_effector_pos(env.physics)` |
| `env.robot.get_end_effector_quat(env.physics)` | 获取末端执行器姿态 (四元数) | `quat = env.robot.get_end_effector_quat(env.physics)` |
| `env.robot.get_ee_open_state(env.physics)` | 获取夹爪状态 (True=闭合) | `closed = env.robot.get_ee_open_state(env.physics)` |
| `env.step(action)` | 执行一个动作步 | `timestep = env.step(action)` |
| `env.get_observation()` | 获取当前观测 | `obs = env.get_observation()` |

#### 1.3 实现模板

```python
@staticmethod
def your_new_operation(env, param1=default_value, gripper_state=None, max_n_substep=30, tolerance=0.01):
    """
    你的新操作的描述

    Args:
        env: LM4manipEnv object
        param1: 参数1的描述
        gripper_state: 夹爪状态 (None 表示自动检测)
        max_n_substep: 每个时间步的最大子步数
        tolerance: 关节位置误差容忍度

    Returns:
        observations: 观测列表
        waypoints: 路径点列表
        stage_success: 操作是否成功
        task_success: 任务是否完成
    """
    # 1. 初始化
    qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
    observations = [env.get_observation()]
    waypoints = []
    task_success = False
    stage_success = False

    # 2. 确定夹爪状态
    if gripper_state is None:
        gripper_closed = env.robot.get_ee_open_state(env.physics)
        if gripper_closed:
            gripper_state = np.zeros(2)
        else:
            gripper_state = np.ones(2) * 0.04

    # 3. 执行操作逻辑
    # ... 你的具体实现 ...

    # 4. 返回结果
    observations.pop(-1)
    assert len(observations) == len(waypoints), \
        f"observations and waypoints should have the same length"

    return observations, waypoints, stage_success, task_success
```

#### 1.4 实战示例：rotate 操作

参考 `skill_lib.py:654-722` 中的 `rotate` 操作实现：

```python
@staticmethod
def rotate(env, rotation_angle=np.pi/2, gripper_state=None,
           target_q_velocity=np.pi/40, max_n_substep=30, tolerance=0.01):
    """
    通过旋转腕关节来旋转抓取的物体
    """
    qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
    observations = [env.get_observation()]
    waypoints = []

    # 自动检测夹爪状态
    if gripper_state is None:
        gripper_closed = env.robot.get_ee_open_state(env.physics)
        if gripper_closed: gripper_state = np.zeros(2)
        else: gripper_state = np.ones(2) * 0.04

    # 计算旋转步数
    timesteps = int(abs(rotation_angle) / target_q_velocity)
    task_success = False
    stage_success = False

    # 逐步旋转
    for i in range(timesteps):
        action = np.array(qpos).copy()
        # 旋转最后一个关节（腕关节）
        action[-1] += target_q_velocity * (i + 1) * np.sign(rotation_angle)
        action = np.concatenate([action, gripper_state])

        # 执行动作
        for _ in range(max_n_substep):
            timestep = env.step(action)
            if timestep.last():
                task_success = True
                break
            current_qpos = np.array(env.task.robot.get_qpos(env.physics)).reshape(-1)
            if np.max(current_qpos - action[:7]) < tolerance \
                and np.min(current_qpos - action[:7]) > -tolerance:
                break

        if task_success:
            break

        # 记录轨迹
        waypoint = np.concatenate([
            env.robot.get_end_effector_pos(env.physics),
            quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
            gripper_state
        ])
        obs = env.get_observation()
        observations.append(obs)
        waypoints.append(waypoint)

    observations.pop(-1)

    # 检查旋转是否成功
    final_qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
    angle_diff = abs(final_qpos[-1] - (qpos[-1] + rotation_angle))
    if angle_diff < tolerance * 10:
        stage_success = True

    return observations, waypoints, stage_success, task_success
```

**关键点**：
- ✅ 使用 `qpos[-1]` 控制腕关节（最后一个关节）
- ✅ 支持正负角度（正=逆时针，负=顺时针）
- ✅ 渐进式旋转，避免突然运动
- ✅ 记录完整的观测和轨迹

---

### 步骤 2: 更新 task_creator.py 的 LLM Prompt

**文件位置**: `scripts/vlabench_agent/nodes/task_creator.py`

#### 2.1 找到操作序列生成的 prompt

定位到 `task_creator_node` 函数中的 `operation_prompt` (约第 144 行)。

#### 2.2 添加新操作到支持列表

在 prompt 的 "重要规则" 部分添加新操作：

```python
operation_prompt = f"""你是机器人操纵任务专家。请根据任务生成操作序列。

任务信息:
- 任务指令: {task_analysis.get('instruction_en')}
- 物体列表: {[o['name'] for o in objects]}
- 操作类型: {task_analysis.get('operation_type')}

请生成 VLABench 操作序列,返回 JSON 格式:
[
    {{"name": "pick", "params": {{"target_entity_name": <物体索引>}}}},
    {{"name": "操作名称", "params": {{...}}}}
]

重要规则:
1. 物体索引从 1 开始 (0 是桌子)
2. 支持的操作名称: pick, place, pour, lift, rotate, flip, push, pull, your_new_op
3. pick 操作: {{"name": "pick", "params": {{"target_entity_name": 索引}}}}
4. place 操作: {{"name": "place", "params": {{"target_container_name": 索引}}}}
...
N. your_new_op 操作: {{"name": "your_new_op", "params": {{"param1": 值}}}}

特殊说明:
- your_new_op 用于 [具体说明使用场景]
- 常用参数值: [列举常用值]

只返回 JSON 数组,不要其他文字。"""
```

#### 2.3 示例：添加 rotate 操作

参考 `task_creator.py:159-172`：

```python
重要规则:
1. 物体索引从 1 开始 (0 是桌子)
2. 支持的操作名称: pick, place, pour, lift, rotate, flip, push, pull
3. pick 操作: {"name": "pick", "params": {"target_entity_name": 索引}}
4. place 操作: {"name": "place", "params": {"target_container_name": 索引}}
5. pour 操作: {"name": "pour", "params": {"target_container_name": 索引}}
6. lift 操作: {"name": "lift", "params": {"target_height": 高度}}
7. rotate 操作: {"name": "rotate", "params": {"rotation_angle": 角度}}  # 弧度, 默认1.57(90度)
8. flip 操作: {"name": "flip", "params": {}}  # 翻转180度
9. push 操作: {"name": "push", "params": {"push_distance": 距离}}
10. pull 操作: {"name": "pull", "params": {"pull_distance": 距离}}

特殊说明:
- rotate 操作用于旋转抓取的物体,需要先 pick 物体
- 常用角度: 1.57 (90度), 3.14 (180度), 0.785 (45度)
- 对于 "旋转烧杯" 类任务,序列应为: pick -> rotate
```

---

### 步骤 3: （可选）添加评测条件

**文件位置**: `scripts/vlabench_agent/nodes/task_creator.py`

#### 3.1 找到评测条件生成逻辑

定位到 `task_creator_node` 函数中的 "生成评测条件" 部分 (约第 195 行)。

#### 3.2 添加新操作的评测条件

```python
# 3. 生成评测条件
conditions = {}
operation_type = task_analysis.get('operation_type', 'pick')

if operation_type == 'pour':
    conditions = {
        "pour": {
            "target_entity": objects[0]["name"],
            "threshold": 0
        }
    }
elif operation_type == 'place' and len(objects) >= 2:
    conditions = {
        "contain": {
            "container": objects[1]["name"],
            "entities": [objects[0]["name"]]
        }
    }
elif operation_type == 'lift':
    conditions = {
        "lift": {
            "entities": [objects[0]["name"]],
            "target_height": 0.9
        }
    }
elif operation_type == 'your_new_op':
    # 为你的新操作定义评测条件
    conditions = {
        "your_condition_type": {
            "param1": value1,
            "param2": value2
        }
    }

logger.info(f"[Task Creator] ✓ 生成评测条件: {list(conditions.keys()) if conditions else '无特定条件'}")
```

#### 3.3 注意事项

- ⚠️ 如果 VLABench 没有为你的操作定义专门的评测条件，可以留空 `conditions = {}`
- ⚠️ 评测条件主要用于自动化测试，对于 VLM 评测任务不是必需的

---

## ✅ 测试新操作

### 1. 单元测试（推荐）

创建测试脚本测试新操作：

```python
import sys
sys.path.insert(0, '/ssd/mkqin/workspace/VLABench')
import os
os.environ['VLABENCH_ROOT'] = '/ssd/mkqin/workspace/VLABench/VLABench'
os.environ['MUJOCO_GL'] = 'osmesa'

import json
from VLABench.utils.interface import create_env_from_config
from VLABench.utils.skill_lib import SkillLib

# 加载测试环境
with open('path/to/env_config.json', 'r') as f:
    config = json.load(f)

env = create_env_from_config(config)

# 测试你的新操作
observations, waypoints, stage_success, task_success = SkillLib.your_new_operation(
    env,
    param1=value1
)

print(f"Stage success: {stage_success}")
print(f"Task success: {task_success}")
print(f"Observations: {len(observations)}")
print(f"Waypoints: {len(waypoints)}")
```

### 2. Agent 端到端测试

使用 VLABench Agent CLI 测试：

```bash
# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlabench_2

# 运行 Agent
python vlabench_agent_cli.py --instruction "执行你的新操作"
```

### 3. 验证检查项

- ✅ 操作序列生成正确（包含新操作）
- ✅ 场景渲染成功
- ✅ 验证报告显示 PASS
- ✅ 生成的轨迹合理

---

## 📊 常见操作类型

### 按控制方式分类

| 类型 | 说明 | 示例操作 |
|------|------|---------|
| **关节控制** | 直接控制机械臂关节角度 | rotate, flip, pour |
| **位置控制** | 控制末端执行器位置 | pick, place, moveto |
| **相对移动** | 基于当前位置的相对移动 | lift, push, pull, move_offset |
| **轨迹跟随** | 沿预定义轨迹移动 | open_door, open_drawer |
| **夹爪控制** | 控制夹爪开合 | close_gripper, open_gripper |

### 参数设计指南

| 参数类型 | 说明 | 示例 |
|---------|------|------|
| **目标对象** | 物体或容器名称 | `target_entity_name`, `target_container_name` |
| **运动参数** | 距离、角度、高度 | `rotation_angle`, `lift_height`, `push_distance` |
| **速度参数** | 运动速度 | `target_velocity`, `target_q_velocity` |
| **控制参数** | 精度和稳定性 | `tolerance`, `max_n_substep` |

---

## 🎓 实战案例：添加 rotate 操作

### 需求分析

- **目标**：实现旋转抓取物体的操作
- **输入**：旋转角度（弧度）
- **输出**：观测序列和成功状态

### 实现过程

#### 1. 在 skill_lib.py 中添加 rotate 方法

文件：`VLABench/utils/skill_lib.py:654-722`

```python
@staticmethod
def rotate(env, rotation_angle=np.pi/2, gripper_state=None,
           target_q_velocity=np.pi/40, max_n_substep=30, tolerance=0.01):
    # ... (完整实现见上文)
    pass
```

#### 2. 更新 task_creator.py 的 prompt

文件：`scripts/vlabench_agent/nodes/task_creator.py:159-172`

添加到支持列表：
```
7. rotate 操作: {"name": "rotate", "params": {"rotation_angle": 角度}}
```

#### 3. 添加评测条件

文件：`scripts/vlabench_agent/nodes/task_creator.py:220-223`

```python
elif operation_type == 'rotate':
    conditions = {}  # rotate 操作暂无专门的评测条件
```

#### 4. 测试

```bash
python vlabench_agent_cli.py --instruction "旋转烧杯"
```

**结果**：
```json
{
    "skill_sequence": [
        {"name": "pick", "params": {"target_entity_name": 1}},
        {"name": "rotate", "params": {"rotation_angle": 1.57}}
    ]
}
```

✅ 成功！

---

## 🔍 调试技巧

### 1. 查看 Agent 日志

日志保存在：`/ssd/mkqin/workspace/VLABench/logs/任务生成日志/`

关键信息：
- 操作序列生成结果
- LLM 响应内容
- 错误信息

### 2. 检查操作序列

读取生成的文件：
```bash
cat dataset/vlm_evaluation_v1.0/M&T/{task_name}/example0/output/operation_sequence.json
```

### 3. 验证场景配置

读取验证报告：
```bash
cat dataset/vlm_evaluation_v1.0/M&T/{task_name}/example0/input/validation_report.json
```

---

## ⚠️ 注意事项

### 1. 保持向后兼容

- ❌ **不要修改现有操作的函数签名**
- ✅ **只添加新操作，不修改旧操作**
- ✅ **新参数使用默认值**

### 2. 安全性考虑

- ✅ 检查关节角度限制，避免超出机械臂运动范围
- ✅ 使用运动规划避免碰撞
- ✅ 验证目标位置的可达性

### 3. 代码规范

- ✅ 添加详细的 docstring
- ✅ 使用类型注解（如果可能）
- ✅ 遵循现有代码风格
- ✅ 添加必要的 assert 检查

---

## 📚 参考资源

### 相关文档

- [VLABench Agent 使用指南](./VLABench_Agent_User_Guide.md)
- [VLM 数据集渲染指南](./render_vlm_guide.md)
- [自定义 VLM 任务指南](./custom_vlm_task_guide.md)

### 代码参考

- **skill_lib.py**: 查看所有现有操作的实现
- **task_creator.py**: 查看操作序列生成逻辑
- **render_vlm_dataset.py**: 查看渲染和验证逻辑

### 工具函数

`VLABench/utils/utils.py` 中的常用函数：
- `quaternion_to_euler()` - 四元数转欧拉角
- `quaternion_from_axis_angle()` - 从轴角创建四元数
- `quaternion_multiply()` - 四元数乘法
- `distance()` - 计算距离

---

## 💡 最佳实践

### 1. 从简单操作开始

- ✅ 先实现基本功能
- ✅ 测试通过后再添加高级特性
- ✅ 参考类似的现有操作

### 2. 渐进式开发

1. 在 skill_lib.py 中实现基础版本
2. 进行单元测试
3. 更新 task_creator.py
4. 进行端到端测试
5. 优化和完善

### 3. 充分测试

- ✅ 测试各种参数组合
- ✅ 测试边界情况
- ✅ 测试与其他操作的组合
- ✅ 验证生成的轨迹合理性

---

## 🆘 常见问题

### Q1: 操作没有出现在生成的序列中？

**检查**：
1. task_creator.py 的 prompt 是否包含该操作？
2. LLM 是否理解了操作的用途？
3. 查看 Agent 日志中的 LLM 响应

### Q2: 操作执行失败？

**检查**：
1. 参数是否在合理范围内？
2. 是否违反了机械臂的运动学约束？
3. 是否发生了碰撞？

### Q3: 如何调整操作的速度？

修改 `target_velocity` 或 `target_q_velocity` 参数：
- 更小的值 = 更慢、更平滑
- 更大的值 = 更快、可能不稳定

### Q4: 如何处理操作失败？

在操作中添加错误处理：
```python
try:
    # 执行操作
    ...
except Exception as e:
    logger.error(f"操作失败: {e}")
    return observations, waypoints, False, False
```

---

## 📝 总结

添加新操作的核心步骤：

1. ✅ **实现操作逻辑** - 在 `skill_lib.py` 中添加静态方法
2. ✅ **更新 Agent prompt** - 在 `task_creator.py` 中更新 LLM prompt
3. ✅ **添加评测条件** - （可选）在 `task_creator.py` 中添加评测逻辑
4. ✅ **测试验证** - 使用 Agent CLI 进行端到端测试

记住：
- 📖 参考现有操作的实现
- 🔧 保持代码简洁和可维护
- 🧪 充分测试各种场景
- 📚 编写清晰的文档

---

**更新日期**: 2026-03-06
**版本**: 1.0
**维护者**: VLABench Team
