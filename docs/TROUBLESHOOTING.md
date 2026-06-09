# VLABench 常见错误与解决方案

本文档记录 VLABench 项目开发过程中遇到的常见错误、根本原因及解决方案，便于团队成员快速定位和解决类似问题。

---

## 目录

1. [场景与机器人位置不匹配导致物体距离过远](#1-场景与机器人位置不匹配导致物体距离过远)
2. [OBJ模型几何中心偏移导致视觉与物理不匹配](#2-obj模型几何中心偏移导致视觉与物理不匹配)
3. [LiftCondition 在 record_initial_state 未调用时误判为已满足](#3-liftcondition-在-record_initial_state-未调用时误判为已满足)
4. [place/drop 技能被 should_terminate 提前终止导致仿真中断](#4-placedrop-技能被-should_terminate-提前终止导致仿真中断)
5. [CapOpenCondition 静平衡形变误触导致 load_env 无限 reset 死循环](#5-capopencondition-静平衡形变误触导致-load_env-无限-reset-死循环)

---

## 1. 场景与机器人位置不匹配导致物体距离过远

**日期**: 2026-03-11
**影响任务**: `pick_petri_dish`, `flip_petri_dish`
**严重程度**: 🔴 高（任务完全无法执行）

### 问题表现

在为 `pick_petri_dish` 任务生成轨迹时出现以下异常：

1. **视觉异常**
   - 渲染视频中机械臂和培养皿都不在视野中
   - 机械臂直接放置在地面上，培养皿消失

2. **位置偏差巨大**
   ```
   DEBUG [pick]: 目标实体 = petri_dish_0
   实体 worldbody 位置: [0.38770943, 8.43650479, 0.09843795]  // Y=8.4米
   机械臂起始位置: [0.23380471, 1.97789653, 1.24297870]       // Y=1.9米
   XY误差: 6.24 米
   位置误差: 6.28 米
   ```

3. **物理仿真问题**
   - 大量 `WARNING: Failed to converge after 99 steps: err_norm=X.XX`
   - 误差持续增长（从 0.5 增长到 6.0+）

4. **任务失败**
   ```
   DEBUG [pick]: ✗ 抓取失败! is_grasped=False
   WARNING: Task failed, skip saving data
   ```

### 根本原因

#### 1. 场景偏移机制

VLABench 中场景配置了候选位置偏移（`VLABench/configs/scene_config.json`）：

```json
"lab_0": {
    "xml_path": "lab_0/lab.xml",
    "candidate_pos": [[0, 2.4, 0], [2, 2.4, 0]]  // 场景会随机偏移
}
```

场景初始化时会随机选择一个 `candidate_pos`，导致整个场景在世界坐标系中平移。

#### 2. 实体独立性原理

关键代码（`VLABench/tasks/dm_task.py`）：

```python
self._arena.attach(scene)      # 场景
self._arena.attach(robot)      # 机器人
self._arena.attach(entity)     # 桌子、物体等
```

**重要**：场景的平移只影响场景内部的视觉元素（实验室装饰等），不会自动带动其他独立附加的实体（机器人、桌子、物体）。

#### 3. 错误的配置方式

问题任务在 `task_config.json` 中显式设置了 robot position：

```json
"pick_petri_dish": {
    "robot": {
        "position": [0, -0.4, 0.6]  // ❌ 显式配置覆盖了默认对齐机制
    },
    "task": {
        "scene": {
            "name": "lab_0"  // 场景会偏移到 Y=2.4米
        }
    }
}
```

导致：
- 场景偏移到 `[0, 2.4, 0]`
- 机器人固定在 `[0, -0.4, 0.6]`
- 培养皿相对场景放置，实际位置 `Y ≈ 2.4 + offset`
- **结果：机器人与物体相距数米，完全超出工作范围**

#### 4. 对比：正确的配置

检查所有使用实验室场景的任务发现：

| 任务名称 | 场景 | Robot Position | 状态 |
|---------|------|----------------|------|
| `select_chemistry_tube_series` | lab_1 | `None` ✅ | 正常 |
| `physical_qa_series` | lab_1 | `None` ✅ | 正常 |
| `seesaw_series` | lab_1 | `None` ✅ | 正常 |
| `pick_petri_dish` | lab_0 | `[0, -0.4, 0.6]` ❌ | **失败** |
| `flip_petri_dish` | lab_0 | `[0, -0.4, 0.6]` ❌ | **失败** |

**结论**：所有正常工作的任务都没有显式设置 robot position。

### 解决方案

**删除 `task_config.json` 中的 robot position 配置**，让机器人使用默认位置并自动与场景对齐。

#### 具体修改

**文件**: `VLABench/configs/task_config.json`

```diff
  "pick_petri_dish": {
-     "robot": {
-         "position": [0, -0.4, 0.6]
-     },
      "task": {
          "asset": {
              "seen_object": ["petri_dish_0"],
              ...
          },
          "components": [...],
          "scene": {
              "name": "lab_0"
          }
      }
  }
```

同样修改 `flip_petri_dish` 任务。

#### 修复效果

| 指标 | 修改前 | 修改后 | 改善 |
|------|--------|--------|------|
| 培养皿 Y 坐标 | 8.43 米 | 0.055 米 | ✅ |
| 机器人 Y 坐标 | 1.98 米 | -0.173 米 | ✅ |
| XY 位置误差 | 6.24 米 | **0.034 米** | ✅ 99.5% |
| Z 位置误差 | 0.65 米 | **0.091 米** | ✅ 86% |
| 总位置误差 | 6.28 米 | **0.098 米** | ✅ 98.4% |
| 物理收敛 | 大量失败 | 偶尔警告 | ✅ |

### 技术细节

#### 场景偏移实现

`VLABench/tasks/components/scene.py`:

```python
class Scene(composer.Entity):
    def initialize_episode(self, physics, random_state):
        if self.candidate_pos is not None:
            new_xpos = random.choice(self.candidate_pos)  # 随机选择偏移
        else:
            new_xpos = self.init_pos
        self.set_pose(physics, new_xpos, new_xquat)
```

#### Robot 配置加载

`VLABench/envs/__init__.py`:

```python
def load_env(task, robot="franka", config=None, **kwargs):
    robot_config = ROBOT_CONFIG.get(robot, None)

    # ⚠️ 任务配置会覆盖默认配置
    robot_config_overide = default_config.get("robot", {})
    robot_config.update(robot_config_overide)

    robot = register.load_robot(robot)(**robot_config)
```

如果 `task_config.json` 中设置了 `robot.position`，会覆盖 `robot_config.json` 中的默认值。

### 预防措施

#### ✅ DO - 推荐做法

1. **使用场景时不设置 robot position**
   ```json
   "your_task": {
       "task": {
           "scene": {"name": "lab_0"}
       }
   }
   ```

2. **参考现有成功任务的配置**
   - 查看 `select_chemistry_tube_series`
   - 查看 `physical_qa_series`
   - 查看其他使用相同场景的任务

3. **物体使用相对坐标**
   ```python
   position=[
       random.uniform(-0.1, 0.1),  # 相对原点
       random.uniform(-0.1, 0.1),
       0.761
   ]
   ```

#### ❌ DON'T - 避免做法

1. **不要在有偏移场景中显式设置 robot position**
   ```json
   "your_task": {
       "robot": {
           "position": [0, -0.4, 0.6]  // ❌
       },
       "task": {
           "scene": {"name": "lab_0"}
       }
   }
   ```

2. **不要修改 `scene_config.json` 的 `candidate_pos`**
   - 会影响所有使用该场景的任务
   - 其他任务可能依赖现有配置

### 调试方法

#### 1. 快速诊断位置问题

查看轨迹生成日志中的 DEBUG 信息：

```bash
grep "DEBUG \[pick\]" generation.log
```

正常情况：
```
实体 worldbody 位置: [-0.137, 0.055, 0.677]   // ✅ Y值合理
机械臂起始位置: [0.0005, -0.173, 1.20]         // ✅ Y值合理
```

异常情况：
```
实体 worldbody 位置: [0.38, 8.43, 0.09]       // ❌ Y=8.4米异常
机械臂起始位置: [0.23, 1.97, 1.24]             // ❌ 距离过远
```

#### 2. 检查任务配置

```python
import json
with open('VLABench/configs/task_config.json') as f:
    config = json.load(f)
    task_name = 'your_task'

    robot_pos = config.get(task_name, {}).get('robot', {}).get('position')
    scene = config.get(task_name, {}).get('task', {}).get('scene', {}).get('name')

    print(f'Robot position: {robot_pos}')    # 应为 None
    print(f'Scene: {scene}')

    # 检查场景偏移
    with open('VLABench/configs/scene_config.json') as sf:
        scene_config = json.load(sf)
        print(f'Candidate positions: {scene_config[scene]["candidate_pos"]}')
```

#### 3. 对比参考任务

```bash
# 查找所有使用 lab_0 或 lab_1 的任务
python3 -c "
import json
with open('VLABench/configs/task_config.json') as f:
    config = json.load(f)
    for task, conf in config.items():
        scene = conf.get('task', {}).get('scene', {}).get('name')
        if scene in ['lab_0', 'lab_1']:
            robot_pos = conf.get('robot', {}).get('position')
            print(f'{task:35s} robot_pos={robot_pos}')
"
```

### 相关文件

- `VLABench/configs/task_config.json` - 任务配置
- `VLABench/configs/scene_config.json` - 场景配置
- `VLABench/configs/robot_config.json` - 机器人默认配置
- `VLABench/tasks/components/scene.py` - 场景加载逻辑
- `VLABench/envs/__init__.py` - 环境初始化
- `VLABench/tasks/dm_task.py` - 任务基类

### 关键经验

> **原则**: 使用带有 `candidate_pos` 偏移的场景时，不要在 `task_config.json` 中显式设置 robot position。

这个问题的本质是**场景偏移机制**与**显式机器人位置配置**的冲突。系统设计了默认的场景对齐机制，但显式配置会覆盖这个机制，导致位置不匹配。

---

## 贡献指南

遇到新问题时，请按以下格式添加到此文档：

```markdown
## N. 问题标题

**日期**: YYYY-MM-DD
**影响任务/组件**: 具体任务或组件名称
**严重程度**: 🔴 高 / 🟡 中 / 🟢 低

### 问题表现
[描述问题的具体现象]

### 根本原因
[解释问题产生的技术原因]

### 解决方案
[提供具体的解决步骤]

### 预防措施
[提供最佳实践和避免事项]

### 调试方法
[提供调试和诊断的方法]
```

---

## 2. OBJ模型几何中心偏移导致视觉与物理不匹配

**日期**: 2026-03-12
**影响任务**: `pick_petri_dish`, `flip_petri_dish`
**严重程度**: 🟡 中（任务可以执行但视觉效果异常）

### 问题表现

在 `pick_petri_dish` 任务修复物理稳定性问题后，出现了"幽灵抓取"(Ghost Grasp)现象：

1. **视觉异常**
   - 机械臂夹爪在培养皿**侧方10-15厘米处**进行了"空抓"动作
   - 培养皿看起来没有被物理接触
   - 但抓取判定成功，培养皿随机械臂抬起

2. **物理判定正常**
   ```
   DEBUG [pick]: ✓ 抓取成功! stage_success=True
   当前末端位置: [-0.04441981  0.03168596  0.77616179]
   目标位置: [-0.04480204  0.03121822  0.76999022]
   XY误差: 0.0006m, Z误差: 0.006m
   ```

3. **视觉-物理分离**
   - 物理仿真：夹爪确实接触到了碰撞几何体
   - 视觉渲染：夹爪位置距离视觉网格很远
   - 抓取点坐标正确，但视觉模型显示在错误位置

### 根本原因

#### 1. OBJ 模型几何中心偏移

培养皿的 OBJ 文件（`petri_dish.obj`）的几何中心不在原点：

```bash
# 检查原始 OBJ 文件的顶点范围
grep "^v " petri_dish_original.obj | awk '{
    if(NR==1){minx=$2;maxx=$2}
    if($2<minx)minx=$2;
    if($2>maxx)maxx=$2
} END{
    print "X range:", minx, "to", maxx, "| Center:", (minx+maxx)/2
}'
# 输出: X range: -0.195 to -0.105 | Center: -0.15
```

**关键发现**: 视觉网格的X轴中心在 **-0.15米** 处，而不是原点！

#### 2. MuJoCo 的 Body 坐标系机制

在 MuJoCo 中，一个 `<body>` 元素可以包含多个 `<geom>`：

```xml
<body name="petri_dish_0">
  <!-- 惯性（质心） -->
  <inertial pos="0 0 0.0075" mass="0.015" diaginertia="0.0005 0.0005 0.0005"/>

  <!-- 视觉网格（用于渲染） -->
  <geom material="Scratched_Glass" mesh="petri_dish" class="visual"/>

  <!-- 碰撞几何（用于物理） -->
  <geom type="cylinder" size="0.045 0.0075" pos="0 0 0.0075" ... />

  <!-- 抓取点 -->
  <site class="grasppoint" pos="0 0 0.01"/>
</body>
```

所有 `pos` 属性都是**相对于 body 原点**的局部坐标。

#### 3. 物理稳定性约束

为了物理仿真稳定，**惯性中心必须与碰撞几何中心对齐**：

```xml
<!-- ✅ 正确：惯性与碰撞中心都在 (0, 0, 0.0075) -->
<inertial pos="0 0 0.0075" .../>
<geom type="cylinder" pos="0 0 0.0075" .../>
```

如果尝试移动碰撞几何去匹配视觉网格：

```xml
<!-- ❌ 错误：碰撞在 X=-0.15，但惯性在 X=0，导致物理不稳定 -->
<inertial pos="0 0 0.0075" .../>
<geom type="cylinder" pos="-0.15 0 0.0075" .../>
```

**结果**: 培养皿会侧立、摇摆、倾斜，无法平稳放置。

#### 4. 不匹配的本质

三者位置关系：

| 组件 | X坐标（body局部坐标系） | 用途 |
|------|------------------------|------|
| 惯性中心 | `0` | 质心，影响力矩平衡 |
| 碰撞圆柱 | `0` | 物理接触检测 |
| **视觉网格** | **`-0.15`** | **仅用于渲染** |
| 抓取点 | `0` | 跟随碰撞几何 |

**问题**: 视觉网格偏移了15厘米，但物理世界中培养皿的实际位置由碰撞几何决定。

#### 5. 为什么不能简单移动碰撞几何

尝试过的失败方案：

```xml
<!-- 尝试1: 移动碰撞几何到 X=-0.15 匹配视觉 -->
<inertial pos="0 0 0.0075" .../>
<geom type="cylinder" pos="-0.15 0 0.0075" .../>
<!-- 结果: 培养皿侧立摇摆，物理不稳定 ❌ -->

<!-- 尝试2: 同时移动惯性和碰撞到 X=-0.15 -->
<inertial pos="-0.15 0 0.0075" .../>
<geom type="cylinder" pos="-0.15 0 0.0075" .../>
<!-- 结果: body整体偏移，桌子位置不对，仍然物理不稳定 ❌ -->
```

**约束条件**:
1. 惯性中心 = 碰撞中心（物理稳定性要求）
2. 碰撞几何与抓取点对齐（抓取成功要求）
3. 视觉网格中心 = 碰撞中心（视觉正确性要求）

前两个是强约束（不满足会失败），第三个可以通过修改 OBJ 文件解决。

### 解决方案

**方案：修改 OBJ 文件，将几何中心平移到原点**

通过平移所有顶点坐标，使视觉网格中心与碰撞几何中心对齐。

#### 具体步骤

**文件**: `VLABench/assets/obj/meshes/lab_equipment/petri_dish/petri_dish_0/petri_dish.obj`

1. **备份原始文件**
   ```bash
   cd VLABench/assets/obj/meshes/lab_equipment/petri_dish/petri_dish_0
   cp petri_dish.obj petri_dish_original.obj
   ```

2. **使用 awk 平移顶点坐标**
   ```bash
   awk '
   BEGIN {offset_x = 0.15}  # 向右平移0.15米，使X中心从-0.15移到0
   {
       if ($1 == "v") {
           printf "v %.8f %.8f %.8f\n", $2 + offset_x, $3, $4
       } else {
           print $0
       }
   }' petri_dish_original.obj > petri_dish.obj
   ```

3. **验证修改结果**
   ```bash
   grep "^v " petri_dish.obj | awk '{
       if(NR==1){minx=$2;maxx=$2}
       if($2<minx)minx=$2;
       if($2>maxx)maxx=$2
   } END{
       print "New X range:", minx, "to", maxx, "| Center:", (minx+maxx)/2
   }'
   # 期望输出: New X range: -0.045 to 0.045 | Center: 0
   ```

#### 修复效果

| 指标 | 修改前 | 修改后 | 说明 |
|------|--------|--------|------|
| 视觉网格 X 中心 | -0.15米 | 0米 | ✅ 对齐原点 |
| 碰撞圆柱 X 中心 | 0米 | 0米 | 保持不变 |
| 惯性 X 中心 | 0米 | 0米 | 保持不变 |
| 抓取点 X 坐标 | 0米 | 0米 | 保持不变 |
| **视觉-物理对齐** | **偏移15cm** | **完全对齐** | ✅ |
| 物理稳定性 | 稳定 | 稳定 | ✅ 无影响 |
| 抓取成功率 | 成功 | 成功 | ✅ 无影响 |
| **视觉效果** | **幽灵抓取** | **正常抓取** | ✅ 修复 |

修复后的轨迹生成日志：

```
DEBUG [pick]: 目标实体 = petri_dish_0
实体 worldbody 位置: [-0.04480194  0.03121821  0.75999022]  # X = -0.045（桌面随机位置）
实体抓取点: [-0.04480204  0.03121822  0.76999022]          # 与worldbody对齐
机械臂起始位置: [ 4.41854309e-04 -1.70331405e-01  1.21176806e+00]
DEBUG [pick]: ✓ 抓取成功! stage_success=True
```

视觉渲染中夹爪准确地夹取培养皿，位置完全匹配。

### 技术细节

#### MuJoCo 不会自动居中网格

**重要澄清**: MuJoCo **不会**自动将 OBJ 文件居中。

验证代码：

```python
from dm_control import mjcf

model = mjcf.from_path("petri_dish.xml")
physics = mjcf.Physics.from_mjcf_model(model)

# 检查所有 geom 的位置
for i in range(physics.model.ngeom):
    geom_name = physics.model.id2name(i, 'geom')
    geom_pos = physics.named.data.geom_xpos[geom_name]
    print(f"Geom[{i}] ({geom_name}): pos={geom_pos}")
```

输出（修复前）：
```
Geom[0] (visual mesh): pos=[-0.15, 0, 0.0065]  # 视觉在 X=-0.15
Geom[1] (collision cylinder): pos=[0, 0, 0.0075]  # 碰撞在 X=0
```

输出（修复后）：
```
Geom[0] (visual mesh): pos=[0, 0, 0.0065]      # ✅ 对齐！
Geom[1] (collision cylinder): pos=[0, 0, 0.0075]
```

#### OBJ 文件格式

OBJ 文件中 `v` 行定义顶点坐标：

```
v -0.19500000 0.08500000 0.01300000  # 顶点在 X=-0.195
v -0.10500000 0.08500000 0.01300000  # 顶点在 X=-0.105
```

所有顶点的X坐标范围 `[-0.195, -0.105]` 决定了网格的几何中心在 `-0.15`。

#### 为什么物理判定仍然成功

`is_grasped()` 方法检测**碰撞几何的接触**：

```python
def is_grasped(self, physics, robot):
    """基于接触力判断是否抓取成功"""
    gripper_geom_ids = [physics.bind(geom).element_id for geom in gripper_geoms]
    entity_geom_ids = [physics.bind(geom).element_id for geom in self.geoms]

    for contact in physics.data.contact:
        if (contact.geom1 in gripper_geom_ids and contact.geom2 in entity_geom_ids):
            return True  # ✅ 检测到碰撞几何接触
    return False
```

- 夹爪确实接触了碰撞圆柱（X=0 位置）
- 接触检测正确，抓取成功
- 但视觉网格在 X=-0.15，看起来没有被触碰

这是典型的**物理正确但视觉错误**的情况。

### 预防措施

#### ✅ DO - 推荐做法

1. **创建新模型时确保几何居中**
   - 在 Blender 等建模软件中将几何中心设置为原点
   - 导出前执行 "Origin to Geometry Center"
   - 验证顶点坐标范围关于原点对称

2. **检查现有 OBJ 文件的中心**
   ```bash
   # 快速检查脚本
   python3 -c "
   import numpy as np
   vertices = []
   with open('model.obj') as f:
       for line in f:
           if line.startswith('v '):
               parts = line.split()
               vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
   vertices = np.array(vertices)
   center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
   print(f'Geometric center: {center}')
   if np.linalg.norm(center) > 0.01:
       print('⚠️  Model is not centered!')
   "
   ```

3. **碰撞几何与惯性对齐**
   ```xml
   <!-- 确保惯性中心 = 碰撞几何中心 -->
   <body name="object">
     <inertial pos="0 0 0.015" mass="0.020" diaginertia="..."/>
     <geom type="cylinder" pos="0 0 0.015" .../>  <!-- 相同 pos -->
   </body>
   ```

4. **使用简单碰撞几何**
   - 复杂网格仅用于视觉 (`class="visual"`, `contype="0"`)
   - 碰撞使用简化形状（box, cylinder, sphere）
   - 减少接触点数量，提高稳定性

#### ❌ DON'T - 避免做法

1. **不要移动碰撞几何去匹配偏移的视觉网格**
   ```xml
   <!-- ❌ 会导致物理不稳定 -->
   <inertial pos="0 0 0.01" .../>
   <geom type="cylinder" pos="-0.15 0 0.01" .../>  <!-- 偏移！ -->
   ```

2. **不要依赖 MuJoCo 自动居中**
   - MuJoCo 会**忠实加载** OBJ 文件的顶点坐标
   - 不会进行任何自动平移或对齐

3. **不要忽略视觉-物理不匹配**
   - 即使物理判定正确，视觉错误会严重影响用户体验
   - 视觉学习模型会学到错误的视觉-动作映射

4. **不要使用过多凸包分解**
   ```xml
   <!-- ❌ 避免：21个凸mesh导致不稳定接触 -->
   <geom mesh="petri_dish_decomp_0" class="collision"/>
   <geom mesh="petri_dish_decomp_1" class="collision"/>
   ...
   <geom mesh="petri_dish_decomp_20" class="collision"/>
   ```

### 调试方法

#### 1. 检查视觉-物理对齐

```python
from dm_control import mjcf

model = mjcf.from_path("your_model.xml")
physics = mjcf.Physics.from_mjcf_model(model)

# 比较视觉和碰撞几何的位置
body_name = 'your_object_name/your_object_name'
visual_geom = None
collision_geom = None

for i in range(physics.model.ngeom):
    geom_type = physics.model.geom_type[i]
    geom_group = physics.model.geom_group[i]

    if geom_group == 2:  # visual
        visual_geom = physics.model.id2name(i, 'geom')
    elif geom_group == 3:  # collision
        collision_geom = physics.model.id2name(i, 'geom')

if visual_geom and collision_geom:
    visual_pos = physics.named.data.geom_xpos[visual_geom]
    collision_pos = physics.named.data.geom_xpos[collision_geom]
    offset = visual_pos - collision_pos

    print(f"Visual:    {visual_pos}")
    print(f"Collision: {collision_pos}")
    print(f"Offset:    {offset}")

    if np.linalg.norm(offset[:2]) > 0.01:  # XY平面偏移超过1cm
        print("⚠️  Visual-physical mismatch detected!")
```

#### 2. 快速验证 OBJ 中心

```bash
# 一行命令检查 OBJ 文件中心
python3 -c "
import sys
import numpy as np
verts = [[float(x) for x in l.split()[1:4]] for l in open(sys.argv[1]) if l.startswith('v ')]
verts = np.array(verts)
center = (verts.min(0) + verts.max(0)) / 2
print(f'Center: {center}')
print('✓ Centered' if np.linalg.norm(center) < 0.01 else '⚠️  Offset: ' + str(np.linalg.norm(center)) + 'm')
" your_model.obj
```

#### 3. 测试物理稳定性

```python
# 简单场景测试：放在地面上500步
from dm_control import mjcf
import numpy as np

test_scene = mjcf.RootElement()
test_scene.worldbody.add('geom', name='floor', type='plane', size=[2, 2, 0.1])

model = mjcf.from_path("your_model.xml")
attachment = test_scene.attach(model)
attachment.pos = [0, 0, 0.02]  # 刚好接触地面

physics = mjcf.Physics.from_mjcf_model(test_scene)

positions = []
for _ in range(500):
    physics.step()
    pos = physics.named.data.xpos['your_object/your_object'].copy()
    positions.append(pos)

positions = np.array(positions)
z_std = np.std(positions[:, 2])

if z_std > 0.001:
    print(f"⚠️  Unstable: Z std = {z_std*1000:.2f}mm")
else:
    print(f"✓ Stable: Z std = {z_std*1000:.3f}mm")
```

#### 4. 可视化检查

查看生成的视频，观察：
- 夹爪是否准确对准物体
- 抓取时是否有明显的空间错位
- 物体是否在夹爪闭合前就开始移动（weld约束错位）

### 相关文件

- `VLABench/assets/obj/meshes/lab_equipment/petri_dish/petri_dish_0/petri_dish.obj` - 视觉网格模型
- `VLABench/assets/obj/meshes/lab_equipment/petri_dish/petri_dish_0/petri_dish.xml` - MuJoCo模型配置
- `VLABench/tasks/components/entity.py` - 实体基类（包含 `is_grasped()` 方法）

### 关键经验

> **原则**: OBJ 模型的几何中心必须在原点，否则视觉渲染与物理仿真会出现空间错位。

> **约束**: 为了物理稳定，惯性中心必须与碰撞几何中心对齐。不能为了视觉效果而牺牲物理稳定性。

> **最佳实践**: 在建模阶段确保几何居中 > 导入后修改 OBJ 文件 > XML中使用 pos 偏移（最后的选择）

这个问题的本质是**三维建模规范**与**物理仿真约束**的冲突。正确的解决路径是修正源数据（OBJ文件），而不是通过配置参数去"绕过"问题。

---

## 3. LiftCondition 在 record_initial_state 未调用时误判为已满足

**日期**: 2026-05-25
**影响组件**: `VLABench/tasks/condition.py` — `LiftCondition`
**影响任务**: 所有使用 `lift_height` 参数的 `LiftCondition`（如 `lift_beaker`）
**严重程度**: 🔴 高（导致 `load_env` 无限死循环，评测完全卡死）

### 问题表现

1. **评测脚本卡死**
   - 调用 `load_env('lift_beaker')` 后无任何输出，进程永久挂起
   - 无报错信息，无法通过日志定位

2. **调试定位**
   ```
   [DEBUG] load_env 开始
   [DEBUG] load_env 完成
   [DEBUG] run_episode 开始
   [DEBUG] env.reset() 完成
   # ← 之后永久卡住
   ```

3. **最终表现**
   - `load_env` 内部调用 `env.reset()`
   - `env.reset()` 调用 `super().reset()` 后执行 `step()` 循环来稳定场景
   - 第一个 `step()` 中 `should_terminate_episode()` 返回 True
   - 设置 `_reset_next_step = True`
   - 下一个 `step()` 触发 `self.reset()` → 再次进入 `step()` 循环 → 再次 terminate → 无限递归

### 根本原因

#### 1. LiftCondition.is_met() 的逻辑缺陷

原始代码（修复前）：

```python
def is_met(self, physics=None):
    for entity in self.entities:
        entity_xpos = physics.bind(entity.mjcf_model.worldbody).xpos
        name = entity.name if hasattr(entity, 'name') else str(id(entity))

        if self.lift_height is not None and self._initial_z:  # ← 关键
            initial_z = self._initial_z.get(name, entity_xpos[-1])
            target_z = initial_z + self.lift_height - self._tolerance
            if entity_xpos[-1] < target_z:
                return False
        elif self.target_height is not None:
            if entity_xpos[-1] < self.target_height - self._tolerance:
                return False
    return True  # ← 当两个分支都不进入时，直接返回 True
```

当 `lift_height` 不为 None 但 `_initial_z` 为空字典 `{}` 时：
- `self._initial_z` 为 falsy → 跳过第一个 if 分支
- `self.target_height` 为 None → 跳过 elif 分支
- 循环体没有 `return False` → 最终 `return True`

**结果**：条件在环境刚初始化、还没记录初始高度时就返回 True。

#### 2. 死循环链条

```
load_env()
  → env.reset()                    # LM4ManipDMEnv.reset()
    → super().reset()               # composer.Environment.reset() ✅
    → cancel_gravity_and_improve_fluid()  ✅
    → for i in range(10): step()    # step 1
      → should_terminate_episode()  # → LiftCondition.is_met() = True ❌
      → _reset_next_step = True
    → step()                        # step 2
      → _reset_next_step == True
      → return self.reset()         # ← 递归！
        → super().reset()           ✅
        → step() 循环...            # 同样的 terminate → reset 循环
```

#### 3. 为什么 record_initial_state 没有被调用

`record_initial_state` 在评测脚本中是在 `env.reset()` 之后手动调用的：

```python
env.reset()  # ← 这里就卡死了，下面的代码不会执行
for condition in env.task.conditions.conditions:
    condition.record_initial_state(env.physics)
```

而 `env.reset()` 内部的 `step()` 循环在 `record_initial_state` 被调用之前就已经检查了 `is_met()`。

### 解决方案

**在 `is_met()` 中增加初始态检查**：当 `lift_height` 模式下 `_initial_z` 为空时，返回 `False`。

文件：`VLABench/tasks/condition.py`

```python
def is_met(self, physics=None):
    for entity in self.entities:
        entity_xpos = physics.bind(entity.mjcf_model.worldbody).xpos
        name = entity.name if hasattr(entity, 'name') else str(id(entity))

        if self.lift_height is not None:
            if not self._initial_z:      # ← 新增：初始态未记录，条件不满足
                return False
            initial_z = self._initial_z.get(name, entity_xpos[-1])
            target_z = initial_z + self.lift_height - self._tolerance
            if entity_xpos[-1] < target_z:
                return False
        elif self.target_height is not None:
            if entity_xpos[-1] < self.target_height - self._tolerance:
                return False
    return True
```

同时让 `LiftCondition.__init__` 调用 `super().__init__()`，`record_initial_state` 调用 `super().record_initial_state()`。

### 预防措施

#### ✅ DO — 添加新 Condition 时的检查清单

1. **依赖初始态的条件，必须在 is_met 中检查初始态是否已记录**
   - 如果条件需要"前态-终态对比"（如举起高度、移动距离），确保 `_initial_z` / `_initial_pos` 等非空时才进行判断
   - 未记录初始态时应返回 `False`

2. **子类 __init__ 必须调用 super().__init__()**
   ```python
   class MyCondition(Condition):
       def __init__(self, ...):
           super().__init__()   # ← 必须
           self._my_state = {}
   ```

3. **子类 record_initial_state 必须调用 super().record_initial_state()**
   ```python
   def record_initial_state(self, physics=None):
       super().record_initial_state(physics)  # ← 必须
       ...
   ```

4. **验证环境能正常加载**：新任务创建后，先测试 `load_env` 能否在 10 秒内完成

#### ❌ DON'T

1. **不要让 is_met 在缺少前置数据时默认返回 True**
2. **不要假设 record_initial_state 一定在 is_met 之前被调用**

### 调试方法

#### 1. 快速验证条件初始判定

```python
from VLABench.robots import *
from VLABench.tasks import *
from VLABench.envs import load_env
import numpy as np, random

np.random.seed(42)
random.seed(42)
task = register.load_task('your_task')(...)
env = LM4ManipDMEnv(task=task, time_limit=float('inf'), reset_wait_step=0)

env._reset_attempt()  # 只做基础 reset
# 此时 record_initial_state 尚未被调用

if hasattr(task, 'conditions') and task.conditions:
    for c in task.conditions.conditions:
        result = c.is_met(env.physics)
        print(f'{type(c).__name__}: is_met={result}  ← 应为 False')
env.close()
```

如果任何条件在此时返回 True，说明存在同类 bug。

#### 2. 定位 load_env 卡死

```python
import signal, sys

def timeout(signum, frame):
    print('TIMEOUT!'); sys.exit(1)

signal.signal(signal.SIGALRM, timeout)
signal.alarm(60)  # 60秒超时

env = load_env('your_task', random_init=True, eval=False, run_mode='eval')
signal.alarm(0)
print('load_env succeeded')
```

如果超时，逐步拆解 `load_env` 内部调用来定位卡住的位置。

#### 3. 检查 _reset_next_step 死循环

在 `dm_env.py` 的 `step()` 方法中添加临时 debug：

```python
if self._reset_next_step:
    print(f'[WARN] _reset_next_step=True, calling reset() (timestep={self.timestep})')
    self._reset_next_step = False
    return self.reset()
```

如果看到大量重复输出，说明存在 terminate → reset 的死循环。

### 其他 Condition 类的风险评估

| 条件类 | 依赖初始态 | 同类风险 | 说明 |
|--------|-----------|---------|------|
| `LiftCondition` | `_initial_z` | **已修复** | lift_height 模式需初始高度 |
| `WaitForCondition` | `_change_applied` | 低 | 逻辑不同，首次调用时 is_met 行为正确 |
| `OrderCondition` | 无 | 无 | 纯位置检查 |
| `ContainCondition` | 无 | 无 | 容器包含检测 |
| `OnCondition` | 无 | 无 | 接触+Z 轴检测 |
| `HeatedCondition` | `accumulated_time` | 无 | 初始为 0，需要累积才能满足 |
| 其他条件 | 无 | 无 | 均为即时状态检查 |

### 相关文件

- `VLABench/tasks/condition.py` — 条件定义
- `VLABench/envs/dm_env.py` — `reset()` 和 `step()` 中的终止检查
- `VLABench/tasks/dm_task.py` — `should_terminate_episode()` 调用 `conditions.is_met()`

### 关键经验

> **原则**：任何依赖"初始状态记录"的 Condition，在初始态未记录时，`is_met()` 必须返回 `False`，绝不能默认返回 `True`。

> **教训**：Python 中空字典 `{}` 是 falsy，放在 `if` 条件中会导致分支被跳过。对于需要记录初始态的条件，应显式检查"是否已记录"，而非依赖容器的 truthiness。

---

## 4. place/drop 技能被 should_terminate 提前终止导致仿真中断

**日期**: 2026-05-27
**影响任务**: 所有包含 `place` 或 `drop` 技能的任务（如 `pick_small_beaker_place_small_beaker`）
**严重程度**: 🔴 高（任务提前终止，open_gripper 不执行，condition 不评估，误报成功）

### 问题表现

1. **仿真提前终止**
   - 运行 `pick the small_beaker and place it on the electronic_scale` 时，place 技能在烧杯刚接触电子秤表面就停止了
   - 视频显示机械臂带着烧杯接触电子秤后立即停止，没有松开夹爪，没有抬升

2. **condition 未被评估**
   - `step_condition_results` 只有 step 0（`is_grasped`）的结果
   - step 1（`on` condition）从未被检查

3. **任务误报成功**
   - `simulation_success: true`，但实际上烧杯还在夹爪里，没有被放下

### 根本原因

#### 1. 事件链

```
place() 执行:
  step_trajectory() 沿路径移动机械臂
    → 烧杯接触电子秤表面
    → OnCondition.is_met() = True（检测到接触 + Z 轴高度满足）
    → should_terminate_episode() = True
    → dm_env.step() 返回 LAST
    → step_trajectory 检测到 timestep.last() → break → 返回 task_success=True

  place() 检测到 task_success=True → return（直接退出）
    → open_gripper() 未执行 ← 烧杯还在夹爪里
    → lift() 未执行
```

#### 2. 两层提前退出

| 层级 | 代码位置 | 行为 |
|------|---------|------|
| `step_trajectory` | `skill_lib.py:87-89` | `timestep.last()` → `task_success=True` → `break` |
| `place()` | `skill_lib.py:466-469` | `if task_success: return` → 跳过 `open_gripper` |
| `simulation_node` | `simulation.py:371-373` | `if skill_task_success: break` → 跳出技能循环 |

#### 3. 为什么 pour 没有这个问题

`pour` 技能的条件（容器底部高于顶部）只在倾斜过程中满足，且 `pour` 用 `break`（跳出循环继续执行后续阶段）而非 `return`（直接退出函数）。`place` 用了 `return`，导致后续动作被跳过。

#### 4. 核心矛盾

- `should_terminate_episode()` 的设计意图是为**强化学习训练循环**服务的：条件满足 → 返回 LAST → 下一帧 reset → 新 episode
- 但 VLABench 的 pipeline 中，一个技能结束后还有后续动作（open_gripper、lift），不能立刻 reset
- 在技能执行期间，`step()` 不应该因为 `should_terminate()` 而返回 LAST

### 解决方案

在 `dm_env.py` 中添加 `_skill_execution_mode` 开关，当开关打开时 `step()` 不返回 LAST，始终返回 MID。

#### 改动1：`dm_env.py` — 添加开关

**文件**: `VLABench/envs/dm_env.py`

在 `__init__` 中添加开关变量：

```python
self._skill_execution_mode = False  # True 时 step() 不返回 LAST
```

修改 `step()` 方法，开关打开时跳过终止逻辑：

```python
# 修改前:
if not terminating:
    return dm_env_lib.TimeStep(dm_env_lib.StepType.MID, reward, discount, obs)
else:
    self._reset_next_step = True
    return dm_env_lib.TimeStep(dm_env_lib.StepType.LAST, reward, discount, obs)

# 修改后:
if not terminating:
    return dm_env_lib.TimeStep(dm_env_lib.StepType.MID, reward, discount, obs)
else:
    if self._skill_execution_mode:
        # 技能执行期间：不终止，不 reset，继续返回 MID
        return dm_env_lib.TimeStep(dm_env_lib.StepType.MID, reward, discount, obs)
    self._reset_next_step = True
    return dm_env_lib.TimeStep(dm_env_lib.StepType.LAST, reward, discount, obs)
```

#### 改动2：`simulation.py` — 控制开关

**文件**: `VLABench/pipeline/nodes/simulation.py`

在技能循环前打开开关，循环结束后关闭：

```python
# 技能循环前
env._skill_execution_mode = True

try:
    for skill_idx, skill in enumerate(skill_seq):
        ...

except SkillTimeoutError:
    ...
finally:
    signal.alarm(0)
    signal.signal(signal.SIGALRM, old_handler)
    env._skill_execution_mode = False  # 恢复
```

#### 改动3：`skill_lib.py` — 无需修改

`place()` 和 `drop()` 的 `if task_success: return` 逻辑保持原样不变。因为 `step()` 不再返回 LAST，`timestep.last()` 不会触发，`task_success` 不会被设为 True，`if task_success` 分支不会进入。

### 修改后的执行流程

```
simulation_node:
  env._skill_execution_mode = True     ← 打开开关

  skill_idx=0: pick()
    step_trajectory:
      env.step() → should_terminate()=False → 返回 MID → 正常移动
    返回 task_success=False → simulation_node 继续

  skill_idx=1: place()
    step_trajectory:
      env.step() → should_terminate()=True → _skill_execution_mode=True
        → 返回 MID（不终止）→ 继续移动
      ...路径走完或到达位置误差容限...
    返回 task_success=False

    open_gripper():
      env.step() → 返回 MID → 正常执行 ✓
      夹爪打开，烧杯释放 ✓

    lift():
      env.step() → 返回 MID → 正常执行 ✓
      机械臂抬升 ✓

    返回 stage_success, task_success=False

  step 1 condition 评估 ✓ → OnCondition 检查烧杯是否在电子秤上

  env._skill_execution_mode = False    ← 关闭开关
```

### 预防措施

#### ✅ DO

1. **新增技能时，检查是否在 `should_terminate()` 触发后有后续动作**
   - 如果有（如 open_gripper、lift），确保 `_skill_execution_mode` 在技能执行期间为 True
   - 如果没有（如 pour 的倾斜阶段），行为不受影响

2. **`_skill_execution_mode` 只在 `simulation_node` 中控制**
   - 不要在其他地方修改此开关
   - `simulation_node` 的 `finally` 块确保开关一定会被关闭

#### ❌ DON'T

1. **不要在 `skill_lib.py` 中修改 `_skill_execution_mode`**
   - 技能函数应该是纯函数式的，不应依赖环境内部状态

2. **不要删除 `place()` / `drop()` 的 `if task_success: return` 逻辑**
   - 这个逻辑在非 pipeline 场景（如强化学习训练）中是正确的
   - 通过 `_skill_execution_mode` 让它在 pipeline 中不触发

### 调试方法

#### 1. 检查 step_condition_results 是否完整

```bash
# 在 pipeline log 中搜索 condition 结果
grep -A5 "step_condition_results" pipeline_*.log
```

如果只有 step 0 没有 step 1，说明仿真提前终止了。

#### 2. 检查 _skill_execution_mode 是否生效

在 `dm_env.py` 的 `step()` 中添加临时 debug：

```python
if self._skill_execution_mode and terminating:
    print(f"[DEBUG] _skill_execution_mode=True, skipping LAST (terminating={terminating})")
    return dm_env_lib.TimeStep(dm_env_lib.StepType.MID, reward, discount, obs)
```

如果在技能执行期间看到此输出，说明开关生效。

#### 3. 对比 pour 和 place 的行为

`pour` 的条件只在过程中满足（容器倾斜时），结束后恢复。`place` 的条件一旦满足就持续满足。如果新技能的条件也有"持续满足"的特性，需要注意此问题。

### 相关文件

- `VLABench/envs/dm_env.py` — `step()` 方法，`_skill_execution_mode` 开关
- `VLABench/pipeline/nodes/simulation.py` — 技能循环控制开关
- `VLABench/utils/skill_lib.py` — `place()`, `drop()`, `step_trajectory()`
- `VLABench/tasks/condition.py` — `OnCondition.is_met()`
- `VLABench/tasks/dm_task.py` — `should_terminate_episode()`

### 关键经验

> **原则**：技能执行期间，`should_terminate_episode()` 不应该导致 `step()` 返回 LAST。条件检查应该只作为"成功判定"使用，而不是"执行控制"。

> **教训**：`should_terminate_episode()` 是为强化学习训练循环设计的（条件满足 → episode 结束 → reset），但在 skill-based pipeline 中，技能需要完整执行所有动作后才能终止。这两个场景的需求是矛盾的，需要通过开关机制来区分。

> **设计选择**：将开关放在 `dm_env.py` 而非 `skill_lib.py`，是因为技能函数不应该关心环境内部状态。`simulation_node` 作为调用方负责控制环境行为。

---

## 5. CapOpenCondition 静平衡形变误触导致 load_env 无限 reset 死循环

**日期**: 2026-06-05
**影响任务**: `unscrew_pill_bottle`（使用 `ContainerWithCap` + `arena.attach()` 固定瓶身时触发）
**影响组件**: `VLABench/tasks/condition.py` — `CapOpenCondition`，`VLABench/tasks/components/specific_entities/interactive_containers.py` — `ContainerWithCap.is_cap_open`
**严重程度**: 🔴 高（`load_env` 无限死循环，任何使用带关节实体 + attach 固定的任务都会触发）

### 问题表现

1. **`load_env` 无限卡死，无任何输出**
   - 调用 `load_env('unscrew_pill_bottle')` 后进程永久挂起
   - 无报错，无日志，只能通过 `kill` 或 timeout 终止

2. **只在 attach 后出现**
   - **Baseline**（无 `build_from_config` 重写）能跑通
   - **加了 `entity.detach() + self._arena.attach(entity)` 后卡死**
   - 同样的 attach 范式在 `select_chemistry_tube`（试管架）等任务上正常

3. **诊断日志特征**
   ```
   reset #1 entered
     initialize_episode called
     CapOpenCondition.is_met returned: True    ← 刚初始化就判定"瓶盖已拧开"！
   step #1 returned type=2 reward=0            ← LAST，触发 reset
   step #2 entered
     reset #2 entered                           ← 再次进入 reset
       initialize_episode called
       CapOpenCondition.is_met returned: True   ← 又是 True
     ...
   # 无限循环
   ```

### 根本原因

#### 1. 完整因果链

```
pill_bottle.xml 定义了两个关节:
  <joint name="door" type="hinge" range="-6.28 6.28" axis="0 0 1"/>   ← 瓶盖绕 z 轴旋转
  <joint name="cap_slide" type="slide" range="0 0.02" axis="0 0 1"/>  ← 瓶盖 z 方向上升

build_from_config 中 attach 后:
  entity.detach()          ← 删除 freejoint
  self._arena.attach(entity) ← 瓶身焊死到 arena（无 freejoint，不动）

env.reset() 调用 composer.Environment.reset():
  → initialize_episode()   ← 设置实体位置
  → mj_forward()           ← 物理求解器做静平衡
  → slide joint 在约束松弛下被压到 qpos=0.0109 m

env.reset() 继续执行 dm_env.py:67-68:
  → for i in range(reset_wait_step): self.step()  ← 第 1 次 step
    → should_terminate_episode()
      → CapOpenCondition.is_met()
        → slide_qpos=0.0109 > lift_threshold=0.01  ← 超过阈值！
        → return True                               ← 误判"瓶盖已拧开"
    → terminating=True
    → _reset_next_step=True
    → return LAST

  → 第 2 次 step
    → _reset_next_step=True
    → return self.reset()     ← 再次进入 reset
      → 同样的 slide_qpos=0.0109 → 同样 True → 无限循环
```

#### 2. 为什么 tube_stand 不触发

`tube_stand.xml` 没有任何 joint（纯静态 mesh），attach 后不存在静平衡形变问题。
`pill_bottle.xml` 有 hinge + slide 两个关节，slide 的 `range="0 0.02"` 在物理编译时
被求解器从 qpos=0 推到 0.0109（约束松弛），恰好超过 `lift_threshold=0.01`。

#### 3. 为什么 baseline（不 attach）不触发

Baseline 时瓶身有 freejoint（6DOF 自由体），下落过程中物理系统整体运动，
slide joint 的形变更小（约 0.001），低于 0.01 阈值。
attach 后瓶身被固定，所有物理扰动集中在 hinge/slide 关节上，形变被放大。

#### 4. 本质：两个设计缺陷叠加

| 缺陷 | ��置 | 描述 |
|------|------|------|
| **绝对 qpos 判断** | `CapOpenCondition.is_met()` | 用 `abs(slide_qpos) > threshold` 而非 `abs(slide_qpos - initial) > threshold` |
| **无初始态门控** | `CapOpenCondition.is_met()` | `record_initial_state` 未被调用时，`is_met` 仍能返回 True |

第一个缺陷导致静平衡的微小形变（0.0109 m）被误判为"被拧开"。
第二个缺陷导致 reset 阶段（`record_initial_state` 尚未被调用）也能返回 True。

### 解决方案

**两处改动**，均在 `condition.py`：

#### 改动 1：`is_met` 改用相对位移 + 初始态门控

```python
@register.add_condition("cap_open")
class CapOpenCondition(Condition):
    def __init__(self, entities, open_threshold=3*np.pi/2, lift_threshold=0.01):
        super().__init__()
        self.entities = entities
        self.open_threshold = open_threshold
        self.lift_threshold = lift_threshold
        self._initial_joint_pos = {}   # entity_name -> hinge qpos at record time
        self._initial_slide_pos = {}   # entity_name -> slide qpos at record time

    def record_initial_state(self, physics):
        super().record_initial_state(physics)
        for entity in self.entities:
            name = entity.mjcf_model.model
            if entity.cap_joint is not None:
                self._initial_joint_pos[name] = physics.bind(entity.cap_joint).qpos.copy()
            if entity.slide_joint is not None:
                self._initial_slide_pos[name] = physics.bind(entity.slide_joint).qpos.copy()

    def is_met(self, physics=None):
        # 门控：record_initial_state 未调用时无法判断相对位移，返回 False
        if not self._initial_state_recorded:
            return False
        for entity in self.entities:
            name = entity.mjcf_model.model
            hinge_open = False
            slide_open = False
            if entity.cap_joint is not None:
                current = physics.bind(entity.cap_joint).qpos
                initial = self._initial_joint_pos.get(name, 0.0)
                hinge_open = abs(current - initial) > self.open_threshold
            if entity.slide_joint is not None:
                current = physics.bind(entity.slide_joint).qpos
                initial = self._initial_slide_pos.get(name, 0.0)
                slide_open = abs(current - initial) > self.lift_threshold
            if not (hinge_open or slide_open):
                return False
        return True
```

#### 改动 2（同步）：`ContainerWithCap.is_cap_open` 加初始态参数

`interactive_containers.py` 中的 `is_cap_open` / `is_cap_closed` 方法也使用绝对 qpos 判断，
改为接受可选的 `initial_joint_qpos` / `initial_slide_qpos` 参数：

```python
def is_cap_open(self, physics, initial_joint_qpos=None, initial_slide_qpos=None):
    if self.cap_joint is None:
        return False
    initial_j = initial_joint_qpos if initial_joint_qpos is not None else 0.0
    if abs(physics.bind(self.cap_joint).qpos - initial_j) > self.open_threshold:
        return True
    if self.slide_joint is not None and initial_slide_qpos is not None:
        if abs(physics.bind(self.slide_joint).qpos - initial_slide_qpos) > 0.01:
            return True
    return False
```

### 与条目 3（LiftCondition）的关联

这是**同一种 bug 模式的第二次出现**：

| 条目 | 条件类 | 判断方式 | 静平衡触发源 | 修复方式 |
|------|--------|---------|------------|---------|
| #3 | `LiftCondition` | 绝对高度 + 空字典门控 | 物体初始 z 不为 0 | 加 `_initial_z` 空检查 |
| **#5** | `CapOpenCondition` | **绝对 qpos + 无门控** | **slide joint 静平衡形变** | **相对位移 + `_initial_state_recorded` 门控** |

**共同模式**：任何依赖"前后状态对比"的条件（高度变化、关节位移、姿态变化），
在 `record_initial_state` 未被调用时，`is_met()` **必须返回 False**。
绝对值判断在物理仿真中不可靠——静平衡、数值漂移、约束松弛都会导致初始值偏离理论值。

### 预防措施

#### ✅ DO — 新增带关节实体 + attach 固定时的检查清单

1. **新 Condition 必须实现 `record_initial_state`**
   ```python
   def record_initial_state(self, physics):
       super().record_initial_state(physics)  # ← 必须，设置 _initial_state_recorded=True
       # 记录初始 qpos / xpos / euler 等
   ```

2. **`is_met` 必须检查 `_initial_state_recorded`**
   ```python
   def is_met(self, physics=None):
       if not self._initial_state_recorded:
           return False  # ← 未记录初态，无法判断变化量
   ```

3. **用相对位移而非绝对值判断"状态变化"**
   ```python
   # ✅ 正确：相对初态的位移
   abs(current - initial) > threshold
   # ❌ 错误：绝对值（会被静平衡形变误触）
   abs(current) > threshold
   ```

4. **带关节实体 attach 后验证 `load_env` 不卡**
   - 有 freejoint 的实体（baseline）：物理运动吸收了关节形变
   - attach 后无 freejoint：关节形变被放大，可能触发条件误判
   - **必须在 attach 后单独测试**

#### ❌ DON'T

1. **不要用绝对 qpos / xpos 判断"某物被操作过"** —— 物理仿真的初始值不是理论值
2. **不要假设 `record_initial_state` 一定在 `is_met` 之前被调用** —— `env.reset()` 内部的 `step()` 循环就会调用 `should_terminate_episode()`

### 调试方法

#### 1. 诊断 load_env 卡死

```python
import signal, sys
def timeout(signum, frame):
    print('TIMEOUT after 60s!'); sys.exit(1)
signal.signal(signal.SIGALRM, timeout)
signal.alarm(60)
env = load_env('your_task')
signal.alarm(0)
print('load_env succeeded')
```

#### 2. 追踪 reset 循环

在 `dm_env.py` 的 `step()` 中临时添加：

```python
if self._reset_next_step:
    print(f'[WARN] _reset_next_step at timestep={self.timestep}')
```

如果看到连续输出，说明存在 terminate → reset 死循环。

#### 3. 检查条件初值

```python
env = load_env('your_task')
# reset() 刚完成，record_initial_state 尚未被 pipeline 调用
for c in env.task.conditions.conditions:
    result = c.is_met(env.physics)
    print(f'{type(c).__name__}: is_met={result}  ← 必须为 False')
env.close()
```

如果任何条件返回 True，存在同类 bug。

### 相关文件

- `VLABench/tasks/condition.py` — `CapOpenCondition` 修复
- `VLABench/tasks/components/specific_entities/interactive_containers.py` — `ContainerWithCap.is_cap_open` 修复
- `VLABench/tasks/autogen_tasks/unscrew_pill_bottle_series.py` — `build_from_config` 中的 attach 逻辑
- `VLABench/assets/review/pill_bottle/pill_bottle/pill_bottle/pill_bottle.xml` — pill_bottle XML（hinge + slide joint 定义）

### 关键经验

> **原则**：物理仿真中关节的初始 qpos 不是理论值 0。静平衡、约束松弛、数值漂移都会导致初始值偏离。任何"检测是否被操作过"的条件，必须用相对位移（当前值 - 初态值），不能用绝对值。

> **教训**：这是 `LiftCondition`（条目 #3）同一 bug 模式的重复出现。当新增 Condition 类时，必须对照条目 #3 的检查清单：实现 `record_initial_state`、`is_met` 中检查 `_initial_state_recorded`、使用相对位移判断。

> **attach 放大了问题**：有 freejoint 的实体下落时物理系统整体运动，关节形变小。attach 后实体被固定，所有扰动集中在关节上，形变被放大到超过阈值。这是为什么"baseline 能跑、attach 卡死"的原因。

---

**最后更新**: 2026-06-05
**维护者**: VLABench Team
