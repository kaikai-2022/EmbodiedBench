# VLABench 手动创建任务指南

本文档说明从零创建一个新任务所需的全部步骤，包括文件创建、注册、模型加载和轨迹渲染。

---

## 整体流程概览

创建一个新任务需要完成以下步骤：

```
1. 编写任务 Series 文件（定义 ConfigManager + Task）
2. 在 primitive/__init__.py 中导入任务文件
3. 在 configs/__init__.py 的 name2config 中注册
4. 在 configs/task_config.json 中添加配置
5. 用 trajectory_generation.py 脚本渲染轨迹
```

---

## 第一步：编写任务 Series 文件

**路径**：`VLABench/tasks/hierarchical_tasks/primitive/<task_name>_series.py`

每个任务文件必须包含两个类，都用装饰器注册：

### 最简模板

```python
import random
import numpy as np
from functools import partial
from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.primitive.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register


@register.add_config_manager("my_task")
class MyTaskConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=[1, 1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)

    def load_objects(self, target_entity):
        """加载物体到场景中"""
        object_config = dict(
            name="my_object",
            xml_path="obj/meshes/path/to/model.xml",
            position=[0.0, 0.0, 0.8],
            orientation=[0, 0, 0],
        )
        object_config["class"] = "CommonGraspedEntity"  # 对应 constant.py 中的类名
        object_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0.05])
        self.config["task"]["components"].append(object_config)

    def get_instruction(self, target_entity, **kwargs):
        instruction = [f"Do something with {target_entity}"]
        self.config["task"]["instructions"] = instruction

    def get_condition_config(self, target_entity, **kwargs):
        conditions_config = dict(
            lift=dict(
                entities=[target_entity],
                target_height=0.9
            )
        )
        self.config["task"]["conditions"] = conditions_config


@register.add_task("my_task")
class MyTaskTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name=self.target_entity),
            partial(SkillLib.lift, lift_height=0.15, gripper_state=np.zeros(2)),
        ]
        return skill_sequence
```

### ConfigManager 中可重写的方法

| 方法 | 必须 | 说明 |
|------|------|------|
| `load_containers(self, target_container)` | 否 | 加载目标容器 |
| `load_init_containers(self, init_container)` | 否 | 加载初始容器 |
| `load_objects(self, target_entity)` | **是** | 加载操作物体 |
| `get_instruction(self, ...)` | **是** | 返回任务指令字符串列表 |
| `get_condition_config(self, ...)` | **是** | 返回成功条件配置字典 |

### Task 类中的关键方法

| 方法 | 说明 |
|------|------|
| `get_expert_skill_sequence(self, physics)` | 返回技能序列，每个元素是 `partial(SkillLib.xxx, ...)` |
| `build_from_config(self, eval, **kwargs)` | 可重写，用于额外的实体初始化（如固定容器） |

### 可用的 SkillLib 技能

| 技能 | 关键参数 | 说明 |
|------|---------|------|
| `SkillLib.pick` | `target_entity_name` | 抓取物体 |
| `SkillLib.lift` | `lift_height`, `gripper_state` | 抬起物体 |
| `SkillLib.place` | `target_container_name` | 放入容器 |
| `SkillLib.moveto` | `target_pos`, `gripper_state` | 移动到目标位置 |
| `SkillLib.rotate` | `rotation_angle`, `gripper_state` | 旋转 |
| `SkillLib.pour` | `target_delta_qpos`, `target_q_velocity` | 倾倒 |
| `SkillLib.open_gripper` | 无 | 打开夹爪 |
| `SkillLib.close_gripper` | 无 | 关闭夹爪 |
| `SkillLib.wait` | `wait_time` | 等待 |

### 可用的条件类型

| 条件类型 | 参数 | 说明 |
|---------|------|------|
| `lift` | `entities`, `target_height` | 物体被抬到指定高度 |
| `contain` | `entities`, `container` | 物体被放入容器 |
| `grasped` | `entities` | 物体被抓取 |

---

## 第二步：在 primitive/__init__.py 中导入

**路径**：`VLABench/tasks/hierarchical_tasks/primitive/__init__.py`

在文件末尾追加一行：

```python
from VLABench.tasks.hierarchical_tasks.primitive.my_task_series import *
```

这一步让 `@register.add_task` 和 `@register.add_config_manager` 装饰器生效。

---

## 第三步：在 configs/__init__.py 中注册

**路径**：`VLABench/configs/__init__.py`

在 `name2config` 字典中添加映射：

```python
name2config = {
    ...,
    "my_task_series": ["my_task"],
}
```

**关键规则**：
- key 必须以 `_series` 结尾（如 `my_task_series`）
- value 是任务名列表（如只有一个任务就写 `["my_task"]`）
- 一个 series 可以包含多个变体任务（如 `select_mahjong_series` 包含 spatial、semantic 等变体）

---

## 第四步：在 task_config.json 中添加配置

**路径**：`VLABench/configs/task_config.json`

添加配置块，key 与 `name2config` 中的 key 一致：

```json
"my_task_series": {
    "task": {
        "asset": {
            "seen_object": ["my_object"],
            "unseen_object": ["my_object"]
        },
        "scene": {
            "name": "laboratory_0"
        },
        "components": [
            {
                "name": "table",
                "xml_path": "obj/meshes/table/table.xml",
                "class": "Table",
                "randomness": {
                    "texture": false
                }
            }
        ]
    }
}
```

### 配置字段说明

| 字段 | 说明 |
|------|------|
| `asset.seen_object` | 训练时使用的物体列表，传给 ConfigManager 的 `seen_object` |
| `asset.unseen_object` | 评估时使用的未见物体列表 |
| `asset.seen_container` | 训练时使用的容器列表 |
| `asset.unseen_container` | 评估时使用的未见容器列表 |
| `asset.seen_init_container` | 训练时使用的初始容器列表 |
| `scene.name` | 场景名称，如 `"laboratory_0"`、`"lab_0"` |
| `components` | 固定组件（通常是桌子） |

---

## 模型加载方式

### 方式一：手动构建 object_config（推荐用于简单任务）

```python
def load_objects(self, target_entity):
    object_config = dict(
        name="chemistry_beaker",               # 实体名称，全局唯一标识
        xml_path="obj/meshes/.../model.xml",    # 模型 XML 相对路径
        solution="CuSO4",                       # 可选的自定义参数
        position=[0.0, 0.0, 0.8],              # 初始位置 [x, y, z]
        orientation=[0, 0, 0],                  # 初始朝向（欧拉角）
    )
    object_config["class"] = "ChemistryBeaker"  # 类名，对应 constant.py 中的注册名
    object_config["randomness"] = dict(
        pos=[0.02, 0.02, 0],    # 位置随机范围 [x, y, z]
        quat=[0, 0, 0.05],       # 朝向随机范围 [roll, pitch, yaw]
    )
    self.config["task"]["components"].append(object_config)
```

### 方式二：使用 get_entity_config（推荐用于已注册的模型）

```python
def load_objects(self, target_entity):
    object_config = self.get_entity_config(
        target_entity,                       # 必须是 constant.py 中 name2class_xml 的 key
        position=[0.0, 0.0, 0.8],
        orientation=[0, 0, 0],
        solution="CuSO4",                    # 额外参数会透传到实体构造函数
    )
    self.config["task"]["components"].append(object_config)
```

### 已注册的模型（在 constant.py 中）

常用的 `name2class_xml` key：

| key | 类名 | 说明 | 是否支持额外参数 |
|-----|------|------|----------------|
| `"beaker"` | CommonGraspedEntity | 普通烧杯 | 否 |
| `"chemistry_beaker"` | ChemistryBeaker | 带溶液渲染的烧杯 | `solution`（溶剂名） |
| `"tube"` | ChemistryTube | 试管 | `solution`（溶剂名） |
| `"chemistry_tube_stand"` | TubeStand | 试管架 | 否 |
| `"nametag"` | NameTag | 标签 | `content`（标签内容） |
| `"flask"` | CommonContainer | 锥形瓶 | 否 |
| `"petri_dish"` | CommonGraspedEntity | 培养皿 | 否 |
| `"mahjong"` | Mahjong | 麻将牌 | `value`, `suite` |
| `"billiards"` | BilliardBall | 台球 | `value`（颜色名） |

### 需要额外参数的模型

**ChemistryTube / ChemistryBeaker** — 需要 `solution` 参数指定溶剂颜色：
```python
xml_path=name2class_xml["tube"][-1]  # 或 "chemistry_beaker"
object_config["class"] = name2class_xml["tube"][0]  # 或 "chemistry_beaker"
# 传入 solution 参数
solution="CuSO4"   # 蓝色
solution="KMnO4"  # 紫色
solution="FeCl3"  # 棕黄色
```

**Mahjong** — 需要 `value` 和 `suite` 参数：
```python
mahjong_config = dict(
    name=f"{value}_{type}",
    xml_path=name2class_xml["mahjong"][-1],
    value=value,      # "1"~"9"
    suite=type,       # "man", "pin", "sou"
)
```

### 作为子实体加载（试管架上的试管等）

```python
# 先添加父容器（如试管架）
init_container_config["subentities"] = []
# 再在 subentities 中添加子实体
object_config = dict(
    name=target_entity,
    solution=target_entity,
    xml_path=name2class_xml["tube"][-1],
    position=[col_pos, row_pos, 0.05],
)
object_config["class"] = name2class_xml["tube"][0]
init_container_config["subentities"].append(object_config)
```

---

## 第五步：渲染轨迹

使用 `scripts/trajectory_generation.py`：

```bash
# 基本用法
MUJOCO_GL=egl python3 scripts/trajectory_generation.py \
    --task-name my_task \
    --n-sample 1 \
    --debug \
    --save-dir /path/to/output

# 常用参数
#   --task-name       任务名（name2config value 中的名字）
#   --n-sample        生成轨迹数量
#   --debug           打印详细调试信息
#   --save-dir        输出目录
#   --robot           机器人（默认 franka）
#   --eval-unseen     使用未见物体评估
#   --record-video    是否录制视频（默认 True）
```

**必须使用 `MUJOCO_GL=egl` 环境变量**，否则在无显示器的服务器上会报 OpenGL 错误。

### 输出文件

```
<save-dir>/my_task/
├── demo_0_success_True.mp4      # 渲染视频
├── demo_0_success_True.hdf5     # 轨迹数据
```

---

## 完整示例：test_chemistry_beaker

### 涉及的修改

| 文件 | 操作 |
|------|------|
| `tasks/.../primitive/test_chemistry_beaker.py` | 新建 |
| `tasks/.../primitive/__init__.py` | 追加导入 |
| `configs/__init__.py` | 追加 name2config 条目 |
| `configs/task_config.json` | 追加配置 |

### 运行命令

```bash
MUJOCO_GL=egl python3 scripts/trajectory_generation.py \
    --task-name test_chemistry_beaker \
    --n-sample 1 \
    --debug \
    --save-dir /ssd/mkqin/workspace/VLABench/test_output
```

---

## 常见问题

### 1. `KeyError: 'my_task'`
任务未被注册。检查：
- `primitive/__init__.py` 是否导入了任务文件
- `@register.add_task("my_task")` 装饰器中的名字是否正确
- `configs/__init__.py` 的 `name2config` 中是否包含该任务

### 2. `KeyError: 'task'`（task_config.json 相关）
`task_config.json` 中的 key 必须与 `name2config` 中的 key（series 名）一致，不是任务名本身。例如 name2config 中是 `"my_task_series": ["my_task"]`，则 task_config.json 的 key 也必须是 `"my_task_series"`。

### 3. `ModuleNotFoundError` 导入失败
`primitive/__init__.py` 中的导入路径必须与实际文件名完全一致（不含 `_series` 后缀也行，取决于文件名）。

### 4. `gladLoadGL error`
缺少 `MUJOCO_GL=egl` 环境变量。

### 5. `mesh 'xxx' not found`
XML 中的 `<mesh>` 必须显式指定 `name` 属性（当 `<geom>` 通过 `mesh="name"` 引用时），否则 MuJoCo 会用文件名作为 mesh 名，可能导致名字不匹配。

**最后更新**: 2026-04-02
