# VLABench Pipeline: `Code Generator` 节点功能设计与实施规范

## 1. 节点定位与职责 (Node Overview)

`Code Generator` 节点是整个 VLABench Pipeline 的"确定性代码合成器（Deterministic Code Synthesizer）"。

它承接 `Skill Planner` 输出的 `global_skill_plan` 和 `Normalizer` 的 `normalized_context`，通过纯模板填充的方式生成一个完整的 Python 任务文件（`<task_name>_series.py`），写入 VLABench 的 primitive 任务目录。

**核心设计原则：**

1. **确定性（Deterministic）**：相同的输入必定生成完全相同的代码，不依赖任何 LLM 调用。
2. **Fail-Fast**：模板填充失败即报错，不降级、不尝试修复。
3. **UID 贯穿**：entity name = uid，Skill 参数中的 uid 直接写入代码，不替换为 spec。
4. **执行完即成功**：生成的 `get_condition_config` 默认为 `pass`（无条件），任务以技能序列执行完毕为成功。

---

## 2. 接口契约 (Interface Contract)

### 2.1 输入 (Input from State)

| 字段 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `state["normalized_context"]` | Dict | Normalizer | 包含 `instances` 和 `steps` |
| `state["asset_status"]` | Dict | Asset Manager | uid → `{class_name, xml_path, properties}` |
| `state["skill_plan"]` | Dict | Skill Planner | 包含 `global_skill_plan` 技能序列 |
| `state["task_analysis"]` | Dict | Analyzer | 包含 `task_name`、`scene` 等元信息 |

### 2.2 输出 (Output to State)

| 字段 | 类型 | 说明 |
|------|------|------|
| `state["generated_code"]` | String | 生成的完整 Python 代码文本 |
| `state["task_module_path"]` | String | 写入的任务文件路径 |
| `state["current_stage"]` | String | 固定为 `"registration"` |

---

## 3. 生成的文件结构

生成文件路径：`$VLABENCH_ROOT/tasks/hierarchical_tasks/primitive/<task_name>_series.py`

文件由以下四个部分拼接而成：

```
[IMPORTS_BLOCK]        # 固定导入头
[extra_imports]        # 按需追加（如 name2class_xml）
[extra_constants]      # 按需追加（如 relative_col_pos 试管坐标常量）
[ConfigManager 类]    # 继承 BenchTaskConfigManager，包含 load_* 方法
[Task 类]             # 继承 PrimitiveTask，包含 get_expert_skill_sequence
```

### 3.1 ConfigManager 类结构

```python
@register.add_config_manager("<task_name>")
class <ClassPrefix>ConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=[1, 1], **kwargs): ...
    def load_containers(self, target_container): ...      # 仅有容器实体时生成
    def load_init_containers(self, init_container): ...  # 仅有 subentity 时生成
    def load_objects(self, target_entity): ...            # 始终生成
    def get_instruction(self, target_entity, ...): ...
    def get_condition_config(self, target_entity, ...): pass
    def get_target_entity(self): return "<target_uid>"
```

### 3.2 Task 类结构

```python
@register.add_task("<task_name>")
class <ClassPrefix>Task(PrimitiveTask):
    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.<skill>, <params>),
            ...
        ]
        return skill_sequence
```

---

## 4. 核心子模块 (Sub-modules)

### 4.1 Entity Loader (`entity_loader.py`)

负责将 `instances` 列表转换为结构化的加载计划（`EntityLoadPlan`），再生成 `load_*` 方法代码。

**加载模式分类：**

| 模式 | 触发条件 | 对应方法 | 说明 |
|------|----------|----------|------|
| `plain` | 普通可操作实体 | `load_objects` | 随机位置放置 |
| `liquid` | properties 含 `solution` | `load_objects` | 附带溶液参数（ChemistryTube） |
| `subentity` | class_name == `ChemistryTube` | `load_objects` | 作为 TubeStand 的子实体，使用相对孔位坐标 |
| container | `_is_container_class()` 为 True | `load_containers` | 单独生成容器加载方法 |
| init_container | subentity 的父容器 | `load_init_containers` | 自动注入 TubeStand 等父容器 |

**加载顺序（框架要求）：**

框架调用顺序固定为 `load_containers → load_init_containers → load_objects`，对应 `components` 数组顺序为 `[table, beaker, tube_stand, ...tubes]`。

**SubEntity 自动注入规则：**

当存在 `ChemistryTube` 时，自动为其注入 `chemistry_tube_stand` 父容器。扩展支持新的 subentity 类型需在 `DEFAULT_PARENT_CONTAINERS` 映射中添加条目：

```python
DEFAULT_PARENT_CONTAINERS = {
    "ChemistryTube": ("chemistry_tube_stand", "TubeStand"),
    # TODO: "ChemistryFlask": ("flask_rack", "FlaskRack"),
}
```

**UID 一致性保证：**

父容器的 UID 必须与 `asset_status` 中的 key 一致。`_gen_init_containers()` 使用 `plan.uid`（而非 `parent_spec`）作为容器名，确保生成的代码中的 entity name 与 skill_plan 中的 UID 匹配。

例如：
- `asset_status` 中的 key：`chemistry_tube_stand_0`
- 生成的代码：`name="chemistry_tube_stand_0"`（使用 UID）
- skill_plan 中的 target：`insert_to_entity(target_entity_name="chemistry_tube_stand_0")`

**试管孔位常量（世界坐标系，已考虑试管架 90° 旋转）：**

```python
TUBE_COL_POS = [-0.16, -0.08, 0, 0.08, 0.16]  # 世界 X 轴方向（5列）
TUBE_ROW_POS = [-0.05, 0.05]                    # 世界 Y 轴方向（2行）
```

### 4.2 Skill Formatter (`skill_formatter.py`)

负责将 `global_skill_plan` 中的 `atomic_sequence` 转换为 `partial(SkillLib.xxx, ...)` 代码行。

**参数名映射规则（`target_uid` → 实际参数名）：**

| 技能 | 参数名 |
|------|--------|
| `place`, `pour_to_entity` | `target_container_name` |
| 其他（`pick`, `insert_to_entity` 等） | `target_entity_name` |

**合法技能白名单（VALID_SKILLS）：**

```
pick, place, lift, moveto, moveto_entity, pour, pour_to_entity,
insert_to_entity, push, press, flip, wait, rotate,
open_gripper, close_gripper, open_door, close_door, open_drawer,
open_laptop, move_offset, reset, shake
```

---

## 5. 代码校验 (Validation)

生成后执行 `_validate_code()` 检查：

1. 含 `@register.add_task` 装饰器
2. 含 `@register.add_config_manager` 装饰器
3. 含 `get_expert_skill_sequence` 方法
4. 继承 `PrimitiveTask`
5. 继承 `BenchTaskConfigManager`
6. Python 语法无误（`compile()` 校验）

校验失败直接返回错误，不写文件。

---

## 6. 常见问题与扩展指引

### 6.1 添加新的 subentity 类型

在 `entity_loader.py` 的 `DEFAULT_PARENT_CONTAINERS` 中添加映射，并在 `_code_subentity` 函数中处理对应的布局常量。

### 6.2 添加新的容器类型

在 `entity_loader.py` 的 `_is_container_class()` 和 `registration.py` 的 `CONTAINER_CLASSES` 中同步添加新的 class_name。

### 6.3 技能参数名扩展

在 `skill_formatter.py` 的 `_format_params()` 中添加 `target_uid` 的新映射规则（`skill_name in (...)` 条件）。
