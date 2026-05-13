# VLABench Pipeline: `Registration` 节点功能设计与实施规范

## 1. 节点定位与职责 (Node Overview)

`Registration` 节点是整个 VLABench Pipeline 的"动态任务注册器（Dynamic Task Registrar）"。

它承接 `Code Generator` 写入的任务 Python 文件，通过动态 `importlib` 导入触发 `@register` 装饰器，将任务类和配置管理器注册到 VLABench 的全局注册表中，并同步更新内存中的 `name2config` 和 `TASK_CONFIG`，使 `load_env()` 能够找到该任务。

**核心职责：**

1. **清理旧注册**：重试场景下清除上一次遗留的模块缓存和注册条目，防止污染。
2. **双重 name2config 更新**：同时更新 `VLABench.configs` 和 `VLABench.envs` 中的 `name2config`，避免 import 顺序导致的不一致。
3. **TASK_CONFIG 内存+持久化**：更新运行时内存中的 `TASK_CONFIG`，并将新条目写入 `task_config.json`（已存在的条目不覆盖）。
4. **注册验证**：确认 `register._tasks` 和 `register._config_managers` 中都存在新注册的任务。

---

## 2. 接口契约 (Interface Contract)

### 2.1 输入 (Input from State)

| 字段 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `state["task_module_path"]` | String | Code Generator | 生成的任务 `.py` 文件路径 |
| `state["task_analysis"]` | Dict | Analyzer | 包含 `task_name`、`scene` |
| `state["normalized_context"]` | Dict | Normalizer | 包含 `instances`，用于推断容器/物体分类 |
| `state["asset_status"]` | Dict | Asset Manager | uid → `{class_name, ...}`，用于推断容器类 |

### 2.2 输出 (Output to State)

| 字段 | 类型 | 说明 |
|------|------|------|
| `state["registration_success"]` | Bool | 注册是否成功 |
| `state["error_feedback"]` | String/None | 失败原因（成功时为 None） |
| `state["current_stage"]` | String | 成功时为 `"simulation"`，失败时为 `"registration"` |

---

## 3. 核心实现逻辑 (Implementation Guidelines)

### 3.1 注册流程

```
1. 清理旧注册（register._tasks, register._config_managers, sys.modules）
2. 更新内存 name2config（configs 模块 + envs 模块）
3. 构造 series_config 并更新内存 TASK_CONFIG
4. 将 series_config 写入 task_config.json（仅新增，不覆盖已有条目）
5. importlib.import_module(module_name) 触发 @register 装饰器
6. 验证 register._tasks 和 register._config_managers 均包含新任务
```

### 3.2 series_config 结构

```python
series_config = {
    "task": {
        "asset": {
            "seen_object": [...],    # 来自 asset_status 中的非容器实体 spec
            "unseen_object": [...],
            "seen_container": [...], # 仅有容器时存在
            "unseen_container": [...],
        },
        "scene": {"name": "<scene>_0"},
        "components": [
            {"name": "table", "xml_path": "...", "class": "Table", ...}
        ],
    }
}
```

### 3.3 容器类判断（CONTAINER_CLASSES 白名单）

```python
CONTAINER_CLASSES = {
    "CommonContainer", "ContainerWithDoor", "ContainerWithDrawer",
    "FlatContainer", "Fridge", "Microwave", "Shelf",
    "TubeStand", "Vase", "Plate", "Mug",
    "ChemistryBeaker", "ChemistryFlask", "ChemistryBottle",
}
```

判断规则：`class_name in CONTAINER_CLASSES or "Stand" in class_name`。

**扩展新容器类型时**，需要同时更新 `CONTAINER_CLASSES`（此处）和 `entity_loader.py` 的 `_is_container_class()`，保持两处同步。

---

## 4. 重要注意事项

### 4.1 `load_env()` 的 KeyError 防御

`VLABench/envs/__init__.py` 第 52 行存在如下判断：

```python
if default_config.get('task') and default_config['task'].get("random_init", None) is not None:
```

（已修复为 `get('task')`，避免 `TASK_CONFIG["default"]` 被 `update()` 污染后不含 `task` key 时的 KeyError）

### 4.2 `configs/__init__.py` 的静态映射

`VLABench/configs/__init__.py` 中的 `name2config` 字典在模块加载时就固化，动态生成的任务需要在 Registration 节点中通过内存注入 `configs_name2config[series_name] = [task_name]` 来使其生效，无需修改文件。

对于框架内置任务，需要在 `configs/__init__.py` 中手动添加静态映射条目。

---

## 5. 错误处理

注册失败时返回 `current_stage: "registration"`，允许上游 Pipeline 编排器（如 LangGraph）在收到错误反馈后触发重试，从 `skill_planner` 或 `code_generator` 阶段重新生成。

完整的错误 traceback 包含在 `error_feedback` 中，供 LLM 分析失败原因。
