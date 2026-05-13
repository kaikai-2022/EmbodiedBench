# Pipeline 节点职责划分与优化分析

> 本文档系统梳理当前 Pipeline 各节点的职责边界，分析职责混乱导致的 Bug，
> 并提出基于"节点间低耦合、节点内高内聚"原则的优化方向。

---

## 1. 当前 Pipeline 全貌

```
START → analyzer → asset_manager → skill_planner → code_generator → registration → simulation → vlm_data → END
                                                                 ↑                           |
                                                            (注册/仿真失败重试)           |
                                                                 +---------------------------+
```

---

## 2. 各节点职责详解

### 2.1 analyzer（任务分析）

**职责**：将自然语言指令解析为结构化的任务元信息。

**接收输入**：
- `state["user_instruction"]`：用户原始指令字符串

**内部工作**：
- 调用 LLM（ChatAnthropic）解析任务
- 输出 JSON：objects、scene、operation_type、instruction_en、spatial_relations、task_name

**输出**：
- `state["task_analysis"]`：包含 `objects`、`scene`、`operation_type`、`instruction_en`、`task_name` 的字典
- `state["current_stage"]` = `"asset_check"`（但实际在 agent.py 中是 `asset_manager`）

**问题**：
- objects 列表中允许出现**同名重复项**（如 `["beaker", "beaker"]`）。LLM 输出 JSON 时没有区分同名物体的语义角色（源/目标）。

---

### 2.2 asset_manager（资产管理）

**职责**：检查物体资产是否存在，下载缺失资产，填充 asset_status。

**接收输入**：
- `state["task_analysis"]["objects"]`：物体名称列表
- `state["errors"]`

**内部工作**：
- 遍历 objects，调用 `check_asset_exists()` 检查每个物体
- 记录 canonical_name（同义词映射后的名称）
- 对缺失资产调用 `download_asset()`
- 对下载的资产使用 `CommonGraspedEntity` 类，动态注册到 `name2class_xml`

**输出**：
- `state["asset_status"]`：dict，key=原始物体名，value={found, xml_path, class, canonical_name, newly_downloaded}
- `state["current_stage"]` = `"task_creation"`（注：agent.py 中实际期望的是 `"skill_planner"`，存在不一致）

**关键问题**：
- 对同名重复物体（如两个 `"beaker"`），只生成**一个** asset_status 条目（后一个覆盖前一个）
- 没有去重、没有区分语义角色
- 对下载的资产统一用 `CommonGraspedEntity`，忽略该物体的实际类型

---

### 2.3 skill_planner（技能规划）

**职责**：LLM 规划技能序列和成功条件。

**接收输入**：
- `state["task_analysis"]`：任务分析结果
- `state["asset_status"]`：资产状态
- `state["error_feedback"]`：上次规划失败的反馈

**内部工作**：
- 调用 LLM，根据 TASK_TYPE_PATTERNS 模板和资产信息生成 skill_sequence、conditions、task_type 等
- 校验生成的 JSON 格式和字段合法性

**输出**：
- `state["skill_plan"]`：dict，包含 task_type、skill_sequence、conditions、instruction_template、needs_container、moveto_target_expr

**已知问题**：
- pour 任务的 `conditions["pour"]["target_entity"]` 被设为容器名（如 `"beaker"`）而非源实体名（如 `"test_tube"`）
- 输出中的实体名是原始名（如 `"beaker"`），未转换为 canonical_name
- conditions 中的占位符使用 `self.target_container` / `self.target_entity`（Python 变量名语法），与 code_generator 的占位符格式不统一

---

### 2.4 code_generator（代码生成）

**职责**：根据 skill_plan、task_analysis、asset_status 模板生成任务 Python 代码。

**接收输入**：
- `state["skill_plan"]`
- `state["task_analysis"]`
- `state["asset_status"]`
- `state["code_generation_attempts"]`

**内部工作**：
1. 依赖注入：`infer_dependencies()` 检测需要注入的父容器（如 ChemistryTube → chemistry_tube_stand）
2. 角色分配：`_determine_entity_roles()` 将 objects[0] 设为 object，objects[1] 设为 container，objects[2+] 设为 init_container
3. 模板路由：三步命中（硬编码 subentity 映射 → MRO 继承链检测 → builtin）选择 load 方法模板
4. 生成代码：IMPORTS + CONFIG_MANAGER_TEMPLATE + TASK_CLASS_TEMPLATE
5. 校验并写入文件

**输出**：
- `state["generated_code"]`：生成的代码字符串
- `state["task_module_path"]`：文件路径
- `state["code_generation_attempts"]` + 1
- 失败时降级到 `_code_generator_legacy()`（旧 LLM 模式）

**关键问题**：
- 对同名重复物体只处理一次，导致**源实体未被加载**
- `_format_conditions_dict` 只处理 `"target_entity"` / `"target_container"` 占位符，不处理 `self.target_entity` / `self.target_container` 语法
- 降级重试时从 skill_planner 重新生成 skill_plan，而不是修复当前问题

---

### 2.5 registration（任务注册）

**职责**：动态注册生成的任务类到 VLABench 任务系统。

**接收输入**：
- `state["task_analysis"]`
- `state["asset_status"]`
- `state["task_module_path"]`

**内部工作**：
1. 清理旧的 task / config_manager 注册和模块缓存
2. 更新 `name2config`（内存）
3. 更新 `envs.TASK_CONFIG`（内存 + 持久化 task_config.json）
4. 动态 `importlib.import_module()` 触发 `@register` 装饰器
5. 验证注册成功

**关键逻辑**：构建 `series_config["task"]["asset"]` 时，根据 class 名中是否含 `"Container"` 或 `"Stand"` 判断容器。

**问题**：
- 用 class 名字符串匹配判断容器类型是不可靠的（如 `CommonGraspedEntity` 的 beaker 被当成普通 object 正确，但逻辑不清晰）
- canonical_name 推断依赖 class 名字符串匹配，扩展性差

---

### 2.6 simulation（物理仿真）

**职责**：在 MuJoCo 中执行技能序列，验证任务是否成功。

**接收输入**：
- `state["task_analysis"]`

**内部工作**：
1. 加载环境 `load_env(task_name)`
2. 获取 `episode_config`、target_entity、entities
3. 执行技能序列（带超时保护）
4. 检查任务条件 `conditions.is_met()`
5. 保存视频、HDF5 数据

**输出**：
- `state["simulation_success"]`
- `state["simulation_video_path"]`
- `state["simulation_hdf5_path"]`
- `state["episode_config"]`
- `state["executed_skill_sequence"]`
- `state["error_feedback"]`（失败时）

**问题**：
- 当 conditions 中实体名无效时，`conditions.is_met()` 会抛出 `AttributeError`，而非优雅地失败

---

### 2.7 vlm_data（VLM 评测数据）

**职责**：将仿真结果输出为 VLM 评测数据集。

**接收输入**：
- `state["task_analysis"]`
- `state["episode_config"]`
- `state["executed_skill_sequence"]`

**内部工作**：
1. 创建目录结构（env_config/、input/、output/）
2. 保存 env_config.json、instruction.txt、operation_sequence.json
3. 调用 render_vlm_dataset.py 渲染图像

**输出**：
- `state["env_config"]`
- `state["task_save_path"]`
- `state["rendered_images"]`
- `state["validation_report"]`
- `state["current_stage"]` = `"done"`

---

## 3. 各节点间数据流

```
user_instruction
    ↓
task_analysis = {
    "objects": ["beaker", "beaker"],      ← Analyzer 输出（重复）
    "task_name": "pour_liquid_from_beaker_to_beaker",
    ...
}
    ↓
asset_status = {
    "beaker": {found, xml_path, class, canonical_name},  ← Asset Manager 去重为一个条目
    "beaker": {...}  ← 后者覆盖前者，只剩一个 beaker！
}
    ↓
skill_plan = {
    "skill_sequence": [...],
    "conditions": {
        "pour": {"target_entity": "self.target_container"},   ← Skill Planner 格式问题
        "above": {"target_entity": "self.target_entity", "platform": "self.target_container"}
    },
    ...
}
    ↓
objects = ["beaker"]        ← Code Generator 只拿到一个 beaker
roles: beaker=object        ← objects[1] 根本没出现在列表中
→ 源 beaker 未被加载！
```

---

## 4. 职责混乱导致的 Bug 链路

### Bug: 两个同名 beaker 导致场景缺少实体

| 节点 | 问题 |
|---|---|
| **analyzer** | 输出 `objects = ["beaker", "beaker"]`，未区分语义角色 |
| **asset_manager** | 对重复 key 只生成一个条目，后一个覆盖前一个 |
| **code_generator** | objects 去重后，objects[1]（target_container 的 beaker）没有出现在任何 load 方法中 |
| **结果** | 场景中只有一个 beaker，要么是源实体，要么是目标实体，必然有一个缺失 |

### Bug: conditions 占位符格式不统一

| 节点 | 问题 |
|---|---|
| **skill_planner** | 输出 `"target_entity": "self.target_container"`（Python 变量语法） |
| **code_generator** | `_format_conditions_dict` 只处理 `"target_entity"` / `"target_container"`（字面量） |
| **结果** | conditions 中写入原始字符串 `self.target_container`，condition checker 查找名为 "self.target_container" 的实体 → None → AttributeError |

### Bug: current_stage 命名不一致

| 节点 | 输出 current_stage | agent.py 期望 |
|---|---|---|
| analyzer | （未设置，依赖默认值） | `"asset_check"`？`"asset_manager"`？实际代码是 `"asset_check"` 但 agent.py 中从 analyzer → asset_manager 是直接边 |
| asset_manager | `"task_creation"` | 实际期望是 `"skill_planner"`（直接边） |

---

## 5. 优化方向分析

### 5.1 同名实体去重与重命名（关键修复）

**问题根源**：Analyzer 是唯一理解任务语义的地方，但它只输出 JSON，不做语义区分。

**方案**：新增一个 **"语义归一化"节点**（或作为 analyzer 的一部分），专门处理：
1. 检测 objects 中的同名重复项
2. 根据 task_analysis 的语义（operation_type、instruction）分配角色
3. 输出去重且语义明确的 objects 列表，如：
   ```
   输入: objects=["beaker", "beaker"], operation_type="pour"
   输出: {"source_entity": "source_beaker", "target_entity": "target_beaker", "container": "target_beaker"}
   ```

**替代方案**：在 analyzer 的 prompt 中要求输出语义明确的名称（如 `"source_beaker"`, `"target_beaker"`），但这会让 Analyzer 做更多推理工作。

### 5.2 Conditions 格式规范化

**问题根源**：Skill Planner 输出的是 Python 运行时变量名（`self.target_container`），Code Generator 期望的是字面占位符（`"target_entity"`）。

**方案**：
- **方案 A（推荐）**：在 Skill Planner 输出后、Code Generator 处理前，增加一个 **"格式归一化"步骤**，将 `self.target_entity` → `target_entity`, `self.target_container` → `target_container`
- **方案 B**：统一 Skill Planner 的 prompt，要求输出字面占位符而非 Python 变量名
- **方案 C**：在 Code Generator 中同时支持两种格式

方案 A 更好，因为它是节点间的契约约束点，不需要修改两个节点各自的核心逻辑。

### 5.3 Asset Manager 的去重职责

**问题**：asset_manager 对 `["beaker", "beaker"]` 只生成一个 asset_status 条目。

**方案**：在 asset_manager 中，对重复物体名生成**独立的资产条目**，使用下标区分：
```python
asset_status = {
    "beaker_0": {found, xml_path, class, canonical_name="beaker", index=0},
    "beaker_1": {found, xml_path, class, canonical_name="beaker", index=1},
}
```
然后 code_generator 根据 index 生成不同的 load 方法。

### 5.4 职责重新划分建议

**当前问题**：
- analyzer 同时负责"语义理解"和"数据结构输出"
- asset_manager 同时负责"资产检查"和"数据整理"
- code_generator 承担了"实体去重"（失败）和"模板选择"两份职责

**建议的结构**：

```
analyzer           → 只负责语义理解，输出原始 objects
       ↓
normalizer         → 新增：处理同名实体去重、角色分配、canonical name 映射
       ↓
asset_manager      → 接收规范化后的 objects，只负责资产检查和下载
       ↓
skill_planner      → 接收规范化的 asset_status，输出规范化的 conditions
       ↓
code_generator     → 接收规范化数据，只负责模板填充
       ↓
registration       → 只负责注册
       ↓
simulation         → 只负责仿真
       ↓
vlm_data           → 只负责数据输出
```

### 5.5 节点边界清洗

| 问题 | 解决方案 |
|---|---|
| `current_stage` 命名混乱 | 统一用节点函数名作为 stage 值，agent.py 中路由直接用函数名匹配 |
| 降级重试链路不清晰 | 实现精准回溯（Targeted Backtracking），见下方 error_classifier |
| error_feedback 携带信息过多 | 将 error_feedback 按错误类型分类，由 error_classifier 分发 |

### 5.6 精准回溯（Error Classifier）

```python
def error_classifier(state):
    error = state.get("error_feedback", "")
    # 语法/属性错误 → code_generator 的锅
    if any(k in error for k in ["SyntaxError", "AttributeError", "TypeError", "NameError"]):
        return "code_generator"
    # 资产缺失 → asset_manager 的锅
    if any(k in error for k in ["asset", "xml", "download", "texture"]):
        return "asset_manager"
    # 物理/条件失败 → skill_planner 的锅
    if any(k in error for k in ["collision", "condition", "timeout", "skill failed"]):
        return "skill_planner"
    # 未知错误 → 从 skill_planner 重试（安全回退）
    return "skill_planner"
```

### 5.7 Context_Map 与表达式映射

**原则**：code_generator 不硬编码 Python 表达式。所有运行时表达式通过 DSL 占位符映射表解析。

```python
# 在 skill_planner 的 prompt 中明确给出表达式模板：
# - $CONTAINER_POS_EXPR → "np.array(self.entities[self.target_container].get_xpos(physics)) + np.array([0, 0, 0.2])"
# - $RANDOM_WORKSPACE_POS → "np.array([random.uniform(-0.15, 0.15), random.uniform(-0.15, 0.15), 0.85])"

# 在 code_generator 中维护映射表：
DSL_EXPRESSION_MAP = {
    "$CONTAINER_POS_EXPR": "np.array(self.entities[self.target_container].get_xpos(physics)) + np.array([0, 0, 0.2])",
    "$RANDOM_WORKSPACE_POS": "np.array([random.uniform(-0.15, 0.15), random.uniform(-0.15, 0.15), 0.85])",
}
```

code_generator 只做 `text.replace("$DSL_TOKEN", python_expr)`，不包含任何推理逻辑。

### 5.8 Normalizer 兜底逻辑

当 Analyzer 输出的 `role_hint` 缺失时，按 operation_type 默认规则分配：

```python
FALLBACK_ROLE_RULES = {
    "pour":  [("object", 0), ("container", 1)],
    "place": [("object", 0), ("container", 1)],
    "lift":  [("object", 0)],
    "pick":  [("object", 0)],
    "move":  [("object", 0)],
    "heat":  [("object", 0), ("container", 1)],
}
```

兜底触发时在 `state["warnings"]` 中追加记录，便于调试追踪。

### 5.9 Asset Manager 下载去重

同 type 的多个 instance（如 beaker_source 和 beaker_target）只下载一次：

```python
# asset_manager 内部逻辑：
seen_types = set()          # 已处理的 canonical_name
for instance in instances:
    canonical = instance["canonical_name"]
    if canonical in seen_types:
        # 复制已有条目，只改 key 为当前 instance_id
        asset_status[instance["instance_id"]] = dict(asset_status[first_instance_of_same_type])
        continue
    # 首次遇到该 type，执行检查/下载
    status = check_asset_exists(canonical)
    asset_status[instance["instance_id"]] = status
    if not status["found"]:
        download_result = download_asset(canonical)
        ...
    seen_types.add(canonical)
```

---

## 6. 核心优化建议优先级

### P0（必须修复，阻塞流程）

1. **新建 normalizer 节点：实体实例化 + 角色绑定 + 同义词映射**
   - 含兜底逻辑（role_hint 缺失时按 operation_type 默认规则分配）

2. **Conditions DSL 占位符统一**
   - Skill Planner 输出 `$TARGET_ENTITY` / `$TARGET_CONTAINER`
   - Code Generator 通过 DSL_EXPRESSION_MAP 映射为 Python 表达式

3. **State Schema 强化**
   - 新增 `task_graph` 字段
   - TypedDict 或 Pydantic 模型约束

### P1（重要，提升鲁棒性）

4. **Asset Manager 接收 instances + 下载去重**

5. **精准回溯（Error Classifier）**
   - 语法错误 → code_generator
   - 资产缺失 → asset_manager
   - 物理/条件失败 → skill_planner

6. **code_generator 瘦身**
   - 移除 `_determine_entity_roles()`
   - 移除所有占位符猜测逻辑
   - 只保留模板填充 + DSL 替换

### P2（长期优化）

7. **规范 current_stage 命名**
8. **注册节点从 task_graph 读取容器分类（不再用 class 名猜测）**
