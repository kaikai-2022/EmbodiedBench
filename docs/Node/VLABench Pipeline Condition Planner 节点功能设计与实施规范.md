# VLABench Pipeline: `Condition Planner` 节点功能设计与实施规范

## 1. 节点定位与职责 (Node Overview)

`Condition Planner`（条件规划器）是整个流水线的"成功标准制定者（Success Criteria Designer）"。

它承接上游 `Normalizer` 和 `Asset Manager` 的输出，为每个 step 生成物理可验证的成功条件（Success Condition），确保仿真不仅执行完技能序列，还能验证任务是否真正达到预期的物理状态。

**核心架构原则（必须严格遵守）：**

1. **Per-Step 粒度**：为每个 step 生成一个 condition，而非整个任务一个总条件。
2. **物理可验证**：condition 必须基于物理状态（位置、接触、包含关系等），而非语义推理。
3. **与 Skill Planner 并行**：两者同时执行（fan-in 架构），互不依赖，提高效率。
4. **容错设计**：如果 LLM 生成失败，返回 `null`，由 simulation 节点回退到旧的 fallback 机制。

**设计动机：**

原有的任务成功判定逻辑是"执行完所有技能即成功"，无法验证物理状态。例如：
- `pick` 技能执行完了，但物体没有被抓住
- `pour` 技能执行完了，但液体没有倒入目标容器
- `insert` 技能执行完了，但试管没有插入试管架

Condition Planner 通过为每个 step 生成明确的物理验证条件，确保任务真正成功。

---

## 2. 接口契约 (Interface Contract)

### 2.1 输入 (Input from State)

- `state["normalized_context"]["instances"]` (List): 实体列表（包含 UID）
- `state["normalized_context"]["steps"]` (List): 步骤列表（包含 action 和 UID）
- `state["asset_status"]` (Dict): 实体的物理信息（class_name, properties）

### 2.2 输出 (Output to State)

- `state["condition_plan"]` (List | None): 每个 step 的 condition 配置列表
- `state["messages"]` (List): LLM 交互记录
- `state["error_feedback"]` (String | None): 错误信息（如果生成失败）
- `state["current_stage"]` (String): 固定为 `"code_generator"`

---

## 3. 输出数据结构定义 (Condition Plan Schema)

`state["condition_plan"]` 是一个列表，每个元素对应一个 step 的 condition：

```json
[
  {
    "step_id": 0,
    "condition_type": "is_grasped",
    "params": {
      "entities": ["tube_0"],
      "robot": "robot"
    },
    "reasoning": "The step picks tube_0, so the end state should be that tube_0 is grasped by the robot"
  },
  {
    "step_id": 1,
    "condition_type": "contain",
    "params": {
      "container": "beaker_0",
      "entities": ["tube_0"]
    },
    "reasoning": "The step pours tube_0 into beaker_0, so the liquid should be contained in beaker_0"
  }
]
```

### 字段说明

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `step_id` | Integer | 对应 normalized_context.steps 中的 step 编号 | `0` |
| `condition_type` | String | Condition 类型（注册在 VLABench/tasks/condition.py） | `"is_grasped"`, `"contain"`, `"lift"` |
| `params` | Dict | Condition 的参数，key 为参数名，value 为 UID 或数值 | `{"entities": ["tube_0"], "robot": "robot"}` |
| `reasoning` | String | LLM 的推理过程（用于调试和审计） | `"The step picks..."` |

---

## 4. 可用 Condition 类型 (Available Condition Types)

Condition Planner 支持以下 condition 类型（注册在 `VLABench/tasks/condition.py`）：

| Condition Type | 参数 | 物理含义 | 适用场景 |
|----------------|------|----------|----------|
| `is_grasped` | `entities`, `robot` | 实体被机械臂抓取 | pick, grasp |
| `contain` | `container`, `entities` | 实体在容器内 | pour, place (in), insert |
| `not_contain` | `container`, `entities` | 实体不在容器内 | remove, take out |
| `on` | `entities`, `container` | 实体在容器表面上 | place (on) |
| `lift` | `entities`, `target_height` | 实体高度超过阈值 | lift, raise |
| `pour` | `target_entity`, `threshold` | 容器倾斜（top_site.z < bottom_site.z） | pour, tilt |
| `contact` | `entity1`, `entity2`, `robot` | 两实体接触 | touch, press |
| `pass` | 无 | 无物理验证，执行完即成功 | shake, wait, move |

**重要参数约定：**
- `is_grasped` 和 `contact` 必须包含 `"robot": "robot"` 参数
- `entities` 参数必须是列表（即使只有一个实体）
- UID 必须与 normalized_context 中的 UID 一致

---

## 5. 节点内部工作流 (Internal Data Pipeline)

### 工序 1：构建 Prompt

从 `normalized_context` 和 `asset_status` 中提取信息，构建 LLM prompt：

1. **实体列表**：列出所有物理实体的 UID、spec、class_name
2. **步骤列表**：列出每个 step 的 action、primary_uid、secondary_uid、grounded_instruction
3. **Condition 类型白皮书**：列出所有可用的 condition 类型及其参数
4. **Action → Condition 映射表**：提供参考（但 LLM 应基于语义理解选择）

### 工序 2：LLM 生成

调用 LLM 生成 JSON 格式的 condition_plan：

```python
llm = ChatAnthropic(**AgentConfig.get_llm_config())
response = llm.invoke(prompt)
condition_plan = _extract_json(response.content)
```

### 工序 3：校验与容错

1. **结构校验**：检查是否为 list，每个元素是否包含必需字段
2. **step_id 校验**：检查 step_id 是否连续且与 steps 数量匹配
3. **condition_type 校验**：检查是否为已注册的 condition 类型
4. **UID 校验**：检查 params 中的 UID 是否存在于 asset_status 中

如果校验失败，返回 `condition_plan: null` 和 `error_feedback`。

---

## 6. 与其他节点的协作 (Node Collaboration)

### 6.1 与 Skill Planner 的关系

- **并行执行**：两者同时执行（fan-in 架构），互不依赖
- **输入相同**：都读取 `normalized_context` 和 `asset_status`
- **输出独立**：Skill Planner 输出 `skill_plan`，Condition Planner 输出 `condition_plan`
- **汇聚点**：Code Generator 同时接收两者的输出

### 6.2 与 Simulation 的关系

- **Simulation 读取 condition_plan**：在每个 step 的 skills 执行完后，检查对应的 condition
- **Per-Step 检查**：Simulation 根据 `step_skill_ends` 判断何时检查 condition
- **Fallback 机制**：如果 `condition_plan` 为 `null`，Simulation 回退到旧的 `task.conditions.is_met()` 机制

---

## 7. 错误处理与容错 (Error Handling)

### 7.1 LLM 生成失败

- **现象**：LLM 返回非 JSON 格式，或 JSON 结构不符合预期
- **处理**：返回 `condition_plan: null`，`error_feedback` 记录错误信息
- **影响**：Simulation 回退到旧机制，不影响整体流程

### 7.2 UID 不存在

- **现象**：params 中的 UID 在 asset_status 中找不到
- **处理**：校验阶段捕获，返回 `condition_plan: null`
- **影响**：同上

### 7.3 Condition 类型未注册

- **现象**：`condition_type` 不在 `register._conditions` 中
- **处理**：Simulation 执行时捕获异常，记录 `error`，该 step 的 condition 判定为 `met: false`
- **影响**：任务判定失败，但不会崩溃

---

## 8. 设计权衡与未来扩展 (Design Trade-offs)

### 8.1 为什么是 Per-Step 而非 Per-Task？

- **粒度更细**：可以精确定位哪个 step 失败
- **调试友好**：日志中可以看到每个 step 的 condition 检查结果
- **复用性强**：同一个 action 在不同任务中可以复用相同的 condition

### 8.2 为什么与 Skill Planner 并行？

- **效率优化**：两者都需要调用 LLM，并行执行可节省时间
- **解耦设计**：两者职责独立，互不依赖

### 8.3 未来扩展方向

1. **Condition 组合**：支持 `and`、`or` 逻辑组合多个 condition
2. **时序 Condition**：支持 `asyn_sequence`（多阶段条件）
3. **自定义 Condition**：允许用户注册自定义 condition 类型
4. **Condition 优化**：基于历史数据优化 condition 选择策略

---

## 9. 实施检查清单 (Implementation Checklist)

- [x] 节点函数签名符合 LangGraph 规范（接收 `state: Dict`，返回 `Dict`）
- [x] 输出包含 `condition_plan`、`messages`、`error_feedback`、`current_stage`
- [x] LLM prompt 包含完整的 condition 类型白皮书
- [x] JSON 提取逻辑支持多种格式（纯 JSON、```json 代码块）
- [x] 校验逻辑覆盖结构、step_id、condition_type、UID
- [x] 容错机制：生成失败时返回 `null` 而非抛出异常
- [x] 日志记录：记录 LLM 交互和校验结果
- [x] 与 agent.py 集成：添加到 LangGraph 图中，与 skill_planner 并行
- [x] 与 simulation.py 集成：添加 per-step condition 检查逻辑
- [x] 测试覆盖：单步任务（pick）、多步任务（pick + pour + insert）

---

## 10. 参考资料 (References)

- **Condition 类定义**：`VLABench/tasks/condition.py`
- **Condition 注册机制**：`VLABench/utils/register.py`
- **Simulation 集成**：`scripts/vlabench_agent/nodes/simulation.py`
- **Agent 图定义**：`scripts/vlabench_agent/agent.py`
