# VLABench Pipeline: `Skill Planner` 节点功能设计与实施规范

## 1. 节点定位与职责 (Node Overview)

`Skill Planner`（技能规划器）是整个流水线的"多步长程编排引擎（Long-Horizon Choreography Engine）"。 它承接了带有明确物理指向的剧本（`steps`）和完整的片场环境字典（`asset_status`），通过阅读系统预设的底层能力白皮书（`SKILL_LIB_DOC`），自主决定如何将宏观语义指令拆解为可执行的原子机械臂动作序列。

**🚨 核心架构原则（必须严格遵守）：**

1. **状态驱动的全局单次规划 (Stateful One-Shot)**：废除旧版的逐句循环。通过在 JSON 中强制引入 `pre_state_assertion` 和 `post_state_assertion`，逼迫大模型利用自回归机制（Attention）自行追踪机器人的抓手状态和环境变化。
2. **消灭硬编码 (Zero Hardcoded Rules)**：废除旧版中类似 `pour 必须包含 pick -> lift -> moveto -> pour` 的硬编码规则。完全信任并要求大模型根据实时的 `pre_state` 自主决策序列（例如：如果手里已经拿着瓶子，就不应该再次生成 `pick`）。
3. **显式传参，废弃宏替换**：废除旧版的 `$TARGET_ENTITY` 占位符玩法。大模型必须直接输出精确的 `uid`（如 `vial_0`），并取消旧版的 `conditions` 字典，将其意图融合进动作参数中。

------

## 2. 接口契约 (Interface Contract)

### 2.1 输入 (Input from State)

- `state["normalized_context"]["steps"]` (List): 宏观步骤剧本。
- `state["asset_status"]` (Dict): 当前物理实例的详细属性（包含类名和物理能力）。

### 2.2 输出 (Output to State)

- `state["skill_plan"]` (Dict): 包含全局详细原子序列的字典。
- `state["current_stage"]` (String): 变更为 `"code_generator"`。

------

## 3. 输出数据结构定义 (Skill Plan Schema)

必须强制大模型输出符合以下结构的严格 JSON：

JSON

```
{
  "global_skill_plan": [
    {
      "step_id": 0,             // 严格对应输入的 step_id
      "semantic_instruction": "", // 原始的重接地自然语言指令

      // 🌟 核心状态追踪 (CoT) 🌟
      "pre_state_assertion": "描述执行此步骤前，机械臂抓手里是否有东西？物品的开关状态如何？",

      "atomic_sequence": [      // 原子动作序列
        {
          "skill": "moveto",
          "params": {"target_uid": "beaker_0"}
        },
        // ... 其他动作
      ],

      // 🌟 核心状态追踪 (CoT) 🌟
      "post_state_assertion": "描述执行完此步骤后，机械臂抓手里的状态和物品位置改变。"
    }
    // ... 依序包含所有的 steps
  ]
}
```

------

## 4. 节点内部工作流与 Prompt 组装策略

相比于前置节点的 Python 代码逻辑，`Skill Planner` 的核心工作流在于**"组装一个完美的 Mega-Prompt"**并进行结果校验。

### 🛠️ 工序 1：组装运行时上下文 (Context Assembly)

在 Python 代码中，提取并格式化以下信息供大模型阅读：

1. **可用原子技能文档 (`SKILL_LIB_DOC`)**：明确列出底层支持的动作及其必填参数（如 `pick(target_uid)`, `pour(source_uid, target_container_uid)`, `place_on_table(target_uid)` 等）。
2. **物理资产清单**：遍历 `state["asset_status"]`，告诉大模型现在桌子上有哪些实体，它们分别属于什么类（如 `ChemistryBeaker` 能装水）。
3. **待执行剧本**：传入完整的 `state["normalized_context"]["steps"]`。

### 🛠️ 工序 2：大模型 One-Shot 调用

调用 LLM (建议使用具备强大长上下文和 JSON 遵循能力的模型)，并在 System Prompt 中重点强调**思维链约束（CoT Constraint）**：

> "你必须在规划每一个 step 的 `atomic_sequence` 之前，在 `pre_state_assertion` 中推演上一步结束后的机械臂状态。如果你的抓手里已经拿着目标物品，请不要重复生成 `pick` 动作；如果在抓取新物品前你的抓手不是空的，你必须先生成一个 `place` 动作。"

### 🛠️ 工序 3：JSON 结构与参数校验 (Validation)

LLM 返回 JSON 后，执行轻量级的结构校验：

1. 校验 `step_id` 是否连续且覆盖了所有输入的 `steps`。
2. 校验 `atomic_sequence` 中的每一个 `skill` 是否存在于 `SKILL_LIB_DOC` 的白名单中。
3. 校验参数中的 `uid` 是否存在于 `asset_status` 中（防止 LLM 幻觉生成不存在的物品 ID）。

------

## 5. 预期 I/O 示例 (Expected Example)

**Input (Context Extract):** 在此示例中，输入步骤（steps）已包含由 `Normalizer` 预处理后的完整结构化信息，为规划提供双重约束。

JSON

```
{
  "asset_status": {
    "pump_0": {"class_name": "CommonGraspedEntity"},
    "vial_0": {"class_name": "ChemistryTube", "properties": {"is_container": true}}
  },
  "steps": [
    {
      "step_id": 0,
      "action": "grab",
      "primary_uid": "pump_0",
      "secondary_uid": null,
      "grounded_instruction": "The robot grabs <pump_0>"
    },
    {
      "step_id": 1,
      "action": "transfer",
      "primary_uid": "pump_0",
      "secondary_uid": "vial_0",
      "grounded_instruction": "and uses <pump_0> to transfer liquid into <vial_0>"
    }
  ]
}
```

**Output (state["skill_plan"]):** 大模型参考 `action` 和 `uid` 字段，结合 `grounded_instruction` 语义，生成带有状态断言的序列。

JSON

```
{
  "global_skill_plan": [
    {
      "step_id": 0,
      "semantic_instruction": "The robot grabs <pump_0>",
      "pre_state_assertion": "The robot gripper is currently empty. <pump_0> is on the table.",
      "atomic_sequence": [
        {"skill": "moveto", "params": {"target_uid": "pump_0"}},
        {"skill": "pick", "params": {"target_uid": "pump_0"}}
      ],
      "post_state_assertion": "The robot is now holding <pump_0>."
    },
    {
      "step_id": 1,
      "semantic_instruction": "and uses <pump_0> to transfer liquid into <vial_0>",
      "pre_state_assertion": "The robot is ALREADY HOLDING <pump_0> from the previous step. It does not need to pick it up again. Current target is <vial_0>.",
      "atomic_sequence": [
        {"skill": "moveto", "params": {"target_uid": "vial_0"}},
        {"skill": "dispense", "params": {"source_tool_uid": "pump_0", "target_container_uid": "vial_0"}}
      ],
      "post_state_assertion": "Liquid is transferred to <vial_0>. The robot is still holding <pump_0>."
    }
  ]
}
```

## 6. 开发实施清单 (Implementation Checklist)

- [ ] 彻底删除旧版代码中关于 `$TARGET_ENTITY` 模板替换和硬编码动作序列的冗余逻辑。
- [ ] 彻底删除旧版 JSON 结构中的 `conditions` 字段，确保所有目标指向全部集成到 `params` 中。
- [ ] 在 `nodes/skill_planner.py` 中编写包含资产状态、步骤列表和 `SKILL_LIB_DOC` 的 Prompt 组装器。
- [ ] 使用 Pydantic 或强类型的 Schema 控制 LLM 输出，确保包含 `pre_state_assertion` 和 `post_state_assertion`。
- [ ] 编写轻量级的校验器 `validate_plan()`，校验生成的 `uid` 是否都在当前环境的合法列表中。
