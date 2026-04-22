- [ ] # VLABench Pipeline: `Analyzer` 节点功能设计与实施规范

  ## 1. 节点定位与职责 (Node Overview)

  `Analyzer` 节点是整个 VLABench Pipeline 的“自然语言理解前端（NLU Frontend）”。

  它扮演**“文本切分与实体绑定器（Text Segmenter & Entity Binder）”**的角色，负责将用户输入的自然语言指令或论文操作片段，无损解析为带有实体引用的**神经符号学抽象语法树（Neuro-Symbolic AST）**。

  **🚨 核心边界（必须严格遵守）：**

  1. **只提取显式意图，不脑补隐式动作**：只切分并提取原文本中写明的动作。绝对不要去脑补前置的（如 `pick`）或后置的（如 `place`）隐含步骤，将其全部留给下游 `Skill Planner` 处理。
  2. **只提取原始动词，不做意图分类**：不再使用预设的动作库。直接提取句子中最能表达物理动作的核心原始动词（如 `remove`, `shake`, `pour`）。
  3. **只提取语义状态，不推断物理角色**：不要为物体分配 `source` 或 `target` 标签，角色的流转由下游节点推断。

  ------

  ## 2. 接口契约 (Interface Contract)

  ### 2.1 输入 (Input from State)

  - `state["user_instruction"]` (String): 包含实验步骤的原始自然语言文本（单步指令或长程多步段落）。

  ### 2.2 输出 (Output to State)

  - `state["task_analysis"]` (Dict): 包含解析后的实体列表和接地动作序列。
  - `state["current_stage"]` (String): 必须设置为 `"normalizer"`。

  ------

  ## 3. 输出数据结构定义 (AST Schema)

  `state["task_analysis"]` 必须严格包含以下两个核心字段：`raw_entities` 和 `raw_steps`。

  ### 3.1 `raw_entities` (List of Dicts)

  场景中所有被提及的物理实体、仪器、材料（包含容器、工具、甚至是文本中隐含的液体/粉末）。

  | **字段名**            | **类型** | **说明**                                                     | **示例**                      |
  | --------------------- | -------- | ------------------------------------------------------------ | ----------------------------- |
  | `raw_id`              | String   | LLM 生成的局部唯一标识符，格式必须为 `名词_数字`。           | `"vial_1"`, `"dispenser_1"`   |
  | `raw_type`            | String   | 提取自原文的精确名词词组。                                   | `"10 mL glass vial"`          |
  | `semantic_attributes` | Dict     | **开放属性包**。仅提取原文中明确提及的状态、颜色、体积、开关等。**禁止脑补，没有就不填 `{}`。** | `{"initial_state": "sealed"}` |

  ### 3.2 `raw_steps` (List of Dicts)

  完全拥抱 LLM-Native 的**接地指令（Grounded Instructions）**序列。

  | **字段名**             | **类型**    | **说明**                                                     | **示例**                                     |
  | ---------------------- | ----------- | ------------------------------------------------------------ | -------------------------------------------- |
  | `step_id`              | Integer     | 从 0 开始的递增序号。                                        | `0`, `1`, `2`                                |
  | `action`               | String      | **原始动词 (Raw Verb)**。直接提取自然语言中的核心动词（建议用原形）。 | `"remove"`, `"shake"`, `"pour"`              |
  | `primary_obj`          | String      | 动作的主体操作对象（直接宾语），必须是 `raw_entities` 中的 `raw_id`。 | `"cap_1"`                                    |
  | `secondary_obj`        | String/Null | 动作的客体、目标容器或参考物（状语），必须是 `raw_entities` 中的 `raw_id`。无则填 `null`。 | `"vial_1"`                                   |
  | `grounded_instruction` | String      | **实体接地指令**。保留原始句型和副词，但必须将名词严格替换为带有 `< >` 包裹的 `<raw_id>`。 | `"The robot removes <cap_1> from <vial_1>."` |

  ------

  ## 4. 核心实现逻辑与 Prompt 约束 (Implementation Guidelines)

  请按照以下逻辑构建给 LLM 的 System Prompt：

  1. **身份设定**：你是一个化学实验文本的“神经符号解析器（Neuro-Symbolic Parser）”。
  2. **任务 1（盘点登记 Entity Extraction）**：识别文本中的所有物理实体，赋予 `raw_id`，并将文本中明确提及的状态修饰语放入 `semantic_attributes`。
  3. **任务 2（文本重写 Instruction Grounding）**：将实验过程拆解为单步。**你必须将原始句子中的名词替换为你生成的 `<raw_id>`（使用尖括号包裹）**，并保留所有的副词和动作细节，输出为 `grounded_instruction`。
  4. **任务 3（提取动作 Raw Action）**：提取每一步的**核心原始动词**填入 `action` 字段（如 'pour', 'shake', 'remove'）。禁止进行任何维度的自我总结或意图分类。
  5. **约束**：强制输出符合上述 Schema 的严格 JSON。

  ------

  ## 5. 预期 I/O 示例 (Expected Example)

  **Input (user_instruction):**

  > "The robot removes the cap from the vial and aggressively places the vial under the solid dispenser. Then the robot aggressively shakes the vial for 10 seconds."

  **Output (state["task_analysis"]):**

  JSON

  ```
  {
    "raw_entities": [
      {
        "raw_id": "vial_1",
        "raw_type": "vial",
        "semantic_attributes": {}
      },
      {
        "raw_id": "cap_1",
        "raw_type": "cap",
        "semantic_attributes": {}
      },
      {
        "raw_id": "dispenser_1",
        "raw_type": "solid dispenser",
        "semantic_attributes": {}
      }
    ],
    "raw_steps": [
      {
        "step_id": 0,
        "action": "removes",
        "primary_obj": "cap_1",
        "secondary_obj": "vial_1",
        "grounded_instruction": "The robot removes <cap_1> from <vial_1>."
      },
      {
        "step_id": 1,
        "action": "places",
        "primary_obj": "vial_1",
        "secondary_obj": "dispenser_1",
        "grounded_instruction": "and aggressively places <vial_1> under <dispenser_1>."
      },
      {
        "step_id": 2,
        "action": "shakes",
        "primary_obj": "vial_1",
        "secondary_obj": null,
        "grounded_instruction": "Then the robot aggressively shakes <vial_1> for 10 seconds."
      }
    ]
  }
  ```

  ## 6. 开发实施清单 (Implementation Checklist)

  - [ ] 更新 `state.py` 中的 `task_analysis` 字段类型注释（如果使用了 TypedDict），移除已被弃用的 `params` 等字段。
  - [ ] 编写 `nodes/analyzer.py`，使用如 Langchain/Instructor 等库保证 LLM 输出严格符合 JSON Schema。
  - [ ] 确保 LLM 的 Prompt 中包含关于 `<raw_id>` **尖括号实体替换（Entity Grounding）**的明确 Few-shot 示例。
  - [ ] 确保节点返回的 `current_stage` 变更为 `"normalizer"`。