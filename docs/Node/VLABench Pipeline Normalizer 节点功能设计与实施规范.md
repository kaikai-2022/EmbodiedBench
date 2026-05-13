# VLABench Pipeline: `Normalizer` 节点功能设计与实施规范

## 1. 节点定位与职责 (Node Overview)

`Normalizer` 节点是整个 VLABench Pipeline 的“语义到物理的网关（Semantic-to-Physical Gateway）”或“符号链接器（Symbol Linker）”。

它承接前端 `Analyzer` 提取的“生肉（Raw AST）”，负责清洗脏数据、拦截非刚体、颁发全局 UID，并输出一张严谨的**全局任务状态图（Global Task Graph / Normalized Context）**。

**🚨 核心边界（必须严格遵守）：**

1. **只发指令，不干苦力**：负责决定资产去哪里找（本地或 Objaverse），但**绝对不能**在节点内执行任何网络下载或文件 I/O 操作（这是下游 Asset Manager 的活）。
2. **只给语义标签，不猜物理数值**：将物质描述转化为 `contains_substance` 的语义标签，**绝对不能**让 LLM 凭空猜测 RGB 颜色、密度、摩擦系数等底层物理数值。
3. **不篡改动词，只替换名词**：在处理指令文本时，只能通过正则替换名词的 `<ID>` 占位符，绝对不能修改原句的动词或句式结构。

------

## 2. 接口契约 (Interface Contract)

### 2.1 输入 (Input from State)

- `state["task_analysis"]` (Dict): 来自 Analyzer 的输出，包含 `raw_entities` 和 `raw_steps`。

### 2.2 输出 (Output to State)

- `state["normalized_context"]` (Dict): 包含标准化后的 `instances`（物料清单）和 `steps`（执行剧本）。
- `state["current_stage"]` (String): 必须设置为 `"asset_manager"`。

------

## 3. 输出数据结构定义 (Normalized Context Schema)

### 3.1 `instances` (List of Dicts) - 传递给 Asset Manager 的物料图纸

| **字段名**    | **类型** | **说明**                                                     | **示例**                                                    |
| ------------- | -------- | ------------------------------------------------------------ | ----------------------------------------------------------- |
| `uid`         | String   | **全局唯一标识符**。结合 `spec` 和全局自增序号生成。         | `"glass_vial_0"`, `"centrifuge_0"`                          |
| `spec`        | String   | 映射到资产库的标准大类名称。                                 | `"glass_vial"`, `"solid_dispenser"`                         |
| `source_type` | String   | **资产来源路由**。必须是 `"local"`（本地资产库存在）或 `"objaverse"`（本地不存在，需要下游去云端下载）。 | `"local"`, `"objaverse"`                                    |
| `is_physical` | Boolean  | **刚体/模型渲染标志**。如果是粉末、溶液、光线等不需要独立 3D 模型渲染的概念实体，设为 `false`；否则为 `true`。 | `true`, `false`                                             |
| `init_params` | Dict     | **强类型初始化参数**。将自然语言属性翻译为底层环境构建的指令字典（见 4.3）。 | `{"is_open": true, "contains_substance": "CuSO4 solution"}` |

### 3.2 `steps` (List of Dicts) - 传递给 Skill Planner 的执行剧本

| **字段名**             | **类型**    | **说明**                                                     | **示例**                                           |
| ---------------------- | ----------- | ------------------------------------------------------------ | -------------------------------------------------- |
| `step_id`              | Integer     | 继承自 Analyzer 的序号。                                     | `0`, `1`                                           |
| `action`               | String      | 继承自 Analyzer 的原始动词。                                 | `"removes"`, `"shakes"`                            |
| `primary_uid`          | String      | 将原来的 `primary_obj` (raw_id) 映射为全局唯一的 `uid`。     | `"cap_0"`                                          |
| `secondary_uid`        | String/Null | 将原来的 `secondary_obj` (raw_id) 映射为全局唯一的 `uid`。   | `"glass_vial_0"`, `null`                           |
| `grounded_instruction` | String      | **重接地指令**。将原句中 `<raw_id>` 的占位符，**全部正则替换**为 `<uid>`。 | `"The robot removes <cap_0> from <glass_vial_0>."` |

------

## 4. 核心实现逻辑 (Implementation Guidelines)

### 4.1 资产映射的双轨制 (Spec Mapping Routing)

针对 `raw_entities` 中的 `raw_type`，必须采用“缓存查表 + 受限 LLM”的降级策略：

1. **轨道 1（本地 Cache）**：读取 `asset_cache.json`，如果 `raw_type` 命中，直接使用缓存的 `spec` 和 `source_type`。
2. **轨道 2（受限 LLM）**：若未命中，调用 LLM 进行分类。
   - **Prompt 约束**：提供本地有的标准资产库列表（如 `[beaker, test_tube, ...] `）。让 LLM 做单选题。
   - 如果匹配本地资产，输出：`LOCAL: <标准名>`。
   - 如果本地绝对没有该类仪器，输出：`OBJAVERSE: <搜索关键词>`。
   - 如果是粉末、液体等非刚体，输出：`NON_PHYSICAL: <物质名>`。

### 4.2 非刚体拦截 (Physical Gatekeeping)

根据上述 LLM 的判断：

- 若返回 `NON_PHYSICAL`，设置 `is_physical = False`。下游的 Asset Manager 看到此标志将直接跳过该实体的 3D 模型加载。

### 4.3 物理状态翻译 (InitParams Translation)

将 `semantic_attributes` 翻译为可选的强类型参数。定义如下的 Pydantic 模型作为转换目标（只提取文本中提到的内容，未提到则留空）：

Python

```
class InitParams(BaseModel):
    is_open: Optional[bool] = Field(description="容器或门是否打开")
    is_powered_on: Optional[bool] = Field(description="电气设备是否通电/开启")
    contains_substance: Optional[str] = Field(description="内部包含的化学物质或液体的名称（如 'CuSO4 solution'）")
    quantity_hint: Optional[str] = Field(description="容纳物的体积或重量描述（如 '50ml', '5g'）")
```

### 4.4 指令重接地 (Regex Re-grounding)

遍历 `raw_steps`，维护一个 `{raw_id: uid}` 的映射字典。

必须使用 Python 的 `re` 模块，对 `grounded_instruction` 进行精确替换：

Python

```
# 示例代码逻辑
text = step["grounded_instruction"]
for raw_id, uid in id_map.items():
    text = re.sub(rf"<{raw_id}>", f"<{uid}>", text)
step["grounded_instruction"] = text
```

------

## 5. 预期 I/O 示例 (Expected Example)

**Input (state["task_analysis"]):**

JSON

```
{
  "raw_entities": [
    {"raw_id": "vial_1", "raw_type": "10 mL glass vial", "semantic_attributes": {"cap": "removed"}},
    {"raw_id": "polymer_1", "raw_type": "photocatalyst polymer", "semantic_attributes": {"amount": "5mg"}}
  ],
  "raw_steps": [
    {
      "step_id": 0, "action": "dispenses", "primary_obj": "polymer_1", "secondary_obj": "vial_1",
      "grounded_instruction": "The robot dispenses <polymer_1> into <vial_1>."
    }
  ]
}
```

**Output (state["normalized_context"]):**

JSON

```
{
  "instances": [
    {
      "uid": "tube_container_0",
      "spec": "Tube Container",
      "source_type": "local",
      "is_physical": true,
      "init_params": {"is_open": true}
    },
    {
      "uid": "polymer_powder_0",
      "spec": "photocatalyst polymer",
      "source_type": "local",
      "is_physical": false,
      "init_params": {"contains_substance": "photocatalyst polymer", "quantity_hint": "5mg"}
    }
  ],
  "steps": [
    {
      "step_id": 0, "action": "dispenses", "primary_uid": "polymer_powder_0", "secondary_uid": "tube_container_0",
      "grounded_instruction": "The robot dispenses <polymer_powder_0> into <tube_container_0>."
    }
  ]
}
```

## 6. 开发实施清单 (Implementation Checklist)

- [ ] 建立 `asset_cache.json` 的读写逻辑，实现映射结果的本地缓存。
- [ ] 在 `normalizer.py` 中实现与 LLM 的交互，强制要求 LLM 输出 `LOCAL`, `OBJAVERSE`, 或 `NON_PHYSICAL` 前缀。
- [ ] 编写基于 Pydantic 的 `InitParams` 解析函数。
- [ ] 实现 `re.sub` 正则替换逻辑，确保尖括号 `< >` 格式不被破坏。
- [ ] 更新 `state.py`，新增 `normalized_context` 类型定义。