# VLABench Pipeline: `Asset Manager` 节点功能设计与实施规范

## 1. 节点定位与职责 (Node Overview)

`Asset Manager`（资产管理器）是整个流水线的“物理实例化总监（World Builder）”。

它承接上游 `Normalizer` 下发的标准物料清单（Instances），负责获取 3D 模型文件、推断并绑定底层 Python 框架类（Mixin），并对模型进行必要的 XML 注入改造，最终输出可以直接供代码生成器（Code Generator）调用的物理资产状态表。

**🚨 核心架构原则（必须严格遵守）：**

1. **语义隔离**：**绝对不要**在此节点引入 LLM 进行自然语言推理或同义词映射。只根据上游传来的强类型 `init_params` 办事。
2. **快速失败 (Fail-Fast)**：彻底抛弃旧版的静默失败（`"found": false`）模式。如果遇到下载失败或本地资产缺失，立即抛出明确的 Python 异常（如 `AssetNotFoundError`），在框架外层拦截。
3. **数据源唯一 (SSOT)**：颜色的具体 RGBA 值计算交由物理类（如 `SolutionMixin`）处理，本节点只负责传递 `solution` 物质名称标签。

------

## 2. 接口契约 (Interface Contract)

### 2.1 输入 (Input from State)

- `state["normalized_context"]["instances"]` (List): 来自 Normalizer 的标准物料清单。

### 2.2 输出 (Output to State)

- `state["asset_status"]` (Dict): 一个以 `uid` 为键的字典，包含每个实体的底层物理信息。
- `state["current_stage"]` (String): 变更为下游的下一个节点（如 `"skill_planner"` 或 `"code_generator"`）。

------

## 3. 输出数据结构定义 (Asset Status Schema)

`state["asset_status"]` 必须严格遵守以下嵌套结构，废弃原有的 `found` 和 `instantiation_kwargs` 字段，采用高度可拓展的 `properties` 字段。

| **字段名 (以 UID 为键)** | **类型** | **说明**                                                     | **示例**                                      |
| ------------------------ | -------- | ------------------------------------------------------------ | --------------------------------------------- |
| `xml_path`               | String   | 准备就绪的 MuJoCo XML 模型文件绝对或相对路径（注入修改后的版本）。 | `"obj/meshes/.../beaker_0_injected.xml"`      |
| `class_name`             | String   | **动态推断出的 VLABench 底层类名**。决定了该实体能调用哪些 Mixin（如能否装水）。 | `"ChemistryBeaker"`, `"CommonContainer"`      |
| `properties`             | Dict     | **实例化属性包**。供下游生成 Python 代码时直接解包或提取使用的参数字典。 | `{"is_container": true, "solution": "CuSO4"}` |

*(注：如果输入中的 `is_physical` 为 `false`，则该实体不会出现在此输出字典中。)*

------

## 4. 节点内部工作流 (Internal Data Pipeline)

节点内部包含四大核心工序，针对每一个 `instance` 按顺序执行：

### 🛠️ 工序 1：物理过滤与资产获取 (Fetch & Route)

1. **拦截幽灵**：检查 `instance["is_physical"]`。如果为 `false`（如溶液本身、粉末），直接 `continue` 跳过，不进行任何模型处理。
2. **本地加载**：如果 `source_type == "local"`，根据 `spec` 从本地 `name2class_xml` 映射表中获取初始 XML 路径。若不存在，抛出 `AssetNotFoundError`。
3. **云端下载**：如果 `source_type == "objaverse"`，调用 `download_and_fix_asset(spec)`。若下载失败，抛出异常。拿到经过几何归一化的原始 XML 路径。

### 🛠️ 工序 2：动态类名推断 (Dynamic Class Inference)

摒弃硬编码，采用基于 `init_params` 的特征工厂模式。判断逻辑如下：

1. **化学容器升级**：如果 `init_params` 中存在 `contains_substance`（溶液），强制将 `class_name` 设为 `ChemistryBeaker`（或对应的管类），以激活 `SolutionMixin`。并将物质名称提取为 `properties["solution"]`。
2. **普通容器判定**：如果 `init_params.get("is_container") == True`，将其设为 `CommonContainer`（如果有门则设为 `ContainerWithDoor`）。
3. **本地兜底**：如果均不满足且是本地模型，使用本地 registry 中注册的默认类名。否则兜底为 `CommonGraspedEntity`。

### 🛠️ 工序 3：XML 外科手术与注入 (XML Surgical Injection)

**这是极其关键的一步！必须通过 `lxml` 等工具在内存中修改 XML 结构，并另存为 `_injected.xml`。**

- **场景 A：注入 Site（针对所有容器类）**

  - **条件**：如果该模型在工序 2 中被判定为任何容器类（包含 `ChemistryBeaker`, `CommonContainer` 等）。

  - **动作**：解析该模型的 Mesh，计算 Bounding Box 的 `z_max` 和 `z_min`。在 XML 的 `<body>` 节点下写入：

    `<site name="top_site" pos="0 0 {z_max}" size="0.01" rgba="1 0 0 0"/>`

    `<site name="bottom_site" pos="0 0 {z_min}" size="0.01" rgba="0 1 0 0"/>`

- **场景 B：注入 Solution 占位符（专治 Objaverse 野生化学容器）**

  - **条件**：如果是从 Objaverse 下载的野生模型，且在工序 2 中被判定为 `ChemistryBeaker`。

  - **动作**：因为野生模型没有自带流体网格，必须在 XML 的 `<body>` 内强行写入一个占位用的几何体（形状可以复制原物体的内部空腔并略微缩小）：

    `<geom name="solution" type="mesh" mesh="inner_cavity" rgba="1 1 1 0" ... />`

  - **目的**：保证底层 `SolutionMixin` 调用 `find("geom", "solution")` 时不会报错返回。

### 🛠️ 工序 4：打包输出 (Pack Status)

将注入修改后的新 XML 路径、推断出的 `class_name` 以及整合后的 `properties` 字典存入 `state["asset_status"][uid]`。

------

## 5. 预期 I/O 示例 (Expected Example)

**Input (state["normalized_context"]["instances"]):**

JSON

```
[
  {
    "uid": "tube_0", "spec": "tube", "source_type": "local", "is_physical": true,
    "init_params": {"is_container": true, "contains_substance": "CuCl2"}
  },
  {
    "uid": "salt_0", "spec": "NaCl", "source_type": "local", "is_physical": false,
    "init_params": {}
  },
  {
    "uid": "centrifuge_0", "spec": "centrifuge", "source_type": "objaverse", "is_physical": true,
    "init_params": {"is_container": true}
  }
]
```

**Output (state["asset_status"]):**

JSON

```
{
  "tube_0": {
    "xml_path": "obj/meshes/tube_0_injected.xml", 
    "class_name": "ChemistryTube",
    "properties": {
      "is_container": true,
      "solution": "CuCl2"
    }
  },
  "centrifuge_0": {
    "xml_path": "assets/downloaded/centrifuge_0_injected.xml", 
    "class_name": "CommonContainer",
    "properties": {
      "is_container": true
    }
  }
}
// salt_0 因 is_physical=false 被跳过
```

## 6. 开发实施清单 (Implementation Checklist)

- [ ] 移除旧版逻辑中所有与 `found: bool` 相关的代码，全面改写为 `try-except` 并抛出自定义 `AssetNotFoundError`。
- [ ] 实现类名推断工厂函数 `infer_class_name(spec, init_params, is_objaverse)`。
- [ ] 编写基于 `lxml` 或 `xml.etree.ElementTree` 的 XML 注入工具箱（计算 BBox 并写入 `top_site` / `bottom_site`）。
- [ ] **高危预警实现**：确保为来源于 Objaverse 的 `ChemistryBeaker` 等化学类容器注入 `<geom name="solution">` 节点，防止 `SolutionMixin` 崩溃。
- [ ] 更新 `state.py`，定义新的 `asset_status` 类型字典。