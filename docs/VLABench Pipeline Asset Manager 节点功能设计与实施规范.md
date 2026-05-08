# VLABench Pipeline: `Asset Manager` 节点功能设计与实施规范

## 1. 节点定位与职责 (Node Overview)

`Asset Manager`（资产管理器）是整个流水线的"物理实例化总监（World Builder）"。

它承接上游 `Normalizer` 下发的标准物料清单（Instances），负责获取 3D 模型文件、推断并绑定底层 Python 框架类（Mixin），并对模型进行必要的 XML 注入改造，最终输出可以直接供代码生成器（Code Generator）调用的物理资产状态表。

**核心架构原则（必须严格遵守）：**

1. **语义隔离**：**绝对不要**在此节点引入 LLM 进行自然语言推理或同义词映射。只根据上游传来的强类型 `init_params` 办事。
2. **快速失败 (Fail-Fast)**：彻底抛弃旧版的静默失败（`"found": false`）模式。如果遇到下载失败或本地资产缺失，立即抛出明确的 Python 异常（如 `AssetNotFoundError`），在框架外层拦截。
3. **数据源唯一 (SSOT)**：颜色的具体 RGBA 值计算交由物理类（如 `SolutionMixin`）处理，本节点只负责传递 `solution` 物质名称标签。

**资产路由优先级（核心逻辑）：**

```
_fetch_asset(spec, source_type):
    1. 检查 constant.py（开发者维护的权威注册表，最高优先级）
       → 如果找到了，直接用，跳过 source_type 判断
    2. constant.py 没有注册
       → source_type="local" → 报错 AssetNotFoundError
       → source_type="objaverse" → 下载
           → 下载成功后，更新缓存 source_type="local"
             （下次缓存命中时直接走 constant.py，不再下载）
```

**关键设计动机：** `asset_cache.json` 中的 `source_type` 字段描述的是"该资产在分类时的预期来源"，而非"当前状态"。由于 `constant.py` 是后来注册上去的，缓存中可能存在 `source_type="objaverse"` 的过时记录（如 `pipette`），因此必须优先查 constant.py。

---

## 2. 接口契约 (Interface Contract)

### 2.1 输入 (Input from State)

- `state["normalized_context"]["instances"]` (List): 来自 Normalizer 的标准物料清单。

### 2.2 输出 (Output to State)

- `state["asset_status"]` (Dict): 一个以 `uid` 为键的字典，包含每个实体的底层物理信息。
- `state["current_stage"]` (String): 变更为下游的下一个节点（如 `"skill_planner"` 或 `"code_generator"`）。

---

## 3. 输出数据结构定义 (Asset Status Schema)

`state["asset_status"]` 必须严格遵守以下嵌套结构，废弃原有的 `found` 和 `instantiation_kwargs` 字段，采用高度可拓展的 `properties` 字段。

| **字段名 (以 UID 为键)** | **类型** | **说明**                                                     | **示例**                                      |
| ------------------------ | -------- | ------------------------------------------------------------ | --------------------------------------------- |
| `xml_path`               | String   | 准备就绪的 MuJoCo XML 模型文件路径（注入修改后的版本）。 | `"obj/meshes/.../beaker_0_injected.xml"`      |
| `class_name`             | String   | **动态推断出的 VLABench 底层类名**。决定了该实体能调用哪些 Mixin（如能否装水）。 | `"ChemistryBeaker"`, `"CommonContainer"`      |
| `properties`             | Dict     | **实例化属性包**。供下游生成 Python 代码时直接解包或提取使用的参数字典。 | `{"is_container": true, "solution": "CuSO4"}` |

*(注：如果输入中的 `is_physical` 为 `false`，则该实体不会出现在此输出字典中。)*

---

## 4. 节点内部工作流 (Internal Data Pipeline)

节点内部包含四大核心工序，针对每一个 `instance` 按顺序执行：

### 工序 1：物理过滤与资产获取 (Fetch & Route)

1. **物理过滤**：检查 `instance["is_physical"]`。如果为 `false`（如溶液本身、粉末），直接 `continue` 跳过，不进行任何模型处理。
2. **动态类名推断**：调用 `_infer_class_name(spec, init_params)` 推断类名。
3. **资产获取**：调用 `_fetch_asset(spec, source_type)` 获取 XML 路径。
   - **最高优先级**：先检查 `constant.py`（`name2class_xml`）注册表。如果找到，直接使用，不再判断 `source_type`。
   - **未注册时**：根据 `source_type` 决定：
     - `source_type="local"` → 抛出 `AssetNotFoundError`
     - `source_type="objaverse"` → 调用 `download_asset` 下载
       - **下载成功后**：更新 `asset_cache.json`，将 `source_type` 改为 `"local"`。这样下次运行时会直接走 `constant.py`，避免重复下载。

### 工序 2：动态类名推断 (Dynamic Class Inference)

摒弃硬编码，采用基于 `init_params` 的特征工厂模式。判断逻辑如下：

1. **化学容器升级**：如果 `init_params` 中存在 `contains_substance`（溶液），强制将 `class_name` 设为 `ChemistryBeaker`（或对应的管类），以激活 `SolutionMixin`。并将物质名称提取为 `properties["solution"]`。
2. **普通容器判定**：如果 `init_params.get("is_container") == True`，将其设为 `CommonContainer`（如果有门则设为 `ContainerWithDoor`）。
3. **注册表兜底**：查询 `constant.py` 的 `name2class_xml`，如果命中，使用注册表中的默认类名。
4. **最终兜底**：如果均不满足，设为 `CommonGraspedEntity`。

### 工序 3：XML 外科手术与注入 (XML Surgical Injection)

**通过 `xml_injector.py` 在原 XML 文件上直接修改。**（不再另存为 `_injected.xml`）

**优化：内置模型跳过注入**
- `BUILTIN_SITES`：记录了哪些 spec 已经内置了 `top_site` / `bottom_site`（如 `tube`）
- `BUILTIN_SOLUTION`：记录了哪些 spec 已经内置了 solution geom（如 `chemistry_beaker`, `tube`）
- 如果模型已在这些内置列表中，跳过相应注入

**场景 A：注入 Site（针对容器类）**

- **条件**：class_name 是容器类（`ChemistryBeaker`, `ChemistryTube`, `CommonContainer` 等）且 spec 不在内置列表中。
- **动作**：解析 OBJ 文件计算 Bounding Box 的 `z_max` 和 `z_min`，注入：
  ```xml
  <site name="top_site" pos="0 0 {z_max}" size="0.01" rgba="1 0 0 0"/>
  <site name="bottom_site" pos="0 0 {z_min}" size="0.01" rgba="0 1 0 0"/>
  ```

**场景 B：注入 Solution 占位符（专治 Objaverse 野生化学容器）**

- **条件**：class_name 是 `ChemistryBeaker` 或 `ChemistryTube` 且 spec 不在内置列表中。
- **动作**：注入透明圆柱体作为 solution 几何体，供 `SolutionMixin` 运行时修改 rgba 颜色：
  ```xml
  <geom name="solution" type="cylinder" size="0.02 0.05" pos="0 0 0.05" rgba="1 1 1 0" class="visual"/>
  ```

### 工序 4：打包输出 (Pack Status)

将注入修改后的 XML 路径、推断出的 `class_name` 以及整合后的 `properties` 字典存入 `state["asset_status"][uid]`。

---

## 5. 预期 I/O 示例 (Expected Example)

**Input (state["normalized_context"]["instances"]):**

```json
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

```json
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

---

## 6. 开发实施清单 (Implementation Checklist)

- [x] ~~移除旧版逻辑中所有与 `found: bool` 相关的代码~~ → 已完成：全面改用 `AssetNotFoundError`
- [x] ~~实现类名推断工厂函数~~ → 已完成：`_infer_class_name(spec, init_params)`
- [x] ~~XML 注入工具箱~~ → 已完成：`xml_injector.py`
- [x] ~~高危预警：ChemistryBeaker solution 注入~~ → 已完成：`_inject_solution_placeholder()`
- [x] **新增**：constant.py 优先路由逻辑
- [x] **新增**：下载成功后同步更新缓存

---

## 7. 相关文件清单

| 文件 | 职责 |
|------|------|
| `scripts/vlabench_agent/nodes/asset_manager.py` | 节点主逻辑：Fetch & Route、类推断、打包输出 |
| `scripts/vlabench_agent/tools/asset_tools.py` | 工具函数：`_register_downloaded_asset()`, `check_asset_exists()`, `download_asset()` |
| `scripts/vlabench_agent/tools/xml_injector.py` | XML 注入工具：`inject_xml()`, `_inject_sites()`, `_inject_solution_placeholder()` |
| `scripts/vlabench_agent/tools/asset_cache.py` | 缓存管理：`load_cache()`, `save_cache()`, `set_cached()`, `get_cached()` |
| `VLABench/configs/constant.py` | 资产注册表：`name2class_xml`（开发者维护的权威注册表） |
| `scripts/VLABench/assets/asset_cache.json` | 缓存文件（Normalizer 写入，Asset Manager 下载后更新） |
