# 模型处理与注册标准工作流指南 (SOP)

## 1. 角色与目标

你是 VLABench 项目的**资产处理专家**。你的职责是将已下载的 GLB 3D 模型转换为 VLABench 仿真框架可直接使用的格式（OBJ + MuJoCo XML），并在项目代码中完成注册，使其能在仿真任务中被正常加载、渲染和操作。

**完整链路**：下载的 GLB → 拆分（如多合一）→ 处理（OBJ/collision/XML）→ 注册（constant.py + normalizer）→ 可用

**项目根目录**：`/ssd/mkqin/workspace/VLABench`（下文所有相对路径均基于此）

---

## 2. 核心工具链与文件路径

### 2.1 脚本工具

| 脚本 | 路径 | 用途 |
|------|------|------|
| **split_glb.py** | `VLABench/pipeline/tests/split_glb.py` | 将包含多个几何体的 GLB 文件拆分为独立 GLB |
| **process_local_glb.py** | `VLABench/pipeline/tests/process_local_glb.py` | GLB → OBJ 转换、collision mesh 生成、尺寸归一化、XML 生成 |
| **register_model.py** | `VLABench/pipeline/tests/register_model.py` | XML 注入（grasppoint/solution 等）+ 注册到 constant.py |
| **validate_asset.py** | `VLABench/pipeline/tests/validate_asset.py` | 验证生成的模型能否被 MuJoCo 正确加载 |

> **注意**：`process_local_glb.py` 依赖 `fix_obj2mjcf_xml` 模块，如果遇到 `ModuleNotFoundError`，需要从 `scripts/` 目录复制该模块到 Python path，或使用场景 C 的简化方法直接用 trimesh 转换。

### 2.2 模型输出目录

所有处理好的模型保存在：
```
VLABench/assets/review/<model_name>/<model_name>/
```

**标准输出结构**（以 `large_beaker` 为例）：
```
VLABench/assets/review/large_beaker/
  └── large_beaker/
        ├── large_beaker.xml          ← MuJoCo XML 配置（最核心）
        ├── large_beaker.obj          ← 主视觉模型
        ├── large_beaker.glb          ← 原始 GLB（保留）
        ├── lab_beaker_a.png          ← 材质贴图（刻度等）
        ├── material.mtl
        └── large_beaker/             ← 子目录（含 collision mesh 的 OBJ 副本）
              ├── large_beaker.obj
              ├── large_beaker_collision_0.obj
              ├── large_beaker_collision_1.obj
              ├── ...
              └── lab_beaker_a.png
```

> **注意**：`process_local_glb.py` 会生成两层嵌套目录（`review/<name>/<name>/<name>/`），XML 文件在最内层。

### 2.3 必须修改的配置文件/代码

| 文件 | 路径 | 作用 |
|------|------|------|
| **constant.py** | `VLABench/configs/constant.py` | 资产注册表 `name2class_xml`，框架通过此文件查找模型 |
| **normalizer.py** | `VLABench/pipeline/nodes/normalizer.py` | `STANDARD_ASSET_LIBRARY` 列表，LLM 分类时匹配模型名称 |
| **asset_cache.json** | `scripts/VLABench/assets/asset_cache.json` | Normalizer 缓存，已分类的模型会缓存 spec 映射 |

### 2.4 实体类对照表

| 类名 | 用途 | 需要的 XML 注入 | 典型物体 |
|------|------|----------------|---------|
| `CommonGraspedEntity` | 普通可抓取物体 | grasppoint sites | 搅拌棒、移液枪、水果、工具 |
| `ChemistryBeaker` | 化学容器（可装液体） | grasppoint + solution geom + solution materials + top/bottom_site | 烧杯、锥形瓶、量筒 |
| `ChemistryTube` | 试管 | 同 ChemistryBeaker + subentity 模式 | 试管 |
| `CommonContainer` | 3D 容器 | keypoints + placepoints | 篮子、盒子 |
| `FlatContainer` | 平面容器 | 4角 keypoints + placepoint | 盘子、培养皿 |
| `ContainerWithDoor` | 带门容器 | keypoints + placepoints + door joint | 微波炉、冰箱 |

---

## 3. 标准操作步骤

### 场景 A：单个 GLB 模型处理

适用于：已有一个独立的 GLB 文件（如从 Objaverse 下载的单一模型）。

#### Step 1：处理 GLB → OBJ + XML

```bash
cd /ssd/mkqin/workspace/VLABench

python VLABench/pipeline/tests/process_local_glb.py \
    --input_dir <GLB所在目录> \
    --keyword <模型关键词> \
    --output_dir VLABench/assets/review/<model_name>
```

**参数说明**：
- `--input_dir`：GLB 文件所在目录
- `--keyword`：GLB 文件名的子串（用于匹配，如目录中有 `flask.glb` 则 keyword 为 `flask`）
- `--output_dir`：输出路径
- `--file`：可选，指定后**只处理**目录中文件名（不含扩展名）匹配此值的单个 GLB 文件；不指定则处理目录下**所有** `.glb` 文件

> **⚠️ 默认行为**：`process_local_glb.py` 不指定 `--file` 时会处理 `--input_dir` 中**所有** `.glb` 文件。如果目录中有多个模型，会被一次性全部处理。
> 当只需要处理单个模型时，**务必使用 `--file`** 限定，避免误处理其他模型。

**只处理单个模型**（推荐）：
```bash
python VLABench/pipeline/tests/process_local_glb.py \
    --input_dir <GLB所在目录> \
    --keyword <模型关键词> \
    --file <模型文件名（不含扩展名）> \
    --output_dir VLABench/assets/review/<model_name>
```

**注意**：大部分待处理模型都需要绕 X 轴旋转 90 度才能正过来：
```bash
--rotate_axis x --rotate_degrees 90
```

#### Step 2：注册模型

```bash
python VLABench/pipeline/tests/register_model.py \
    --model_dir VLABench/assets/review/<model_name>/<model_name>/<model_name> \
    --class_name <实体类名> \
    --name <注册名>
```

**`--model_dir` 必须指向 XML 文件所在的目录**（通常是三层嵌套的最内层目录）。

**`--class_name` 必须从以下选项中选择**：
```
CommonGraspedEntity | ChemistryBeaker | ChemistryTube |
CommonContainer | FlatContainer | ContainerWithDoor
```

**`--name`**：在 constant.py 中注册的 key，也是 Normalizer 分类时匹配的 spec 名称。建议用英文小写下划线命名（如 `large_beaker`、`glass_stirring_rod`）。

此脚本会自动完成：
1. XML 注入（grasppoint 验证、solution geom/materials 注入、top/bottom_site 注入等）
2. 注册到 `constant.py` 的 `name2class_xml` 字典

#### Step 3：检查并修复 XML 材质透明度

**register_model.py 不会处理材质透明度！** 这一步必须手动完成。

打开生成的 XML 文件，检查并修改：

**透明玻璃材质（适用于烧杯、量筒、锥形瓶等）**：
```xml
<!-- ✅ 正确：保留 texture 贴图（刻度等），添加 rgba 透明度 -->
<texture type="2d" name="lab_beaker_a" file="lab_beaker_a.png"/>
<material name="lab_beaker_a" texture="lab_beaker_a" specular="0.4" shininess="0.001" rgba="1 1 1 0.3"/>

<!-- ❌ 错误：删除了 texture，导致刻度贴纸消失！ -->
<material name="glass_beaker" specular="0.9" shininess="0.9" rgba="0.85 0.92 0.97 0.25"/>
```

**纯透明材质（适用于无贴图的玻璃物体，如搅拌棒）**：
```xml
<material name="glass_stirring_rod" specular="0.9" shininess="0.9" rgba="0.85 0.92 0.97 0.3"/>
```

**collision geom 必须去掉 rgba 颜色**（让它们不可见）：
```xml
<!-- ✅ 正确：collision 无颜色 -->
<geom mesh="large_beaker_collision_0" class="collision"/>

<!-- ❌ 错误：collision 有随机颜色，会在渲染中显示出来 -->
<geom mesh="large_beaker_collision_0" rgba="0.45 0.99 0.58 1" class="collision"/>
```

可用以下命令批量修复 collision 的 rgba：
```python
import re
content = re.sub(
    r'(<geom\s+mesh="[^"]*collision[^"]*"\s+)rgba="[^"]*"(\s+class="collision"\s*/>)',
    r'\1\2',
    content
)
```

#### Step 4：检查并修复 solution geom 尺寸

**register_model.py 注入的 solution geom 尺寸经常是原始单位（未按 scale 换算）！**

打开 XML，找到 `<geom name="solution" .../>` ，检查其 `size` 和 `pos` 是否合理：

```xml
<!-- ❌ 错误：尺寸是原始单位（毫米级别），scale 未换算 -->
<geom name="solution" type="cylinder" size="2.689 2.675" pos="0 0 4.459" material="CaSO4" class="visual"/>

<!-- ✅ 正确：已按 scale 换算为米 -->
<geom name="solution" type="cylinder" size="0.0269 0.0268" pos="0 0 0.0446" material="CaSO4" class="visual"/>
```

**换算方法**：从 XML 的 `<mesh>` 标签中读取 `scale` 值，将 solution geom 的 size 和 pos 都乘以 scale。

#### Step 5：注册到 Normalizer 标准资产库

编辑 `VLABench/pipeline/nodes/normalizer.py`，在 `STANDARD_ASSET_LIBRARY` 列表中添加新模型名：

```python
STANDARD_ASSET_LIBRARY = [
    "beaker", "chemistry_beaker", "tube", "chemistry_tube_stand",
    "flask", "conical_flask", "large_beaker", "small_beaker",
    "cylinder_small", "cylinder_mid", "cylinder_big",
    # ↑ 在此列表中添加新模型名
    ...
]
```

#### Step 6：更新 asset_cache.json（可选但推荐）

如果 `scripts/VLABench/assets/asset_cache.json` 中已有该模型的旧记录（spec 指向错误的名称），需要修正：

```python
import json
with open('scripts/VLABench/assets/asset_cache.json', 'r') as f:
    cache = json.load(f)

# 修正 spec 映射
if 'small_beaker' in cache:
    cache['small_beaker']['spec'] = 'small_beaker'  # 确保指向正确的注册名

with open('scripts/VLABench/assets/asset_cache.json', 'w') as f:
    json.dump(cache, f, indent=2)
```

#### Step 7：验证

```bash
# 1. 验证模型能被 MuJoCo 加载
python3 -c "
import mujoco, os
os.environ['MUJOCO_GL'] = 'osmesa'
model = mujoco.MjModel.from_xml_path('VLABench/assets/review/<model_name>/<model_name>/<model_name>/<model_name>.xml')
print(f'OK: {model.nbody} bodies, {model.ngeom} geoms, {model.nsite} sites')
"

# 2. 验证 constant.py 注册
grep '<model_name>' VLABench/configs/constant.py

# 3. 端到端测试
cd /ssd/mkqin/workspace/VLABench
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python scripts/test_e2e.py "Lift the <model_name>"

# 使用 validate_asset.py 验证模型
python VLABench/pipeline/tests/validate_asset.py --asset_dir VLABench/assets/review/<model_name>
```

---

### 场景 B：多合一 GLB 拆分处理

适用于：一个 GLB 文件包含多个独立几何体（如 `chemistry_glassware.glb` 包含烧杯、量筒、试管等）。

#### Step 1：拆分 GLB

由于 `split_glb.py` 需要交互式输入，建议用 Python 脚本直接拆分：

```python
import trimesh
from pathlib import Path

glb_path = Path('VLABench/assets/review/chemistry_glassware.glb')
output_dir = Path('/ssd/mkqin/download')  # 临时输出目录
output_dir.mkdir(parents=True, exist_ok=True)

scene = trimesh.load(str(glb_path))

for i, (name, geom) in enumerate(scene.geometry.items()):
    size = geom.bounds[1] - geom.bounds[0]
    vertices = len(geom.vertices) if hasattr(geom, 'vertices') else 0

    print(f"[{i}] {name}: {size[0]:.1f}x{size[1]:.1f}x{size[2]:.1f}, {vertices} verts")

    # 手动指定输出文件名
    filename = "your_model_name.glb"  # ← 根据实际物体修改

    single_scene = trimesh.Scene()
    single_scene.add_geometry(geom, node_name=name)
    glb_data = single_scene.export(file_type="glb")
    (output_dir / filename).write_bytes(glb_data)
```

#### Step 2~7：与场景 A 的 Step 1~7 相同

对每个拆分后的 GLB 文件，分别执行 `process_local_glb.py` → `register_model.py` → 修复材质 → 注册。

**注意**：每个模型需要单独放到一个临时输入目录中，因为 `process_local_glb.py` 按 keyword 匹配文件名：

```bash
# 使用 split_glb.py 拆分多合一 GLB
python VLABench/pipeline/tests/split_glb.py /path/to/combined.glb --output_dir /tmp/split

# 为每个模型创建临时目录
mkdir -p /tmp/<model_name>_input
cp /ssd/mkqin/download/<split_file>.glb /tmp/<model_name>_input/<model_name>.glb

python VLABench/pipeline/tests/process_local_glb.py \
    --input_dir /tmp/<model_name>_input \
    --keyword <model_name> \
    --output_dir VLABench/assets/review/<model_name>
```

---

### 场景 C：简单 GLB 只需转 OBJ（无需 collision mesh 拆分）

适用于：极简模型（如搅拌棒，只有几十个顶点），不需要 VHACD collision 拆分。

直接用 trimesh 转换后手写 XML：

```python
import trimesh
import numpy as np
from pathlib import Path

glb_path = Path('path/to/model.glb')
output_dir = glb_path.parent

scene = trimesh.load(str(glb_path))
for name, geom in scene.geometry.items():
    # 底部对齐：Z_min = 0
    z_min = geom.bounds[0][2]
    geom.vertices[:, 2] -= z_min

    # 导出 OBJ
    obj_data = geom.export(file_type='obj')
    (output_dir / f"{name}.obj").write_bytes(
        obj_data.encode() if isinstance(obj_data, str) else obj_data
    )
```

然后手动创建 XML（参考已有模型的格式），再用 `register_model.py` 注册。

---

## 4. 关键避坑指南与隐性知识

### 4.1 材质与贴图（最容易出错的部分）

**绝对不能删除 texture 定义！**

```xml
<!-- ✅ 正确：保留 texture + 添加 rgba -->
<texture type="2d" name="lab_beaker_a" file="lab_beaker_a.png"/>
<material name="lab_beaker_a" texture="lab_beaker_a" specular="0.4" shininess="0.001" rgba="1 1 1 0.3"/>

<!-- ❌ 致命错误：删除 texture 后刻度贴纸消失，变成纯色透明 -->
<material name="glass_beaker" specular="0.9" shininess="0.9" rgba="0.85 0.92 0.97 0.25"/>
```

**rgba 第四位控制透明度**：
- `1.0` = 完全不透明
- `0.3` = 30% 不透明度（玻璃推荐值）
- `0.0` = 完全透明

**collision geom 必须无颜色**：它们的 rgba 是程序生成的随机颜色，在渲染时会透过透明物体显示出来。必须删除。

### 4.2 solution geom 尺寸单位

`register_model.py` 注入 solution geom 时，从 OBJ 文件的 bounding box 计算尺寸，但**不会考虑 XML 中的 mesh scale**。

典型错误：
```xml
<mesh file="small_beaker.obj" scale="0.01009 ..."/>
<!-- scale ≈ 0.01，但 solution geom 的 size 是原始毫米值 -->
<geom name="solution" size="2.689 2.675" pos="0 0 4.459"/>
```

**修复方法**：所有 solution geom 的 `size` 和 `pos` 值都要乘以 mesh 的 `scale`。

### 4.3 抓取点 (grasppoint) 设置

- 抓取点位于 `<site class="grasppoint" pos="X Y Z"/>`，在 `<body>` 内
- Z 值是相对于 body 原点的（body 原点在模型底部）
- 通常设置 1-3 个抓取点，分布在不同高度
- 抓取点的 group=4，只在调试时可见
- **抓取点位置应避免**：物体最顶部（夹爪会推到物体）和最底部（夹爪够不到）
- **建议位置**：物体高度的 60%-95% 处

### 4.4 命名规范

- **注册名**（constant.py 的 key）：英文小写 + 下划线，如 `large_beaker`、`cylinder_small`
- **目录名**：与注册名一致
- **三处必须统一**：
  1. `constant.py` 中的 key
  2. `normalizer.py` 的 `STANDARD_ASSET_LIBRARY` 列表
  3. `asset_cache.json` 中的 `spec` 字段

如果三处不一致，会导致：
- Normalizer 分类到错误的 spec → 加载错误的模型
- Asset Manager 在 constant.py 中找不到 → 尝试从 Objaverse 重新下载

### 4.5 目录嵌套陷阱

`process_local_glb.py` 的 `--output_dir` 如果指定 `VLABench/assets/review/<name>`，实际 XML 会在：
```
VLABench/assets/review/<name>/<name>/<name>/<name>.xml
```

三层嵌套！`register_model.py` 的 `--model_dir` 必须指向最内层目录：
```bash
python scripts/register_model.py \
    --model_dir VLABench/assets/review/cylinder_small/cylinder_small/cylinder_small \
    ...
```

### 4.6 size_fix 模块缺失

`process_local_glb.py` 内部调用 `size_fix.py` 做 LLM 尺寸修正，但该模块可能不在 Python path 中。这不影响核心处理，只是尺寸不会自动优化。如果需要精确尺寸，可以：
1. 手动修改 XML 中的 mesh scale
2. 重新运行 `bottom_align_all` + `fix_xml_after_postprocess`

### 4.7 collision mesh 的 friction 参数

不同物体需要不同的摩擦力设置：
- **玻璃容器**（烧杯、量筒）：`friction="1 1 0.001"` + `solimp="0.998 0.998 0.001"` + `solref="0.001 2"`（低摩擦，夹爪容易滑，需要更精确的抓取）
- **普通物体**：`friction="2 2 0.5"` + `solimp="0.9 0.95 0.001"` + `solref="0.02 1"`（默认）

### 4.8 测试命令注意事项

仿真测试需要设置 headless 渲染环境变量：
```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python scripts/test_e2e.py "Lift the <model_name>"
```

如果不设置，会出现 `gladLoadGL error` 或 `GLFW library is not initialized` 错误。

### 4.9 常见错误速查

| 错误现象 | 原因 | 解决方案 |
|----------|------|---------|
| 仿真加载了旧模型而非新模型 | `asset_cache.json` 中 spec 指向错误 | 修正 cache 中的 spec 为正确注册名 |
| 模型是灰色的，没有刻度 | texture 被删除 | 恢复 `<texture>` 和 material 的 `texture=` 属性 |
| 碰撞 mesh 显示为彩色方块 | collision geom 有 rgba 颜色 | 删除 collision geom 的 rgba 属性 |
| solution 圆柱体巨大/极小 | size 未按 scale 换算 | 所有值乘以 mesh scale |
| KeyError: 'table' | 模型未注册到 constant.py | 运行 register_model.py |
| 夹爪抓空（没碰到物体就松开） | `pick` 函数提前返回（task_success=True） | 确保 pick 返回 `task_success=False` |
| `AssetNotFoundError` | constant.py 中没有注册该模型 | 检查 name2class_xml 中是否有对应条目 |
| Normalizer 把 "small_beaker" 分类成 "beaker" | STANDARD_ASSET_LIBRARY 没有新名称 | 添加到列表中 |

---

## 5. 快速参考：完整命令序列

以下是一个从零开始处理新模型的完整命令序列模板：

```bash
cd /ssd/mkqin/workspace/VLABench

# ===== Step 1: 处理 GLB =====
python VLABench/pipeline/tests/process_local_glb.py \
    --input_dir /path/to/glb/directory \
    --keyword model_keyword \
    --output_dir VLABench/assets/review/my_new_model \
    --rotate_axis x --rotate_degrees 90

# ===== Step 2: 注册 =====
python VLABench/pipeline/tests/register_model.py \
    --model_dir VLABench/assets/review/my_new_model/my_new_model \
    --class_name CommonGraspedEntity \
    --name my_new_model

# ===== Step 3: 修复 XML（手动编辑） =====
# 3a. 材质透明度：在 material 标签加 rgba="1 1 1 0.3"
# 3b. 删除 collision geom 的 rgba
# 3c. 检查 solution geom 尺寸（乘以 scale）
# 3d. 调整 grasppoint 位置

# ===== Step 4: 注册到 Normalizer =====
# 编辑 scripts/vlabench_agent/nodes/normalizer.py
# 在 STANDARD_ASSET_LIBRARY 列表中添加 "my_new_model"

# ===== Step 5: 验证 =====
python3 -c "
import mujoco, os
os.environ['MUJOCO_GL'] = 'osmesa'
m = mujoco.MjModel.from_xml_path('VLABench/assets/review/my_new_model/my_new_model/my_new_model/my_new_model.xml')
print(f'OK: {m.nbody} bodies, {m.ngeom} geoms, {m.nsite} sites')
"
grep 'my_new_model' VLABench/configs/constant.py

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python scripts/test_e2e.py "Lift the my_new_model"
```
