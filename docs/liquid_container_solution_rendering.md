# 液体容器溶液渲染功能

## 背景

VLABench 中只有试管（ChemistryTube）支持通过 `solution` 参数动态渲染不同颜色的溶剂。本项目将此能力扩展到烧杯等其他容器，创建了一套可复用的 `SolutionMixin` 架构，所有容器共享统一的 `solution` 参数接口。

## 架构设计

```
SolutionMixin (Mixin基类)
├── solution2rgba          # 全局共享的溶剂→颜色映射表
├── set_solution_rgba()    # 设置颜色的统一方法
├── get_solution()         # 获取当前溶剂名
└── initialize_episode()    # 初始化时自动设置颜色

ChemistryTube (试管)
ChemistryBeaker (烧杯)
ChemistryFlask  (锥形瓶，未来扩展)
ChemistryPetriDish (培养皿，未来扩展)
```

---

## 将普通容器转换为液体容器 — 手动流程

以下步骤将一个普通容器模型（如烧杯、锥形瓶）转换为支持溶剂渲染的液体容器。

### 准备工作：分析原容器模型

在开始之前，需要了解原容器的几何参数。以烧杯为例：

```bash
# 查看容器 mesh 文件
ls assets/obj/meshes/lab_equipment/beaker/beaker_0/beaker/

# 查看 XML 模型结构
cat assets/obj/meshes/lab_equipment/beaker/beaker_0/beaker/beaker.xml
```

需要关注的参数：
- **mesh scale**：XML 中 mesh 的 scale 值（如 beaker 是 `0.012`）
- **容器朝向**：body 的 euler 旋转角度（影响液体 mesh 的放置方向）
- **容器内壁尺寸**：用于计算液体 mesh 的半径和高度
- **底部位置**：液体 mesh 的 Z 轴起点

### Step 1：创建新目录并复制文件

假设原容器路径为 `obj/meshes/<category>/<name>/<name>/<name>.xml`，新容器放在同级目录：

```bash
# 创建新目录（命名为 chemistry_<name>）
mkdir -p assets/obj/meshes/<category>/chemistry_<name>/<name>/

# 复制原容器的所有文件
cp assets/obj/meshes/<category>/<name>/<name>/*.obj \
   assets/obj/meshes/<category>/chemistry_<name>/<name>/

# 如果有 material.mtl 也复制
cp assets/obj/meshes/<category>/<name>/<name>/*.mtl \
   assets/obj/meshes/<category>/chemistry_<name>/<name>/
```

### Step 2：分析容器几何参数

用 Python 脚本分析原容器的尺寸：

```python
import math

# 读取原容器的 mesh 文件
with open('assets/obj/meshes/<category>/<name>/<name>/<name>.obj') as f:
    lines = f.readlines()

xs, ys, zs = [], [], []
for line in lines:
    if line.startswith('v '):
        parts = line.split()
        xs.append(float(parts[1]))
        ys.append(float(parts[2]))
        zs.append(float(parts[3]))

print(f'X: {min(xs):.4f} to {max(xs):.4f}')
print(f'Y: {min(ys):.4f} to {max(ys):.4f}')
print(f'Z: {min(zs):.4f} to {max(zs):.4f}')
print(f'Height: {max(zs)-min(zs):.4f}')

# 估算内壁半径（按 Z 高度分段查看）
# 圆柱形容器的半径在各个高度应该相对一致
```

查看原容器 XML 中的关键信息：

```xml
<!-- 从原 XML 中提取 -->
<mesh file="<name>.obj" scale="0.012 0.012 0.012"/>
<body name="<name>" euler="0 0 1.57">  <!-- 朝向 -->

<!-- 查找 grasppoint site 的位置，确定容器在 MuJoCo 中的高度 -->
<site class="grasppoint" pos="0 0 0.14"/>
```

### Step 3：创建液体 mesh 文件

液体 mesh 是一个简单的几何体（圆柱体或锥形），用于填充容器底部到一定高度。

#### 3.1 生成圆柱体 OBJ

```python
import math

# 参数（基于 Step 2 的分析结果）
n_segments = 32       # 圆周分段数，越多越光滑
radius = 2.8         # 液体半径（略小于容器内壁，留 0.1~0.2 间隙）
z_bottom = 0.5       # 液体底部 Z（高于容器实际底部）
z_top = 5.5           # 液体顶部 Z（液面高度，通常为容器高度的 40%~70%）

vertices = []
faces = []

# 顶面圆心
vertices.append((0, 0, z_top))
# 底面圆心
vertices.append((0, 0, z_bottom))

# 顶面圆环顶点
for i in range(n_segments):
    angle = 2 * math.pi * i / n_segments
    vertices.append((radius * math.cos(angle), radius * math.sin(angle), z_top))

# 底面圆环顶点
for i in range(n_segments):
    angle = 2 * math.pi * i / n_segments
    vertices.append((radius * math.sin(angle), radius * math.cos(angle), z_bottom))

# 顶面三角形（扇形）
for i in range(n_segments):
    next_i = (i + 1) % n_segments
    faces.append((0, 2 + i, 2 + next_i))

# 底面三角形（反向）
for i in range(n_segments):
    next_i = (i + 1) % n_segments
    faces.append((1, 2 + n_segments + next_i, 2 + n_segments + i))

# 侧面四边形
for i in range(n_segments):
    next_i = (i + 1) % n_segments
    top_cur = 2 + i
    top_next = 2 + next_i
    bot_cur = 2 + n_segments + i
    bot_next = 2 + n_segments + next_i
    faces.append((top_cur, bot_cur, bot_next))
    faces.append((top_cur, bot_next, top_next))

# 写入 OBJ 文件
output_path = 'assets/obj/meshes/<category>/chemistry_<name>/<name>/<name>_liquid.obj'
with open(output_path, 'w') as f:
    f.write('usemtl liquid\n')
    for v in vertices:
        f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
    for face in faces:
        f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')

print(f'Generated: {output_path}, {len(vertices)} vertices, {len(faces)} faces')
```

#### 3.2 关键参数选择原则

- **radius**：液体半径 = 容器内壁半径 - 0.1~0.2 的间隙（防止穿透容器壁）
- **z_bottom**：液体底部略高于容器实际底部（留出底部厚度）
- **z_top**：液面高度 = 容器高度的 40%~70%（太满视觉上不好，太少看不出有液体）
- **scale**：与原容器 mesh 的 scale 一致（如 `0.012 0.012 0.012`）

### Step 4：修改 XML 模型文件

复制原 XML 并重命名，然后添加以下内容：

#### 4.1 在 `<asset>` 中添加溶剂 material

在 `<asset>` 末尾（所有现有 material 和 mesh 之后）添加：

```xml
<!-- Solution materials: shared across all chemistry containers -->
<material name="CuCl2" rgba="0.141000 1.000000 0.174043 0.400000"/>
<material name="CuSO4" rgba="0 0.45 1 0.400000"/>
<material name="FeCl3" rgba="0.6475 0.5686 0.023 0.400000"/>
<material name="KMnO4" rgba="0.5 0 0.5 0.4"/>
<material name="I2" rgba="0.3 0.13 0.0 0.4"/>
<material name="K2CrO4" rgba="0.57 0.12 0.013 0.40000"/>
<material name="NaCl" rgba="1 1 1 0.3"/>
<material name="AgNO3" rgba="1 1 1 0.3"/>
<material name="BaCl2" rgba="1 1 1 0.3"/>
<material name="H2SO4" rgba="1 1 1 0.3"/>
<material name="NaOH" rgba="1 1 1 0.3"/>
<material name="Ba(NO3)2" rgba="1 1 1 0.3"/>
<material name="Pb(NO3)2" rgba="1 1 1 0.3"/>
<material name="Na2CO3" rgba="1 1 1 0.3"/>
<material name="CaCl2" rgba="1 1 1 0.3"/>
<material name="HCl" rgba="1 1 1 0.3"/>
<material name="CaSO4" rgba="1 1 1 0.7"/>
```

#### 4.2 在 `<asset>` 中添加液体 mesh

在所有 `<mesh>` 声明之后添加：

```xml
<!-- Liquid mesh (必须显式指定 name="solution"，供 Python 侧查找) -->
<mesh name="solution" file="<name>_liquid.obj" scale="0.012 0.012 0.012"/>
```

**注意**：必须写 `name="solution"`，否则 `<geom mesh="solution">` 找不到 mesh。

#### 4.3 在 `<worldbody>` 中添加溶液 geom

在容器 body 中找到 `<geom>` 声明处，在容器壁 geom 之后添加：

```xml
<!-- Solution visual geom -->
<geom name="solution" type="mesh" mesh="solution" material="CaSO4" class="visual"/>
<!-- Solution collision geom (用于物理碰撞) -->
<geom type="mesh" mesh="solution" class="collision"/>
```

**注意事项**：
- `material="CaSO4"` 是默认颜色，可以是任意一种溶剂名，运行时会被 Python 覆盖
- `<geom name="solution">` 中的 `name="solution"` 必须与 mesh 的 `name="solution"` 一致
- `class="visual"` 和 `class="collision"` 与原容器的 geom class 保持一致

### Step 5：创建 Python 实体类

在 `tasks/components/specific_entities/ml_liquid_containers.py` 中添加新类：

```python
@register.add_entity("Chemistry<Name>")
class Chemistry<Name>(SolutionMixin, CommonGraspedEntity):
    """
    带溶剂渲染的<中文容器名>。
    传入 solution 参数（如 solution="CuSO4"）显示对应颜色的液体。
    不传 solution 则为空容器。
    """
    pass
```

### Step 6：在 constant.py 中注册

在 `configs/constant.py` 中添加注册条目：

```python
"chemistry_<name>": [components.Chemistry<Name>, "obj/meshes/<category>/chemistry_<name>/<name>/<name>.xml"],
```

### Step 7：测试验证

```bash
# 用已有任务脚本测试，或创建简单测试任务
MUJOCO_GL=egl python3 scripts/trajectory_generation.py \
    --task-name test_chemistry_<name> \
    --n-sample 1 \
    --debug \
    --save-dir /path/to/output
```

---

## 溶剂颜色表

| 溶剂名 | 视觉颜色 | RGBA |
|-------|---------|------|
| CuCl2 | 绿色 | [0.141, 1.0, 0.174, 0.4] |
| CuSO4 | 蓝色 | [0, 0.45, 1, 0.4] |
| FeCl3 | 棕黄色 | [0.648, 0.569, 0.023, 0.4] |
| KMnO4 | 紫色 | [0.5, 0, 0.5, 0.4] |
| I2 | 棕色 | [0.3, 0.13, 0.0, 0.4] |
| K2CrO4 | 橙红色 | [0.57, 0.12, 0.013, 0.4] |
| CaSO4 | 白色（高不透明度） | [1, 1, 1, 0.7] |
| NaCl, AgNO3, BaCl2, H2SO4, NaOH, Ba(NO3)2, Pb(NO3)2, Na2CO3, CaCl2, HCl | 白色透明 | [1, 1, 1, 0.3] |

---

## 已完成的转换

| 容器 | 状态 | 备注 |
|------|------|------|
| 烧杯 (beaker) | ✅ 已完成 | chemistry_beaker，含 beaker_liquid.obj |
| 试管 (tube) | ✅ 已有（已改造为 SolutionMixin） | 原 ChemistryTube 继承自 SolutionMixin |

---

## 当前进度

- [x] SolutionMixin 架构设计
- [x] ChemistryBeaker 模型转换
- [x] ChemistryTube 改造为复用 SolutionMixin
- [x] 渲染验证通过
- [x] 手动转换流程文档
- [ ] Flask 锥形瓶转换
- [ ] Petri Dish 转换
- [ ] 自动化转换脚本

---

生成日期：2026-04-03
