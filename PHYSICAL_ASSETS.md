# VLAbench_LZR 物理模拟资产说明文档

## 概述

VLAbench_LZR 项目包含丰富的3D物理模拟资产，用于机器人视觉-语言-动作（VLA）模型的评估。这些资产包括实验室器材、机器人模型、场景环境等，所有资产均为本地实体文件，可直接用于MuJoCo/PyBullet等物理仿真环境。

## 资产目录结构

```
/ssd/liuzirui/VLAbench_LZR/VLABench/assets/
├── base/                  # 基础配置文件
├── obj/                   # 通用3D模型（符号链接）
├── review/                # 实验器材资产（36个类别，468MB）
├── robots/                # 机器人模型（2种，43MB）
└── scenes/                # 场景环境（20个场景，2.1GB）
```

## 一、实验器材资产（review/）

实验器材资产是本项目核心资产，包含36个类别的实验室物品，共**905个文件**，总大小**468MB**。

### 1.1 液体容器类（9种）

| 资产名称 | 目录路径 | 功能描述 | 变体数量 |
|---------|---------|---------|---------|
| **烧杯-小** | `beaker_small/` | 小型玻璃烧杯，用于少量液体操作 | 1 |
| **烧杯-大** | `beaker_large/` | 大型玻璃烧杯，用于大量液体操作 | 1 |
| **锥形瓶-小** | `conical_flask_small/` | 小型锥形瓶，用于滴定实验 | 1 |
| **锥形瓶-中** | `conical_flask_mid/` | 中型锥形瓶，通用实验容器 | 1 |
| **锥形瓶-大** | `conical_flask_large/` | 大型锥形瓶，用于混合反应 | 1 |
| **量筒-小** | `cylinder_small/` | 小型量筒，精确量取液体 | 1 |
| **量筒-中** | `cylinder_mid/` | 中型量筒，通用量液工具 | 1 |
| **量筒-大** | `cylinder_large/` | 大型量筒，大量液体测量 | 1 |
| **培养皿** | `petri_dish/` | **培养皿**，用于培养微生物或液体实验 | 1 |

#### 典型资产结构（以烧杯为例）
```
beaker_small/
└── small_beaker/
    ├── small_beaker.xml              # MuJoCo物理配置
    ├── small_beaker.obj              # 主3D网格
    ├── small_beaker.glb              # GLB格式（导入优化）
    ├── small_beaker_octagon.obj      # 八边形碰撞网格
    ├── material.mtl                   # 材质定义
    ├── lab_beaker_a.png              # 纹理贴图
    ├── metadata.json                  # 资产元数据
    ├── size_report.json              # 尺寸报告
    ├── validation_report.json        # 物理验证报告
    ├── postprocess_report.json       # 后处理报告
    ├── validation.mp4                # 验证视频
    ├── sources.txt                    # 资产来源
    └── small_beaker/                 # 碰撞网格子目录
        ├── small_beaker_collision_0.obj
        ├── small_beaker_collision_1.obj
        ├── ...
        └── small_beaker_collision_22.obj  # 23个精确碰撞网格
```

---

### 1.2 实验器材类（27种）

#### 加热与温控设备（6种）

| 资产名称 | 目录路径 | 功能描述 |
|---------|---------|---------|
| **本生灯** | `bunsen_burner/` | 火焰加热设备，**3种变体** |
| **酒精灯** | `alcohol_lamp/` | 酒精燃料加热灯 |
| **磁力搅拌器** | `magnetic_stir_plate/` | 磁力搅拌加热板 |
| **加热设备** | `heat_device/` | 通用加热装置 |
| **热板** | `hot_plate/` | 电加热板 |
| **水浴锅** | `water_bath/` | 恒温水浴设备 |

**本生灯变体**:
```
bunsen_burner/
├── 7532be8f501d435194e3feec33a3addf/    # 变体1
├── 5185e41b2beb48fa8f15ca3707f43e10/    # 变体2
└── 214242c1de024978a9db5a7c032b9845/    # 变体3
```

#### 实验操作工具（8种）

| 资产名称 | 目录路径 | 功能描述 |
|---------|---------|---------|
| **移液枪** | `pipette/` | 精确液体转移工具 |
| **机械移液器** | `mechanical_pipette/` | 机械式移液设备 |
| **漏斗** | `funnel/` | 液体转移漏斗 |
| **玻璃搅拌棒** | `glass_stirring_rod/` | 玻璃搅拌工具 |
| **抹布** | `rag/` | 实验室清洁用布 |
| **方垫** | `square_mat/` | 实验台防护垫 |
| **三脚架** | `tripod/` | 实验器材支撑架 |
| **万能支架** | `universal_support/` | 多功能实验支架 |

#### 测量与检测设备（5种）

| 资产名称 | 目录路径 | 功能描述 |
|---------|---------|---------|
| **温度计** | `thermometer/` | 温度测量仪器 |
| **电子秤** | `electronic_scale/` | 精密电子天平 |
| **分光光度计** | `spectrophotometer.glb` | 光学测量设备 |

#### 实验容器与存储（8种）

| 资产名称 | 目录路径 | 功能描述 |
|---------|---------|---------|
| **化学试管** | `chemistry_tube/` | 试管实验容器 |
| **试管架** | `chemistry_tube_rack/` | 试管存放架 |
| **药瓶** | `pill_bottle/` | 小型药品瓶 |
| **阿司匹林药瓶** | `aspirin_pill_bottle/` | 特定药品容器 |
| **干燥箱** | `drying_box/` | 器材干燥存储 |
| **抽屉** | `drawer/` | 实验室储物抽屉 |
| **佛罗伦萨烧瓶** | `florence_flask/` | 蒸馏用长颈烧瓶 |
| **烧瓶** | `flask/` | 通用实验烧瓶 |

#### 实验家具（2种）

| 资产名称 | 目录路径 | 功能描述 |
|---------|---------|---------|
| **实验台** | `lab_table/` | 标准实验室操作台 |
| **化学实验台** | `chemistry_lab_table/` | 化学专用实验台 |
| **移液枪架** | `pipettes_stand/` | 移液枪存放架 |

---

### 1.3 批量资产文件

部分资产包含GLB格式的批量文件：
- `beakers.glb` - 烧杯批量文件
- `lab_table.glb` - 实验台批量文件
- `spectrophotometer.glb` - 分光光度计

---

### 1.4 资产文件类型说明

| 文件类型 | 扩展名 | 用途 | 物理引擎 |
|---------|--------|------|---------|
| **物理配置** | `.xml` | MuJoCo物理属性定义 | MuJoCo |
| **3D网格** | `.obj` | 主几何模型 | PyBullet/MuJoCo |
| **优化模型** | `.glb` | 二进制压缩模型 | 通用 |
| **材质** | `.mtl` | 材质和纹理 | 渲染引擎 |
| **纹理** | `.png` | 表面纹理贴图 | 视觉渲染 |
| **元数据** | `.json` | 资产属性和验证 | 框架加载 |
| **验证** | `.mp4` | 物理验证视频 | 人工审核 |

---

## 二、机器人模型资产（robots/）

包含2种工业机器人，总大小**43MB**。

### 2.1 Franka Emika Panda

**目录**: `robots/franka_emika_panda/`

**文件清单**:
```
franka_emika_panda/
├── panda.xml              # 完整机器人配置（含手爪）
├── panda_nohand.xml       # 无手爪版本配置
├── hand.xml               # 手爪单独配置
├── hand_bak.xml           # 手爪备份配置
├── scene.xml              # 场景配置
├── panda.png              # 机器人外观图（2.2MB）
├── README.md              # 说明文档
├── LICENSE                # 许可证
└── assets/                # 附加资产
```

**机器人参数**:
- **自由度**: 7关节 + 2手指夹爪
- **工作半径**: 855mm
- **负载能力**: 0.3kg (推荐) / 3kg (最大)
- **重复定位精度**: ±0.1mm
- **配置文件**: MuJoCo XML格式

**用途**: 精密装配、实验室操作、液体处理

---

### 2.2 xArm7

**目录**: `robots/xarm7/`

**文件清单**:
```
xarm7/
├── xarm7.xml              # 完整机器人配置（含手爪）
├── xarm7_nohand.xml       # 无手爪版本
├── hand.xml               # 手爪配置
└── scene.xml              # 场景配置
```

**机器人参数**:
- **自由度**: 7关节
- **工作半径**: 700mm
- **负载能力**: 3kg (最大)
- **重复定位精度**: ±0.2mm

**用途**: 通用机器人任务、抓取操作

---

## 三、场景环境资产（scenes/）

包含20个场景，共**2039个文件**，总大小**2.1GB**。

### 3.1 场景分类

#### A. 实验室场景（2个）

| 场景名称 | 目录路径 | 用途 | 文件格式 |
|---------|---------|------|---------|
| **实验室-0** | `lab_0/` | 基础实验室 | XML + 纹理 |
| **实验室-1** | `lab_1/` | 高级实验室 | XML + 视觉资源 |

**实验室场景结构**:
```
lab_0/
├── lab.xml                    # MuJoCo场景配置
├── textures/                  # 纹理贴图
└── [其他资源]
```

#### B. 厨房场景（3个）

| 场景名称 | 目录路径 | 用途 | 文件格式 |
|---------|---------|------|---------|
| **现代厨房** | `kitchen_0/` | 斯堪的纳维亚风格 | OBJ (18MB) |
| **简约厨房** | `kitchen_1/` | 简约风格厨房 | OBJ (12MB) |
| **餐厅厨房** | `kitchen_2/` | 餐厅厨房一体化 | XML + 纹理 |

#### C. 生活场景（5个）

| 场景名称 | 目录路径 | 用途 | 文件格式 |
|---------|---------|------|---------|
| **未来公寓** | `apartment_0/` | 现代公寓 | OBJ (35MB) |
| **唐人街公寓** | `apartment_1/` | 传统风格公寓 | OBJ (22MB) |
| **卧室** | `bedroom_0/` | 居住卧室 | XML + 纹理 |
| **客厅** | `living_room_0/` | 日常起居室 | XML + 纹理 |
| **书房** | `studyroom_0/` | 办公学习环境 | XML + 纹理 |

#### D. 专业场景（4个）

| 场景名称 | 目录路径 | 用途 | 文件格式 |
|---------|---------|------|---------|
| **医院手术室** | `medical_room_1/` | Charite大学医院 | OBJ (28MB) |
| **办公室** | `office_0/` | 现代简约办公 | OBJ (15MB) |
| **商店** | `store_0/` | 零售商店 | XML + 纹理 |
| **餐厅** | `dining_hall_0/` | 酒店餐厅 | OBJ (14MB) |
| **豪华餐厅** | `dining_hall_2/` | 高端餐厅 | OBJ (19MB) |

#### E. 特殊场景（3个）

| 场景名称 | 目录路径 | 用途 | 文件格式 |
|---------|---------|------|---------|
| **温室公园** | `greenhouse_0/` | 植物生长环境 | OBJ (21MB) |
| **空实验室** | `lab_copy/` | 自定义场景基础 | XML + 纹理 |
| **默认场景** | `default/` | 空白测试场景 | XML + 纹理 |

---

### 3.2 场景资产详情

#### 高频使用场景

**实验室场景（lab_0/）**
```xml
<!-- 典型配置 -->
<worldbody>
    <geom name="floor" type="plane" />
    <geom name="table" type="mesh" mesh="table_mesh" />
    <body name="robot_base">
        <include file="panda.xml" />
    </body>
</worldbody>
```

**场景用途**:
- 液体转移实验
- 化学反应模拟
- 器材操作训练

---

## 四、资产使用指南

### 4.1 在MuJoCo中加载资产

```python
import mujoco

# 加载机器人
model = mujoco.MjModel.from_xml_path(
    "VLABench/assets/robots/franka_emika_panda/panda.xml"
)

# 加载场景
model = mujoco.MjModel.from_xml_path(
    "VLABench/assets/scenes/lab_0/lab.xml"
)

# 加载器材
model = mujoco.MjModel.from_xml_path(
    "VLABench/assets/review/beaker_small/small_beaker/small_beaker.xml"
)
```

### 4.2 在PyBullet中加载资产

```python
import pybullet as p

# 加载机器人
robot_id = p.loadURDF(
    "VLABench/assets/robots/franka_emika_panda/panda.urdf",
    basePosition=[0, 0, 0]
)

# 加载OBJ模型
visual_shape = p.createVisualShape(
    shapeType=p.GEOM_MESH,
    meshFileName="VLABench/assets/review/petri_dish/petri_dish.obj"
)

collision_shape = p.createCollisionShape(
    shapeType=p.GEOM_MESH,
    meshFileName="VLABench/assets/review/petri_dish/petri_dish.obj"
)

obj_id = p.createMultiBody(
    baseVisualShapeIndex=visual_shape,
    baseCollisionShapeIndex=collision_shape
)
```

### 4.3 资产组合示例

**液体转移实验场景配置**:
```xml
<worldbody>
    <include file="lab_0/lab.xml" />
    <body name="beaker" pos="0.5 0 0.8">
        <include file="../review/beaker_small/small_beaker/small_beaker.xml" />
    </body>
    <body name="pipette" pos="0.3 0 1.0">
        <include file="../review/pipette/pipette/pipette_-_laboratory_essential_tool.xml" />
    </body>
    <body name="robot_base" pos="0 0 0.76">
        <include file="../../robots/franka_emika_panda/panda.xml" />
    </body>
</worldbody>
```

---

## 五、资产规格统计

### 5.1 总体统计

| 资产类型 | 类别数 | 文件数 | 总大小 | 格式 |
|---------|--------|--------|--------|------|
| **实验器材** | 36 | 905 | 468MB | XML/OBJ/GLB |
| **机器人** | 2 | 10 | 43MB | XML/URDF |
| **场景** | 20 | 2039 | 2.1GB | XML/OBJ |
| **总计** | **58** | **2954** | **2.6GB** | 多格式 |

### 5.2 文件格式分布

| 格式 | 数量 | 用途 |
|------|------|------|
| `.xml` | ~500 | MuJoCo物理配置 |
| `.obj` | ~1500 | 3D几何模型 |
| `.glb` | ~46 | 优化二进制模型 |
| `.png` | ~500 | 纹理贴图 |
| `.mtl` | ~200 | 材质定义 |
| `.json` | ~200 | 元数据和验证 |

### 5.3 资产完整度

**标准资产包含**:
- ✓ 物理配置文件（XML）
- ✓ 3D网格文件（OBJ/GLB）
- ✓ 碰撞检测网格
- ✓ 材质和纹理
- ✓ 元数据文档
- ✓ 验证报告

---

## 六、特殊资产说明

### 6.1 培养皿（Petri Dish）

**路径**: `review/petri_dish/`

**用途**:
- 微生物培养实验
- 液体表面张力实验
- 细胞观察实验

**特点**:
- 精确的圆形几何
- 透明材质配置
- 多精度碰撞网格

---

### 6.2 本生灯变体

**变体数量**: 3种

**变体差异**:
- 不同高度调节范围
- 不同的火焰喷嘴
- 稳定性配置差异

**用途**:
- 高温加热实验
- 火焰观察实验
- 玻璃加工操作

---

### 6.3 符号链接资产

**路径**: `assets/obj -> /ssd/liuzirui/VLABench/VLABench/assets/obj`

**说明**: obj目录是符号链接，指向外部仓库的共享资产，用于减少重复存储。

---

## 七、资产开发规范

### 7.1 新增资产流程

1. **创建3D模型**（OBJ格式）
2. **生成碰撞网格**（多个精度级别）
3. **编写物理配置**（XML格式）
4. **定义材质纹理**（MTL/PNG）
5. **验证物理属性**（生成验证报告）
6. **添加元数据**（JSON格式）
7. **单元测试**（使用validation.mp4验证）

### 7.2 命名规范

- **目录名**: 小写下划线分隔（如 `petri_dish`）
- **文件名**: 描述性名称（如 `small_beaker.xml`）
- **碰撞网格**: `object_collision_N.obj`（N=0-22）

### 7.3 单位规范

- **长度**: 米（m）
- **质量**: 千克（kg）
- **角度**: 弧度（rad）
- **力**: 牛顿（N）

---

## 八、常见问题

### Q1: 如何添加新的实验器材？

A: 参考现有资产结构，在`review/`目录下创建新目录，按照标准资产结构添加必需文件。

### Q2: OBJ文件无法在MuJoCo中加载？

A: 需要确保OBJ文件在XML中正确引用，并且有对应的MTL材质文件。

### Q3: 如何修改机器人的初始姿态？

A: 编辑对应机器人的XML文件，调整`<key>`或`<default>`元素的关节角度值。

### Q4: 场景纹理丢失怎么办？

A: 检查场景XML中的纹理路径，确保纹理文件在正确的相对路径下。

---

## 九、资产维护

### 9.1 版本控制

所有资产均已纳入Git版本控制，大型二进制文件使用Git LFS管理。

### 9.2 更新日志

- **2025-07-07**: 创建初始资产文档
- 后续更新将在此记录

### 9.3 贡献指南

新增资产需遵循以下规范：
1. 保持与现有资产一致的目录结构
2. 提供完整的验证报告
3. 包含必要的元数据
4. 通过物理仿真测试

---

## 十、联系方式

如有资产使用问题，请联系项目维护者。

---

**文档生成时间**: 2025-07-07
**资产总大小**: 2.6GB
**资产总数**: 2954个文件
**资产类别**: 58种
