# 模型资产下载与处理 Pipeline 文档

**日期**: 2026-03-16
**状态**: 已实现并测试通过

---

## 概述

本 Pipeline 实现了从 Objaverse 开源 3D 模型库自动下载模型、转换为 MuJoCo MJCF 格式、后处理修复、物理验证的完整流程。目标是让下载的模型能够直接加载到 VLABench 仿真场景中使用。

### 核心问题

从 Objaverse 下载的原始模型无法直接在 VLABench 中使用，存在以下问题：

1. **搜索不精确** — 搜索 "beaker" 返回花瓶等文化遗产模型
2. **缺少物理属性** — 无 `<compiler>`、`<inertial>`、friction/solref/solimp
3. **碰撞 mesh 过多** — 一个烧杯模型有 126 个 collision mesh
4. **几何中心偏移** — OBJ 顶点不以原点为中心，导致视觉-物理错位
5. **比例尺未知** — 模型可能是 13 米或 111 米大
6. **缺少抓取点** — 无 `<site group="4">` grasp sites

---

## 文件清单

| 文件 | 用途 | 类型 |
|------|------|------|
| `scripts/get_assets.py` | 主 Pipeline：搜索、下载、转换、后处理、验证 | 修改 |
| `fix_obj2mjcf_xml.py` | OBJ 居中、尺寸归一化、XML 物理属性修复 | 修改 |
| `scripts/validate_asset.py` | MuJoCo 加载验证 + 视频录制 | 新建 |

---

## 1. get_assets.py — 主下载 Pipeline

### 功能

从 Objaverse（~80万个3D模型）搜索并下载模型，经过完整处理后输出可用于 VLABench 的 MJCF 模型。

### 处理流程

```
1. 搜索 Objaverse 元数据，按相关性排序
2. 下载 GLB 文件
3. GLB → OBJ (trimesh)
4. OBJ → MJCF (obj2mjcf，含凸分解)
5. 修复 XML 资源路径
6. 几何中心归零（修改 OBJ 文件）
7. 尺寸归一化（XML 中添加 scale）
8. 添加物理属性（compiler/default/inertial/grasppoints）
9. 渲染预览图 (PNG)
10. MuJoCo 加载验证 + 录制验证视频 (MP4)
11. 保存后处理报告 + 验证报告
```

### 用法

```bash
cd /ssd/mkqin/workspace/VLABench/scripts

# 基本用法：下载 5 个烧杯模型
python get_assets.py --keyword beaker --max_downloads 5

# 使用实验室上下文（推荐）：优先筛选实验器材
python get_assets.py --keyword beaker --max_downloads 5 --context lab

# 其他参数
python get_assets.py --keyword flask --max_downloads 10 --context lab --output_dir ./assets/review
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--keyword` | (必填) | 搜索关键词 |
| `--max_downloads` | 10 | 最大下载数量 |
| `--context` | None | 搜索上下文：`lab`（实验室）或 `home`（家居） |
| `--output_dir` | `./assets/review` | 输出目录 |
| `--skip_existing` | False | 跳过已存在的模型 |

### 上下文搜索机制

`--context lab` 会启用以下排序策略：

- **类别加分**: `science-technology` +2 分
- **类别减分**: `cultural-heritage-history`、`art-abstract` -2 分
- **关键词加分**: 模型名称/标签/描述中包含实验室相关词（lab, chemistry, glass, equipment 等），最多 +3 分
- **质量因子**: `likeCount` 作为同等条件下的排序依据

### 输出目录结构

```
assets/review/beaker/
├── {uid}/                          # 每个模型一个目录
│   ├── {uid}.glb                   # 原始 GLB 文件
│   ├── {uid}.obj                   # 转换后的 OBJ（已居中）
│   ├── {uid}.xml                   # 最终 MJCF XML（含物理属性）
│   ├── {uid}/                      # 碰撞 mesh 子目录
│   │   ├── {uid}_collision_0.obj
│   │   ├── {uid}_collision_1.obj
│   │   └── ...
│   ├── material_0.png              # 贴图
│   ├── metadata.json               # Objaverse 元数据
│   ├── sources.txt                 # 来源信息
│   ├── preview.png                 # 模型预览图
│   ├── postprocess_report.json     # 后处理报告
│   ├── validation_report.json      # 验证报告
│   └── validation.mp4              # 验证视频（模型落到地面）
```

---

## 2. fix_obj2mjcf_xml.py — 后处理工具

### 功能

解决 obj2mjcf 转换后 XML 缺少物理属性的问题，同时处理 OBJ 几何中心偏移和尺寸归一化。

### 完整后处理流程

1. **几何中心归零** — 读取所有 OBJ 文件（主 OBJ + collision OBJs），计算 bounding box 中心，平移所有顶点使中心归零。阈值 1cm，低于此不修改
2. **朝向检查** — 判断最长轴是否为 Z 轴，如果不是则输出警告（不自动修复）
3. **尺寸归一化** — 如果最大维度超出 [0.02m, 0.3m] 范围，计算缩放因子使其归到 0.15m。通过在 XML `<mesh>` 标签添加 `scale` 属性实现，不修改 OBJ 文件
4. **XML 物理属性修复** — 添加 `<compiler>`、`<default>` classes（visual/collision/grasppoint）、`<inertial>`、grasp sites

### 用法

```bash
cd /ssd/mkqin/workspace/VLABench

# 完整后处理（推荐）：居中 + 归一化 + 物理属性
python fix_obj2mjcf_xml.py --model-dir scripts/assets/review/beaker/{uid} --uid {uid}

# 仅修复单个 XML 的物理属性
python fix_obj2mjcf_xml.py path/to/model.xml --mass 0.05 --height 0.15

# 批量修复目录中所有 XML
python fix_obj2mjcf_xml.py --dir scripts/assets/review/beaker/

# 完整后处理 + 移除 freejoint
python fix_obj2mjcf_xml.py --model-dir path/to/model --uid xxx --remove-freejoint
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-dir` | - | 模型目录（完整后处理模式） |
| `--uid` | - | 模型 UID（完整后处理模式） |
| `--mass` | 0.02 | 物体质量 (kg) |
| `--target-max-dim` | 0.15 | 目标最大尺寸 (m) |
| `--remove-freejoint` | False | 是否移除 freejoint |
| `--no-grasppoints` | False | 不添加抓取点 |

### 修复后 XML 结构对比

**修复前**（obj2mjcf 原始输出）：
```xml
<mujoco>
  <default>
    <default class="visual"><geom group="2" type="mesh" contype="0" conaffinity="0"/></default>
    <default class="collision"><geom group="3" type="mesh"/></default>
  </default>
  <asset>...</asset>
  <worldbody>
    <body name="xxx">
      <freejoint/>
      <geom ... class="visual"/>
      <geom ... class="collision"/>  <!-- 126 个 -->
    </body>
  </worldbody>
</mujoco>
```

**修复后**：
```xml
<mujoco>
  <compiler boundmass="0.015" boundinertia="1e-05" angle="radian"/>
  <default>
    <default class="visual">
      <geom group="2" type="mesh" contype="0" conaffinity="0" density="50"/>
    </default>
    <default class="collision">
      <geom type="mesh" density="50" friction="1.5 0.1 0.1"
            solimp="0.9 0.95 0.001" solref="0.02 1"/>
    </default>
    <default class="grasppoint">
      <site type="sphere" size="0.005" group="4" rgba="0 0 1 1"/>
    </default>
  </default>
  <asset>
    <mesh file="xxx.obj" scale="0.029859 0.029859 0.029859"/>  <!-- 尺寸归一化 -->
    ...
  </asset>
  <worldbody>
    <body name="xxx">
      <freejoint/>
      <inertial pos="0 0 0.075" mass="0.02" diaginertia="0.0008 0.0008 0.0005"/>
      <geom ... class="visual"/>
      <geom ... class="collision"/>
      <site class="grasppoint" pos="0 0 0.120"/>
      <site class="grasppoint" pos="0 0 0.130"/>
      <site class="grasppoint" pos="0 0 0.140"/>
    </body>
  </worldbody>
</mujoco>
```

### 幂等性

`fix_obj2mjcf_xml` 支持多次运行，会先移除已有的 `<compiler>`、`<default>`、`<inertial>`、grasppoint sites 再重新添加。

### Python API

```python
from fix_obj2mjcf_xml import postprocess_model, get_obj_dimensions, center_obj_file

# 完整后处理
report = postprocess_model(model_dir="path/to/model", uid="xxx")

# 单独使用工具函数
dims = get_obj_dimensions("model.obj")
# dims = {'center_x': -0.15, 'x_range': 0.09, 'max_dim': 13.35, ...}

was_modified, offset = center_obj_file("model.obj", threshold=0.01)
```

---

## 3. validate_asset.py — MuJoCo 验证脚本

### 功能

创建最小 MuJoCo 场景（地面 + 待验证模型），进行物理仿真，检查模型是否能正常加载和渲染。可生成验证视频（MP4），直观查看模型从空中落到地面的过程。

### 验证项

| 检查项 | 说明 | PASS 条件 |
|--------|------|-----------|
| xml_load | dm_control mjcf.from_path() | 无异常 |
| physics_compile | Physics.from_mjcf_model() | 无异常 |
| body_found | 模型 body 存在 | 找到非 world 的 body |
| stability | 仿真后 z 坐标合理 | -0.5 < z < 5 |
| xy_drift | XY 方向漂移 | drift < 1.0m |
| video | 验证视频录制 | 录制成功 |
| render | 验证截图 | 渲染成功 |

### 用法

```bash
cd /ssd/mkqin/workspace/VLABench

# 验证单个模型（仅物理检查）
python scripts/validate_asset.py path/to/model.xml

# 验证 + 录制视频（推荐）
python scripts/validate_asset.py path/to/model.xml --video

# 验证 + 截图
python scripts/validate_asset.py path/to/model.xml --render

# 批量验证 + 录制视频
python scripts/validate_asset.py --dir scripts/assets/review/beaker/ --video

# 自定义视频参数
python scripts/validate_asset.py path/to/model.xml --video --video-duration 5.0 --video-fps 60
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--video` | False | 录制验证视频 (MP4) |
| `--render` | False | 渲染验证截图 (PNG) |
| `--steps` | 200 | 物理检查仿真步数 |
| `--video-duration` | 3.0 | 视频时长 (秒) |
| `--video-fps` | 30 | 视频帧率 |
| `--dir` | - | 批量验证目录 |

### 输出文件

- `validation_report.json` — 验证报告（JSON）
- `validation.mp4` — 验证视频（模型从空中落到地面）
- `validation_preview.png` — 验证截图（使用 `--render` 时）

### Python API

```python
from validate_asset import validate_asset, validate_directory

# 单个验证
report = validate_asset("model.xml", render_video=True)
# report = {"status": "PASS", "checks": {...}, "video_path": "..."}

# 批量验证
results = validate_directory("scripts/assets/review/beaker/", render_video=True)
```

### Pipeline 集成

在 `get_assets.py` 的 `process_model()` 中，验证默认开启视频录制（`render_video=True`），每个下载的模型自动生成 `validation.mp4`。

---

## 测试结果

对 6 个已下载的 beaker 模型进行后处理 + 验证，结果：

| 模型 UID | 原始尺寸 | 中心偏移 | 缩放后 | 朝向警告 | 验证 |
|----------|----------|----------|--------|----------|------|
| 0c3fcd77... | 0.28m | 0m | 无需缩放 | Y轴最长 | PASS |
| 18910c5d... | 5.02m | 0.95m | 0.15m | 无 | PASS |
| 2bcc5a13... | 111.07m | 7.24m | 0.15m | X轴最长 | PASS |
| b63ae471... | 0.64m | 0.09m | 0.15m | 无 | PASS |
| b8594f7d... | 46.06m | 13.95m | 0.15m | X轴最长 | PASS |
| e684c968... | 13.35m | 6.84m | 0.15m | 无 | PASS |

全部 6/6 通过验证。

---

## 依赖

```
trimesh          # GLB → OBJ 转换
obj2mjcf         # OBJ → MJCF 转换 (含凸分解)
mujoco           # 物理仿真 + 渲染
dm-control       # MJCF 高级 API
objaverse        # 模型搜索和下载
imageio          # 视频写入 (imageio.v3)
Pillow           # 图片处理
tqdm             # 进度条 (可选)
```

---

## 已知限制

1. **朝向不自动修复** — 部分模型最长轴不是 Z 轴（如 0c3fcd77 的 Y 轴最长），当前只输出警告，需要人工判断是否旋转
2. **碰撞 mesh 数量未减少** — obj2mjcf 的凸分解可能产生大量碰撞 mesh（如 126 个），当前未做简化
3. **质量固定为 0.02kg** — 不同尺寸的物体应有不同质量，当前统一使用默认值
4. **验证场景简化** — 仅使用地面，未放置桌面等真实场景元素

---

**最后更新**: 2026-03-16
