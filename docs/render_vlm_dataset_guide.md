# VLM 数据集渲染工具使用指南

## 概述

`render_vlm_dataset.py` 是 VLABench 的核心渲染工具,用于为 VLM 评测任务生成高质量的多视角图像和实例分割掩码。该工具集成了自动化场景验证系统,确保渲染前模型放置正确。

**主要功能:**
- 🎨 多视角 RGB 图像渲染 (2x2 网格布局)
- 🎭 彩色实例分割掩码生成 (带物体标签)
- ✅ 自动场景验证 (高度、朝向、碰撞、可见性检查)
- 📊 验证报告生成 (JSON 格式)
- 🔄 批量处理支持
- 🚫 离线渲染 (OSMesa, 无需 X11 显示)

---

## 快速开始

### 基础用法

```bash
# 激活 conda 环境
source /ssd/mkqin/miniconda3/bin/activate vlabench_2

# 渲染单个任务 (自动检测样本数量)
python scripts/render_vlm_dataset.py --task insert_tube_centrifuge --dimension "M&T"

# 渲染指定数量的样本
python scripts/render_vlm_dataset.py --task pour_tube --dimension "M&T" --num-examples 5

# 覆盖已存在的图像
python scripts/render_vlm_dataset.py --task lift_object --dimension "M&T" --overwrite
```

### 完整工作流

```bash
# 1. 创建任务配置
python scripts/create_vlm_task.py --task insert_tube_centrifuge --num-examples 3

# 2. 渲染图像 + 自动验证
python scripts/render_vlm_dataset.py --task insert_tube_centrifuge --dimension "M&T" --overwrite

# 3. 检查验证报告
cat dataset/vlm_evaluation_v1.0/M&T/insert_tube_centrifuge/example0/input/validation_report.json

# 4. 查看渲染结果
ls dataset/vlm_evaluation_v1.0/M&T/insert_tube_centrifuge/example0/input/
# 输出: input.png  input_mask.png  validation_report.json
```

---

## 命令行参数

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--task` | 任务名称 (必须在 create_vlm_task.py 中定义) | `--task insert_tube_centrifuge` |

### 可选参数

| 参数 | 默认值 | 说明 | 示例 |
|------|--------|------|------|
| `--dimension` | `"M&T"` | 评测维度 (M&T, Spatial, etc.) | `--dimension "Spatial"` |
| `--num-examples` | 自动检测 | 渲染的样本数量 | `--num-examples 10` |
| `--overwrite` | False | 覆盖已存在的图像 | `--overwrite` |
| `--cameras` | `[0,1,2,3]` | 相机 ID 列表 | `--cameras 0 1 2 3` |

### 参数详解

#### `--task` (必需)
- 指定要渲染的任务名称
- 任务必须在 `scripts/create_vlm_task.py` 的 `BUILTIN_TEMPLATES` 中定义
- 对应的 `env_config.json` 文件必须已存在于数据集目录

**可用任务列表:**
```bash
# 查看所有可用任务
python scripts/create_vlm_task.py --list-templates
```

#### `--dimension`
- 评测维度,决定数据集存放路径
- 默认: `"M&T"` (Manipulation & Tool use)
- 其他常见值: `"Spatial"`, `"Reasoning"`, `"Visual"`

#### `--num-examples`
- 如果不指定,会自动检测任务目录中的样本数量
- 手动指定可用于只渲染前 N 个样本

#### `--overwrite`
- 默认跳过已存在的图像文件
- 使用此参数强制重新渲染所有样本

#### `--cameras`
- 指定使用哪些相机角度
- VLABench 默认有 4 个固定相机 (ID: 0, 1, 2, 3)
- 可自定义子集: `--cameras 0 2` (只渲染相机 0 和 2)

---

## 输出文件

### 目录结构

```
dataset/vlm_evaluation_v1.0/
└── M&T/
    └── insert_tube_centrifuge/
        ├── example0/
        │   ├── env_config/
        │   │   └── env_config.json          # 环境配置
        │   ├── input/
        │   │   ├── input.png                # RGB 图像 (512x512)
        │   │   ├── input_mask.png           # 分割掩码 (512x512)
        │   │   ├── validation_report.json   # 验证报告 (NEW)
        │   │   └── instruction.txt          # 任务指令
        │   └── output/
        │       └── operation_sequence.json  # 操作序列
        ├── example1/
        │   └── ...
        └── example2/
            └── ...
```

### 文件说明

#### 1. `input.png` - RGB 图像
- **格式**: PNG, 512x512 像素
- **布局**: 2x2 网格 (4 个相机视角)
- **内容**: 原始 RGB 渲染,无标注
- **用途**: 作为 VLM 输入图像

**视角布局:**
```
┌─────────┬─────────┐
│ 相机 0  │ 相机 1  │  相机 0: 前视角
│         │         │  相机 1: 侧视角
├─────────┼─────────┤  相机 2: 顶视角
│ 相机 2  │ 相机 3  │  相机 3: 后视角
│         │         │
└─────────┴─────────┘
```

#### 2. `input_mask.png` - 实例分割掩码
- **格式**: PNG, 512x512 像素
- **布局**: 2x2 网格 (与 input.png 对应)
- **内容**:
  - 半透明彩色高亮 (任务物体)
  - 黑色数字标签 (物体 ID: 0, 1, 2, ...)
  - 保留背景纹理 (70% 原图 + 30% 彩色 overlay)
- **用途**: 辅助 VLM 识别物体边界和 ID

**颜色映射:**
- 物体 0: 红色
- 物体 1: 绿色
- 物体 2: 蓝色
- 物体 3: 黄色
- ...

**标签位置:**
- 放置在物体上方,避免遮挡主体
- 白色数字 + 黑色背景框
- 字体大小: 0.5x

#### 3. `validation_report.json` - 验证报告 (NEW)
- **格式**: JSON
- **内容**: 场景验证结果 (高度、朝向、碰撞、可见性)
- **用途**:
  - 自动检测模型放置问题
  - 提供修复建议
  - 质量控制

**报告结构:**
```json
{
    "overall_status": "PASS" | "FAILED",
    "checks": {
        "height": [...],        // 高度检查结果
        "orientation": [...],   // 朝向检查结果
        "collision": {...},     // 碰撞检查结果
        "visibility": {...}     // 可见性检查结果
    },
    "warnings": [],            // 警告列表
    "errors": []               // 错误列表
}
```

详见 [场景验证系统文档](VALIDATION_SYSTEM_README.md)

---

## 渲染流程

### 执行步骤

```
1. 加载环境配置 (env_config.json)
   ↓
2. 创建 MuJoCo 物理环境
   ↓
3. ID 映射 (geom ID → 顺序 ID)
   ↓
4. 场景验证 (NEW)
   ├─ 高度检查
   ├─ 朝向检查
   ├─ 碰撞检查
   └─ 可见性检查
   ↓
5. 保存验证报告
   ↓
6. 多视角渲染
   ├─ 相机 0: RGB + 分割
   ├─ 相机 1: RGB + 分割
   ├─ 相机 2: RGB + 分割
   └─ 相机 3: RGB + 分割
   ↓
7. 图像拼接 (2x2 网格)
   ↓
8. 生成彩色掩码 + 标签
   ↓
9. 保存 PNG 文件
```

### 关键函数

#### `render_vlm_example(env_config_path, output_image_path, output_mask_path, camera_ids)`
**功能**: 渲染单个样本

**流程**:
1. 加载环境配置
2. 创建物理环境
3. ID 映射 (识别任务物体)
4. **场景验证** (检查模型放置)
5. 多视角渲染
6. 生成彩色掩码
7. 保存图像和验证报告

**参数**:
- `env_config_path`: 环境配置文件路径
- `output_image_path`: RGB 输出路径
- `output_mask_path`: 掩码输出路径
- `camera_ids`: 相机 ID 列表

#### `render_vlm_task(task_name, num_examples, dimension, overwrite)`
**功能**: 批量渲染整个任务

**流程**:
1. 确定数据集路径
2. 遍历所有样本
3. 检查是否需要渲染 (跳过/覆盖)
4. 调用 `render_vlm_example()` 渲染
5. 统计成功/失败/跳过数量

**参数**:
- `task_name`: 任务名称
- `num_examples`: 样本数量
- `dimension`: 评测维度
- `overwrite`: 是否覆盖

---

## 场景验证系统

### 概述

渲染前自动检查场景配置,确保:
- ✅ 物体未嵌入桌面或悬浮
- ✅ 物体朝向正确 (未倒置)
- ✅ 物体间无异常碰撞
- ✅ 物体在相机视野内

### 验证检查项

#### 1. 高度检查
**检测**: 物体 z 坐标是否合理

**阈值**: ±0.05m (5cm)

**示例输出**:
```
✓ test_tube_0: 高度正常 (0.805m)
✗ centrifuge_1: 物体嵌入桌面 (实际高度 0.750m < 期望 0.820m)
  建议: 建议在 XML 中增加 pos z 偏移约 0.070m
```

#### 2. 朝向检查
**检测**: 物体局部 z 轴与世界 z 轴夹角

**阈值**:
- < 45°: 正向 ✅
- 45° ~ 135°: 倾斜 ⚠️
- > 135°: 倒置 ❌

**示例输出**:
```
✓ test_tube_0: 朝向正常 (0.0°)
✗ centrifuge_1: 物体倒置 (z 轴与垂直方向夹角 180.0°)
  建议: 建议在 XML 中添加或调整 euler 旋转，尝试 euler='1.57 0 0'
```

#### 3. 碰撞检查
**检测**: 任务物体间穿透深度

**阈值**: < -0.001m (1mm)

**示例输出**:
```
✓ 无异常碰撞
✗ 检测到 1 处物体重叠: test_tube_0-centrifuge_1
  建议: 建议调整物体位置，增大 position_base 中 x/y 的间距
```

#### 4. 可见性检查
**检测**: 物体是否至少在一个相机中可见

**阈值**: > 100 像素

**示例输出**:
```
✓ 所有物体均可见
⚠ 以下物体在所有相机中均不可见: test_tube_0
  建议: 建议检查物体位置是否超出工作区域，或被其他物体完全遮挡
```

### 验证报告示例

```json
{
    "overall_status": "PASS",
    "checks": {
        "height": [
            {
                "component": "test_tube_0",
                "expected_z": 0.805,
                "actual_z": 0.805,
                "diff": 0.0,
                "status": "OK"
            }
        ],
        "orientation": [
            {
                "component": "test_tube_0",
                "local_z": [0.0, 0.0, 1.0],
                "angle_with_vertical": 0.0,
                "status": "OK"
            }
        ],
        "collision": {
            "status": "OK",
            "collisions": []
        },
        "visibility": {
            "status": "OK",
            "visibility": {
                "test_tube_0": [0, 1, 2, 3],
                "centrifuge_1": [0, 1, 2, 3]
            }
        }
    },
    "warnings": [],
    "errors": []
}
```

详细文档: [场景验证系统使用指南](VALIDATION_SYSTEM_README.md)

---

## 图像生成算法

### RGB 图像生成

```python
# 1. 渲染单视角
rgb = env.render(camera_id=cam_id, width=256, height=256)

# 2. 拼接 2x2 网格
row1 = np.hstack([rgb_images[0], rgb_images[1]])  # 上行: 相机 0, 1
row2 = np.hstack([rgb_images[2], rgb_images[3]])  # 下行: 相机 2, 3
stacked_rgb = np.vstack([row1, row2])              # 垂直拼接

# 3. RGB → BGR (OpenCV 格式)
stacked_bgr = cv2.cvtColor(stacked_rgb, cv2.COLOR_RGB2BGR)

# 4. 保存
cv2.imwrite(output_image_path, stacked_bgr)
```

### 分割掩码生成

```python
# 1. 渲染分割图
seg = env.render(camera_id=cam_id, width=256, height=256, segmentation=True)
instance_ids = seg[:, :, 0].astype(int)  # 提取 instance ID

# 2. 创建半透明 overlay
overlay = np.zeros_like(rgb)
for obj_id in target_geom_ids:
    seq_id = target_geom_to_seq[obj_id]  # 映射到顺序 ID
    color = COLOR_PALETTE[seq_id % len(COLOR_PALETTE)]
    mask = (instance_ids == obj_id)
    overlay[mask] = color[:3]  # RGB 颜色

# 3. 混合图像 (70% 原图 + 30% 彩色)
blended = cv2.addWeighted(rgb, 0.70, overlay, 0.30, 0)

# 4. 添加数字标签
for obj_id, seq_id in target_geom_to_seq.items():
    # 计算 bounding box
    y_coords, x_coords = np.where(instance_ids == obj_id)
    center_x = int((np.min(x_coords) + np.max(x_coords)) / 2)
    label_y = np.min(y_coords) - padding  # 标签放在物体上方

    # 绘制黑框 + 白字
    cv2.rectangle(labeled, top_left, bottom_right, (0, 0, 0), -1)
    cv2.putText(labeled, str(seq_id), text_pos, font, 0.5, (255, 255, 255), 2)

# 5. 拼接和保存
# (与 RGB 图像相同)
```

### ID 映射逻辑

**问题**: MuJoCo 的 geom ID 是全局的,包括场景、机器人等所有物体,但我们只需要标注任务物体。

**解决方案**:
1. 统计任务物体数量: `num_objects = len(env_config['task']['components'])`
2. 从分割图获取所有唯一 ID: `unique_ids = np.unique(instance_ids)`
3. 取最大的 N 个 ID 作为任务物体: `target_geom_ids = sorted(unique_ids)[-num_objects:]`
4. 映射到顺序 ID: `{geom_id: seq_id}` (0, 1, 2, ...)

**示例**:
```python
# 场景中所有 geom ID: [2, 7, 14, 50, 78, 88, 110, 114]
# 任务物体数量: 4
# → 取最大的 4 个: [78, 88, 110, 114]
# → 映射: {78: 0, 88: 1, 110: 2, 114: 3}
```

---

## 使用示例

### 示例 1: 渲染单个任务

```bash
# 场景: 离心机试管插入任务, 3 个样本
python scripts/render_vlm_dataset.py \
    --task insert_tube_centrifuge \
    --dimension "M&T" \
    --num-examples 3 \
    --overwrite
```

**输出**:
```
============================================================
VLM 数据集渲染工具 - 分割掩码版
============================================================
任务: insert_tube_centrifuge
样本数: 3
维度: M&T

渲染 example0...
  任务物体数量: 4
  ID 映射: {78: 0, 88: 1, 110: 2, 114: 3}

==================================================
  开始场景验证...
==================================================
  [验证] 检查物体高度...
    ✓ test_tube_0: 高度正常 (0.805m)
    ✓ centrifuge_1: 高度正常 (0.820m)

  [验证] 检查物体朝向...
    ✓ test_tube_0: 朝向正常 (0.0°)
    ✓ centrifuge_1: 朝向正常 (0.0°)

  [验证] 检查物体碰撞...
    ✓ 无异常碰撞

  [验证] 检查相机可见性...
    ✓ 所有物体均可见

  ✓ 验证报告已保存
==================================================
  验证结果: ✓ 通过 - 场景配置正常
==================================================

  ✓ 保存: .../example0/input/input.png (512, 512, 3)
  ✓ 保存: .../example0/input/input_mask.png (512, 512, 3)

渲染 example1...
  ...

渲染 example2...
  ...

============================================================
完成! 成功:3 跳过:0 失败:0
============================================================
```

### 示例 2: 批量渲染多个任务

```bash
#!/bin/bash
# batch_render.sh

DIMENSION="M&T"
TASKS=(
    "pour_tube"
    "lift_object"
    "place_in_container"
    "pick_tube"
    "insert_tube_centrifuge"
)

for task in "${TASKS[@]}"; do
    echo "渲染任务: $task"
    python scripts/render_vlm_dataset.py \
        --task "$task" \
        --dimension "$DIMENSION" \
        --overwrite
    echo "---"
done

echo "所有任务渲染完成!"
```

运行:
```bash
chmod +x batch_render.sh
./batch_render.sh
```

### 示例 3: 自定义相机角度

```bash
# 只渲染前视角和顶视角
python scripts/render_vlm_dataset.py \
    --task pour_tube \
    --dimension "M&T" \
    --cameras 0 2
```

### 示例 4: 验证失败后修复

```bash
# 1. 首次渲染 (发现问题)
python scripts/render_vlm_dataset.py --task my_task --dimension "M&T"

# 输出:
# ✗ centrifuge_1: 物体倒置 (z 轴与垂直方向夹角 180.0°)
#   建议: 建议在 XML 中添加或调整 euler 旋转，尝试 euler='1.57 0 0'

# 2. 修改模型 XML
vim VLABench/assets/obj/meshes/lab_equipment/centrifuge/centrifuge.xml
# 添加: <body name="centrifuge" euler="1.57 0 0" pos="0 0 0.12">

# 3. 重新渲染验证
python scripts/render_vlm_dataset.py --task my_task --dimension "M&T" --overwrite

# 输出:
# ✓ centrifuge_1: 朝向正常 (0.0°)
# ✓ 验证结果: ✓ 通过 - 场景配置正常
```

---

## 常见问题

### Q1: 渲染时报错 "ModuleNotFoundError: No module named 'numpy'"

**原因**: 未激活正确的 conda 环境

**解决**:
```bash
source /ssd/mkqin/miniconda3/bin/activate vlabench_2
python scripts/render_vlm_dataset.py --task your_task --dimension "M&T"
```

### Q2: 渲染出来的图像全黑或异常

**原因**:
- OSMesa 渲染器未正确初始化
- 相机位置错误
- 光照问题

**解决**:
```bash
# 检查环境变量
echo $MUJOCO_GL  # 应该是 "osmesa"

# 如果不是,手动设置
export MUJOCO_GL=osmesa
python scripts/render_vlm_dataset.py --task your_task --dimension "M&T"
```

### Q3: 验证报告显示 "物体在所有相机中均不可见"

**原因**:
- 物体位置超出相机视野
- 物体被其他大型物体完全遮挡
- 物体尺寸过小

**解决**:
```python
# 调整任务模板中的 position_base
"position_base": [0.0, 0.0, 0.8],  # 确保在桌面中心附近 (-0.3 ~ 0.3)
```

### Q4: 分割掩码中物体 ID 不正确

**原因**: ID 映射逻辑取了错误的 geom

**解决**: 检查任务物体数量是否正确
```python
# 在 create_vlm_task.py 中检查
"components": [...],  # 确保 components 列表准确
"task_objects": [...] # 确保 task_objects 包含所有任务物体
```

### Q5: 渲染速度很慢

**原因**:
- 场景复杂度高
- 渲染分辨率过大
- 验证检查耗时

**优化**:
```bash
# 1. 降低渲染分辨率 (修改代码中的 width=256, height=256)
# 2. 减少相机数量
python scripts/render_vlm_dataset.py --task your_task --cameras 0 2

# 3. 跳过已渲染的样本 (不使用 --overwrite)
python scripts/render_vlm_dataset.py --task your_task --dimension "M&T"
```

### Q6: 验证报告显示碰撞,但实际看起来没问题

**原因**:
- 穿透阈值设置过严格
- 物理引擎初始化时的微小接触

**解决**: 检查验证报告中的 `penetration` 值
```json
"penetration": -0.0015  // < -0.001m (1mm) 才算异常碰撞
```

如果穿透深度很小 (< 2mm),可以忽略。

---

## 技术细节

### 环境配置

渲染工具依赖以下环境变量:

```python
os.environ['MUJOCO_PY_NO_XR'] = '1'      # 禁用 VR 扩展
os.environ['DISPLAY'] = ''               # 无头模式
os.environ['MUJOCO_GL'] = 'osmesa'       # 使用 OSMesa 离线渲染
os.environ['VLABENCH_ROOT'] = project_root  # 项目根目录
```

### 相机配置

VLABench 默认使用 4 个固定相机:

| ID | 位置 | 朝向 | 用途 |
|----|------|------|------|
| 0 | 前方 | 向后 | 正面视角 |
| 1 | 侧方 | 向左 | 侧面视角 |
| 2 | 上方 | 向下 | 顶视角 |
| 3 | 后方 | 向前 | 背面视角 |

### 渲染参数

```python
# RGB 图像
env.render(
    camera_id=cam_id,
    width=256,          # 单视角宽度
    height=256          # 单视角高度
)

# 分割图
env.render(
    camera_id=cam_id,
    width=256,
    height=256,
    segmentation=True   # 启用分割模式
)
```

输出尺寸:
- 单视角: 256x256
- 2x2 网格: 512x512

### 颜色调色板

```python
COLOR_PALETTE_WITH_ALPHA = [
    [255, 0, 0, 100],      # 0: 半透明红
    [0, 255, 0, 100],      # 1: 半透明绿
    [0, 0, 255, 100],      # 2: 半透明蓝
    [255, 255, 0, 100],    # 3: 半透明黄
    [255, 0, 255, 100],    # 4: 半透明品红
    [0, 255, 255, 100],    # 5: 半透明青
    # ... 更多颜色
]
```

Alpha 值 (100/255 ≈ 40%) 用于半透明 overlay。

### 混合算法

```python
# 70% 原始 RGB + 30% 彩色 overlay
blended = cv2.addWeighted(rgb, 0.70, overlay, 0.30, 0)
```

保留纹理细节的同时添加彩色高亮。

---

## 性能优化

### 批量渲染

```python
# 不推荐: 逐个样本手动运行
python scripts/render_vlm_dataset.py --task my_task --num-examples 1
python scripts/render_vlm_dataset.py --task my_task --num-examples 2
...

# 推荐: 一次性批量渲染
python scripts/render_vlm_dataset.py --task my_task --num-examples 100
```

### 跳过已渲染样本

```python
# 默认行为: 跳过已存在的文件
if not overwrite and os.path.exists(out_img) and os.path.exists(out_mask):
    print(f"example{i}: 已存在,跳过")
    skip += 1
    continue
```

### 并行渲染 (高级)

```bash
# 将任务分组,使用 GNU Parallel
parallel -j 4 python scripts/render_vlm_dataset.py --task {} --dimension "M&T" ::: \
    pour_tube lift_object place_in_container pick_tube
```

---

## 扩展开发

### 添加新的验证规则

在 `render_vlm_dataset.py` 中添加:

```python
def check_object_size(env, component_name, max_size=0.5):
    """检查物体尺寸是否合理"""
    physics = env.physics
    body_id = find_body_for_component(physics, component_name)

    # 获取 geom 尺寸
    geom_adr = physics.model.body_geomadr[body_id]
    geom_num = physics.model.body_geomnum[body_id]

    max_dim = 0
    for i in range(geom_num):
        geom_id = geom_adr + i
        geom_size = physics.model.geom_size[geom_id]
        max_dim = max(max_dim, max(geom_size))

    if max_dim > max_size:
        return False, f"物体过大 ({max_dim:.3f}m)", {
            "status": "FAILED",
            "max_dimension": max_dim
        }

    return True, "", {"status": "OK", "max_dimension": max_dim}

# 在 validate_scene() 中调用
report['checks']['size'] = []
for obj in task_objects:
    passed, problem, details = check_object_size(env, obj['name'])
    report['checks']['size'].append(details)
```

### 自定义渲染分辨率

修改渲染调用:

```python
# 原代码 (256x256 → 512x512)
rgb = env.render(camera_id=cam_id, width=256, height=256)

# 修改为高分辨率 (512x512 → 1024x1024)
rgb = env.render(camera_id=cam_id, width=512, height=512)
```

### 添加新的相机角度

在场景配置中添加相机:

```xml
<!-- VLABench/assets/scenes/laboratory_0/laboratory_0.xml -->
<worldbody>
  <camera name="camera_4" pos="0 -2 1.5" quat="..." fovy="45"/>
</worldbody>
```

使用新相机:
```bash
python scripts/render_vlm_dataset.py --task my_task --cameras 0 1 2 3 4
```

---

## 相关文档

- **场景验证系统**: [VALIDATION_SYSTEM_README.md](VALIDATION_SYSTEM_README.md)
- **任务创建指南**: [vlm_task_pipeline_guide.md](vlm_task_pipeline_guide.md)
- **VLABench 官方文档**: [README.md](../README.md)

---

## 版本历史

### v2.0 (2026-03-04)
- ✨ 新增自动化场景验证系统
- ✨ 新增验证报告生成 (validation_report.json)
- ✨ 支持高度、朝向、碰撞、可见性检查
- 🐛 修复 body 名称映射问题 (VLABench 分层结构)
- 📝 完善文档和使用示例

### v1.0 (2026-02-06)
- 🎉 初始版本
- 🎨 多视角 RGB 图像渲染
- 🎭 彩色实例分割掩码生成
- 📊 ID 映射逻辑
- 🖼️ 2x2 网格布局

---

## 技术支持

遇到问题或建议改进?

- **Issues**: [https://github.com/anthropics/vlabench/issues](https://github.com/anthropics/vlabench/issues)
- **文档**: `/ssd/mkqin/workspace/VLABench/docs/`
- **代码**: `/ssd/mkqin/workspace/VLABench/scripts/render_vlm_dataset.py`

---

**最后更新**: 2026-03-04
**作者**: Claude + QMK
**版本**: 2.0
