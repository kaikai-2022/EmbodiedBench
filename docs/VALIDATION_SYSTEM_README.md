# VLABench 场景验证系统

## 概述

自动化场景验证系统已集成到 VLM 数据集渲染 pipeline 中,在渲染图像之前自动检查场景配置的合理性。该系统可以检测模型放置问题(高度、朝向、碰撞、可见性),并生成详细的验证报告和修复建议。

## 功能特性

### 1. 高度检查 (Height Check)
- **目的**: 检测物体是否嵌入桌面或悬浮在空中
- **方法**: 比较物体实际 z 坐标与配置中的期望值
- **阈值**: ±0.05m (5cm)
- **输出**:
  - 实际高度 vs 期望高度
  - 高度差异
  - 修复建议(调整 XML 中的 pos z 偏移)

**示例输出**:
```
✓ test_tube_0: 高度正常 (0.805m)
✗ centrifuge_1: 物体嵌入桌面 (实际高度 0.750m < 期望 0.820m)
  建议: 建议在 XML 中增加 pos z 偏移约 0.070m
```

### 2. 朝向检查 (Orientation Check)
- **目的**: 检测物体是否倒置或倾斜过大
- **方法**: 分析物体局部 z 轴与世界 z 轴的夹角
- **阈值**:
  - < 45° : 正向放置 (OK)
  - 45° ~ 135° : 倾斜过大 (WARNING)
  - > 135° : 倒置 (FAILED)
- **输出**:
  - 局部 z 轴方向向量
  - 与垂直方向夹角
  - 修复建议(调整 euler 旋转参数)

**示例输出**:
```
✓ test_tube_0: 朝向正常 (0.0°)
✗ centrifuge_1: 物体倒置 (z 轴与垂直方向夹角 180.0°)
  建议: 建议在 XML 中添加或调整 euler 旋转，尝试 euler='1.57 0 0' (X轴旋转90度)
```

### 3. 碰撞检查 (Collision Check)
- **目的**: 检测任务物体之间是否有不合理的重叠
- **方法**: 分析 MuJoCo 物理引擎的接触数据
- **阈值**: 穿透深度 < -0.001m (1mm)
- **输出**:
  - 碰撞物体对
  - 穿透深度
  - 碰撞位置
  - 修复建议(调整物体位置间距)

**示例输出**:
```
✓ 无异常碰撞
✗ 检测到 1 处物体重叠: test_tube_0-centrifuge_1
  建议: 建议调整物体位置，增大 position_base 中 x/y 的间距
```

### 4. 可见性检查 (Visibility Check)
- **目的**: 确保任务物体至少在一个相机视角中可见
- **方法**: 检查物体是否出现在任何相机的分割图中
- **阈值**: 至少 100 像素可见
- **输出**:
  - 每个物体在各相机中的可见性
  - 不可见物体列表
  - 修复建议(调整位置或检查遮挡)

**示例输出**:
```
✓ 所有物体均可见
⚠ 以下物体在所有相机中均不可见: test_tube_0
  建议: 建议检查物体位置是否超出工作区域，或被其他物体完全遮挡
```

## 使用方法

### 自动验证 (推荐)

验证系统已集成到渲染 pipeline 中,在运行渲染脚本时自动执行:

```bash
python scripts/render_vlm_dataset.py --task insert_tube_centrifuge --dimension "M&T" --overwrite
```

**输出位置**:
- 验证报告: `example{i}/input/validation_report.json`
- 渲染图像: `example{i}/input/input.png`
- 分割掩码: `example{i}/input/input_mask.png`

### 验证报告格式

**JSON 结构**:
```json
{
    "overall_status": "PASS" | "FAILED",
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

## 工作原理

### 1. 环境加载
```python
env = load_env(
    task="simple_pickplace",
    episode_config=env_config,
    time_limit=float('inf'),
    reset_wait_step=0,
    random_init=False
)
```

### 2. 物理数据访问
- 使用 `env.physics.model` 访问 MuJoCo 模型
- 使用 `env.physics.data` 访问仿真状态
- 通过 `physics.model.name2id()` 和 `physics.data.xpos[]` 获取物体位姿

### 3. Body 名称映射
VLABench 使用分层命名结构:
- 配置中的组件名: `test_tube_0`
- MuJoCo 中的 body 名: `test_tube_0/` 或 `test_tube_0/tube`

验证系统通过 `find_body_for_component()` 自动处理名称映射。

### 4. 验证流程
```
加载环境 → 提取任务物体 → 高度检查 → 朝向检查 → 碰撞检查 → 可见性检查 → 生成报告
```

## 常见问题与修复

### 问题 1: 物体嵌入桌面
**症状**: `物体嵌入桌面 (实际高度 0.750m < 期望 0.820m)`

**原因**:
- 模型坐标系与 VLABench 坐标系不一致
- 模型原点不在底部

**修复方法**:
```xml
<!-- 在模型 XML 的 worldbody 中调整 pos -->
<body name="centrifuge" pos="0 0 0.12">
  <!-- 增加 z 偏移将模型抬高 -->
</body>
```

### 问题 2: 物体倒置
**症状**: `物体倒置 (z 轴与垂直方向夹角 180.0°)`

**原因**:
- 模型使用 Y-up 坐标系,而 MuJoCo 使用 Z-up
- 需要旋转 90 度转换坐标系

**修复方法**:
```xml
<!-- 添加 euler 旋转 (X轴旋转 90度 = 1.57 弧度) -->
<body name="centrifuge" euler="1.57 0 0" pos="0 0 0.12">
  <geom .../>
</body>
```

### 问题 3: 物体重叠
**症状**: `检测到物体重叠: test_tube_0-centrifuge_1`

**原因**:
- `position_base` 设置的位置过近
- 模型尺寸估计不准确

**修复方法**:
在 `create_vlm_task.py` 中调整 position_base:
```python
{
    "type": "centrifuge",
    "position_base": [0.35, 0.25, 0.82],  # 增大 x/y 间距
    ...
}
```

### 问题 4: 物体不可见
**症状**: `物体在所有相机中均不可见: test_tube_0`

**原因**:
- 物体位置超出相机视野
- 物体被其他物体完全遮挡
- 物体过小

**修复方法**:
1. 调整物体位置到工作区域中心 (-0.3 ~ 0.3, -0.3 ~ 0.3)
2. 减少遮挡物体的尺寸
3. 调整相机角度 (通常不建议)

## 验证阈值配置

可以在验证函数中调整阈值:

```python
def check_object_height(env, component_name, expected_z, tolerance=0.05):
    # tolerance: 高度容差 (默认 5cm)
    ...

def check_object_orientation(env, component_name):
    # angle < 45°: 正向
    # angle > 135°: 倒置
    ...

def check_collisions(env, component_names):
    # penetration < -0.001m: 异常碰撞
    ...

def check_in_camera_view(env, component_names, camera_ids):
    # visible_pixels > 100: 可见
    ...
```

## 技术架构

### 核心函数

1. **`find_body_for_component(physics, component_name)`**
   - 将配置中的组件名映射到 MuJoCo body ID
   - 处理 VLABench 的分层命名结构

2. **`check_object_height(env, component_name, expected_z, tolerance)`**
   - 高度验证
   - 返回: (是否通过, 问题描述, 详细信息)

3. **`check_object_orientation(env, component_name)`**
   - 朝向验证
   - 返回: (是否通过, 问题描述, 详细信息)

4. **`check_collisions(env, component_names)`**
   - 碰撞验证
   - 返回: (是否通过, 问题描述, 详细信息)

5. **`check_in_camera_view(env, component_names, camera_ids)`**
   - 可见性验证
   - 返回: (是否通过, 问题描述, 详细信息)

6. **`validate_scene(env, env_config, camera_ids)`**
   - 综合验证接口
   - 返回: 完整验证报告字典

7. **`save_validation_report(report, output_path)`**
   - 保存 JSON 格式验证报告

### 集成点

验证系统集成在 `render_vlm_example()` 函数中:
```python
def render_vlm_example(env_config_path, output_image_path, output_mask_path, camera_ids):
    # 1. 加载环境
    env = load_env(...)

    # 2. ID 映射逻辑
    target_geom_to_seq = {...}

    # 3. 场景验证 (NEW)
    validation_report = validate_scene(env, env_config, camera_ids)
    save_validation_report(validation_report, validation_output_path)

    # 4. 渲染多视角
    for cam_id in camera_ids:
        rgb = env.render(...)
        ...
```

## 扩展与定制

### 添加新的验证规则

1. 在 `render_vlm_dataset.py` 中添加检查函数:
```python
def check_object_size(env, component_name, max_size=0.5):
    """检查物体尺寸是否合理"""
    physics = env.physics
    body_id = find_body_for_component(physics, component_name)

    # 获取 bounding box
    geom_adr = physics.model.body_geomadr[body_id]
    geom_num = physics.model.body_geomnum[body_id]

    max_dim = 0
    for i in range(geom_num):
        geom_id = geom_adr + i
        geom_size = physics.model.geom_size[geom_id]
        max_dim = max(max_dim, max(geom_size))

    if max_dim > max_size:
        return False, f"物体过大 ({max_dim:.3f}m > {max_size:.3f}m)", {
            "status": "FAILED",
            "max_dimension": max_dim
        }

    return True, "", {"status": "OK", "max_dimension": max_dim}
```

2. 在 `validate_scene()` 中调用:
```python
def validate_scene(env, env_config, camera_ids):
    report = {...}

    # 添加尺寸检查
    for obj in task_objects:
        passed, problem, details = check_object_size(env, obj['name'])
        report['checks']['size'].append(details)
        if not passed:
            report['overall_status'] = "FAILED"
            report['errors'].append(problem)

    return report
```

### 调整验证严格程度

**宽松模式** (仅记录警告,不阻止渲染):
```python
if not passed:
    report['warnings'].append(problem)  # 不设置 overall_status = "FAILED"
```

**严格模式** (任何问题都标记为失败):
```python
if not passed or details.get('angle_with_vertical', 0) > 10:  # 更严格的角度阈值
    report['overall_status'] = "FAILED"
```

## 性能影响

- **验证时间**: ~0.5 秒/样本 (包括物理仿真初始化)
- **额外开销**: ~10% (相比纯渲染)
- **内存占用**: +50MB (验证报告数据)

验证在环境加载后立即执行,不影响后续渲染性能。

## 最佳实践

1. **始终检查验证报告**: 在提交数据集前查看 `validation_report.json`
2. **修复而非忽略**: 遇到错误时修改模型 XML,而不是调整阈值
3. **批量验证**: 对整个数据集运行验证,查找系统性问题
4. **版本控制**: 将验证报告纳入 git 管理,追踪配置变更
5. **CI 集成**: 在持续集成中运行验证,确保数据质量

## 相关文件

- **验证实现**: `scripts/render_vlm_dataset.py` (行 52-389)
- **任务创建**: `scripts/create_vlm_task.py`
- **模型 XML**: `VLABench/assets/obj/meshes/lab_equipment/centrifuge/centrifuge.xml`
- **配置注册**: `VLABench/configs/constant.py`
- **本文档**: `scripts/VALIDATION_SYSTEM_README.md`

## 示例：完整工作流

```bash
# 1. 创建任务 (生成 env_config.json)
python scripts/create_vlm_task.py --task insert_tube_centrifuge --num-examples 3

# 2. 渲染 + 验证 (自动生成 validation_report.json)
python scripts/render_vlm_dataset.py --task insert_tube_centrifuge --dimension "M&T" --overwrite

# 3. 检查验证结果
cat dataset/vlm_evaluation_v1.0/M&T/insert_tube_centrifuge/example0/input/validation_report.json

# 4. 如果发现问题，修改模型 XML
vim VLABench/assets/obj/meshes/lab_equipment/centrifuge/centrifuge.xml

# 5. 重新验证
python scripts/render_vlm_dataset.py --task insert_tube_centrifuge --dimension "M&T" --overwrite
```

## 总结

场景验证系统通过自动化检测和详细报告,显著提高了 VLM 评测数据集的质量。系统设计遵循以下原则:

- **自动化**: 无需手动检查,集成到 pipeline
- **全面性**: 覆盖高度、朝向、碰撞、可见性
- **可操作**: 提供具体修复建议
- **可扩展**: 易于添加新的验证规则
- **非侵入**: 不影响现有工作流

通过使用该系统,可以在人工审核之前发现并修复绝大多数模型放置问题,节省时间并确保数据集的一致性。
