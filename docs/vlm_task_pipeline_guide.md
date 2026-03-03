# VLM 任务生成 Pipeline 使用指南

## 概述

`create_vlm_task.py` 是一个功能强大的自动化工具，用于批量生成 VLM 评测任务数据集。

## 核心特性

- ✅ **命令行工具** - 支持丰富的命令行参数
- ✅ **内置模板** - 预定义常用任务模板（pour, lift, place）
- ✅ **自定义模板** - 支持 YAML/JSON 格式的自定义模板
- ✅ **批量生成** - 一次生成多个样本，自动添加位置变化
- ✅ **资产验证** - 自动检查 XML 文件路径是否存在
- ✅ **自动渲染** - 集成渲染脚本，一键生成图像
- ✅ **干运行模式** - 预览生成结果，无需创建文件

## 快速开始

### 1. 查看可用模板

```bash
python scripts/create_vlm_task.py --list-templates
```

### 2. 使用内置模板创建任务

```bash
# 创建倾倒试管任务（1个样本）
python scripts/create_vlm_task.py --task pour_tube

# 创建10个样本并自动渲染
python scripts/create_vlm_task.py --task pour_tube --num-examples 10 --render

# 预览（不创建文件）
python scripts/create_vlm_task.py --task lift_object --num-examples 5 --dry-run
```

### 3. 使用自定义模板

```bash
# 从 YAML 文件创建任务
python scripts/create_vlm_task.py --config templates/my_task.yaml --render

# 从 JSON 文件创建
python scripts/create_vlm_task.py --config templates/my_task.json --num-examples 20
```

## 内置模板

### 1. `pour_tube` - 倾倒试管

**描述**: 抓取并倾倒试管

**物体**:
- 试管（ChemistryTube）- 支持4种溶液变体：CuSO4, CuCl2, FeCl3, KMnO4

**操作序列**: pick → pour

**使用场景**: 实验室操作、液体转移

### 2. `lift_object` - 举起物体

**描述**: 抓取物体并举起到指定高度

**物体**:
- 礼盒（CommonContainer）

**操作序列**: pick → lift

**使用场景**: 高度控制、精确抓取

### 3. `place_in_container` - 放置任务

**描述**: 将物体放入容器

**物体**:
- 扑克牌（Poker）
- 礼盒容器（CommonContainer）

**操作序列**: pick → place

**使用场景**: 分类、整理

## 自定义模板格式

### YAML 模板示例

创建 `templates/my_task.yaml`:

```yaml
description: "我的自定义任务 - 描述内容"
dimension: "M&T"
scene: "laboratory_0"
floor_texture: "lab_floor"

# 固定组件（如桌子、试管架等）
components:
  - name: "table"
    xml_path: "obj/meshes/table/table.xml"
    position: [0, 0, 0]
    orientation: [1, 0, 0, 0]
    class: "Table"
    materials: ["wood3"]

# 任务物体（会自动添加位置变化）
task_objects:
  - type: "test_tube"
    xml_path: "obj/meshes/tube/tube/tube.xml"
    class: "ChemistryTube"
    position_base: [0.0, 0.0, 0.8]
    orientation: [1, 0, 0, 0]
    extra_params:
      solution: "CuSO4"
    # 可选：定义变体，不同 example 使用不同变体
    variations:
      - solution: "CuSO4"
      - solution: "CuCl2"
      - solution: "FeCl3"

# 指令模板（支持占位符）
instruction_template: "Pour the {object_name}"

# 操作序列
operation_sequence:
  - name: "pick"
    params:
      target_entity_name: 1  # 第一个任务物体的索引
  - name: "pour"
    params:
      target_container_name: 0  # 桌子的索引

# 成功条件
conditions:
  pour:
    target_entity: "{object_name}"
    threshold: 0
```

### JSON 模板示例

创建 `templates/my_task.json`:

```json
{
  "description": "我的自定义任务",
  "dimension": "M&T",
  "scene": "laboratory_0",
  "floor_texture": "lab_floor",
  "components": [
    {
      "name": "table",
      "xml_path": "obj/meshes/table/table.xml",
      "position": [0, 0, 0],
      "orientation": [1, 0, 0, 0],
      "class": "Table",
      "materials": ["wood3"]
    }
  ],
  "task_objects": [
    {
      "type": "target_object",
      "xml_path": "obj/meshes/poker/poker_asset.xml",
      "class": "Poker",
      "position_base": [0.0, 0.0, 0.8],
      "orientation": [1, 0, 0, 0]
    }
  ],
  "instruction_template": "Pick the {object_name}",
  "operation_sequence": [
    {
      "name": "pick",
      "params": {
        "target_entity_name": 1
      }
    },
    {
      "name": "lift",
      "params": {
        "target_height": 0.9
      }
    }
  ],
  "conditions": {
    "is_grasped": {
      "entities": ["{object_name}"],
      "robot": "franka"
    },
    "lift": {
      "entities": ["{object_name}"],
      "target_height": 0.9
    }
  }
}
```

## 占位符说明

模板中支持以下占位符：

- `{object_name}` - 第一个任务物体的名称
- `{target}` - 目标物体（等同于第一个任务物体）
- `{container}` - 容器物体（第二个任务物体）

占位符会在生成时自动替换为实际的物体名称。

## 可用的物体类

| 类名 | 说明 | 常用路径示例 |
|------|------|--------------|
| `Table` | 桌子 | `obj/meshes/table/table.xml` |
| `ChemistryTube` | 试管 | `obj/meshes/tube/tube/tube.xml` |
| `CommonContainer` | 通用容器 | `obj/meshes/containers/boxes/giftbox/...` |
| `Poker` | 扑克牌 | `obj/meshes/poker/poker_asset.xml` |
| `BilliardBall` | 台球 | `obj/meshes/billiard_balls/...` |
| `Toy` | 玩具 | `obj/meshes/toys/...` |
| `Book` | 书籍 | `obj/meshes/books/...` |
| `Drink` | 饮料 | `obj/meshes/drinks/...` |
| `Fruit` | 水果 | `obj/meshes/fruits/...` |

更多类定义请参考：`VLABench/tasks/components/specific_entities/`

## 支持的操作类型

| 操作名称 | 说明 | 参数 |
|---------|------|------|
| `pick` | 抓取物体 | `target_entity_name`: 物体索引 |
| `place` | 放置物体 | `target_container_name`: 容器索引 |
| `pour` | 倾倒容器 | `target_container_name`: 接收容器索引 |
| `lift` | 举起物体 | `target_height`: 目标高度 |

## 支持的成功条件

| 条件名称 | 说明 | 参数示例 |
|---------|------|----------|
| `is_grasped` | 物体被抓取 | `entities`, `robot` |
| `lift` | 物体举起到高度 | `entities`, `target_height` |
| `contain` | 物体在容器内 | `container`, `entities` |
| `pour` | 容器被倾倒 | `target_entity`, `threshold` |
| `on` | 物体在表面上 | `entities`, `container` |
| `on_position` | 物体在位置 | `entities`, `positions`, `tolerance_distance` |

完整条件列表请参考：`VLABench/tasks/condition.py`

## 命令行参数完整列表

```
--task TASK                任务名称（使用内置模板）
--config PATH              自定义模板文件路径（YAML/JSON）
--dimension DIM            评测维度（默认: M&T）
--num-examples N           生成样本数量（默认: 1）
--render                   创建后自动渲染图像
--overwrite                覆盖已存在的图像
--dry-run                  干运行模式（预览，不创建文件）
--list-templates           列出所有可用的内置模板
```

## 工作流程

完整的任务创建流程：

```bash
# 1. 预览任务（干运行）
python scripts/create_vlm_task.py --task pour_tube --num-examples 10 --dry-run

# 2. 创建任务文件
python scripts/create_vlm_task.py --task pour_tube --num-examples 10

# 3. 渲染图像（可以合并到步骤2）
python scripts/create_vlm_task.py --task pour_tube --num-examples 10 --render

# 4. 测试评估
python scripts/evaluate_vlm.py --vlm_name Qwen2_VL --eval-dimension "M&T" --tasks pour_tube
```

或者一步完成：

```bash
python scripts/create_vlm_task.py --task pour_tube --num-examples 10 --render --overwrite
```

## 高级用法

### 批量创建多个任务

```bash
# 使用 bash 循环
for task in pour_tube lift_object place_in_container; do
    python scripts/create_vlm_task.py --task $task --num-examples 50 --render
done
```

### 位置变化控制

脚本会自动为每个样本添加位置变化（±15cm范围内随机），这确保：
- 避免物体重叠
- 增加数据多样性
- 保持可重现性（使用 example_idx 作为随机种子）

如需调整变化范围，修改 `add_position_variation()` 函数中的 `variation_range` 参数。

### 溶液变体循环

对于试管任务，脚本会自动循环使用不同溶液：
- example0: CuSO4（蓝色）
- example1: CuCl2（绿色）
- example2: FeCl3（黄色）
- example3: KMnO4（紫色）
- example4: CuSO4（循环）
- ...

## 故障排除

### 问题1: 资产路径验证失败

**错误**: `无效的资产路径: obj/meshes/...`

**解决方案**:
1. 检查 XML 文件是否存在于 `VLABench/assets/` 目录下
2. 使用相对于 `assets/` 的路径（不需要 `assets/` 前缀）
3. 查看现有任务的 `env_config.json` 作为参考

### 问题2: 渲染失败

**错误**: 图像渲染失败，显示 "gladLoadGL error" 或 "X11: Failed to open display"

**解决方案**:
1. **已自动配置**: 渲染脚本已自动配置使用 OSMesa 离线渲染（`MUJOCO_GL=osmesa`），无需 X11 显示
2. 如果仍然失败，手动运行渲染脚本测试：
   ```bash
   python scripts/render_vlm_dataset.py --task YOUR_TASK --dimension "M&T"
   ```
3. 检查环境配置是否正确（物体类、条件等）
4. 确认 MuJoCo 和 dm_control 已正确安装

### 问题3: 物体类未注册

**错误**: `KeyError: 'SomeClass'`

**解决方案**:
1. 检查类名是否正确（区分大小写）
2. 查看 `VLABench/tasks/components/` 中可用的类
3. 确保使用项目支持的类名

## 最佳实践

1. **先使用 --dry-run 预览** - 避免创建错误的文件
2. **从内置模板开始** - 复制并修改，而非从零开始
3. **使用描述性的任务名** - 便于后续管理
4. **验证资产路径** - 在模板中使用已存在的 XML 文件
5. **测试小批量** - 先生成1-2个样本测试，再批量生成
6. **定期备份** - 保存成功的模板配置

## 贡献

欢迎贡献新的任务模板！请：
1. 在 `BUILTIN_TEMPLATES` 中添加你的模板
2. 测试模板是否正常工作
3. 更新本文档

---

**作者**: Claude + VLABench Team
**更新日期**: 2026-02-06
**版本**: 2.0
