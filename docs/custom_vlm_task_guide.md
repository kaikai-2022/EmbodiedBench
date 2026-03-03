# 创建自定义 VLM 评测任务完整指南

## 📋 概述

本指南详细说明如何为 VLABench 创建自定义的 VLM（Vision-Language Model）评测任务。

## 🎯 任务示例

我们以 **实验室实验任务 (lab_experiment)** 为例：
- **场景**：实验室环境
- **任务**：选择特定的实验器材并放入指定位置
- **评测维度**：M&T (Manipulation & Tools)

---

## 📁 VLM 数据集结构

```
dataset/vlm_evaluation_v1.0/
├── M&T/                           # 评测维度
│   ├── select_toy/               # 现有任务
│   │   ├── example0/
│   │   │   ├── input/
│   │   │   │   ├── input.png          # RGB 图像 (必需)
│   │   │   │   ├── input_mask.png     # 分割掩码 (必需)
│   │   │   │   └── instruction.txt    # 任务指令 (必需)
│   │   │   ├── output/
│   │   │   │   └── operation_sequence.json  # 标准答案 (必需)
│   │   │   └── env_config/
│   │   │       └── env_config.json    # 环境配置 (必需)
│   │   ├── example1/
│   │   └── ...
│   └── lab_experiment/           # 自定义任务 ⭐
│       └── example0/
│           ├── input/
│           ├── output/
│           └── env_config/
```

---

## 🔧 完整创建流程

### 步骤 1: 创建目录结构

```bash
# 手动创建
mkdir -p /ssd/mkqin/workspace/VLABench/dataset/vlm_evaluation_v1.0/M&T/lab_experiment/example0/{input,output,env_config}

# 或使用自动化脚本
python /ssd/mkqin/workspace/VLABench/scripts/create_vlm_task.py
```

### 步骤 2: 编写必需文件

#### 2.1 instruction.txt（任务指令）

**位置**: `example0/input/instruction.txt`

**格式**: 纯文本，描述任务要求

**示例**:
```
Please select the test tube from the laboratory table. Please put it into the test tube rack.
```

**要点**:
- 清晰描述目标物体
- 明确指定放置位置
- 使用简单的自然语言

#### 2.2 operation_sequence.json（标准答案）

**位置**: `example0/output/operation_sequence.json`

**格式**: JSON 数组，包含技能序列

**示例**:
```json
{
    "skill_sequence": [
        {
            "name": "pick",
            "params": {
                "target_entity_name": 0
            }
        },
        {
            "name": "place",
            "params": {
                "target_container_name": 1
            }
        }
    ]
}
```

**可用技能**:
- `pick`: 抓取物体
  - `target_entity_name`: 目标物体索引（数字）
- `place`: 放置物体
  - `target_container_name`: 容器索引（数字）
- `press`: 按压
- `open_door`: 开门
- `close_door`: 关门
- `pour`: 倾倒
- `move`: 移动

**重要**: `target_entity_name` 和 `target_container_name` 使用**数字索引**，不是物体名称！

#### 2.3 env_config.json（环境配置）

**位置**: `example0/env_config/env_config.json`

**格式**: JSON，描述场景、物体和任务条件

**示例**:
```json
{
    "task": {
        "components": [
            {
                "name": "lab_table",
                "xml_path": "obj/meshes/table/table.xml",
                "position": [0, 0, 0],
                "orientation": [1, 0, 0, 0],
                "class": "Table",
                "materials": ["metal"]
            },
            {
                "name": "test_tube_0",
                "xml_path": "obj/meshes/lab_equipment/test_tube/test_tube.xml",
                "position": [-0.2, 0.0, 0.8],
                "orientation": [1, 0, 0, 0],
                "class": "LabEquipment"
            },
            {
                "name": "test_tube_rack",
                "xml_path": "obj/meshes/lab_equipment/test_tube_rack/test_tube_rack.xml",
                "position": [0.3, 0.2, 0.8],
                "orientation": [1, 0, 0, 0],
                "class": "Container"
            }
        ],
        "scene": {
            "name": "laboratory_0",
            "position": [0, 0, 0],
            "orientation": [1, 0, 0, 0],
            "floor_textures": ["lab_floor"]
        },
        "instructions": [
            "Put the test_tube_0 into the test_tube_rack"
        ],
        "conditions": {
            "contain": {
                "container": "test_tube_rack",
                "entities": ["test_tube_0"]
            }
        }
    }
}
```

**关键字段**:
- `components`: 场景中的物体列表
- `scene`: 场景信息（房间、地板等）
- `instructions`: 内部指令（用于验证）
- `conditions`: 任务完成条件

### 步骤 3: 生成渲染图像（关键！）

**VLM 评测需要预渲染的图像**，这是最重要的步骤。

#### 方案 A: 使用现有渲染系统（推荐）

```python
# 使用 VLABench 的渲染工具
from VLABench.utils.render_utils import render_vlm_scene

render_vlm_scene(
    env_config_path="path/to/env_config.json",
    output_image="input.png",
    output_mask="input_mask.png",
    camera_position=[0.5, -0.5, 1.5],
    resolution=[1024, 1024]
)
```

#### 方案 B: 复制现有样本（快速测试）

```bash
# 复制 select_toy 的图像作为示例
cp /ssd/mkqin/workspace/VLABench/dataset/vlm_evaluation_v1.0/M&T/select_toy/example0/input/*.png \
   /ssd/mkqin/workspace/VLABench/dataset/vlm_evaluation_v1.0/M&T/lab_experiment/example0/input/
```

#### 方案 C: 手动渲染（需要 MuJoCo）

```bash
# 使用 MuJoCo 渲染器
python /ssd/mkqin/workspace/VLABench/scripts/render_vlm_dataset.py
```

**输出文件**:
- `input.png`: RGB 图像，尺寸通常为 1024x1024 或更大
- `input_mask.png`: 分割掩码，用于标识不同物体

### 步骤 4: 注册任务到评测系统

编辑 `VLABench/configs/evaluation/dim2task.json`:

```json
{
    "M&T": [
        "select_poker",
        "select_drink",
        "select_book",
        "select_toy",
        "select_fruit",
        "select_toy_spatial",
        "add_condiment",
        "insert_flower",
        "select_chemistry_tube",
        "lab_experiment"  // ⭐ 添加你的任务
    ]
}
```

### 步骤 5: 测试任务

```bash
cd /ssd/mkqin/workspace/VLABench

# 运行 VLM 评测
python scripts/evaluate_vlm.py \
    --vlm_name Qwen2_VL \
    --eval-dimension "M&T" \
    --tasks lab_experiment \
    --n-episodes 10
```

**预期输出**:
```
working on  Qwen2_VL
working start at 2026-02-03 12:00:00
....................................................................................................100%( 10/ 10) using: 0h 1m23s, remain:0h 0m 0s
working end at 2026-02-03 12:01:23
```

### 步骤 6: 查看结果

```bash
# 评测结果
cat /ssd/mkqin/workspace/VLABench/logs/vlm/Qwen2_VL/en/0_shot/M&T/lab_experiment_result.json
```

**结果格式**:
```json
{
    "lab_experiment": {
        "0": {
            "skill_match_score": 1.0,
            "entity_match_score": 1.0,
            "skill_with_entity_match_score": 1.0,
            "exact_match_score": 1.0,
            "total_score": 1.0
        },
        "1": { ... }
    }
}
```

---

## 🚀 快速开始模板

### 使用自动化脚本

```bash
# 1. 生成任务结构（10 个样本）
python /ssd/mkqin/workspace/VLABench/scripts/create_vlm_task.py

# 2. 运行完整测试流程
bash /ssd/mkqin/workspace/VLABench/scripts/test_custom_vlm_task.sh
```

### 创建多个样本

```python
# 修改 create_vlm_task.py 中的数量
create_vlm_task("lab_experiment", num_examples=100)
```

---

## 📊 评测指标说明

VLM 评测使用以下指标：

1. **skill_match_score**: 技能序列匹配度
   - 只比较技能类型（pick/place 等），不比较参数

2. **entity_match_score**: 实体参数匹配度
   - 只比较参数（目标物体、容器），不比较技能顺序

3. **skill_with_entity_match_score**: 技能+实体匹配度
   - 同时比较技能类型和参数，但不考虑顺序

4. **exact_match_score**: 精确匹配度
   - 完全匹配，包括顺序和参数

5. **total_score**: 总分
   - 综合以上所有指标

**计算方式**: 基于编辑距离和序列相似度

---

## 🎨 自定义任务变体

### 变体 1: 更复杂的实验室任务

```json
{
    "skill_sequence": [
        {"name": "pick", "params": {"target_entity_name": 0}},
        {"name": "move", "params": {"target_position": [0.5, 0.5, 0.8]}},
        {"name": "pour", "params": {"target_container_name": 1}},
        {"name": "place", "params": {"target_container_name": 2}}
    ]
}
```

### 变体 2: 多步骤实验

```json
{
    "skill_sequence": [
        {"name": "pick", "params": {"target_entity_name": 0}},
        {"name": "place", "params": {"target_container_name": 1}},
        {"name": "pick", "params": {"target_entity_name": 2}},
        {"name": "place", "params": {"target_container_name": 1}}
    ]
}
```

### 变体 3: 条件分支

你可以创建不同难度的样本：
- **简单**: 清晰的指令，少数干扰物体
- **中等**: 多个相似物体
- **困难**: 复杂场景，需要常识推理

---

## ⚠️ 常见问题

### Q1: target_entity_name 应该用数字还是名称？

**A**: **必须用数字索引**，从 0 开始编号。

- `env_config.json` 中的 `components` 列表顺序决定索引
- 第一个物体（index 0）通常是表
- 第二个物体（index 1）是第一个可操作物体
- 以此类推

### Q2: 如何获取物体的索引？

```python
# 读取 env_config.json
import json
with open("env_config.json") as f:
    config = json.load(f)

# 打印物体列表
for i, comp in enumerate(config["task"]["components"]):
    print(f"{i}: {comp['name']} ({comp['class']})")
```

### Q3: 图像分辨率有要求吗？

**A**: 建议使用 1024x1024 或更高分辨率。VLM 模型通常能处理各种尺寸，但一致性很重要。

### Q4: 可以使用自己的 3D 模型吗？

**A**: 可以！需要：
1. 创建 MuJoCo XML 格式的模型文件
2. 放置在 `obj/meshes/` 目录下
3. 在 `env_config.json` 中正确引用路径

### Q5: 如何批量生成多个样本？

```python
# 使用自动化脚本
for i in range(100):
    create_example(i, task_config)
    render_scene(i)
```

---

## 📚 参考资料

### VLM 评测相关文件

- **评测器**: `VLABench/evaluation/evaluator/vlm.py`
- **VLM 模型**: `VLABench/evaluation/model/vlm/`
- **评测配置**: `VLABench/configs/evaluation/`
- **Prompt 模板**: `VLABench/configs/prompt/eval_vlm_en.txt`

### 渲染相关

- **渲染工具**: `VLABench/utils/render_utils.py`
- **相机配置**: MuJoCo XML 中的 `<camera>` 标签
- **渲染后端**: `MUJOCO_GL` 环境变量 (egl/osmesa/glfw)

### 数据集工具

- **任务生成器**: `scripts/create_vlm_task.py`
- **渲染脚本**: `scripts/render_vlm_dataset.py`
- **测试脚本**: `scripts/test_custom_vlm_task.sh`

---

## 🎯 完整示例

查看我们创建的示例任务：
```bash
# 查看任务结构
ls -R /ssd/mkqin/workspace/VLABench/dataset/vlm_evaluation_v1.0/M&T/lab_experiment/

# 查看配置文件
cat /ssd/mkqin/workspace/VLABench/dataset/vlm_evaluation_v1.0/M&T/lab_experiment/example0/input/instruction.txt
cat /ssd/mkqin/workspace/VLABench/dataset/vlm_evaluation_v1.0/M&T/lab_experiment/example0/output/operation_sequence.json
```

---

## 📞 获取帮助

如果遇到问题：
1. 检查文件格式是否正确（JSON 语法）
2. 确认图像文件存在
3. 验证任务已在 `dim2task.json` 中注册
4. 查看 VLM 输出的错误信息

祝你成功创建自定义 VLM 评测任务！🎉
