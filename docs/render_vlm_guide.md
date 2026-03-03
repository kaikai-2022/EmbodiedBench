# VLM 数据集渲染指南

## 📋 概述

本指南说明如何为自定义 VLM 评测任务生成渲染图像。

## ⚠️ 重要说明

**渲染需要真实的 3D 模型文件！**

当前 `lab_experiment` 任务的 `env_config.json` 中引用的 3D 模型路径可能不存在。在运行渲染之前，你需要：

1. 确保所有 3D 模型文件存在（.xml 格式）
2. 或者修改 `env_config.json` 使用现有的模型

## 🔍 检查现有模型

```bash
# 查看可用的模型
ls /ssd/mkqin/workspace/VLABench/VLABench/assets/obj/meshes/

# 查看容器类模型
ls /ssd/mkqin/workspace/VLABench/VLABench/assets/obj/meshes/containers/

# 查看其他模型
ls /ssd/mkqin/workspace/VLABench/VLABench/assets/obj/meshes/
```

## 🚀 方法 1: 使用现有模型（推荐）

修改 `env_config.json` 使用项目中已有的模型：

### 步骤 1: 查找合适的模型

```bash
# 查找容器
find /ssd/mkqin/workspace/VLABench/VLABench/assets/obj/meshes/containers/ -name "*.xml" | head -5

# 查找水果（作为操作对象）
find /ssd/mkqin/workspace/VLABench/VLABench/assets/obj/meshes/ -name "*fruit*" -name "*.xml" | head -5

# 查找桌子
find /ssd/mkqin/workspace/VLABench/VLABench/assets/obj/meshes/ -name "*table*" -name "*.xml"
```

### 步骤 2: 更新 env_config.json

将 `lab_experiment/example0/env_config/env_config.json` 中的 `xml_path` 改为实际存在的模型路径。

例如：
```json
{
    "name": "lab_table",
    "xml_path": "obj/meshes/table/table.xml",  // ✅ 存在
    ...
}
```

### 步骤 3: 运行渲染

```bash
cd /ssd/mkqin/workspace/VLABench
python scripts/render_vlm_dataset.py
```

## 🚀 方法 2: 复制现有样本的图像（快速测试）

如果只是想快速测试 VLM 评测流程，可以直接复制现有任务的图像：

```bash
# 复制 select_toy 的图像
for i in {0..9}; do
  mkdir -p dataset/vlm_evaluation_v1.0/M&T/lab_experiment/example$i/input
  cp dataset/vlm_evaluation_v1.0/M&T/select_toy/example0/input/*.png \
     dataset/vlm_evaluation_v1.0/M&T/lab_experiment/example$i/input/
done
```

## 🚀 方法 3: 使用原始任务环境（高级）

如果你想完全从头生成任务和图像，需要：

1. **定义任务类**（Python）
2. **使用 MuJoCo 环境**
3. **调用渲染接口**

这涉及到创建继承自 `BaseTask` 的任务类，比较复杂。

## 📊 渲染输出

成功渲染后会生成：

- **input.png**: 堆叠的多视角 RGB 图像（通常是 2x2 网格的 4 个视角）
- **input_mask.png**: 对应的深度/分割掩码

图像尺寸示例：
- 单个视角: 256x256
- 堆叠后 (4视角): 512x512

## 🔧 渲染脚本配置

`render_vlm_dataset.py` 的关键参数：

```python
# 相机配置
camera_ids = [0, 1, 2, 3]  # 使用的相机 ID

# 渲染选项
render_options = {
    "width": 256,      # 图像宽度
    "height": 256,     # 图像高度
    "mode": "depth",   # 渲染模式
}
```

## ⚠️ 常见问题

### Q1: 找不到模型文件

**错误**: `FileNotFoundError: model.xml not found`

**解决**:
1. 检查 `env_config.json` 中的 `xml_path` 是否正确
2. 确保模型文件存在于 `VLABench/assets/obj/meshes/` 目录
3. 使用相对路径（从项目根目录开始）

### Q2: 渲染出的图像是黑屏

**原因**: 相机位置不对，或者物体不在相机视野内

**解决**:
1. 检查 `env_config.json` 中的物体位置
2. 调整相机参数或物体位置
3. 使用不同的 `camera_id`

### Q3: 渲染很慢

**原因**: 渲染 4 个视角需要较长时间

**解决**:
1. 减少相机数量（使用 2 个视角）
2. 降低图像分辨率
3. 使用更简单的场景

### Q4: 没有渲染工具

**说明**: 项目中**没有现成的渲染工具**，`render_vlm_dataset.py` 是我根据项目的接口创建的。

**替代方案**:
1. 使用方法 2（复制现有图像）
2. 手动使用 MuJoCo 渲染
3. 联系 VLABench 维护者获取官方渲染脚本

## 📚 参考资料

- **VLABench 接口**: `VLABench/utils/interface.py`
- **环境渲染**: `VLABench/envs/dm_env.py`
- **相机工具**: `VLABench/utils/camera_utils.py`
- **数据集结构**: `dataset/vlm_evaluation_v1.0/README.md`

## 🎯 推荐流程

对于快速测试，推荐：

1. ✅ 使用方法 2（复制现有图像）
2. ✅ 先验证 VLM 评测流程
3. ✅ 再考虑生成真实渲染图像

这样可以快速验证你的任务配置是否正确！

## 💡 提示

- 渲染图像只是用于 VLM 输入，**不影响评测逻辑**
- 即使使用"错误"的图像（如复制的图像），评测框架仍然可以工作
- 关键是 `instruction.txt` 和 `operation_sequence.json` 的正确性
- 图像主要用于 VLM 理解场景，高质量的图像有助于提高 VLM 性能
