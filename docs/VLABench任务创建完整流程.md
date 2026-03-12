# VLABench创建新任务的完整流程

## 一、核心概念理解

VLABench的任务系统基于**专家策略 + 物理模拟**的设计理念：程序员在代码中预定义专家技能序列，通过Mujoco物理引擎实际执行后生成标准答案和训练数据。

## 二、完整流程（7个阶段）

### 阶段1：准备3D资产

1. 准备obj格式的3D模型
2. 使用obj2mjcf转换为Mujoco的xml格式
3. 标注关键点：
   - **grasppoint**：可抓取物体的抓取点
   - **placepoint**：容器的放置点
   - **keypoint**：容器的包含判断边界框

### 阶段2：注册实体类

**文件位置**：`VLABench/tasks/components/`

1. 创建实体类，继承自`GraspedObject`或`Container`
2. 使用`@register.add_entity`装饰器注册
3. 在`VLABench/configs/constant.py`的`name2class_xml`中添加映射

### 阶段3：创建ConfigManager

**文件位置**：`VLABench/tasks/config_manager.py`

- 负责任务随机性：随机选择目标物体、生成物体组合、设置初始位置
- 使用`@register.add_config_manager`注册

### 阶段4：定义任务类和专家技能序列

**文件位置**：`VLABench/tasks/hierarchical_tasks/`

这是最核心的步骤，需要实现`get_expert_skill_sequence()`方法：

```python
def get_expert_skill_sequence(self, physics):
    skill_sequence = [
        partial(SkillLib.pick, target_entity_name=self.target_entity),
        partial(SkillLib.lift),
        partial(SkillLib.place, target_container_name=self.target_container)
    ]
    return skill_sequence
```

可用的技能包括：`pick`, `place`, `lift`, `moveto`, `pour`, `push`, `pull`, `open_door`, `close_drawer`, `press`, `flip`, `wait`等（详见`VLABench/utils/skill_lib.py`）

### 阶段5：生成轨迹数据

**使用脚本**：`scripts/trajectory_generation.py`

```bash
python scripts/trajectory_generation.py \
    --task_name your_task_name \
    --n_sample 100 \
    --save_dir ./dataset/training_data \
    --record_video \
    --early_stop
```

**执行流程**：
1. 初始化环境
2. 获取专家技能序列
3. 在Mujoco中逐个执行技能
4. 收集观察数据（RGB、深度图、机器人状态、轨迹）
5. 保存为HDF5格式

### 阶段6：数据格式转换（可选）

根据训练需求转换数据格式：

#### TFDS格式（用于OpenVLA/Octo）

```bash
python scripts/convert_to_rlds.py --save_dir /path/to/dataset --task task_name
cd /path/to/dataset/task_name
tfds build --overwrite
```

#### Lerobot格式（用于π0）

```bash
python scripts/convert_to_lerobot.py --dataset-name xxx --dataset-path /path/to/hdf5 --task-list task1 task2
```

### 阶段7：评测准备

生成的数据包含：
- `env_config/env_config.json`：环境配置
- `input/four_view.png`：四视角输入图像
- `input/instruction.json`：任务指令
- `output/operation_sequence.json`：标准答案技能序列

## 三、任务类型

- **Primitive Tasks（原始任务）**：单一操作，如`select_fruit`, `add_condiment`
- **Composite Tasks（复合任务）**：多步骤操作，如`cluster_book`, `cook_dishes`

## 四、测试和调试

- 使用`tutorials/1.load_task.ipynb`快速加载和可视化任务
- 使用`tutorials/4.teleoperate.ipynb`进行人工遥操作测试
- 使用`--record_video`参数生成执行视频
- 使用`--debug`模式查看详细日志

## 五、评测

使用`tutorials/5.run_evaluation.ipynb`：
- **VLA模型评测**：测试动作策略（如OpenVLA）
- **VLM模型评测**：测试视觉语言模型的技能序列预测能力

---

这个流程确保了生成的数据既符合物理规律（通过Mujoco验证），又具有可重复性和标准化的评测标准。
