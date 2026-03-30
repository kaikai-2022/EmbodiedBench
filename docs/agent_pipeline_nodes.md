# VLABench Agent Pipeline 节点功能说明

**日期**: 2026-03-19
**状态**: 已实现并测试通过

---

## 概述

VLABench Agent 是基于 LangGraph 的自动化任务生成系统。用户输入自然语言指令后，系统自动完成：任务理解 → 资产获取 → 代码生成 → 任务注册 → 物理仿真 → 数据输出。

### Pipeline 架构

```
START → analyzer → asset_manager → code_generator → registration → simulation → vlm_data → END
                                         ↑                  |              |
                                         |    (注册失败)      |   (仿真失败)  |
                                         +------------------+--------------+
                                               (重试，最多3次)
```

### CLI 入口

```bash
cd /ssd/mkqin/workspace/VLABench
python vlabench_agent_cli.py --instruction "移动显微镜" --verbose
```

---

## 节点详情

### 1. Analyzer Node（任务理解）

**文件**: `scripts/vlabench_agent/nodes/analyzer.py`

**功能**: 使用 LLM 解析用户自然语言指令，提取任务结构化信息。

**是否调用 LLM**: 是 — 调用 `ChatAnthropic` 解析用户意图

**读取 State**: `user_instruction`

**写入 State**: `task_analysis`
```json
{
  "objects": ["microscope"],
  "scene": "laboratory",
  "operation_type": "pick",
  "instruction_en": "Move the microscope",
  "task_name": "move_microscope"
}
```

---

### 2. Asset Manager Node（资产管理）

**文件**: `scripts/vlabench_agent/nodes/asset_manager.py`

**功能**: 检查所需 3D 模型资产是否存在，缺失时自动下载。

**是否调用 LLM**: 否 — 使用资产工具函数（`check_asset_exists`、`download_asset`）

**读取 State**: `task_analysis`（`objects` 字段）

**写入 State**: `asset_status`
```json
{
  "microscope": {
    "xml_path": "review/microscope/<uid>/<uid>.xml",
    "class": "CommonGraspedEntity",
    "canonical_name": "microscope",
    "newly_downloaded": true
  }
}
```

**资产下载流程**: 调用 `get_assets.py` 子进程，其中 `AssetPipeline.process_model()` 包含完整处理链：

```
GLB下载 → OBJ转换 → MJCF转换 → 路径修复 → 后处理(居中+缩放+物理属性)
→ 渲染预览 → MuJoCo验证 → 朝向修正 → 尺寸修正
```

#### 2.1 朝向修正（LLM 多模态视觉）

**文件**: `scripts/orientation_fix.py`

**触发时机**: 资产下载的 `validate_model()` 之后自动执行

**流程**:
1. 从 `validation.mp4` 提取首帧图片
2. 将图片发送给 LLM（multimodal），判断物体朝向是否正确
3. 如果物体倒置/侧躺 → 旋转所有 OBJ 文件顶点和法线
4. 重新执行后处理（居中+缩放+物理属性）
5. 重新生成 validation.mp4
6. 二次发送新渲染帧给 LLM 确认修正结果

**LLM 调用**: 2 次（首次判断 + 二次确认），使用 Anthropic SDK multimodal API

**输出**: `orientation_report.json`
```json
{
  "action": "rotated",
  "rotation": {"axis": "x", "degrees": 90},
  "verified": true
}
```

#### 2.2 尺寸修正（LLM 常识知识）

**文件**: `scripts/size_fix.py`

**触发时机**: 朝向修正之后自动执行

**背景**: `postprocess_model()` 将所有模型统一缩放到 `max_dim=0.15m`，不区分物体类型。这导致大型物体（如显微镜）在仿真中比例失真。

**流程**:
1. 用纯文本 prompt 请求 LLM 判断物体的合理真实世界高度
2. 根据目标高度重新计算 XML 中的 mesh scale
3. 同时更新 `<inertial>` 质心位置和 `<site>` 抓取点位置
4. 重新生成 validation.mp4

**LLM 调用**: 1 次，纯文本（无图片），利用 LLM 常识知识

**参考信息（包含在 prompt 中）**:
- Franka 机械臂总高约 1.1m
- 桌面高度约 0.75m
- 夹爪张开宽度约 0.08m

**输出**: `size_report.json`
```json
{
  "action": "rescaled",
  "llm_suggestion": {"height_m": 0.35, "width_m": 0.15},
  "old_height_m": 0.15,
  "new_height_m": 0.35
}
```

---

### 3. Code Generator Node（代码生成）

**文件**: `scripts/vlabench_agent/nodes/code_generator.py`

**功能**: 使用 LLM 生成包含 ConfigManager + Task 类的 Python 文件。

**是否调用 LLM**: 是 — 生成完整的 VLABench 任务 Python 代码

**读取 State**: `task_analysis`、`asset_status`、`error_feedback`（重试时）

**写入 State**: `generated_code`、`task_module_path`、`code_generation_attempts`

**生成文件路径**: `VLABench/tasks/hierarchical_tasks/primitive/{task_name}_series.py`

**Prompt 包含**:
- 任务信息（名称、操作类型、指令）
- 资产信息（xml_path、class）
- SkillLib 完整方法签名（pick、place、lift、moveto、pour 等）
- 可用 condition 类型（contain、pour、lift、is_grasped 等）
- 2 个参考示例代码
- 上次失败的错误信息（重试时）

---

### 4. Registration Node（任务注册）

**文件**: `scripts/vlabench_agent/nodes/registration.py`

**功能**: 将生成的任务类动态注册到 VLABench 运行时。

**是否调用 LLM**: 否

**读取 State**: `task_analysis`、`asset_status`、`task_module_path`

**写入 State**: `registration_success`

**注册步骤**:
1. 清理旧注册（重试时）— `register._tasks`、`register._config_managers`、`sys.modules`
2. 更新 `name2config` 内存映射
3. 更新 `TASK_CONFIG` 内存配置 + 持久化写入 `task_config.json`
4. `importlib.import_module()` 动态导入
5. 验证 `register._tasks[task_name]` 存在

---

### 5. Simulation Node（物理仿真）

**文件**: `scripts/vlabench_agent/nodes/simulation.py`

**功能**: 在 MuJoCo 中加载任务、执行技能序列、生成训练数据。

**是否调用 LLM**: 否

**读取 State**: `task_analysis`（task_name）

**写入 State**: `simulation_success`、`simulation_video_path`、`simulation_hdf5_path`、`episode_config`、`executed_skill_sequence`

**执行流程**:
1. `load_env(task_name, robot="franka")` 加载环境
2. `env.reset()` 初始化
3. `env.save()` 保存初始配置
4. `env.get_expert_skill_sequence()` 获取技能序列
5. 逐个执行 skill，收集 observations
6. `env.task.conditions.is_met(physics)` 判断成功
7. 保存 HDF5 数据和演示视频

**输出路径**:
- 视频: `dataset/training_data/{task_name}/demo_0_success_{True/False}.mp4`
- 数据: `dataset/training_data/{task_name}/data_0.hdf5`

---

### 6. VLM Data Node（评测数据生成）

**文件**: `scripts/vlabench_agent/nodes/vlm_data.py`

**功能**: 从仿真结果生成 VLM 评测数据集。

**是否调用 LLM**: 否

**读取 State**: `task_analysis`、`episode_config`、`executed_skill_sequence`、`simulation_success`

**写入 State**: `task_save_path`、`rendered_images`

**输出目录结构**:
```
dataset/vlm_evaluation_v1.0/M&T/{task_name}/example0/
├── env_config/
│   └── env_config.json
├── input/
│   ├── instruction.txt
│   └── *.png              # 多视角渲染图像
└── output/
    └── operation_sequence.json
```

---

### 7. Error Handler Node

**文件**: `scripts/vlabench_agent/agent.py`（内联）

**功能**: 处理 pipeline 错误，记录失败信息。

---

## 路由逻辑

| 路由点 | 条件 | 目标 |
|--------|------|------|
| asset_manager → | 资产全部可用 | code_generator |
| asset_manager → | 资产缺失且无法下载 | error_handler |
| registration → | 注册成功 | simulation |
| registration → | 失败且 attempts < 3 | code_generator（重试） |
| registration → | 失败且 attempts >= 3 | error_handler |
| simulation → | 仿真成功 | vlm_data |
| simulation → | 失败且 attempts < 3 | code_generator（重试） |
| simulation → | 失败且 attempts >= 3 | error_handler |

---

## LLM 调用汇总

| 节点/模块 | LLM 调用次数 | 类型 | 用途 |
|-----------|-------------|------|------|
| Analyzer | 1 | 文本 | 解析用户指令 |
| Code Generator | 1（每次尝试） | 文本 | 生成任务 Python 代码 |
| orientation_fix | 2 | 多模态（图+文） | 判断+确认模型朝向 |
| size_fix | 1 | 文本 | 判断物体合理尺寸 |

**总计**: 单次成功执行最少 **5 次** LLM 调用（analyzer 1 + code_generator 1 + orientation_fix 2 + size_fix 1）。重试时 code_generator 额外调用 1 次/次。

---

## 关键配置文件

| 文件 | 用途 |
|------|------|
| `scripts/vlabench_agent/config.py` | LLM API 配置（模型名、API Key、端点） |
| `scripts/vlabench_agent/state.py` | Pipeline 全局状态定义 |
| `scripts/vlabench_agent/agent.py` | 图结构定义和路由逻辑 |
| `scripts/vlabench_agent/tools/asset_tools.py` | 资产检查/下载工具 |
| `scripts/get_assets.py` | 资产下载主 Pipeline |
| `scripts/orientation_fix.py` | 朝向修正模块 |
| `scripts/size_fix.py` | 尺寸修正模块 |
| `fix_obj2mjcf_xml.py` | OBJ→MJCF 后处理 |
| `scripts/validate_asset.py` | MuJoCo 加载验证 |

---

**最后更新**: 2026-03-19
