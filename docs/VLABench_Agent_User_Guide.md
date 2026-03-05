# VLABench Agent 使用指南

## 简介

VLABench Agent 是一个基于 LangGraph 的智能任务生成系统,可以通过自然语言指令自动创建科研场景任务。

## 安装

### 1. 安装依赖

```bash
cd /ssd/mkqin/workspace/VLABench
pip install -r requirements_agent.txt
```

### 2. 设置 API 密钥

VLABench Agent 使用 Claude Sonnet 4.5 作为核心 LLM,需要配置 Anthropic API 密钥:

```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

或者在运行时指定:

```bash
ANTHROPIC_API_KEY='your-api-key' python vlabench_agent_cli.py --instruction "..."
```

### 3. 设置环境变量

```bash
export VLABENCH_ROOT=/ssd/mkqin/workspace/VLABench
```

如果不设置,CLI 会自动使用项目根目录。

## 使用方法

### 基础用法

```bash
python vlabench_agent_cli.py --instruction "创建一个倾倒试管的任务"
```

### 更多示例

```bash
# 盖玻片和显微镜任务
python vlabench_agent_cli.py --instruction "创建一个将盖玻片放在显微镜台上的任务"

# 试管放入试管架
python vlabench_agent_cli.py --instruction "创建一个将试管放入试管架的任务"

# 烧杯举起任务
python vlabench_agent_cli.py --instruction "创建一个举起烧杯的任务"

# 调试模式
python vlabench_agent_cli.py --instruction "创建一个倾倒烧杯的任务" --verbose
```

## 工作流程

Agent 会自动执行以下步骤:

1. **Analyzer Node**: 理解用户指令,提取物体、场景、操作类型等信息
2. **Asset Manager Node**: 检查所需模型资产,自动下载缺失的模型
3. **Task Creator Node**: 生成 env_config.json 和任务目录结构
4. **Render Executor Node**: 调用渲染脚本生成图像并验证场景

## 输出结构

成功执行后,会在以下位置生成任务:

```
VLABench/dataset/vlm_evaluation_v1.0/M&T/{task_name}/example0/
├── env_config/
│   ├── env_config.json          # 环境配置
│   └── validation_report.json   # 场景验证报告
├── input/
│   ├── instruction.txt          # 任务指令
│   ├── input.png                # 渲染图像 (RGB)
│   └── input_mask.png           # 分割掩码
└── output/
    └── operation_sequence.json  # 操作序列
```

## 自然语言指令格式

支持中文和英文指令,Agent 会自动理解以下要素:

- **物体**: 科研器材名称 (试管、显微镜、烧杯、盖玻片等)
- **操作**: 动作类型 (放置、倾倒、举起、滑动等)
- **场景**: 环境类型 (实验室、厨房、客厅等)

### 示例指令

✅ 好的指令:
- "创建一个倾倒试管的任务"
- "生成一个将盖玻片放在显微镜台上的任务"
- "制作一个从试管架取下试管的场景"

❌ 避免的指令:
- "帮我做个任务" (太模糊)
- "试管" (缺少动作)
- "随便生成一个场景" (缺少具体需求)

## 故障排查

### 问题 1: `ANTHROPIC_API_KEY` 未设置

**错误**:
```
❌ 错误: 未设置 ANTHROPIC_API_KEY 环境变量
```

**解决**:
```bash
export ANTHROPIC_API_KEY='your-api-key'
```

### 问题 2: 模块导入失败

**错误**:
```
ModuleNotFoundError: No module named 'langgraph'
```

**解决**:
```bash
pip install -r requirements_agent.txt
```

### 问题 3: 资产下载失败

**错误**:
```
无法下载资产: microscope
```

**原因**: Objaverse 搜索未找到匹配的模型

**解决**:
1. 检查网络连接
2. 尝试使用更通用的英文关键词
3. 手动下载资产并放入 `assets/obj/meshes/` 目录

### 问题 4: 场景验证失败

**警告**:
```
⚠ 场景验证: FAILED
```

**查看详情**:
```bash
cat dataset/vlm_evaluation_v1.0/M&T/{task_name}/example0/env_config/validation_report.json
```

**常见问题**:
- 物体位置重叠 → 调整 `position_base`
- 物体嵌入桌面 → 增加 z 坐标
- 物体不在视野内 → 调整物体位置到工作区域

## 高级用法

### 编程接口

```python
from scripts.vlabench_agent import build_vlabench_agent, create_initial_state

# 构建 Agent
agent = build_vlabench_agent()

# 创建初始状态
state = create_initial_state("创建一个倾倒试管的任务")

# 运行
result = agent.invoke(state)

# 检查结果
if result['current_stage'] == 'done':
    print(f"任务保存在: {result['task_save_path']}")
    print(f"渲染图像: {result['rendered_images']}")
```

### 批量生成

```python
instructions = [
    "创建一个倾倒试管的任务",
    "创建一个举起烧杯的任务",
    "创建一个放置盖玻片的任务"
]

for instruction in instructions:
    state = create_initial_state(instruction)
    result = agent.invoke(state)
    print(f"{instruction}: {result['current_stage']}")
```

## 已知限制

1. **模型可用性**: 依赖 Objaverse 数据集,某些特定模型可能无法找到
2. **API 成本**: 每个任务会调用 2-3 次 Claude API (约 0.01-0.02 USD)
3. **渲染时间**: 复杂场景的渲染可能需要 1-3 分钟
4. **场景验证**: 自动生成的位置可能需要人工微调

## 项目结构

```
VLABench/
├── scripts/
│   └── vlabench_agent/          # Agent 核心代码
│       ├── __init__.py
│       ├── agent.py             # 主逻辑和图构建
│       ├── state.py             # 状态定义
│       ├── nodes/               # 节点实现
│       │   ├── analyzer.py
│       │   ├── asset_manager.py
│       │   ├── task_creator.py
│       │   └── render_executor.py
│       └── tools/               # 工具函数
│           └── asset_tools.py
├── vlabench_agent_cli.py        # CLI 入口
├── requirements_agent.txt       # 依赖
└── docs/
    ├── LangGraph_Agent_Integration_Plan.md  # 设计文档
    └── VLABench_Agent_User_Guide.md         # 本文档
```

## 贡献和反馈

如有问题或建议,请在项目仓库中提交 Issue。

## License

遵循 VLABench 项目的 License。
