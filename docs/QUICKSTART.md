# VLABench Agent - 快速开始

## ✅ Phase 1 已完成!

VLABench Agent 的基础流水线已经完全实现,包括:

### 已实现的功能

1. ✅ **Analyzer Node** - 使用 Claude Sonnet 4.5 理解自然语言指令
2. ✅ **Asset Manager Node** - 自动检查和下载模型资产
3. ✅ **Task Creator Node** - 智能生成任务配置和目录结构
4. ✅ **Render Executor Node** - 调用渲染脚本生成场景图像
5. ✅ **LangGraph 流程编排** - 完整的状态管理和错误处理
6. ✅ **CLI 命令行接口** - 友好的用户交互界面
7. ✅ **完整文档** - 设计文档和用户指南

---

## 🚀 快速开始 (3 步)

### 步骤 1: 安装依赖

```bash
cd /ssd/mkqin/workspace/VLABench
pip install -r requirements_agent.txt
```

**需要安装的包**:
- `langgraph>=0.2.0` - 核心流程引擎
- `langchain>=0.3.0` - LLM 框架
- `langchain-anthropic>=0.2.0` - Claude API 集成
- `anthropic>=0.40.0` - Anthropic SDK
- `pydantic>=2.0.0` - 数据验证

### 步骤 2: 设置 API 密钥

```bash
export ANTHROPIC_API_KEY='your-anthropic-api-key-here'
```

> 💡 **提示**: 可以从 [Anthropic Console](https://console.anthropic.com/) 获取 API 密钥

### 步骤 3: 运行 Agent

```bash
python3 vlabench_agent_cli.py --instruction "创建一个倾倒试管的任务"
```

---

## 📖 示例用法

### 示例 1: 简单任务 - 倾倒试管

```bash
python3 vlabench_agent_cli.py --instruction "创建一个倾倒试管的任务"
```

**预期输出**:
```
🤖 VLABench Agent - 自然语言任务生成智能体
======================================================================

📝 用户指令: 创建一个倾倒试管的任务
📂 项目根目录: /ssd/mkqin/workspace/VLABench

🚀 开始执行任务...

[Analyzer] 正在分析任务...
[Analyzer] ✓ 识别到物体: ['test_tube']
[Analyzer] ✓ 场景类型: laboratory
[Analyzer] ✓ 操作类型: pour

[Asset Manager] 开始检查资产...
[Asset Manager] ✓ test_tube: obj/meshes/tube/tube/tube.xml

[Task Creator] 开始生成任务配置...
[Task Creator] ✓ 保存 env_config.json
[Task Creator] ✓ 保存 instruction.txt
[Task Creator] ✓ 保存 operation_sequence.json

[Render Executor] 开始渲染场景...
[Render Executor] ✓ 渲染完成

✅ 任务完成!
📁 保存路径: /ssd/mkqin/workspace/VLABench/dataset/vlm_evaluation_v1.0/M&T/pour_test_tube/example0
🖼️  渲染图像:
  - .../input.png
  - .../input_mask.png
```

### 示例 2: 复杂任务 - 盖玻片和显微镜

```bash
python3 vlabench_agent_cli.py --instruction "创建一个将盖玻片放在显微镜台上的任务"
```

**Agent 会自动**:
1. 识别需要 `coverslip` 和 `microscope` 两个物体
2. 检测到这些资产缺失,自动从 Objaverse 下载
3. 生成合理的物体位置布局
4. 推断操作序列 (pick + place)
5. 渲染场景并验证

### 示例 3: 调试模式

```bash
python3 vlabench_agent_cli.py --instruction "创建一个举起烧杯的任务" --verbose
```

---

## 📂 生成的任务结构

```
dataset/vlm_evaluation_v1.0/M&T/{task_name}/example0/
├── env_config/
│   ├── env_config.json          # 完整的场景配置
│   │   ├── task.components      # 物体列表 (桌子 + 任务物体)
│   │   ├── task.scene           # 场景信息
│   │   ├── task.instructions    # 任务指令
│   │   └── task.conditions      # 评测条件
│   └── validation_report.json   # 场景验证报告
│       ├── overall_status       # PASS/FAILED
│       ├── checks               # 高度/朝向/碰撞/可见性检查
│       └── errors/warnings      # 问题列表
├── input/
│   ├── instruction.txt          # 英文任务指令
│   ├── input.png                # 四视角 RGB 图像
│   └── input_mask.png           # 彩色分割掩码
└── output/
    └── operation_sequence.json  # 操作序列
        └── skill_sequence       # [pick, place, pour, ...]
```

---

## 🧪 测试 Agent

运行基础测试 (不需要 API 密钥):

```bash
python3 test_agent_basic.py
```

---

## 📚 详细文档

### 1. 设计文档
📄 [docs/LangGraph_Agent_Integration_Plan.md](docs/LangGraph_Agent_Integration_Plan.md)

包含:
- 完整架构设计
- 每个节点的详细实现
- LangGraph 图构建代码
- 高级特性设计 (Phase 2 & 3)

### 2. 用户指南
📄 [docs/VLABench_Agent_User_Guide.md](docs/VLABench_Agent_User_Guide.md)

包含:
- 安装说明
- 使用方法和示例
- 故障排查
- 编程接口

---

## 🎯 当前限制和已知问题

### Phase 1 的限制:

1. **单任务生成**: 目前一次只能生成一个 example (example0)
2. **物体位置**: 使用简单的随机分布,可能需要人工微调
3. **场景验证**: 验证失败时不会自动重试和修复
4. **批量生成**: 暂不支持一次生成多个任务

### 计划在 Phase 2 和 Phase 3 实现:

- ✨ 场景验证反馈循环和自动修复
- ✨ 人工审核节点 (Human-in-the-Loop)
- ✨ 批量任务生成
- ✨ 任务模板库扩展
- ✨ Web UI 界面

---

## 🔧 故障排查

### 问题 1: 导入错误

```
ModuleNotFoundError: No module named 'langgraph'
```

**解决**: 安装依赖
```bash
pip install -r requirements_agent.txt
```

### 问题 2: API 密钥未设置

```
❌ 错误: 未设置 ANTHROPIC_API_KEY 环境变量
```

**解决**: 设置环境变量
```bash
export ANTHROPIC_API_KEY='sk-ant-...'
```

### 问题 3: VLABENCH_ROOT 未设置

**解决**: Agent 会自动设置,但也可以手动指定
```bash
export VLABENCH_ROOT=/ssd/mkqin/workspace/VLABench
```

---

## 💡 使用技巧

### 1. 好的指令格式

✅ **推荐**:
- "创建一个倾倒试管的任务"
- "生成一个将盖玻片放在显微镜台上的任务"
- "制作一个从试管架取下试管的场景"

❌ **避免**:
- "帮我做个任务" (太模糊)
- "试管" (缺少动作)
- "随便生成一个场景" (缺少具体需求)

### 2. 支持的操作类型

- **pick**: 抓取物体
- **place**: 放置到容器中
- **pour**: 倾倒
- **lift**: 举起到指定高度
- **slide**: 滑动 (暂未充分测试)

### 3. 支持的场景

- **laboratory**: 实验室场景 (推荐用于科研器材)
- **kitchen**: 厨房场景
- **living_room**: 客厅场景

---

## 📊 项目统计

```
创建的文件:
  - 10 个 Python 模块
  - 2 个文档文件
  - 2 个测试脚本
  - 1 个 CLI 入口
  - 1 个依赖配置

代码行数: ~1500 行
文档行数: ~1200 行
```

---

## 🎉 下一步

Phase 1 已完成! 你现在可以:

1. **立即使用**: 按照快速开始指南运行 Agent
2. **测试各种任务**: 尝试不同的自然语言指令
3. **查看生成结果**: 检查 `dataset/vlm_evaluation_v1.0/M&T/` 目录
4. **规划 Phase 2**: 根据实际使用反馈,实现场景优化和人工审核功能

---

**祝使用愉快! 🚀**

如有问题,请查看:
- [用户指南](docs/VLABench_Agent_User_Guide.md)
- [设计文档](docs/LangGraph_Agent_Integration_Plan.md)
