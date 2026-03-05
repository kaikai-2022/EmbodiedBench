# VLABench Agent API 配置指南

## 📍 配置文件位置

### 当前 API 调用位置

1. **[scripts/vlabench_agent/nodes/analyzer.py:41](../scripts/vlabench_agent/nodes/analyzer.py#L41)**
   - 用于分析用户自然语言指令

2. **[scripts/vlabench_agent/nodes/task_creator.py:140](../scripts/vlabench_agent/nodes/task_creator.py#L140)**
   - 用于生成操作序列

3. **新增: [scripts/vlabench_agent/config.py](../scripts/vlabench_agent/config.py)**
   - 集中配置管理模块 (可选使用)

---

## 🔧 配置方法

### 方法 1: 环境变量 (推荐,默认方式)

**优点**:
- 不需要修改代码
- 安全,密钥不会提交到代码仓库
- 符合最佳实践

**使用方法**:

```bash
# 设置 API Key (必需)
export ANTHROPIC_API_KEY='sk-ant-api03-...'

# 设置自定义端点 (可选)
export ANTHROPIC_BASE_URL='https://your-custom-endpoint.com'

# 运行 Agent
python3 vlabench_agent_cli.py --instruction "创建一个倾倒试管的任务"
```

**永久设置** (添加到 shell 配置文件):

```bash
# 对于 bash
echo 'export ANTHROPIC_API_KEY="sk-ant-api03-..."' >> ~/.bashrc
source ~/.bashrc

# 对于 zsh
echo 'export ANTHROPIC_API_KEY="sk-ant-api03-..."' >> ~/.zshrc
source ~/.zshrc
```

---

### 方法 2: 使用 .env 文件

**优点**:
- 配置集中管理
- 支持多个环境变量
- 可以添加到 `.gitignore` 保护密钥

**步骤**:

1. **创建 `.env` 文件**:

```bash
cd /ssd/mkqin/workspace/VLABench
cat > .env << 'EOF'
# Anthropic API 配置
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
ANTHROPIC_BASE_URL=https://api.anthropic.com  # 可选,使用默认值

# 其他配置
VLABENCH_ROOT=/ssd/mkqin/workspace/VLABench
EOF
```

2. **安装 python-dotenv**:

```bash
pip install python-dotenv
```

3. **修改 CLI 入口加载配置**:

编辑 `vlabench_agent_cli.py`,在开头添加:

```python
from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件
```

4. **添加到 .gitignore**:

```bash
echo ".env" >> .gitignore
```

---

### 方法 3: 代码中直接指定 (不推荐)

**警告**: 密钥会暴露在代码中,不安全!

修改 `analyzer.py` 和 `task_creator.py`:

```python
# 修改前 (第41行 / 第140行)
llm = ChatAnthropic(model="claude-sonnet-4-5")

# 修改后
llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    anthropic_api_key="sk-ant-api03-your-key-here",  # ⚠️ 不安全!
    base_url="https://custom-endpoint.com"  # 可选
)
```

---

### 方法 4: 使用配置模块 (推荐,灵活方式)

**优点**:
- 集中管理配置
- 易于维护和修改
- 支持不同环境

**步骤**:

1. **配置模块已创建**: [scripts/vlabench_agent/config.py](../scripts/vlabench_agent/config.py)

2. **修改配置** (编辑 `config.py`):

```python
class AgentConfig:
    # LLM 配置
    MODEL_NAME: str = "claude-sonnet-4-5"  # 可改为其他模型
    TEMPERATURE: float = 0.7               # 温度参数
    MAX_TOKENS: int = 4096                 # 最大输出
    TIMEOUT: float = 60.0                  # 超时时间
    MAX_RETRIES: int = 2                   # 重试次数

    # API 配置 (从环境变量读取)
    ANTHROPIC_API_KEY: Optional[str] = os.environ.get("ANTHROPIC_API_KEY")
    BASE_URL: Optional[str] = os.environ.get("ANTHROPIC_BASE_URL")
```

3. **修改节点代码使用配置**:

编辑 `analyzer.py` 和 `task_creator.py`:

```python
# 在文件开头添加
from ..config import AgentConfig

# 修改 LLM 初始化 (第41行 / 第140行)
llm = ChatAnthropic(**AgentConfig.get_llm_config())
```

---

## 🌐 自定义 API 端点场景

### 场景 1: 使用官方 Anthropic API

```bash
export ANTHROPIC_API_KEY='sk-ant-api03-...'
# 使用默认 https://api.anthropic.com
```

### 场景 2: 使用代理服务器

```bash
export ANTHROPIC_API_KEY='your-key'
export ANTHROPIC_BASE_URL='https://your-proxy.com/v1'
```

### 场景 3: 使用 Azure OpenAI (需要改代码)

```python
# 改用 langchain-openai
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_endpoint="https://your-resource.openai.azure.com/",
    api_version="2024-02-15-preview",
    deployment_name="gpt-4",
    api_key="your-azure-key"
)
```

### 场景 4: 使用 OpenRouter

```bash
export ANTHROPIC_API_KEY='your-openrouter-key'
export ANTHROPIC_BASE_URL='https://openrouter.ai/api/v1'
```

---

## 📊 参数说明

### 必需参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `model` | 模型名称 | `claude-sonnet-4-5` |
| `anthropic_api_key` | API 密钥 | 从环境变量读取 |

### 可选参数

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `temperature` | 输出随机性 (0-1) | 1.0 | 0.7 (更稳定) |
| `max_tokens` | 最大输出长度 | 4096 | 4096 |
| `timeout` | 请求超时(秒) | 60.0 | 60.0-120.0 |
| `max_retries` | 重试次数 | 2 | 2-3 |
| `base_url` | 自定义端点 | `https://api.anthropic.com` | - |

---

## ✅ 验证配置

运行验证脚本:

```bash
python3 verify_installation.py
```

或手动测试:

```bash
source /ssd/mkqin/miniconda3/etc/profile.d/conda.sh
conda activate vlabench_2

python3 -c "
import os
print('ANTHROPIC_API_KEY:', 'SET' if os.environ.get('ANTHROPIC_API_KEY') else 'NOT SET')
print('ANTHROPIC_BASE_URL:', os.environ.get('ANTHROPIC_BASE_URL', 'Using default'))

from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model='claude-sonnet-4-5')
print('✓ LLM initialized successfully')
"
```

---

## 🔒 安全建议

1. **永远不要** 将 API Key 提交到 Git 仓库
2. **使用** 环境变量或 `.env` 文件
3. **添加** `.env` 到 `.gitignore`
4. **定期轮换** API Key
5. **设置** API 使用限额和告警

```bash
# 添加到 .gitignore
echo ".env" >> .gitignore
echo "*.key" >> .gitignore
```

---

## 💰 成本控制

### 监控使用量

访问 [Anthropic Console](https://console.anthropic.com/settings/usage) 查看使用情况。

### 设置预算告警

在 Anthropic Console 中设置每月预算限制。

### 估算成本

| 操作 | API 调用次数 | 估算成本 |
|------|-------------|---------|
| 单个任务生成 | 2-3 次 | $0.01-0.02 |
| 批量 10 个任务 | 20-30 次 | $0.10-0.20 |
| 批量 100 个任务 | 200-300 次 | $1.00-2.00 |

---

## 🆘 常见问题

### Q1: 出现 "Invalid API key" 错误

**检查**:
```bash
echo $ANTHROPIC_API_KEY  # 检查是否设置
```

**解决**:
```bash
export ANTHROPIC_API_KEY='sk-ant-api03-...'  # 重新设置
```

### Q2: 连接超时

**可能原因**: 网络问题或需要代理

**解决**:
```bash
# 设置代理
export HTTPS_PROXY='http://your-proxy:port'

# 或增加超时时间
# 修改 config.py: TIMEOUT = 120.0
```

### Q3: 想使用其他模型

**支持的模型**:
- `claude-opus-4-5` (更强大但更贵)
- `claude-sonnet-4-5` (推荐,性价比最高)
- `claude-haiku-4` (更快但能力较弱)

**修改方法**:
```python
# config.py
MODEL_NAME: str = "claude-opus-4-5"  # 改为 Opus
```

---

## 📚 相关文档

- [Anthropic API 文档](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)
- [LangChain Anthropic 集成](https://python.langchain.com/docs/integrations/chat/anthropic)
- [用户指南](./VLABench_Agent_User_Guide.md)
- [快速开始](./QUICKSTART.md)
