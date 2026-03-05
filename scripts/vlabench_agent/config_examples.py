"""
VLABench Agent 配置示例

复制此文件内容到 config.py 中进行配置
"""

# ============================================================================
# 配置示例 1: 使用环境变量 (推荐)
# ============================================================================
"""
保持 config.py 默认设置,只需设置环境变量:

export ANTHROPIC_API_KEY='sk-ant-api03-your-key-here'

# 可选: 使用自定义端点
export ANTHROPIC_BASE_URL='https://your-proxy.com/v1'
"""

# ============================================================================
# 配置示例 2: 直接在 config.py 中设置
# ============================================================================
"""
编辑 scripts/vlabench_agent/config.py:

class AgentConfig:
    # 模型配置
    MODEL_NAME: str = "claude-sonnet-4-5"
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 4096
    TIMEOUT: float = 60.0
    MAX_RETRIES: int = 2

    # API 配置 - 直接设置
    ANTHROPIC_API_KEY: str = "sk-ant-api03-your-key-here"
    BASE_URL: Optional[str] = None  # 或设置自定义端点
"""

# ============================================================================
# 配置示例 3: 使用 OpenAI (需要改代码)
# ============================================================================
"""
如果要改用 OpenAI GPT-4:

1. 修改 config.py:
   - 添加 OPENAI_API_KEY 配置
   - 修改 MODEL_NAME = "gpt-4"

2. 修改 analyzer.py 和 task_creator.py:
   from langchain_openai import ChatOpenAI
   llm = ChatOpenAI(**AgentConfig.get_llm_config())
"""

# ============================================================================
# 配置示例 4: 使用代理服务器
# ============================================================================
"""
编辑 config.py:

class AgentConfig:
    MODEL_NAME: str = "claude-sonnet-4-5"
    ANTHROPIC_API_KEY: str = "your-key"
    BASE_URL: str = "https://your-proxy.example.com/v1"
"""

# ============================================================================
# 配置示例 5: 不同环境使用不同配置
# ============================================================================
"""
可以根据环境变量动态切换:

class AgentConfig:
    ENV = os.environ.get("VLABENCH_ENV", "production")

    if ENV == "development":
        MODEL_NAME = "claude-haiku-4"  # 开发用更便宜的模型
        TEMPERATURE = 0.5
    else:
        MODEL_NAME = "claude-sonnet-4-5"  # 生产用更好的模型
        TEMPERATURE = 0.7

    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
"""

# ============================================================================
# 常用模型选择
# ============================================================================
"""
Anthropic 模型对比:

1. claude-opus-4-5
   - 最强大,推理能力最好
   - 最贵: ~$15/1M 输入, ~$75/1M 输出
   - 适用: 复杂任务,需要最高质量

2. claude-sonnet-4-5 (推荐)
   - 性能和成本平衡最好
   - 价格: ~$3/1M 输入, ~$15/1M 输出
   - 适用: 大多数任务

3. claude-haiku-4
   - 最快速,最便宜
   - 价格: ~$0.25/1M 输入, ~$1.25/1M 输出
   - 适用: 简单任务,大批量处理
"""
