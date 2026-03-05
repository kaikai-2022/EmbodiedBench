"""
VLABench Agent 统一配置模块

集中管理 LLM API 配置,修改此文件即可全局生效
"""

import os
from typing import Optional


class AgentConfig:
    """
    Agent 配置类

    修改此类的属性即可配置 Agent 的行为
    """

    # ==================== LLM 模型配置 ====================

    # 模型名称 (可选: claude-sonnet-4-6, claude-sonnet-4-5-20250929, claude-sonnet-4-20250514)
    MODEL_NAME: str = "claude-sonnet-4-6"

    # 温度参数 (0-1, 控制输出随机性, 0=确定性, 1=创造性)
    TEMPERATURE: float = 0.7

    # 最大输出 token 数
    MAX_TOKENS: int = 4096

    # 请求超时时间 (秒)
    TIMEOUT: float = 60.0

    # 失败重试次数
    MAX_RETRIES: int = 2

    # ==================== API 配置 ====================

    # API Key (已配置)
    ANTHROPIC_API_KEY: str = "sk-7bjXQqAWmekKzBPC"

    # 自定义 API 端点 (去掉 /v1 后缀,库会自动添加)
    BASE_URL: str = "https://ck67.top"

    # ==================== 其他配置 ====================

    # 是否启用调试模式
    DEBUG: bool = False

    @classmethod
    def get_llm_config(cls) -> dict:
        """
        获取 LLM 初始化配置

        自动组装所有配置参数,供 ChatAnthropic 使用

        Returns:
            配置字典
        """
        config = {
            "model": cls.MODEL_NAME,
            "temperature": cls.TEMPERATURE,
            "max_tokens": cls.MAX_TOKENS,
            "timeout": cls.TIMEOUT,
            "max_retries": cls.MAX_RETRIES,
        }

        # 只在设置时添加 API Key
        if cls.ANTHROPIC_API_KEY:
            config["anthropic_api_key"] = cls.ANTHROPIC_API_KEY

        # 只在设置时添加自定义端点
        if cls.BASE_URL:
            config["base_url"] = cls.BASE_URL

        return config

    @classmethod
    def validate(cls) -> tuple[bool, str]:
        """
        验证配置是否正确

        Returns:
            (是否有效, 错误信息)
        """
        if not cls.ANTHROPIC_API_KEY:
            return False, "ANTHROPIC_API_KEY 未设置,请设置环境变量或在 config.py 中配置"

        # 支持多种 API Key 格式 (官方 sk-ant- 或第三方服务)
        if not (cls.ANTHROPIC_API_KEY.startswith("sk-") or len(cls.ANTHROPIC_API_KEY) > 10):
            return False, "ANTHROPIC_API_KEY 格式不正确"

        return True, ""


# 使用示例:
# from scripts.vlabench_agent.config import AgentConfig
# from langchain_anthropic import ChatAnthropic
#
# # 方法 1: 直接使用
# llm = ChatAnthropic(**AgentConfig.get_llm_config())
#
# # 方法 2: 先验证再使用
# is_valid, error_msg = AgentConfig.validate()
# if is_valid:
#     llm = ChatAnthropic(**AgentConfig.get_llm_config())
# else:
#     print(f"配置错误: {error_msg}")
