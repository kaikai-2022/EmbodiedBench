"""
VLABench Agent - 自动化任务生成智能体

基于 LangGraph 的科研场景任务自动化生成系统
"""

from .agent import build_vlabench_agent, VLABenchAgentState

__version__ = "0.1.0"
__all__ = ["build_vlabench_agent", "VLABenchAgentState"]
