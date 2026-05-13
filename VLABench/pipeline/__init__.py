"""
VLABench Agent - 自动化任务生成智能体

基于 LangGraph 的科研场景任务自动化生成系统（路径一：物理仿真验证）
"""

from .agent import build_vlabench_agent, create_initial_state
from .state import VLABenchAgentState

__version__ = "0.2.0"
__all__ = ["build_vlabench_agent", "create_initial_state", "VLABenchAgentState"]
