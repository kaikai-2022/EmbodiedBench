"""
VLABench Agent - 主逻辑

基于 LangGraph 构建的自动化任务生成流水线
"""

import logging
from typing import Dict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import VLABenchAgentState
from .nodes import (
    analyzer_node,
    asset_manager_node,
    task_creator_node,
    render_executor_node
)

logger = logging.getLogger(__name__)


def error_handler_node(state: Dict) -> Dict:
    """
    错误处理节点

    Args:
        state: VLABenchAgentState

    Returns:
        更新后的状态字典
    """
    errors = state.get("errors", [])
    error_msg = "\n".join(errors)

    logger.error("=" * 60)
    logger.error("❌ Agent 执行失败:")
    logger.error(error_msg)
    logger.error("=" * 60)

    return {
        "current_stage": "error",
        "messages": state.get("messages", []) + [
            {"role": "assistant", "content": f"任务失败:\n{error_msg}"}
        ]
    }


def should_continue_to_task_creator(state: Dict) -> str:
    """
    条件判断: asset_manager 之后是否继续到 task_creator

    Args:
        state: VLABenchAgentState

    Returns:
        下一个节点名称
    """
    if state.get("current_stage") == "error":
        return "error"
    return "task_creator"


def build_vlabench_agent():
    """
    构建 VLABench 自动化流水线 Agent

    Returns:
        编译后的 LangGraph 对象
    """
    logger.info("构建 VLABench Agent...")

    # 创建状态图
    workflow = StateGraph(state_schema=VLABenchAgentState)

    # 添加节点
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("asset_manager", asset_manager_node)
    workflow.add_node("task_creator", task_creator_node)
    workflow.add_node("render_executor", render_executor_node)
    workflow.add_node("error_handler", error_handler_node)

    # 添加边
    # START -> analyzer
    workflow.add_edge(START, "analyzer")

    # analyzer -> asset_manager
    workflow.add_edge("analyzer", "asset_manager")

    # asset_manager -> task_creator (成功) 或 error_handler (失败)
    workflow.add_conditional_edges(
        "asset_manager",
        should_continue_to_task_creator,
        {
            "task_creator": "task_creator",
            "error": "error_handler"
        }
    )

    # task_creator -> render_executor
    workflow.add_edge("task_creator", "render_executor")

    # render_executor -> END
    workflow.add_edge("render_executor", END)

    # error_handler -> END
    workflow.add_edge("error_handler", END)

    # 编译图 (添加 checkpointer 用于状态持久化)
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    logger.info("✓ VLABench Agent 构建完成")

    return graph


def create_initial_state(user_instruction: str) -> Dict:
    """
    创建初始状态

    Args:
        user_instruction: 用户自然语言指令

    Returns:
        初始状态字典
    """
    return {
        "messages": [],
        "user_instruction": user_instruction,
        "task_analysis": {},
        "asset_status": {},
        "env_config": None,
        "task_save_path": None,
        "rendered_images": None,
        "validation_report": None,
        "current_stage": "analyzing",
        "errors": [],
        "warnings": []
    }
