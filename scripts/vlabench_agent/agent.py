"""
VLABench Agent - 主逻辑

基于 LangGraph 构建的自动化任务生成流水线（路径一：物理仿真验证）

Pipeline:
  START → analyzer → asset_manager → skill_planner → code_generator → registration → simulation → vlm_data → END
                                          ↑                                  |              |
                                          |            (注册失败)              |   (仿真失败)  |
                                          +----------------------------------+--------------+
                                                      (重试，最多3次)
"""

import logging
from typing import Dict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import VLABenchAgentState
from .nodes import (
    analyzer_node,
    asset_manager_node,
)
from .nodes.skill_planner import skill_planner_node
from .nodes.code_generator import code_generator_node
from .nodes.registration import registration_node
from .nodes.simulation import simulation_node
from .nodes.vlm_data import vlm_data_node

logger = logging.getLogger(__name__)

MAX_CODE_GENERATION_ATTEMPTS = 3


def error_handler_node(state: Dict) -> Dict:
    """
    错误处理节点
    """
    errors = state.get("errors", [])
    error_msg = "\n".join(errors)

    logger.error("=" * 60)
    logger.error("Agent 执行失败:")
    logger.error(error_msg)
    logger.error("=" * 60)

    return {
        "current_stage": "error",
        "messages": state.get("messages", []) + [
            {"role": "assistant", "content": f"任务失败:\n{error_msg}"}
        ]
    }


def route_after_asset_manager(state: Dict) -> str:
    """
    asset_manager 之后的路由：成功 → skill_planner，失败 → error
    """
    if state.get("current_stage") == "error":
        return "error"
    return "skill_planner"


def route_after_registration(state: Dict) -> str:
    """
    registration 之后的路由：
    - 注册成功 → simulation
    - 注册失败且尝试次数 < MAX → skill_planner（重试）
    - 注册失败且超过重试限制 → error
    """
    if state.get("registration_success"):
        return "simulation"

    attempts = state.get("code_generation_attempts", 0)
    if attempts < MAX_CODE_GENERATION_ATTEMPTS:
        logger.info(f"[Router] 注册失败，第 {attempts} 次重试...")
        return "skill_planner"

    return "error"


def route_after_simulation(state: Dict) -> str:
    """
    simulation 之后的路由：
    - 仿真成功 → vlm_data
    - 仿真失败且尝试次数 < MAX → skill_planner（重试）
    - 仿真失败且超过重试限制 → error
    """
    if state.get("simulation_success"):
        return "vlm_data"

    attempts = state.get("code_generation_attempts", 0)
    if attempts < MAX_CODE_GENERATION_ATTEMPTS:
        logger.info(f"[Router] 仿真失败，第 {attempts} 次重试...")
        return "skill_planner"

    return "error"


def build_vlabench_agent():
    """
    构建 VLABench 自动化流水线 Agent（路径一：物理仿真验证）

    Returns:
        编译后的 LangGraph 对象
    """
    logger.info("构建 VLABench Agent (物理仿真路径)...")

    workflow = StateGraph(state_schema=VLABenchAgentState)

    # 添加节点
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("asset_manager", asset_manager_node)
    workflow.add_node("skill_planner", skill_planner_node)
    workflow.add_node("code_generator", code_generator_node)
    workflow.add_node("registration", registration_node)
    workflow.add_node("simulation", simulation_node)
    workflow.add_node("vlm_data", vlm_data_node)
    workflow.add_node("error_handler", error_handler_node)

    # 添加边
    workflow.add_edge(START, "analyzer")
    workflow.add_edge("analyzer", "asset_manager")

    workflow.add_conditional_edges(
        "asset_manager",
        route_after_asset_manager,
        {"skill_planner": "skill_planner", "error": "error_handler"}
    )

    workflow.add_edge("skill_planner", "code_generator")
    workflow.add_edge("code_generator", "registration")

    workflow.add_conditional_edges(
        "registration",
        route_after_registration,
        {"simulation": "simulation", "skill_planner": "skill_planner", "error": "error_handler"}
    )

    workflow.add_conditional_edges(
        "simulation",
        route_after_simulation,
        {"vlm_data": "vlm_data", "skill_planner": "skill_planner", "error": "error_handler"}
    )

    workflow.add_edge("vlm_data", END)
    workflow.add_edge("error_handler", END)

    # 编译图
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    logger.info("VLABench Agent 构建完成 (物理仿真路径)")
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
        # 代码生成
        "generated_code": None,
        "task_module_path": None,
        # 技能规划
        "skill_plan": None,
        # 注册
        "registration_success": None,
        # 仿真
        "simulation_success": None,
        "simulation_video_path": None,
        "simulation_hdf5_path": None,
        "episode_config": None,
        "executed_skill_sequence": None,
        # VLM 评测数据
        "env_config": None,
        "task_save_path": None,
        "rendered_images": None,
        "validation_report": None,
        # 重试控制
        "code_generation_attempts": 0,
        "error_feedback": None,
        # 流程控制
        "current_stage": "analyzing",
        "errors": [],
        "warnings": [],
    }
