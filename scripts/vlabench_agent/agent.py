"""
VLABench Agent - 主逻辑

基于 LangGraph 构建的自动化任务生成流水线

Pipeline:
  START → analyzer → normalizer → asset_manager → skill_planner → code_generator → registration → simulation → vlm_data → END
                                                                              ↑
                                                              (注册/仿真失败: retry_router 精准回溯)
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
from .nodes.normalizer import normalizer_node
from .nodes.skill_planner import skill_planner_node
from .nodes.code_generator import code_generator_node
from .nodes.registration import registration_node
from .nodes.simulation import simulation_node
from .nodes.vlm_data import vlm_data_node
from .nodes.node_logger import init_run_log

logger = logging.getLogger(__name__)

MAX_CODE_GENERATION_ATTEMPTS = 3


def error_handler_node(state: Dict) -> Dict:
    """错误处理节点"""
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


def _classify_error(error: str) -> str:
    """
    精准回溯分类：根据错误类型分发到最可能修复的节点。
    返回目标节点名。
    """
    code_gen_errors = ["SyntaxError", "AttributeError", "TypeError", "NameError",
                       "IndentationError", "SyntaxWarning"]
    asset_errors = ["asset", "xml", "download", "texture", "name2class_xml",
                    "missing", "not found", "找不到"]
    planning_errors = ["collision", "condition", "timeout", "skill failed",
                       "skill failed to carry out", "does not exist"]

    for keyword in code_gen_errors:
        if keyword.lower() in error.lower():
            logger.info(f"[Router] 错误分类: code_generator (关键词: {keyword})")
            return "code_generator"

    for keyword in asset_errors:
        if keyword.lower() in error.lower():
            logger.info(f"[Router] 错误分类: asset_manager (关键词: {keyword})")
            return "asset_manager"

    for keyword in planning_errors:
        if keyword.lower() in error.lower():
            logger.info(f"[Router] 错误分类: skill_planner (关键词: {keyword})")
            return "skill_planner"

    logger.info("[Router] 错误分类: skill_planner (未知错误，安全回退)")
    return "skill_planner"


def retry_router_node(state: Dict) -> Dict:
    """
    精准回溯节点：根据错误类型决定回退到哪个节点重新开始。
    作为 LangGraph 节点执行，实际只做分类然后将路由信息写入 state，
    由 route_retry 路由函数决定下一步。
    """
    error = state.get("error_feedback", "")
    target = _classify_error(error)
    return {"_retry_target": target}


def route_retry(state: Dict) -> str:
    """retry_router 节点之后的路由"""
    return state.get("_retry_target", "skill_planner")


def route_after_registration(state: Dict) -> str:
    """registration 之后的路由"""
    if state.get("registration_success"):
        return "simulation"
    attempts = state.get("code_generation_attempts", 0)
    if attempts < MAX_CODE_GENERATION_ATTEMPTS:
        logger.info(f"[Router] 注册失败，尝试次数 {attempts}，进入 retry_router...")
        return "retry_router"
    return "error_handler"


def route_after_simulation(state: Dict) -> str:
    """simulation 之后的路由"""
    if state.get("simulation_success"):
        return "vlm_data"
    attempts = state.get("code_generation_attempts", 0)
    if attempts < MAX_CODE_GENERATION_ATTEMPTS:
        logger.info(f"[Router] 仿真失败，尝试次数 {attempts}，进入 retry_router...")
        return "retry_router"
    return "error_handler"


def build_vlabench_agent():
    """构建 VLABench 自动化流水线 Agent"""
    logger.info("构建 VLABench Agent...")

    workflow = StateGraph(state_schema=VLABenchAgentState)

    # 注册所有节点
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("normalizer", normalizer_node)
    workflow.add_node("asset_manager", asset_manager_node)
    workflow.add_node("skill_planner", skill_planner_node)
    workflow.add_node("code_generator", code_generator_node)
    workflow.add_node("registration", registration_node)
    workflow.add_node("simulation", simulation_node)
    workflow.add_node("vlm_data", vlm_data_node)
    workflow.add_node("error_handler", error_handler_node)
    workflow.add_node("retry_router", retry_router_node)

    # 正常链路
    workflow.add_edge(START, "analyzer")
    workflow.add_edge("analyzer", "normalizer")
    workflow.add_edge("normalizer", "asset_manager")
    workflow.add_edge("asset_manager", "skill_planner")
    workflow.add_edge("skill_planner", "code_generator")
    workflow.add_edge("code_generator", "registration")
    workflow.add_edge("registration", "simulation")
    workflow.add_edge("simulation", "vlm_data")
    workflow.add_edge("vlm_data", END)

    # 失败路由：失败 → retry_router → _retry_target → ...
    workflow.add_conditional_edges(
        "registration",
        route_after_registration,
        {"simulation": "simulation", "retry_router": "retry_router", "error_handler": "error_handler"}
    )
    workflow.add_conditional_edges(
        "simulation",
        route_after_simulation,
        {"vlm_data": "vlm_data", "retry_router": "retry_router", "error_handler": "error_handler"}
    )
    # retry_router 之后根据 _retry_target 分发
    workflow.add_conditional_edges(
        "retry_router",
        route_retry,
        {
            "code_generator": "code_generator",
            "skill_planner": "skill_planner",
            "asset_manager": "asset_manager",
            "error_handler": "error_handler",
        }
    )
    workflow.add_edge("error_handler", END)

    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    logger.info("VLABench Agent 构建完成")
    return graph


def create_initial_state(user_instruction: str) -> Dict:
    """创建初始状态"""
    log_filepath = init_run_log(user_instruction)
    return {
        "messages": [],
        "user_instruction": user_instruction,
        "task_analysis": {},
        "normalized_context": {},
        "task_graph": {},
        "asset_status": {},
        "skill_plan": None,
        "generated_code": None,
        "task_module_path": None,
        "registration_success": None,
        "simulation_success": None,
        "simulation_video_path": None,
        "simulation_hdf5_path": None,
        "episode_config": None,
        "executed_skill_sequence": None,
        "env_config": None,
        "task_save_path": None,
        "rendered_images": None,
        "validation_report": None,
        "code_generation_attempts": 0,
        "error_feedback": None,
        "current_stage": "analyzing",
        "errors": [],
        "warnings": [],
        "asset_cache": {},
        "_log_filepath": log_filepath,
    }
