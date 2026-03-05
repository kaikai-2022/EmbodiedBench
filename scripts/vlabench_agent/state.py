"""
VLABench Agent State 定义
"""

from typing import TypedDict, Annotated, List, Dict, Optional
from langgraph.graph.message import add_messages


class VLABenchAgentState(TypedDict):
    """VLABench Agent 全局状态"""

    # 对话和任务输入
    messages: Annotated[list, add_messages]  # LLM 对话历史
    user_instruction: str                     # 用户原始自然语言指令

    # 任务分析结果 (Analyzer Node 输出)
    task_analysis: Dict[str, any]  # {
    #   "task_type": "place_coverslip_on_microscope",
    #   "objects": ["coverslip", "microscope"],
    #   "scene": "laboratory",
    #   "operation_type": "place",
    #   "instruction_parsed": "Place the coverslip onto the microscope stage"
    # }

    # 资产管理结果 (Asset Manager Node 输出)
    asset_status: Dict[str, any]  # {
    #   "coverslip": {
    #       "found": True,
    #       "xml_path": "obj/meshes/coverslip/coverslip.xml",
    #       "class": "LabEquipment"
    #   },
    #   "microscope": {
    #       "found": False,
    #       "需要下载": True,
    #       "search_keyword": "microscope"
    #   }
    # }

    # 任务配置 (Task Creator Node 输出)
    env_config: Optional[Dict]       # 完整的 env_config.json 内容
    task_save_path: Optional[str]    # 任务保存路径

    # 渲染结果 (Render Executor Node 输出)
    rendered_images: Optional[List[str]]  # 渲染图像路径列表
    validation_report: Optional[Dict]     # 场景验证报告

    # 流程控制
    current_stage: str  # 'analyzing' | 'asset_check' | 'asset_download' | 'task_creation' | 'rendering' | 'done' | 'error'
    errors: List[str]   # 错误记录
    warnings: List[str] # 警告记录
