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

    # 代码生成 (code_generator_node 输出)
    generated_code: Optional[str]               # 生成的 Python 任务类源码
    task_module_path: Optional[str]             # 写入磁盘的文件路径

    # 技能规划 (skill_planner_node 输出)
    skill_plan: Optional[Dict]                  # 结构化技能序列和条件 JSON

    # 注册 (registration_node 输出)
    registration_success: Optional[bool]        # 动态注册是否成功

    # 仿真 (simulation_node 输出)
    simulation_success: Optional[bool]          # 轨迹生成是否成功
    simulation_video_path: Optional[str]        # 录制的演示视频路径
    simulation_hdf5_path: Optional[str]         # 保存的 HDF5 训练数据路径
    episode_config: Optional[Dict]              # env.save() 的输出（环境配置快照）
    executed_skill_sequence: Optional[List[Dict]]  # 实际执行的技能序列（operation_sequence 格式）

    # VLM 评测数据 (vlm_data_node 输出)
    env_config: Optional[Dict]       # 完整的 env_config.json 内容
    task_save_path: Optional[str]    # VLM 评测任务保存路径
    rendered_images: Optional[List[str]]  # 渲染图像路径列表

    # 重试控制
    code_generation_attempts: int               # 代码生成尝试次数（默认 0）
    error_feedback: Optional[str]               # 失败时的错误上下文，反馈给 code_generator 重试

    # 流程控制
    current_stage: str  # 'analyzing' | 'asset_check' | 'code_generation' | 'registration' | 'simulation' | 'vlm_data' | 'done' | 'error'
    errors: List[str]   # 错误记录
    warnings: List[str] # 警告记录
