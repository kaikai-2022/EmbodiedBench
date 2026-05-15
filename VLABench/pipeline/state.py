"""
VLABench Agent State 定义

Pipeline 基于四份设计规范文档重构后的状态 schema：
  - Analyzer: raw_entities + raw_steps (神经符号 AST)
  - Normalizer: normalized_context (instances + steps)
  - Asset Manager: asset_status (xml_path + class_name + properties)
  - Skill Planner: global_skill_plan (带 CoT 的原子序列)
  - Code Generator: generated_code + task_module_path
  - 控制: current_stage, errors, warnings, error_feedback, asset_cache
"""

from typing import TypedDict, Annotated, List, Dict, Optional, Any
from langgraph.graph.message import add_messages


class VLABenchAgentState(TypedDict):
    """VLABench Agent 全局状态"""

    # ========== 输入 ==========
    messages: Annotated[list, add_messages]  # LLM 对话历史
    user_instruction: str                     # 用户原始自然语言指令

    # ========== Analyzer 输出 ==========
    # task_analysis 包含:
    #   "raw_entities": [  # 所有物理实体的原始标识列表
    #       {
    #           "raw_id": str,       # LLM 生成的局部唯一标识，格式: 名词_数字
    #           "raw_type": str,     # 提取自原文的精确名词词组
    #           "semantic_attributes": Dict,  # 开放属性包，仅提取原文明确提及的状态
    #       }
    #   ],
    #   "raw_steps": [  # 接地指令序列
    #       {
    #           "step_id": int,      # 从 0 开始的递增序号
    #           "action": str,       # 原始动词（建议用原形）
    #           "primary_obj": str,  # 动作主体，必须是 raw_id
    #           "secondary_obj": str | None,  # 客体/目标容器/参考物
    #           "grounded_instruction": str,  # 实体接地指令，名词替换为 <raw_id>
    #       }
    #   ]
    # 向后兼容字段（同时写入，供暂未适配的代码路径使用）：
    #   "instruction_en": str = user_instruction
    #   "scene": str = "laboratory"
    #   "task_name": str = 从 raw_steps 推断
    task_analysis: Dict[str, Any]

    # ========== Normalizer 输出 ==========
    # normalized_context 包含:
    #   "instances": [  # 传递给 Asset Manager 的物料图纸
    #       {
    #           "uid": str,           # 全局唯一标识符，格式: spec简化_序号
    #           "spec": str,          # 映射到资产库的标准大类名称
    #           "source_type": str,   # "local" | "objaverse"
    #           "is_physical": bool, # 是否需要独立 3D 模型渲染
    #           "init_params": Dict,  # 强类型初始化参数
    #       }
    #   ],
    #   "steps": [  # 传递给 Skill Planner 的执行剧本
    #       {
    #           "step_id": int,      # 继承自 Analyzer 的序号
    #           "action": str,       # 继承自 Analyzer 的原始动词
    #           "primary_uid": str,  # raw_id → uid 的映射
    #           "secondary_uid": str | None,
    #           "grounded_instruction": str,  # <raw_id> 已替换为 <uid>
    #       }
    #   ]
    normalized_context: Optional[Dict[str, Any]]

    # ========== Asset Manager 输出 ==========
    # key = uid (全局唯一), value = {
    #     "xml_path": str,       # 注入修改后的 MuJoCo XML 路径
    #     "class_name": str,     # 动态推断出的 VLABench 底层类名
    #     "properties": Dict,    # 实例化属性包（is_container, solution 等）
    # }
    asset_status: Dict[str, Any]

    # ========== Skill Planner 输出 ==========
    # skill_plan 包含:
    #   "global_skill_plan": [
    #       {
    #           "step_id": int,
    #           "semantic_instruction": str,  # 原始重接地指令
    #           "pre_state_assertion": str,  # 执行前的机器人/环境状态
    #           "atomic_sequence": [
    #               {"skill": str, "params": Dict},
    #           ],
    #           "post_state_assertion": str,  # 执行后的机器人/环境状态
    #       }
    #   ]
    skill_plan: Optional[Dict[str, Any]]

    # ========== Condition Planner 输出 ==========
    # condition_plan: per-step 的成功条件配置，格式:
    #   [
    #     {"step_id": 0, "condition_type": "contain", "params": {"container": "beaker_0", "entities": ["tube_0"]}},
    #     {"step_id": 1, "condition_type": "pass"},  # 无条件通过
    #   ]
    # condition_type 为 condition.py 中注册的 17 种之一，或 "pass" 表示跳过
    condition_plan: Optional[List[Dict[str, Any]]]

    # ========== Code Generator 输出 ==========
    generated_code: Optional[str]               # 生成的 Python 任务类源码
    task_module_path: Optional[str]             # 写入磁盘的文件路径

    # ========== Registration 输出 ==========
    registration_success: Optional[bool]        # 动态注册是否成功

    # ========== Simulation 输出 ==========
    simulation_success: Optional[bool]            # 轨迹生成是否成功
    simulation_video_path: Optional[str]        # 录制的演示视频路径
    simulation_hdf5_path: Optional[str]         # 保存的 HDF5 训练数据路径
    episode_config: Optional[Dict]              # env.save() 的输出
    executed_skill_sequence: Optional[List[Dict]]  # 实际执行的技能序列

    # ========== VLM Data 输出 ==========
    env_config: Optional[Dict]                  # 完整的 env_config.json 内容
    task_save_path: Optional[str]              # VLM 评测任务保存路径
    rendered_images: Optional[List[str]]         # 渲染图像路径列表
    validation_report: Optional[Dict]            # 验证报告

    # ========== 控制 ==========
    code_generation_attempts: int               # 代码生成尝试次数（默认 0）
    error_feedback: Optional[str]               # 失败时的错误上下文
    current_stage: str                          # 当前阶段标识
    errors: List[str]                           # 错误记录
    warnings: List[str]                          # 警告记录
    asset_cache: Dict                           # Normalizer 本地缓存（raw_type → {spec, source_type}）
    _log_filepath: Optional[str]               # 当前运行日志文件路径

    # ========== Reviewer 相关 ==========
    # step_timestamps: 记录每个大 step 和原子操作的执行时间戳
    # 格式: [{"step_id": int, "atomic_timestamps": [{"atomic_idx": int, "start": float, "end": float}, ...], "step_end": float}]
    step_timestamps: Optional[List[Dict]]
    # review_results: Reviewer 节点对每个 step 的审核结果
    # 格式: [{"step_id": int, "passed": bool, "reason": str}]
    review_results: Optional[List[Dict]]
    # review_passed: 所有 step 都通过则为 True
    review_passed: Optional[bool]
