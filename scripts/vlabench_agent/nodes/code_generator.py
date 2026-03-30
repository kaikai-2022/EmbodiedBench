"""
Code Generator Node - 模板化代码生成

从 Skill Planner 输出的结构化 JSON 生成 VLABench 任务类 Python 代码。
使用确定性模板填充，不调用 LLM。当模板生成失败时降级到 LLM 生成。
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

from ..config import AgentConfig

logger = logging.getLogger(__name__)

# ==================== 模板定义 ====================

IMPORTS_TEMPLATE = """\
import random
import numpy as np
from functools import partial

from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.primitive.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
"""

CONFIG_MANAGER_TEMPLATE = '''\
@register.add_config_manager("{task_name}")
class {class_prefix}ConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=[1, 1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)

{load_objects_method}
    def get_instruction(self, target_entity, {extra_params}**kwargs):
        instruction = ["{instruction_template}"]
        self.config["task"]["instructions"] = instruction

    def get_condition_config(self, target_entity, {extra_params}**kwargs):
{conditions_block}

    def get_target_entity(self):
        return "{target_entity_name}"

{get_task_config_method}'''

TASK_CLASS_TEMPLATE = '''\
@register.add_task("{task_name}")
class {class_prefix}Task(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
{moveto_target_line}        skill_sequence = [
{skill_lines}        ]
        return skill_sequence
'''


# ==================== 模板填充函数 ====================

def _to_class_prefix(task_name: str) -> str:
    """将 task_name 转为 CamelCase 类名前缀"""
    return "".join(word.capitalize() for word in task_name.split("_"))


def _format_skill_param(key: str, value) -> str:
    """格式化单个技能参数为 Python 表达式"""
    if isinstance(value, str):
        # 识别 Python 表达式（self.xxx, np.xxx, target_pos 等）
        if any(value.startswith(prefix) for prefix in (
            "self.", "np.", "target_pos", "random.", "euler_to_quaternion"
        )):
            return f"{key}={value}"
        # 普通字符串
        return f'{key}="{value}"'
    elif isinstance(value, list):
        return f"{key}={value}"
    elif isinstance(value, bool):
        return f"{key}={value}"
    elif isinstance(value, (int, float)):
        return f"{key}={value}"
    else:
        return f"{key}={repr(value)}"


def _format_skill_line(entry: Dict) -> str:
    """将一个技能条目转为 partial(...) 调用字符串"""
    skill = entry["skill"]
    params = entry.get("params", {})
    param_parts = [_format_skill_param(k, v) for k, v in params.items()]
    params_str = ", ".join(param_parts)
    if params_str:
        return f"            partial(SkillLib.{skill}, {params_str}),"
    else:
        return f"            partial(SkillLib.{skill}),"


def _format_conditions_dict(conditions: Dict, target_entity_name: str) -> str:
    """将条件字典转为 Python dict 字面量字符串"""
    if not conditions:
        return "dict()"

    parts = []
    for cond_type, cond_params in conditions.items():
        inner_parts = []
        for k, v in cond_params.items():
            if isinstance(v, str):
                # 替换占位符
                if v == "target_entity":
                    inner_parts.append(f'{k}="{target_entity_name}"')
                elif v == "target_container":
                    inner_parts.append(f'{k}=f"{{target_container}}"')
                else:
                    inner_parts.append(f'{k}="{v}"')
            elif isinstance(v, list):
                # 列表中的字符串替换
                formatted_items = []
                for item in v:
                    if isinstance(item, str) and item == "target_entity":
                        formatted_items.append(f'"{target_entity_name}"')
                    elif isinstance(item, str):
                        formatted_items.append(f'"{item}"')
                    else:
                        formatted_items.append(repr(item))
                inner_parts.append(f"{k}=[{', '.join(formatted_items)}]")
            else:
                inner_parts.append(f"{k}={repr(v)}")
        parts.append(f"{cond_type}=dict({', '.join(inner_parts)})")

    return "dict(\n            " + ",\n            ".join(parts) + "\n        )"


def _generate_load_objects_method(asset_status: Dict, objects: List[str]) -> str:
    """生成 load_objects 方法（非内置资产需要自定义加载）

    只为 target_entity（objects[0]）生成自定义加载。
    其他物体（容器、加热源等）由 load_containers / load_init_containers 处理。
    """
    if not objects:
        return ""

    target_entity = objects[0]
    info = asset_status.get(target_entity, {})
    if not isinstance(info, dict) or not info.get("xml_path") or info.get("builtin", False):
        return ""

    xml_path = info.get("xml_path", "UNKNOWN")
    entity_class = info.get("class", "CommonGraspedEntity")

    return f'''
    def load_objects(self, target_entity):
        object_config = dict(
            name=target_entity,
            xml_path="{xml_path}",
            position=[0.0, 0.0, 0.8],
            orientation=[0, 0, 0],
        )
        object_config["class"] = "{entity_class}"
        object_config["randomness"] = dict(pos=[0.05, 0.05, 0], quat=[0, 0, 0.1])
        self.config["task"]["components"].append(object_config)
'''


def _generate_load_containers_method(asset_status: Dict, objects: List[str], needs_container: bool) -> str:
    """生成 load_containers 方法（非内置的 target_container 需要自定义加载）

    当 target_container 对应的资产不在 name2class_xml 中（如 Objaverse 下载的本生灯）时，
    需要重写 load_containers 来直接构造 entity config，避免 get_entity_config 的 KeyError。
    """
    if not needs_container or len(objects) < 2:
        return ""

    # 查找非内置的容器/目标容器资产
    # 在 heat 任务中，target_container 通常是 objects 列表中的非 target_entity 且非 init_container 的物体
    for obj_name in objects[1:]:  # skip target_entity (objects[0])
        info = asset_status.get(obj_name, {})
        if isinstance(info, dict) and info.get("xml_path") and not info.get("builtin", False):
            xml_path = info.get("xml_path", "UNKNOWN")
            entity_class = info.get("class", "CommonGraspedEntity")
            canonical_name = info.get("canonical_name", obj_name)
            return f'''
    def load_containers(self, target_container):
        """自定义加载 target_container（非内置资产，从 Objaverse 下载）"""
        if target_container is not None:
            container_config = dict(
                name=target_container,
                xml_path="{xml_path}",
                position=[random.uniform(-0.15, 0.15), random.uniform(0, 0.15), 0.8],
                orientation=[0, 0, 0],
            )
            container_config["class"] = "{entity_class}"
            container_config["randomness"] = dict(pos=[0.05, 0.05, 0], quat=[0, 0, 0.1])
            self.config["task"]["components"].append(container_config)
'''
    return ""


def generate_code_from_template(skill_plan: Dict, task_analysis: Dict, asset_status: Dict) -> str:
    """
    从 skill_plan JSON 和模板生成完整的 Python 任务文件。

    Args:
        skill_plan: Skill Planner 输出的结构化计划
        task_analysis: 任务分析结果
        asset_status: 资产状态

    Returns:
        完整的 Python 源代码字符串
    """
    task_name = task_analysis.get("task_name", "custom_task")
    objects = task_analysis.get("objects", [])
    class_prefix = _to_class_prefix(task_name)
    target_entity_name = objects[0] if objects else "object"

    # 是否需要容器
    needs_container = skill_plan.get("needs_container", False)

    # get_instruction / get_condition_config 的额外参数
    extra_params = "target_container, " if needs_container else ""

    # 生成 load_objects 方法
    load_objects_method = _generate_load_objects_method(asset_status, objects)

    # ---- get_task_config_method ----
    # 不重写 get_task_config()，让父类 BenchTaskConfigManager.get_task_config() 处理。
    # 父类会正确初始化 self.config["task"]，并按顺序调用
    # load_containers / load_init_containers / load_objects / get_condition_config / get_instruction。
    get_task_config_method = ""

    # 生成技能序列行
    skill_lines = "\n".join(
        _format_skill_line(entry) for entry in skill_plan["skill_sequence"]
    )

    # 生成 moveto target 行（如果有）
    task_type = skill_plan.get("task_type", "")
    moveto_expr = skill_plan.get("moveto_target_expr")
    # 对 move 类任务：如果 moveto_target_expr 缺失、为 null、或只是变量名 "target_pos"，
    # 使用工作空间范围内的随机位置表达式
    if task_type == "move" and (not moveto_expr or moveto_expr == "target_pos"):
        moveto_expr = "np.array([random.uniform(-0.15, 0.15), random.uniform(-0.15, 0.15), 0.85])"
    if moveto_expr and moveto_expr != "target_pos":
        moveto_target_line = f"        target_pos = {moveto_expr}\n"
    else:
        moveto_target_line = ""

    # 生成条件字典
    # 对 move 类任务，不设置 on_position 条件
    # 因为 target_pos 是运行时随机生成的，配置阶段无法确定，
    # 且固定在原点附近会导致初始状态就满足条件，触发 should_terminate_episode 无限 reset
    conditions = skill_plan.get("conditions", {})
    if task_type == "move":
        conditions.pop("on_position", None)

    # 生成 conditions 代码块
    # 注意：空条件必须用 pass（不设 conditions），而不是赋值空 dict
    # 因为 ConditionSet([]).is_met() 返回 all([]) == True → 触发无限 reset
    if conditions:
        conditions_dict = _format_conditions_dict(conditions, target_entity_name)
        conditions_block = f"        conditions_config = {conditions_dict}\n        self.config[\"task\"][\"conditions\"] = conditions_config"
    else:
        conditions_block = "        pass"

    # 指令模板
    instruction_template = skill_plan.get("instruction_template", f"Perform task on {target_entity_name}")

    # 组装代码
    code = IMPORTS_TEMPLATE + "\n"
    code += CONFIG_MANAGER_TEMPLATE.format(
        task_name=task_name,
        class_prefix=class_prefix,
        load_objects_method=load_objects_method,
        extra_params=extra_params,
        instruction_template=instruction_template,
        conditions_block=conditions_block,
        target_entity_name=target_entity_name,
        get_task_config_method=get_task_config_method,
    )
    code += "\n\n"
    code += TASK_CLASS_TEMPLATE.format(
        task_name=task_name,
        class_prefix=class_prefix,
        moveto_target_line=moveto_target_line,
        skill_lines=skill_lines,
    )

    return code


# ==================== 代码校验 ====================

def validate_generated_code(code: str) -> tuple:
    """基础校验生成的代码"""
    required_patterns = [
        ("@register.add_task", "缺少 @register.add_task 装饰器"),
        ("@register.add_config_manager", "缺少 @register.add_config_manager 装饰器"),
        ("get_expert_skill_sequence", "缺少 get_expert_skill_sequence 方法"),
        ("PrimitiveTask", "缺少 PrimitiveTask 继承"),
        ("BenchTaskConfigManager", "缺少 BenchTaskConfigManager 继承"),
    ]

    for pattern, error_msg in required_patterns:
        if pattern not in code:
            return False, error_msg

    # 尝试编译检查语法
    try:
        compile(code, "<generated>", "exec")
    except SyntaxError as e:
        return False, f"语法错误: {e}"

    return True, ""


# ==================== Legacy LLM 代码生成（降级用） ====================

# 保留旧的 SKILL_SIGNATURES 和 CONDITION_TYPES 供 legacy 使用
SKILL_SIGNATURES = """
SkillLib.pick(target_entity_name: str, prior_eulers: list = None)
SkillLib.place(target_container_name: str, target_pos: np.ndarray = None)
SkillLib.lift(lift_height: float = 0.15, gripper_state: np.ndarray = None)
SkillLib.moveto(target_pos: np.ndarray, target_quat: np.ndarray = None, gripper_state: np.ndarray = None)
SkillLib.pour(target_delta_qpos: np.ndarray = None, target_q_velocity: np.ndarray = None)
SkillLib.push(target_pos: np.ndarray, push_distance: float = 0.1)
SkillLib.pull(target_pos: np.ndarray, pull_distance: float = 0.1)
SkillLib.press(target_pos: np.ndarray, target_quat: np.ndarray = None)
SkillLib.flip(gripper_state: np.ndarray = None)
SkillLib.wait(wait_time: int = 50)
"""

CONDITION_TYPES = """
1. contain: {"contain": {"container": "容器名", "entities": ["物体名"]}}
2. pour: {"pour": {"target_entity": "物体名"}}
3. lift: {"lift": {"entities": ["物体名"], "target_height": 0.9}}
4. above: {"above": {"target_entity": "物体名", "platform": "平台名"}}
5. is_grasped: {"is_grasped": {"entities": ["物体名"], "robot": "robot"}}
6. on_position: {"on_position": {"entity": "物体名", "target_pos": [x, y, z], "threshold": 0.05}}
7. press_button: {"press_button": {"target_button": "按钮名"}}
8. heated: {"heated": {"target_entity": "物体名", "heat_source": "加热源名", "duration": 5.0}}
"""


def _build_legacy_prompt(task_analysis: Dict, asset_status: Dict, error_feedback: str = None) -> str:
    """构建 legacy LLM 代码生成 prompt（降级用）"""
    objects_info = []
    for obj_name in task_analysis.get("objects", []):
        info = asset_status.get(obj_name, {})
        canonical = info.get("canonical_name", obj_name)
        objects_info.append({
            "name": obj_name,
            "canonical_name": canonical,
            "xml_path": info.get("xml_path", "UNKNOWN"),
            "class": info.get("class", "CommonGraspedEntity"),
            "newly_downloaded": info.get("newly_downloaded", False),
            "use_canonical": canonical != obj_name,
        })

    prompt = f"""你是 VLABench 任务代码生成专家。请根据以下信息生成一个完整的 VLABench 任务 Python 文件。

## 任务信息
- 任务名称: {task_analysis.get("task_name", "custom_task")}
- 操作类型: {task_analysis.get("operation_type", "pick")}
- 英文指令: {task_analysis.get("instruction_en", "")}
- 物体: {json.dumps(objects_info, indent=2, ensure_ascii=False)}

## 可用技能库
{SKILL_SIGNATURES}

## 可用条件类型
{CONDITION_TYPES}

## 关键规则
1. 文件必须包含一个 ConfigManager 类和一个 Task 类
2. ConfigManager 用 @register.add_config_manager 装饰，继承 BenchTaskConfigManager
3. Task 用 @register.add_task 装饰，继承 PrimitiveTask
4. 必须实现 get_expert_skill_sequence(self, physics) 方法
5. 如果物体是新下载的，重写 load_objects() 直接构造 entity config dict
6. "移动"任务应该用 [pick, lift, moveto, open_gripper] 序列
7. prior_eulers 默认用 [[-np.pi, 0, 0]]（从上方竖直抓取）

请只输出完整的 Python 代码，用 ```python 和 ``` 包裹。
"""

    if error_feedback:
        prompt += f"\n## 错误反馈\n```\n{error_feedback}\n```\n请修复问题。\n"

    return prompt


def _extract_code_from_response(content: str) -> str:
    """从 LLM 响应中提取 Python 代码"""
    content = content.strip()
    if "```python" in content:
        parts = content.split("```python")
        if len(parts) > 1:
            code_part = parts[1]
            if "```" in code_part:
                code_part = code_part.split("```")[0]
            return code_part.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return content


def _code_generator_legacy(state: Dict) -> Dict:
    """Legacy LLM 代码生成（降级用）"""
    logger.info("[Code Generator] 使用 legacy LLM 模式...")

    task_analysis = state.get("task_analysis", {})
    asset_status = state.get("asset_status", {})
    error_feedback = state.get("error_feedback")
    task_name = task_analysis.get("task_name", "custom_task")
    attempts = state.get("code_generation_attempts", 0)

    prompt = _build_legacy_prompt(task_analysis, asset_status, error_feedback)
    llm = ChatAnthropic(**AgentConfig.get_llm_config())

    try:
        response = llm.invoke(prompt)
        code = _extract_code_from_response(response.content)

        is_valid, error_msg = validate_generated_code(code)
        if not is_valid:
            return {
                "generated_code": code,
                "code_generation_attempts": attempts + 1,
                "error_feedback": f"代码校验失败: {error_msg}\n\n生成的代码:\n{code}",
                "current_stage": "code_generation",
            }

        vlabench_root = os.environ.get("VLABENCH_ROOT")
        task_file = Path(vlabench_root) / "tasks" / "hierarchical_tasks" / "primitive" / f"{task_name}_series.py"
        task_file.parent.mkdir(parents=True, exist_ok=True)
        task_file.write_text(code, encoding="utf-8")

        logger.info(f"[Code Generator Legacy] ✓ 代码已写入: {task_file}")
        return {
            "generated_code": code,
            "task_module_path": str(task_file),
            "code_generation_attempts": attempts + 1,
            "error_feedback": None,
            "current_stage": "registration",
        }

    except Exception as e:
        logger.error(f"[Code Generator Legacy] ✗ 失败: {e}")
        return {
            "code_generation_attempts": attempts + 1,
            "error_feedback": f"LLM 调用失败: {e}",
            "current_stage": "error",
            "errors": state.get("errors", []) + [f"代码生成失败: {e}"],
        }


# ==================== 主节点函数 ====================

def code_generator_node(state: Dict) -> Dict:
    """
    代码生成节点 - 从 skill_plan 模板生成代码，失败时降级到 LLM

    Args:
        state: VLABenchAgentState

    Returns:
        更新后的状态字典
    """
    logger.info("=" * 60)
    attempts = state.get("code_generation_attempts", 0)
    logger.info(f"[Code Generator] 开始生成任务代码 (第 {attempts + 1} 次尝试)...")

    task_analysis = state.get("task_analysis", {})
    asset_status = state.get("asset_status", {})
    skill_plan = state.get("skill_plan")
    task_name = task_analysis.get("task_name", "custom_task")

    # 如果没有 skill_plan，降级到 legacy LLM 模式
    if skill_plan is None:
        logger.warning("[Code Generator] 无 skill_plan，降级到 legacy 模式")
        return _code_generator_legacy(state)

    # 尝试模板生成
    try:
        code = generate_code_from_template(skill_plan, task_analysis, asset_status)

        logger.info(f"[Code Generator] 模板生成代码长度: {len(code)} 字符")

        # 校验
        is_valid, error_msg = validate_generated_code(code)
        if not is_valid:
            logger.warning(f"[Code Generator] 模板代码校验失败: {error_msg}，降级到 legacy")
            return _code_generator_legacy(state)

        # 写入文件
        vlabench_root = os.environ.get("VLABENCH_ROOT")
        if not vlabench_root:
            return {
                "current_stage": "error",
                "errors": state.get("errors", []) + ["VLABENCH_ROOT 未设置"],
            }

        task_file = Path(vlabench_root) / "tasks" / "hierarchical_tasks" / "primitive" / f"{task_name}_series.py"
        task_file.parent.mkdir(parents=True, exist_ok=True)
        task_file.write_text(code, encoding="utf-8")

        logger.info(f"[Code Generator] ✓ 模板代码已写入: {task_file}")
        logger.info("=" * 60 + "\n")

        return {
            "generated_code": code,
            "task_module_path": str(task_file),
            "code_generation_attempts": attempts + 1,
            "error_feedback": None,
            "current_stage": "registration",
        }

    except Exception as e:
        logger.warning(f"[Code Generator] 模板生成异常: {e}，降级到 legacy")
        return _code_generator_legacy(state)
