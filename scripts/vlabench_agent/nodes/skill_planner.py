"""
Skill Planner Node - 用 LLM 规划技能序列

专注于任务语义理解和技能序列规划，输出结构化 JSON。
不生成代码，代码由 code_generator 节点通过模板填充完成。
"""

import json
import logging
import re
from typing import Dict, Optional

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

from ..config import AgentConfig

logger = logging.getLogger(__name__)

# 已知的任务类型及其标准技能序列模式
TASK_TYPE_PATTERNS = {
    "pick": {
        "description": "抓取物体（仅抓取，不移动到其他位置）",
        "typical_skills": ["pick"],
        "typical_conditions": {"is_grasped": {"entities": ["target_entity"], "robot": "robot"}},
    },
    "lift": {
        "description": "抓起并举起物体",
        "typical_skills": ["pick", "lift"],
        "typical_conditions": {"lift": {"entities": ["target_entity"], "target_height": 0.9}},
    },
    "place": {
        "description": "抓取物体并放入容器",
        "typical_skills": ["pick", "place"],
        "typical_conditions": {"contain": {"container": "target_container", "entities": ["target_entity"]}},
    },
    "pour": {
        "description": "抓取物体、举起、移动到目标上方、倾倒",
        "typical_skills": ["pick", "lift", "moveto", "pour"],
        "typical_conditions": {
            "pour": {"target_entity": "target_entity"},
            "above": {"target_entity": "target_entity", "platform": "target_container"},
        },
    },
    "push": {
        "description": "推动物体或按压按钮",
        "typical_skills": ["press"],
        "typical_conditions": {"press_button": {"target_button": "target_entity"}},
    },
    "rotate": {
        "description": "抓取物体并旋转",
        "typical_skills": ["pick", "lift", "rotate", "moveto", "open_gripper"],
        "typical_conditions": {
            "on_orientation": {"entities": ["target_entity"], "orientations": [[0, 0, 1.5708]], "tolerance_angle": 0.5236},
            "on_position": {"entities": ["target_entity"], "positions": [[0, 0, 0.8]], "tolerance_distance": 0.08},
        },
    },
    "move": {
        "description": "抓取物体并移动到指定位置后放下",
        "typical_skills": ["pick", "lift", "moveto", "open_gripper"],
        "typical_conditions": {},
    },
    "heat": {
        "description": "抓取物体、举起、移动到加热源上方加热",
        "typical_skills": ["pick", "lift", "moveto", "wait"],
        "typical_conditions": {
            "heated": {"target_entity": "target_entity", "heat_source": "target_container", "duration": 5.0},
        },
    },
}

VALID_SKILLS = {
    "pick", "place", "lift", "moveto", "pour", "push", "pull",
    "press", "flip", "wait", "rotate", "open_gripper", "close_gripper",
    "open_door", "close_door", "open_drawer", "open_laptop", "move_offset",
    "reset",
}

VALID_CONDITIONS = {
    "contain", "not_contain", "is_grasped", "press_button", "on", "above",
    "pour", "on_position", "contact", "joint_in_range", "lift", "order",
    "on_orientation", "asyn_sequence", "or", "heated",
}


def build_skill_planner_prompt(task_analysis: Dict, asset_status: Dict, error_feedback: str = None) -> str:
    """构建 Skill Planner 的 LLM prompt"""

    task_name = task_analysis.get("task_name", "custom_task")
    operation_type = task_analysis.get("operation_type", "pick")
    instruction_en = task_analysis.get("instruction_en", "")
    objects = task_analysis.get("objects", [])

    # 组装资产信息
    asset_info_lines = []
    for obj_name, info in asset_status.items():
        if isinstance(info, dict):
            xml_path = info.get("xml_path", "unknown")
            entity_class = info.get("class", "CommonGraspedEntity")
            newly_downloaded = info.get("newly_downloaded", False)
            asset_info_lines.append(
                f"  - {obj_name}: xml_path={xml_path}, class={entity_class}, "
                f"newly_downloaded={newly_downloaded}"
            )

    asset_info = "\n".join(asset_info_lines) if asset_info_lines else "  （无资产信息）"

    # 任务类型模式描述
    patterns_desc = []
    for ttype, pattern in TASK_TYPE_PATTERNS.items():
        skills_str = " → ".join(pattern["typical_skills"])
        conds_str = ", ".join(pattern["typical_conditions"].keys())
        patterns_desc.append(f"  - {ttype}: {pattern['description']}\n    技能: [{skills_str}], 条件: {{{conds_str}}}")
    patterns_text = "\n".join(patterns_desc)

    prompt = f"""你是一个机器人任务规划专家。根据给定的任务描述，规划出合理的技能序列和成功条件。

## 任务信息
- 任务名称: {task_name}
- 操作类型（仅供参考，以英文指令的语义为准）: {operation_type}
- 英文指令: {instruction_en}
- 涉及物体: {objects}

注意：操作类型可能不准确。请根据英文指令的实际语义判断任务类型。
例如 "Move the X" 应该是 move 类型（抓取+移动+放下），而不是 pick 类型。

## 资产信息
{asset_info}

## 可用的任务类型模式
{patterns_text}

## 可用技能（参数说明）
- pick(target_entity_name, prior_eulers=[[-pi, 0, 0]]): 从上方抓取物体
- place(target_container_name): 放置到容器
- lift(lift_height=0.15, gripper_state=np.zeros(2)): 举起物体
- moveto(target_pos, gripper_state=np.zeros(2)): 移动末端执行器到位置
- pour(): 倾倒
- press(target_pos): 按压
- push(target_pos, push_distance=0.1): 推动
- rotate(rotation_angle=pi/2, gripper_state=np.zeros(2)): 旋转
- open_gripper(): 松开夹爪
- flip(gripper_state): 翻转
- wait(wait_time=50): 等待

## 可用条件类型
- is_grasped: 物体被夹爪抓住 → {{"entities": ["物体名"], "robot": "robot"}}
- lift: 物体举到指定高度 → {{"entities": ["物体名"], "target_height": 0.9}}
- contain: 物体在容器内 → {{"container": "容器名", "entities": ["物体名"]}}
- pour: 物体被倾倒 → {{"target_entity": "物体名"}}
- above: 物体在平台上方 → {{"target_entity": "物体名", "platform": "平台名"}}
- on_position: 物体到达位置 → {{"entities": ["物体名"], "positions": [[x,y,z]], "tolerance_distance": 0.05}}
- press_button: 按钮被按下 → {{"target_button": "按钮名"}}
- on_orientation: 物体方向正确 → {{"entities": ["物体名"], "orientations": [[r,p,y]], "tolerance_angle": 0.5236}}
- heated: 物体被加热源加热（在上方停留足够时间） → {{"target_entity": "物体名", "heat_source": "加热源名", "duration": 5.0}}

## Franka 机械臂工作空间范围
- X: -0.3 ~ 0.3 (左右)
- Y: -0.2 ~ 0.3 (前后)
- Z: 0.75 ~ 1.5 (上下)
- 桌面高度约 0.78m，物体通常在 z=0.80~0.85
- 机器人基座位置: [0, -0.4, 0.78]

## 规则
1. "移动"类任务（move）应包含 pick → lift → moveto → open_gripper，条件用 on_position
2. "抓取"类任务（pick）只需 pick，条件用 is_grasped
3. "举起"类任务（lift）用 pick → lift，条件用 lift
4. "放置"类任务（place）用 pick → place，条件用 contain
5. 所有 pick 操作默认 prior_eulers=[[-3.14159, 0, 0]]（从上方竖直抓取）
6. lift 和 moveto 之后必须加 gripper_state=np.zeros(2) 保持夹爪闭合
7. target_entity_name 填 "self.target_entity"
8. target_container_name 填 "self.target_container"
9. moveto 的 target_pos 如果是容器上方，用表达式 "np.array(self.entities[self.target_container].get_xpos(physics)) + np.array([0, 0, 0.2])"
10. moveto 的 target_pos 如果是随机桌面位置，用 "target_pos"（由外部变量提供）
11. **重要**: move 类任务中，moveto 的 target_pos 和 on_position 条件的 positions 必须使用同一个目标位置变量 "target_pos"
12. **重要**: on_position 条件的 tolerance_distance 应设为 0.1（物体放下后有偏移），dimension 默认检查 XY 平面（2维）
13. **重要**: heat 类任务（加热）用 pick → lift → moveto → wait，条件用 heated。heat_source 填加热工具（如本生灯），target_entity 填被加热物体。moveto 目标是加热源上方。

请输出 JSON（不要任何其他文字）：
{{
  "task_type": "pick|lift|place|pour|push|rotate|move|heat",
  "skill_sequence": [
    {{"skill": "技能名", "params": {{"参数名": "参数值"}}}}
  ],
  "conditions": {{
    "条件类型": {{条件参数}}
  }},
  "instruction_template": "英文指令模板，用 {{target_entity}} 和 {{target_container}} 占位",
  "needs_container": true/false,
  "moveto_target_expr": "moveto 的 target_pos 表达式（如果有）或 null"
}}"""

    if error_feedback:
        prompt += f"""

## 上次规划失败的反馈
{error_feedback}
请根据反馈调整你的规划。"""

    return prompt


def validate_skill_plan(plan: Dict) -> tuple:
    """
    校验 skill plan 的结构合法性

    Returns:
        (is_valid, error_msg)
    """
    # 检查必要字段
    required_fields = ["task_type", "skill_sequence", "conditions", "instruction_template", "needs_container"]
    for field in required_fields:
        if field not in plan:
            return False, f"缺少必要字段: {field}"

    # 检查 task_type
    valid_types = set(TASK_TYPE_PATTERNS.keys())
    if plan["task_type"] not in valid_types:
        return False, f"未知 task_type: {plan['task_type']}，可选: {valid_types}"

    # 检查 skill_sequence
    if not isinstance(plan["skill_sequence"], list) or len(plan["skill_sequence"]) == 0:
        return False, "skill_sequence 必须是非空列表"

    for i, entry in enumerate(plan["skill_sequence"]):
        if not isinstance(entry, dict) or "skill" not in entry:
            return False, f"skill_sequence[{i}] 格式错误，需要 {{skill: ..., params: ...}}"
        if entry["skill"] not in VALID_SKILLS:
            return False, f"未知技能: {entry['skill']}，可选: {VALID_SKILLS}"

    # 检查 conditions
    if not isinstance(plan["conditions"], dict):
        return False, "conditions 必须是 dict"
    for cond_type in plan["conditions"]:
        if cond_type not in VALID_CONDITIONS:
            return False, f"未知条件类型: {cond_type}，可选: {VALID_CONDITIONS}"

    return True, ""


def extract_json_from_response(text: str) -> Optional[Dict]:
    """从 LLM 响应中提取 JSON"""
    # 尝试直接解析
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 尝试从 markdown 代码块中提取
    patterns = [
        r'```json\s*\n(.*?)\n\s*```',
        r'```\s*\n(.*?)\n\s*```',
        r'\{[\s\S]*\}',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                json_str = match.group(1) if match.lastindex else match.group(0)
                return json.loads(json_str)
            except (json.JSONDecodeError, IndexError):
                continue

    return None


def skill_planner_node(state: Dict) -> Dict:
    """
    技能规划节点 - 用 LLM 生成结构化的技能序列和成功条件

    输入: task_analysis, asset_status
    输出: skill_plan (结构化 JSON)
    """
    if ChatAnthropic is None:
        logger.error("[Skill Planner] langchain_anthropic 未安装")
        return {
            "current_stage": "error",
            "errors": state.get("errors", []) + ["langchain_anthropic 未安装"],
        }

    logger.info("=" * 60)
    logger.info("[Skill Planner] 开始规划技能序列...")

    task_analysis = state.get("task_analysis", {})
    asset_status = state.get("asset_status", {})
    error_feedback = state.get("error_feedback")

    prompt = build_skill_planner_prompt(task_analysis, asset_status, error_feedback)
    llm = ChatAnthropic(**AgentConfig.get_llm_config())

    max_retries = 2
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                logger.info(f"[Skill Planner] 第 {attempt + 1} 次尝试...")
                # 将上次的错误反馈追加到 prompt
                retry_prompt = prompt + f"\n\n## 上次解析失败: {last_error}\n请确保输出合法 JSON。"
                response = llm.invoke(retry_prompt)
            else:
                response = llm.invoke(prompt)

            plan = extract_json_from_response(response.content)
            if plan is None:
                last_error = f"无法从 LLM 响应中解析 JSON: {response.content[:200]}"
                logger.warning(f"[Skill Planner] {last_error}")
                continue

            is_valid, error_msg = validate_skill_plan(plan)
            if not is_valid:
                last_error = f"规划校验失败: {error_msg}"
                logger.warning(f"[Skill Planner] {last_error}")
                continue

            # 成功
            logger.info(f"[Skill Planner] ✓ 规划完成:")
            logger.info(f"  任务类型: {plan['task_type']}")
            skills = [s['skill'] for s in plan['skill_sequence']]
            logger.info(f"  技能序列: {' → '.join(skills)}")
            logger.info(f"  条件: {list(plan['conditions'].keys())}")
            logger.info("=" * 60 + "\n")

            return {
                "skill_plan": plan,
                "current_stage": "code_generation",
            }

        except Exception as e:
            last_error = str(e)
            logger.error(f"[Skill Planner] LLM 调用失败: {e}")

    # 所有重试都失败
    logger.error(f"[Skill Planner] ✗ 规划失败: {last_error}")
    return {
        "skill_plan": None,
        "error_feedback": f"技能规划失败: {last_error}",
        "current_stage": "code_generation",  # 降级到 legacy code_generator
    }
