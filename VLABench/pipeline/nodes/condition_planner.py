"""
Condition Planner Node - 为每个 step 生成成功条件

与 skill_planner 并列，都从 normalizer 获取 steps，各自独立工作后汇入 code_generator。

职责：
  - 分析每个 step 的 action verb 和 entity UIDs
  - 从现有 17 种 condition 类型中选择最合适的一个
  - 输出 condition_plan 供 simulation 节点使用

设计原则：
  - 纯 LLM 驱动，选择逻辑需要语义理解
  - 不执行任何物理操作，只生成配置
  - action → condition 映射是参考而非硬编码
"""

import json
import logging
import re
from typing import Dict, List, Optional

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

from ..config import AgentConfig
from .node_logger import log_node_output_file

logger = logging.getLogger(__name__)

# ========== 可用 Condition 类型白皮书 (CONDITION_LIB_DOC) ==========
CONDITION_LIB_DOC = """
## Available Success Conditions

For each step, select ONE condition that best captures the expected end state after that step executes.
Select from these condition types (registered in VLABench/tasks/condition.py):

- **contain**(container=<uid>, entities=[<uid>], **kwargs): Entity is inside the container.
  Use for: "place X in Y", "pour liquid into Y", "insert X into Y"

- **not_contain**(container=<uid>, entities=[<uid>]): Entity is NOT inside the container.
  Use for: "remove X from Y", "take X out of Y"

- **on**(entities=[<uid>], container=<uid>): Entity is in contact with and resting on top of container.
  Use for: "place X on Y", "set X on top of Y"

- **above**(target_entity=<uid>, platform=<uid>): Entity is floating above a platform.
  Use for: "hover X above Y"

- **pour**(target_entity=<uid>, threshold=0): Container's top site z < bottom site z (container is tilted).
  Use for: "pour from X", "tilt X"

- **heated**(target_entity=<uid>, heat_source=<uid>, duration=5.0, xy_tolerance=0.08):
  Target entity has been above heat source for cumulative duration seconds.
  Use for: "heat X over Y", "warm X with Y"

- **lift**(entities=[<uid>], lift_height=0.15): Entity is lifted above its initial height.
  Use for: "lift X", "raise X", "elevate X"
  **IMPORTANT**: Use `lift_height` parameter (relative height in meters, default 0.15) instead of `target_height`.
  The condition will check if entity_z >= initial_z + lift_height.

- **on_position**(entities=[<uid>], positions=[[x,y,z],...], tolerance_distance=0.03, dimension=2):
  Entity is near target position. Use for: "move X to position", "place X at location"

- **is_grasped**(entities=[<uid>], robot="robot"): Entity is held by robot gripper.
  Use for: "grasp X", "pick up X"
  **IMPORTANT**: Always include `"robot": "robot"` in params.

- **contact**(entity1=<uid>, entity2=<uid>, robot="robot"): Two entities are touching.
  Use for: "touch X to Y", "contact X with Y"
  **IMPORTANT**: Always include `"robot": "robot"` in params.

- **on_orientation**(entities=[<uid>], orientations=[[r,p,y],...], tolerance_angle=pi/6):
  Entity orientation matches target. Use for: "rotate X to orientation", "orient X"

- **order**(entities=[<uid>], axis=[0], offset=0.1): Entities are in positional order.
  Use for: "arrange X before Y", "line up X, Y, Z"

- **press_button**(target_button=<uid>): Button is pressed.
  Use for: "press X", "push button"

- **joint_in_range**(entities=[<uid>], target_pos_range=[min,max]): Joint is within range.
  Use for: "rotate joint X to range", "slide X within range"

- **asyn_sequence**(condition_sets=[...]): Multiple conditions must be met over time.
  Use for: complex multi-phase success criteria (advanced, prefer simpler types above)

- **or**(condition_sets=[...]): Any one condition in the set is met.
  Use for: alternative success criteria (advanced, prefer simpler types above)

- **shake**(entities=[<uid>], robot="robot", min_direction_changes=3, min_angle_threshold=0.1, check_axis=1):
  Target entity has been shaken: it must be grasped and its orientation has oscillated back and forth at least min_direction_changes times.
  Use for: "shake X", "oscillate X"
  **IMPORTANT**: Always include `"robot": "robot"` in params. `check_axis` defaults to 1 (Y-axis pitch). `min_angle_threshold` filters out small vibrations.

- **pass**: No physical condition check needed. Step succeeds simply by completing execution.
  Use for: "move to position without final placement goal", "open gripper"

- **wait_for**(entity=<uid>, robot="robot", wait_duration=2.0, change_type="add_solution", solution="CuSO4"):
  Entity is not being touched/grasped by the robot gripper after the wait period.
  Use for: "wait for human to add X", "wait for external change", "wait for solution to change color"
  **IMPORTANT**: Always include `"robot": "robot"` in params. `change_type` options: "add_solution" (needs `solution`), "solution_change_color" (needs `color` as RGBA list like [1, 0, 0, 0.4]), "change_color" (needs `color`).
  The condition checks that the robot gripper is not touching the entity (i.e., the entity was left alone during the wait).

## Action → Condition Selection Guide

Use this as reference, but the LLM should use semantic understanding:

| Action | Typical Condition | Reasoning |
|--------|-------------------|-----------|
| pour | **pour** (source tilted) | Check if source container is tilted (top_site below bottom_site) |
| place (in/into) | **contain** | Entity inside container |
| place (on) | **on** | Entity resting on surface |
| remove (from) | **not_contain** | Entity no longer inside |
| lift/raise | **lift** or **above** | Entity above height |
| heat | **heated** | Accumulated heating time |
| press | **press_button** | Button pressed |
| insert | **contain** | Entity inside target |
| shake | **shake** | Object grasped and orientation oscillated |
| wait_for | **wait_for** | Entity stillness triggers auto change |
| wait | **pass** | Just waiting, no state change |
| move | **on_position** or **pass** | Depends on if position matters |

## Parameter Notes

- All entity parameters MUST use UIDs from the Physical Asset Inventory (e.g., "tube_0", "beaker_0"). Do NOT use substance/solution names (e.g., "CuSO4_0", "NaCl_1") as entity parameters — they are not physical objects in the simulation.
- Numeric parameters (duration, tolerance, target_height, etc.) use reasonable defaults unless the instruction specifies exact values
- Some conditions accept additional kwargs (e.g., contain accepts "layer" for multi-layer containers)
- **pour action**: Always use the **pour** condition (checks if source container is tilted). Do NOT use contain for pour actions, since liquids are not simulated as physical entities.
"""

# ========== 合法 Condition 类型集合 ==========
VALID_CONDITION_TYPES = {
    "contain", "not_contain", "on", "above", "pour", "heated",
    "on_position", "lift", "contact", "is_grasped", "on_orientation",
    "order", "press_button", "joint_in_range", "asyn_sequence", "or",
    "pass", "wait_for", "shake"
}


def _extract_json(text: str) -> Optional[List[Dict]]:
    """从 LLM 响应中提取 JSON 数组"""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    patterns = [
        r'```json\s*\n(.*?)\n\s*```',
        r'```\s*\n(.*?)\n\s*```',
        r'\[[\s\S]*\]',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue
    return None


def _validate_condition_plan(plan: List[Dict], num_steps: int, valid_uids: set) -> tuple:
    """
    校验 condition_plan 的完整性

    Returns:
        (is_valid, error_msg)
    """
    if not isinstance(plan, list):
        return False, "condition_plan 必须是 list"

    if len(plan) != num_steps:
        return False, f"condition_plan 长度 ({len(plan)}) 与 step 数量 ({num_steps}) 不匹配"

    covered_step_ids = set()
    for i, entry in enumerate(plan):
        if not isinstance(entry, dict):
            return False, f"condition_plan[{i}] 必须是 dict"

        if "step_id" not in entry:
            return False, f"condition_plan[{i}] 缺少 step_id"

        step_id = entry["step_id"]
        if step_id in covered_step_ids:
            return False, f"condition_plan 中 step_id {step_id} 重复"
        covered_step_ids.add(step_id)

        if step_id not in range(num_steps):
            return False, f"condition_plan[{i}] step_id {step_id} 超出范围 (0-{num_steps-1})"

        if "condition_type" not in entry:
            return False, f"condition_plan[{i}] 缺少 condition_type"

        cond_type = entry["condition_type"]
        if cond_type not in VALID_CONDITION_TYPES:
            return False, f"condition_plan[{i}] 无效的 condition_type: {cond_type}，可选: {VALID_CONDITION_TYPES}"

        if cond_type != "pass":
            if "params" not in entry:
                return False, f"condition_plan[{i}] 非 pass 条件缺少 params"

            params = entry.get("params", {})
            for k, v in params.items():
                if k in ["robot"]:
                    continue
                # 跳过数值参数
                if k in ["positions", "target_pos_range", "orientations",
                         "duration", "xy_tolerance", "target_height", "tolerance_distance",
                         "tolerance_angle", "dimension", "offset", "threshold", "check_axes",
                         "layer", "tilt_angle", "wait_time", "insert_depth", "lift_height",
                         "push_distance", "rotation_angle", "gripper_state",
                         "min_direction_changes", "min_angle_threshold", "check_axis"]:
                    continue
                    continue
                # entity 参数应该是字符串 UID
                if isinstance(v, str) and v not in valid_uids and v != "":
                    # 警告但不阻塞（可能是动态计算的参数）
                    pass
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, str) and item not in valid_uids and item != "":
                            pass

    return True, ""


def _build_asset_info(asset_status: Dict) -> str:
    """Build asset inventory text for the prompt."""
    if not asset_status:
        return "  (no asset info)"

    lines = []
    for uid, info in asset_status.items():
        class_name = info.get("class_name", "Unknown")
        properties = info.get("properties", {})
        props_str = ", ".join(f"{k}={v}" for k, v in properties.items()) if properties else "none"
        lines.append(f"  - {uid}: class={class_name}, properties={{{props_str}}}")
    return "\n".join(lines)


def _build_steps_info(steps: List[Dict]) -> str:
    """Build execution script text for the prompt."""
    lines = []
    for step in steps:
        action = step.get("action", "?")
        primary = step.get("primary_uid", step.get("primary_obj", "?"))
        secondary = step.get("secondary_uid", step.get("secondary_obj", None))
        instruction = step.get("grounded_instruction", "")
        step_id = step.get("step_id", "?")
        lines.append(f"  Step {step_id}: [{action}] {primary} -> {secondary}")
        lines.append(f"    instruction: {instruction}")
    return "\n".join(lines)


def build_condition_planner_prompt(
    normalized_context: Dict,
    asset_status: Dict,
    error_feedback: str = None
) -> str:
    """Build the Condition Planner Mega-Prompt."""
    steps = normalized_context.get("steps", [])
    asset_info = _build_asset_info(asset_status)
    steps_info = _build_steps_info(steps)

    prompt = f"""You are a robot task condition planning expert. Given execution steps and physical assets, determine the success condition for each step.

## Your Task

For each step in the execution script, select the appropriate condition type that checks whether the step's goal was achieved. The condition will be evaluated immediately after the step's atomic actions complete.

## Core Principles

1. **Step-Level Condition**: Select the condition that verifies the END STATE of THIS step, not the overall task.
2. **Physical Verification**: Conditions check actual physical state (position, contact, containment, etc.), not skill execution.
3. **Use "pass" for ambiguous steps**: If the step has no clear physical success criterion (e.g., "shake", "wait"), use "pass".
4. **Prefer simple conditions**: Use the simplest condition type that captures the step's goal.
5. **UID References**: Use entity UIDs from the asset inventory in condition params.

## Execution Script (from Normalizer)
{steps_info}

## Physical Asset Inventory
{asset_info}

{CONDITION_LIB_DOC}

## Output Format (strict JSON array, no other text):
[
  {{
    "step_id": 0,
    "condition_type": "contain",
    "params": {{"container": "beaker_0", "entities": ["tube_0"]}},
    "reasoning": "The step places tube_0 into beaker_0, so contain is appropriate"
  }},
  {{
    "step_id": 1,
    "condition_type": "pass",
    "params": {{}},
    "reasoning": "This step is a shake operation with no specific position goal"
  }}
]

## Important Rules

- The output MUST be a JSON array with one entry per step
- Each entry MUST have: step_id, condition_type, params, reasoning
- If condition_type is "pass", params should be an empty object {{}}
- All entity references in params MUST use UIDs from the asset inventory
- Numeric parameters should use reasonable defaults unless the instruction specifies exact values"""

    if error_feedback:
        prompt += f"""

## Previous Planning Error Feedback
{error_feedback}
Please adjust your plan based on this feedback."""

    return prompt


def condition_planner_node(state: Dict) -> Dict:
    """
    Condition Planner 节点 - 为每个 step 生成成功条件

    输入: state["normalized_context"]["steps"] + state["asset_status"]
    输出: state["condition_plan"]
    """
    logger.info("=" * 60)
    logger.info("[Condition Planner] 节点被调用")
    logger.info(f"[Condition Planner] 当前 state keys: {list(state.keys())}")

    if ChatAnthropic is None:
        logger.error("[Condition Planner] langchain_anthropic 未安装")
        result = {
            "errors": state.get("errors", []) + ["langchain_anthropic 未安装"],
            "current_stage": "code_generator",
        }
        log_node_output_file("condition_planner", state, result)
        return result

    normalized_context = state.get("normalized_context", {})
    asset_status = state.get("asset_status", {})
    error_feedback = state.get("error_feedback")

    steps = normalized_context.get("steps", [])
    if not steps:
        logger.warning("[Condition Planner] steps 为空，跳过")
        result = {
            "condition_plan": [],
            "current_stage": "code_generator",
        }
        log_node_output_file("condition_planner", state, result)
        return result

    num_steps = len(steps)
    valid_uids = set(asset_status.keys())

    logger.info(f"[Condition Planner] 开始为 {num_steps} 个步骤生成条件...")

    llm = ChatAnthropic(**AgentConfig.get_llm_config())
    prompt = build_condition_planner_prompt(normalized_context, asset_status, error_feedback)

    max_retries = 2
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                retry_prompt = prompt + f"\n\n## 上次校验失败: {last_error}\n请严格遵循 JSON 格式和规则。"
                response = llm.invoke(retry_prompt)
            else:
                response = llm.invoke(prompt)

            raw = response.content
            if isinstance(raw, list):
                text_block = next((b for b in raw if isinstance(b, dict) and b.get("type") == "text"), None)
                if text_block is None:
                    text_block = next((b for b in raw if isinstance(b, dict) and "text" in b), None)
                text = (text_block["text"] if text_block else "").strip()
            else:
                text = raw
            plan = _extract_json(text)
            if plan is None:
                last_error = f"无法解析 JSON: {text[:200]}"
                logger.warning(f"[Condition Planner] {last_error}")
                continue

            is_valid, error_msg = _validate_condition_plan(plan, num_steps, valid_uids)
            if not is_valid:
                last_error = error_msg
                logger.warning(f"[Condition Planner] 校验失败: {error_msg}")
                continue

            # 日志输出
            logger.info(f"[Condition Planner] ✓ 条件规划完成:")
            for entry in plan:
                cond_type = entry.get("condition_type", "?")
                params = entry.get("params", {})
                reasoning = entry.get("reasoning", "")[:50]
                logger.info(f"  Step {entry['step_id']}: {cond_type} - {reasoning}...")
            logger.info("=" * 60 + "\n")

            output = {
                "condition_plan": plan,
                "messages": state.get("messages", []) + [response],
                "current_stage": "code_generator",
            }
            log_node_output_file("condition_planner", state, output)
            return output

        except Exception as e:
            last_error = str(e)
            logger.error(f"[Condition Planner] LLM 调用失败: {e}")

    logger.error(f"[Condition Planner] ✗ 条件规划失败: {last_error}")
    result = {
        "condition_plan": None,
        "error_feedback": f"条件规划失败: {last_error}",
        "current_stage": "code_generator",
    }
    log_node_output_file("condition_planner", state, result)
    return result