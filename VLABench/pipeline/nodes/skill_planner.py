"""
Skill Planner Node - 多步长程编排引擎

按 Skill Planner 设计规范文档实现：
  - 废除 operation_type → 改为从 action 字段推断
  - 废除 DSL 占位符 ($TARGET_ENTITY) → 直接输出 uid
  - 废除 conditions 字典 → 执行完即成功
  - 新增 pre/post_state_assertion (CoT 追踪)
  - SKILL_LIB_DOC 底层原子技能白皮书

设计原则:
  - 状态驱动的全局单次规划 (Stateful One-Shot)
  - 消灭硬编码规则，完全信任 LLM 决策
  - 显式传参，无宏替换
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

# ========== 底层原子技能白皮书 (SKILL_LIB_DOC) ==========
SKILL_LIB_DOC = """
## 可用原子技能 (Atomic Skills)

- pick(target_uid, prior_eulers=[[-pi, 0, 0]]): 从上方抓取物体。prior_eulers 决定抓取朝向。
- place(target_container_uid): **将当前抓取的物体精确放置到目标容器/表面上**。target_container_uid 必须是场景中具体的实体（如 hot_plate_0, beaker_0, shelf_0）。place 会使用目标实体的 place_point 作为放置位置。**当任务要求将物体放到某个特定目标上时，必须使用 place，不要用 drop**。
- drop(): **将抓取的物体放到桌面上**。仅用于"使用完物品后腾出抓夹"的场景，即物体不需要放到任何特定位置，只需放到桌面即可。**drop 不接受 target 参数**。如果任务要求将物体放到某个特定实体上（如加热板、架子），必须使用 place(target_container_uid)，不要用 drop。
- pour(): 倾倒动作（假设手里已抓着容器）。仅旋转腕部关节，末端位置会偏移。
- pour_to_entity(target_uid, tilt_angle=1.8, wait_time=10): 倾倒到指定容器上方。使用 IK 保持末端位置不变，通过逐步倾斜实现稳定倾倒。**pour 操作优先使用此技能**。tilt_angle 默认 1.8 rad (~103°) 足以让液体流出。
- insert_to_entity(target_uid, insert_depth=0.05): 将抓取的物体插入目标实体的孔位（如试管插入试管架）。自动松开夹爪，无需再添加 open_gripper。
- lift(lift_height=0.15, gripper_state=np.zeros(2)): 举起当前抓取的物体。
- moveto(target_pos, gripper_state=np.zeros(2)): 移动末端执行器到目标位置。
- moveto_entity(target_uid, offset=[0,0,0.2], gripper_state=np.zeros(2)): 移动到指定实体上方，自动从实体运行时位置计算目标。
- open_gripper(): 松开夹爪。用于在当前位置直接放下物体（不推荐用于精确放置）。
- close_gripper(): 闭合夹爪。
- wait(wait_time=50): 等待指定步数。
- shake(n_shakes=3, shake_angle=0.7, steps_per_swing=5): **独立的摇晃技能**。通过四元数球面插值（slerp）生成正负角度的平滑摇摆轨迹。n_shakes=3 表示完整往返 3 次，shake_angle=0.7 表示每次摆动 ±0.7 rad。
- stir_entity_with_tool(target_uid, stir_radius=0.02, stir_duration=5, insert_ratio=2/3): **使用搅拌工具搅动容器内液体**。假设当前夹爪已抓取搅拌工具。步骤：①获取容器的 place_point；②移动到 place_point 正上方 25cm；③下降到插入位置（深度 = 容器高度 × insert_ratio）；④以 place_point XY 为圆心做圆周运动。stir_radius=0.02 表示半径 2cm，stir_duration=5 表示持续 5 秒。
- rotate(rotation_angle=pi/2): 旋转腕部关节实现物体翻转或小幅摇晃。适合单次大幅旋转。
- unscrew_cap(target_uid, rotation_angle=4*pi, target_q_velocity=pi/40, max_n_substep=30, tolerance=0.01, lift_height=0.02): **拧开带盖容器（如 pill_bottle）的瓶盖**。内部已封装 pick + 旋转腕关节 + 自动松夹，模拟人手拧开瓶盖动作。target_uid 是带盖容器的 uid（如 pill_bottle_0）。**任务要求拧开/打开瓶盖时，必须使用此技能，不要拆解为 pick+rotate+open_gripper**。rotation_angle 默认 4π。
- press(target_pos): 按压目标位置。
- push(target_pos, push_distance=0.1): 推动物体。
- reset(): 重置环境。
- wait_for(wait_duration=2.0, entity_name=None, change_type=None, solution=None, color=None): **等待外部状态变化的技能**。用于人机协同场景，机械臂保持不动，等待指定时间后自动应用环境变化。change_type 可选 "add_solution"（添加溶液，需要 solution 参数如 "CuSO4"）、"solution_change_color"（改变溶液颜色，需要 color 参数如 [1, 0, 0, 0.4] 表示红色）、或 "change_color"（通用颜色变化）。

## Skill Usage Guidelines

- **place**: 将物体放置到指定的目标实体上（加热板、容器、架子等）。必须传 target_container_uid 参数。适用场景："put A on B", "place A onto B", "put A in B"。
- **drop**: 仅用于将物体放到桌面上（无特定目标位置）。适用场景：使用完物品后腾出抓夹，需要抓取下一个物品时。**如果任务指定了放置目标，必须用 place，不能用 drop**。
- **open_gripper**: 在当前位置直接松开夹爪，物体会掉落。仅用于不需要精确放置的场景。
- **shake 任务的正确序列**:
  - 试管类物体 (ChemistryTube): pick -> lift -> shake(n_shakes=3) -> insert_to_entity(target_uid=chemistry_tube_stand)。insert_to_entity 已包含松开夹爪，不需要额外 open_gripper。
  - 其他物体 (烧杯等): pick -> lift -> shake(n_shakes=3) -> drop。用 drop 把物体放到桌面。
"""

# ========== 原子技能白名单 ==========
VALID_SKILLS = {
    "pick", "place", "drop", "lift", "moveto", "moveto_entity", "pour", "pour_to_entity", "push", "press",
    "flip", "wait", "rotate", "open_gripper", "close_gripper",
    "open_door", "close_door", "open_drawer", "open_laptop",
    "move_offset", "reset", "insert_to_entity", "shake", "stir_entity_with_tool",
    "wait_for",
    "unscrew_cap",
}

# ========== 动作 → 技能模式映射 (LLM 参考，非硬编码) ==========
ACTION_TO_SKILL_HINT = {
    "pour": ["pick", "lift", "moveto_entity", "pour"],
    "remove": ["moveto", "pick", "open_gripper"],
    "place": ["moveto", "place"],
    "lift": ["moveto", "pick", "lift"],
    "shake": ["moveto", "pick", "wait"],
    "dispense": ["moveto", "pick", "pour"],
    "heat": ["moveto", "pick", "lift", "moveto", "wait"],
    "move": ["moveto", "pick", "lift", "moveto", "open_gripper"],
    "open": ["moveto", "pick", "open_gripper"],
    "wait_for": ["wait_for"],  # wait_for 是独立的技能
}


def _extract_json(text: str) -> Optional[Dict]:
    """从 LLM 响应中提取 JSON"""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    patterns = [
        r'```json\s*\n(.*?)\n\s*```',
        r'```\s*\n(.*?)\n\s*```',
        r'\{[\s\S]*\}',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue
    return None


def _validate_skill_plan(plan: Dict, valid_uids: set, max_steps: int) -> tuple:
    """
    校验 skill_plan 的完整性

    Returns:
        (is_valid, error_msg)
    """
    if "global_skill_plan" not in plan:
        return False, "缺少 global_skill_plan 字段"

    if not isinstance(plan["global_skill_plan"], list):
        return False, "global_skill_plan 必须是 list"

    if len(plan["global_skill_plan"]) == 0:
        return False, "global_skill_plan 不能为空"

    # 检查 step_id 覆盖
    step_ids = {s["step_id"] for s in plan["global_skill_plan"]}
    expected = set(range(max_steps))
    if step_ids != expected:
        return False, f"step_id 不连续或不全: 期望 {expected}, 实际 {step_ids}"

    for i, step in enumerate(plan["global_skill_plan"]):
        if not isinstance(step, dict):
            return False, f"global_skill_plan[{i}] 必须是 dict"

        required = ["step_id", "semantic_instruction", "pre_state_assertion",
                    "atomic_sequence", "post_state_assertion"]
        for field in required:
            if field not in step:
                return False, f"global_skill_plan[{i}] 缺少 {field}"

        if not isinstance(step["atomic_sequence"], list):
            return False, f"global_skill_plan[{i}] atomic_sequence 必须是 list"

        for j, skill_entry in enumerate(step["atomic_sequence"]):
            if not isinstance(skill_entry, dict):
                return False, f"global_skill_plan[{i}].atomic_sequence[{j}] 必须是 dict"
            if "skill" not in skill_entry:
                return False, f"global_skill_plan[{i}].atomic_sequence[{j}] 缺少 skill"
            if skill_entry["skill"] not in VALID_SKILLS:
                return False, f"未知技能: {skill_entry['skill']}，可选: {VALID_SKILLS}"

            # 校验 params 中的 uid 是否存在
            params = skill_entry.get("params", {})
            for k, v in params.items():
                if isinstance(v, str) and v in valid_uids:
                    continue  # uid 命中
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, str) and item in valid_uids:
                            continue

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
        primary = step.get("primary_uid", "?")
        secondary = step.get("secondary_uid", "null")
        instruction = step.get("grounded_instruction", "")
        lines.append(f"  Step {step.get('step_id', '?')}: [{action}] {primary} -> {secondary}")
        lines.append(f"    instruction: {instruction}")
    return "\n".join(lines)


def build_skill_planner_prompt(
    normalized_context: Dict,
    asset_status: Dict,
    error_feedback: str = None
) -> str:
    """Build the Skill Planner Mega-Prompt per design doc."""
    steps = normalized_context.get("steps", [])
    asset_info = _build_asset_info(asset_status)
    steps_info = _build_steps_info(steps)

    prompt = f"""You are a robot task planning expert. Given an execution script and physical assets, autonomously decide the atomic action sequence.

## Core Constraints

1. **State Tracking (CoT)**: Before planning each step's atomic_sequence, you MUST first write pre_state_assertion to reason about the robot state after the previous step.
   - If the gripper is ALREADY HOLDING the target object, do NOT generate another pick.
   - If the gripper is NOT empty before picking a new object, you MUST first generate open_gripper.

2. **Explicit UIDs**: All uid parameters in atomic_sequence params MUST use the real UIDs from the asset inventory (e.g. beaker_0, tube_0).
   Do NOT use any placeholder like $TARGET_ENTITY.

3. **Direct Output**: In atomic_sequence params, write uid strings or numeric values directly without extra quotes.

4. **Condition Selection**: Success conditions are handled by the Condition Planner node (parallel to Skill Planner). Focus on generating the atomic skill sequence.

5. **Use moveto_entity for targeting objects**: When you need to move to a specific object (e.g. to pour into a beaker), use moveto_entity(target_uid=<container_uid>) instead of moveto with hardcoded coordinates. NEVER use moveto with hardcoded target_pos for pour operations.

6. **pour_to_entity replaces moveto+pour**: pour_to_entity already handles moving to the container and tilting. After pour_to_entity, just use open_gripper to release the container. Do NOT use place after pour_to_entity.

## Execution Script
{steps_info}

## Physical Asset Inventory
{asset_info}

{SKILL_LIB_DOC}

## CoT State Tracking Rules

For each step, you MUST fill in:
- pre_state_assertion: Describe robot gripper state and object positions BEFORE this step.
- atomic_sequence: Autonomously decide skills based on action type. Reference hints:
  - pour -> pick -> lift -> pour_to_entity -> insert_to_entity (insert_to_entity already includes open_gripper, do NOT add another open_gripper after it)
  - remove -> pick -> open_gripper
  - lift -> pick -> lift
  - place -> place
- post_state_assertion: Describe robot gripper state and object position changes AFTER this step.

## Special Rule for Test Tubes (ChemistryTube)
- After any pour/pour_to_entity operation, you MUST use insert_to_entity to insert the held test tube back into the tube stand.
- After any shake operation on a test tube, you MUST use insert_to_entity to insert the held test tube back into the tube stand.
- insert_to_entity already includes open_gripper internally, so do NOT add another open_gripper after it.
- **CRITICAL**: insert_to_entity target_uid MUST be "chemistry_tube_stand" (the tube stand entity), NOT "tube_0" or "tube_1".

## Special Rule for wait_for
- When a step has action="wait_for", the robot should stay still and wait.
- Use wait_for skill with appropriate parameters:
  - wait_for(wait_duration=2.0) - simple waiting
  - wait_for(wait_duration=3.0, entity_name="beaker_0", change_type="add_solution", solution="CuSO4") - wait and add solution
- The robot gripper should NOT release the held object during wait_for.

## Coordinate Reference
- Franka workspace: X: -0.3~0.3, Y: -0.2~0.3, Z: 0.75~1.5
- Table height ~0.78m
- moveto target position uses np.array() format

## Output (strict JSON, no other text):
{{
  "global_skill_plan": [
    {{
      "step_id": 0,
      "semantic_instruction": "<grounded_instruction original text>",
      "pre_state_assertion": "Describe robot/environment state before execution",
      "atomic_sequence": [
        {{"skill": "skill_name", "params": {{"param_name": param_value}}}},
        ...
      ],
      "post_state_assertion": "Describe robot/environment state after execution"
    }}
  ]
}}"""

    if error_feedback:
        prompt += f"""

## Previous Planning Error Feedback
{error_feedback}
Please adjust your plan based on this feedback."""

    return prompt



def skill_planner_node(state: Dict) -> Dict:
    """
    Skill Planner 节点 - 多步长程编排引擎

    输入: state["normalized_context"]["steps"] + state["asset_status"]
    输出: state["skill_plan"]["global_skill_plan"]
    """
    if ChatAnthropic is None:
        logger.error("[Skill Planner] langchain_anthropic 未安装")
        return {
            "errors": state.get("errors", []) + ["langchain_anthropic 未安装"],
            "current_stage": "code_generator",
        }

    normalized_context = state.get("normalized_context", {})
    asset_status = state.get("asset_status", {})
    error_feedback = state.get("error_feedback")

    steps = normalized_context.get("steps", [])
    if not steps:
        logger.warning("[Skill Planner] steps 为空，跳过")
        return {
            "skill_plan": {"global_skill_plan": []},
            "current_stage": "code_generator",
        }

    logger.info("=" * 60)
    logger.info(f"[Skill Planner] 开始规划 {len(steps)} 个步骤...")

    llm = ChatAnthropic(**AgentConfig.get_llm_config())
    prompt = build_skill_planner_prompt(normalized_context, asset_status, error_feedback)

    valid_uids = set(asset_status.keys())
    max_steps = len(steps)

    max_retries = 2
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                retry_prompt = prompt + f"\n\n## 上次校验失败: {last_error}\n请严格遵循 JSON 格式和 CoT 约束。"
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
                logger.warning(f"[Skill Planner] {last_error}")
                continue

            is_valid, error_msg = _validate_skill_plan(plan, valid_uids, max_steps)
            if not is_valid:
                last_error = error_msg
                logger.warning(f"[Skill Planner] 校验失败: {error_msg}")
                continue

            # 日志输出
            logger.info(f"[Skill Planner] ✓ 规划完成:")
            for step in plan["global_skill_plan"]:
                skills = [s["skill"] for s in step["atomic_sequence"]]
                logger.info(f"  Step {step['step_id']}: {' → '.join(skills)}")
            logger.info("=" * 60 + "\n")

            output = {
                "skill_plan": plan,
                "messages": state.get("messages", []) + [response],
                "current_stage": "code_generator",
            }
            log_node_output_file("skill_planner", state, output)
            return output

        except Exception as e:
            last_error = str(e)
            logger.error(f"[Skill Planner] LLM 调用失败: {e}")

    logger.error(f"[Skill Planner] ✗ 规划失败: {last_error}")
    return {
        "skill_plan": None,
        "error_feedback": f"技能规划失败: {last_error}",
        "current_stage": "code_generator",
    }
