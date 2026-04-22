"""
Analyzer Node - 神经符号解析器

按 Analyzer 设计规范文档实现：
  - 任务 1: raw_entities 提取 (raw_id, raw_type, semantic_attributes)
  - 任务 2: grounded_instruction (尖括号 <raw_id> 替换)
  - 任务 3: raw_steps 提取 (action, primary_obj, secondary_obj)

设计原则:
  - 只提取显式意图，不脑补隐式动作
  - 只提取原始动词，不做意图分类
  - 只提取语义状态，不推断物理角色
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
from .node_logger import log_node_output_file

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> Optional[Dict]:
    """从 LLM 响应中提取 JSON，支持 markdown 代码块包裹"""
    text = text.strip()

    # 尝试直接解析
    try:
        return json.loads(text)
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
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue
    return None


def _validate_analysis(analysis: Dict) -> tuple:
    """
    校验输出结构的完整性

    Returns:
        (is_valid, error_msg)
    """
    if not isinstance(analysis, dict):
        return False, "输出必须是 dict"

    # 检查 raw_entities
    if "raw_entities" not in analysis:
        return False, "缺少 raw_entities 字段"
    if not isinstance(analysis["raw_entities"], list):
        return False, "raw_entities 必须是 list"
    for i, entity in enumerate(analysis["raw_entities"]):
        if not isinstance(entity, dict):
            return False, f"raw_entities[{i}] 必须是 dict"
        if "raw_id" not in entity:
            return False, f"raw_entities[{i}] 缺少 raw_id"
        if "raw_type" not in entity:
            return False, f"raw_entities[{i}] 缺少 raw_type"
        if not isinstance(entity.get("semantic_attributes", {}), dict):
            return False, f"raw_entities[{i}] semantic_attributes 必须是 dict"

    # 检查 raw_steps
    if "raw_steps" not in analysis:
        return False, "缺少 raw_steps 字段"
    if not isinstance(analysis["raw_steps"], list):
        return False, "raw_steps 必须是 list"
    for i, step in enumerate(analysis["raw_steps"]):
        if not isinstance(step, dict):
            return False, f"raw_steps[{i}] 必须是 dict"
        if "step_id" not in step:
            return False, f"raw_steps[{i}] 缺少 step_id"
        if "action" not in step:
            return False, f"raw_steps[{i}] 缺少 action"
        if "primary_obj" not in step:
            return False, f"raw_steps[{i}] 缺少 primary_obj"
        if "grounded_instruction" not in step:
            return False, f"raw_steps[{i}] 缺少 grounded_instruction"

    # 校验所有 primary_obj 和 secondary_obj 都是 raw_entities 中的 raw_id
    raw_ids = {e["raw_id"] for e in analysis["raw_entities"]}
    for i, step in enumerate(analysis["raw_steps"]):
        if step["primary_obj"] not in raw_ids:
            return False, f"raw_steps[{i}] primary_obj '{step['primary_obj']}' not in raw_entities"
        # 防御性处理：过滤掉 None 和字符串 "null"/"None"
        sec_obj = step.get("secondary_obj")
        if sec_obj is not None and str(sec_obj).lower() not in ("null", "none"):
            if sec_obj not in raw_ids:
                return False, f"raw_steps[{i}] secondary_obj '{sec_obj}' not in raw_entities"

    # 校验 grounded_instruction 中使用了尖括号包裹的 raw_id
    for i, step in enumerate(analysis["raw_steps"]):
        instruction = step["grounded_instruction"]
        mentioned_ids = set(re.findall(r'<([^>]+)>', instruction))
        if not mentioned_ids:
            return False, f"raw_steps[{i}] grounded_instruction 中没有尖括号包裹的 <raw_id>"
        for raw_id in mentioned_ids:
            if raw_id not in raw_ids:
                return False, f"raw_steps[{i}] grounded_instruction 中的 <{raw_id}> 不在 raw_entities 中"

    return True, ""


def _infer_task_name(analysis: Dict) -> str:
    """从 raw_steps 推断 task_name（框架注册标识符）"""
    task_name_parts = []
    for step in analysis.get("raw_steps", []):
        verb = step.get("action", "task").lower()
        verb = re.sub(r's$', '', verb)
        verb = re.sub(r'ed$', '', verb)
        task_name_parts.append(verb)
    return "_".join(task_name_parts) if task_name_parts else "custom_task"


def analyzer_node(state: Dict) -> Dict:
    """
    Analyzer 节点 - 神经符号解析器

    将自然语言指令解析为带实体引用的 AST：
      - raw_entities: 所有物理实体的原始标识列表
      - raw_steps: 接地指令序列
    """
    if ChatAnthropic is None:
        logger.error("[Analyzer] langchain_anthropic 未安装")
        return {
                    "errors": state.get("errors", []) + ["langchain_anthropic 未安装"]
        }

    user_instruction = state["user_instruction"]

    logger.info("=" * 60)
    logger.info("[Analyzer] 开始分析任务...")
    logger.info(f"[Analyzer] 用户指令: {user_instruction}")

    llm = ChatAnthropic(**AgentConfig.get_llm_config())

    prompt = f"""You are a Neuro-Symbolic Parser for chemistry lab instructions.

## Task Description

### Task 1: Entity Extraction
Identify all physical entities (containers, tools, materials) in the text, assign a locally unique raw_id (format: noun_number), and put any explicitly mentioned state modifiers into semantic_attributes.

### Task 2: Instruction Grounding
Break the experiment process into single steps. You MUST replace nouns in the original sentences with your generated <raw_id> (wrapped in angle brackets), preserving all adverbs and action details, output as grounded_instruction.

### Task 3: Raw Action Extraction
Extract the core raw verb for each step (prefer infinitive form), fill into the action field.

## Constraints
1. **Do NOT hallucinate**: semantic_attributes only contains states explicitly mentioned in the original text. If not mentioned, leave as {{}}.
2. **Do NOT assign roles**: Do NOT assign source/target or other physical role labels to objects. Role transitions are inferred by downstream nodes.
3. **Do NOT classify intent**: Do NOT classify verbs into pick/lift/pour operation types. Output raw verbs directly.
4. **Strict JSON**: Output must be valid JSON parseable by json.loads().
5. **Physical Entities Include Non-Rigids**: You MUST explicitly extract liquids, powders, gases, or chemicals (e.g., "liquids", "water", "powder") as independent entities in `raw_entities`. Do not ignore them.
6. **Capture Quantities**: If the text specifies an amount, volume, or weight (e.g., "5 mg", "10 ml"), you MUST capture it in the `semantic_attributes` of that entity (e.g., {{"amount": "5 mg"}}). 
7. **Ignore the Actor**: Do NOT extract "the robot" or "the robotic arm" as an entity unless it is being explicitly manipulated by another agent. Focus only on the objects, tools, and materials being handled.

## Input
"{user_instruction}"

## Output Schema (strict JSON, no other text)
{{
    "raw_entities": [
        {{
            "raw_id": "noun_number",
            "raw_type": "exact noun phrase from original text",
            "semantic_attributes": {{}}
        }}
    ],
    "raw_steps": [
        {{
            "step_id": 0,
            "action": "raw verb (prefer infinitive)",
            "primary_obj": "raw_id",
            "secondary_obj": "raw_id or null",
            "grounded_instruction": "complete sentence with <raw_id> replacing nouns"
        }}
    ]
}}

## Few-shot Examples

Input: "The robot removes the cap from the vial and places the vial under the solid dispenser. Then the robot shakes the vial for 10 seconds."

Output:
{{
    "raw_entities": [
        {{"raw_id": "vial_1", "raw_type": "vial", "semantic_attributes": {{}}}},
        {{"raw_id": "cap_1", "raw_type": "cap", "semantic_attributes": {{}}}},
        {{"raw_id": "dispenser_1", "raw_type": "solid dispenser", "semantic_attributes": {{}}}}
    ],
    "raw_steps": [
        {{
            "step_id": 0,
            "action": "remove",
            "primary_obj": "cap_1",
            "secondary_obj": "vial_1",
            "grounded_instruction": "The robot removes <cap_1> from <vial_1>."
        }},
        {{
            "step_id": 1,
            "action": "place",
            "primary_obj": "vial_1",
            "secondary_obj": "dispenser_1",
            "grounded_instruction": "Then the robot places <vial_1> under <dispenser_1>."
        }},
        {{
            "step_id": 2,
            "action": "shake",
            "primary_obj": "vial_1",
            "secondary_obj": null,
            "grounded_instruction": "Then the robot shakes <vial_1> for 10 seconds."
        }}
    ]
}}

Input: "Lift the beaker"

Output:
{{
    "raw_entities": [
        {{"raw_id": "beaker_1", "raw_type": "beaker", "semantic_attributes": {{}}}}
    ],
    "raw_steps": [
        {{
            "step_id": 0,
            "action": "lift",
            "primary_obj": "beaker_1",
            "secondary_obj": null,
            "grounded_instruction": "Lift the <beaker_1>."
        }}
    ]
}}

Input: "Grab the beaker with liquid on the table and pour the liquid into the empty beaker"

Output:
{{
    "raw_entities": [
        {{"raw_id": "beaker_1", "raw_type": "beaker", "semantic_attributes": {{"state": "with liquid"}}}},
        {{"raw_id": "beaker_2", "raw_type": "beaker", "semantic_attributes": {{"state": "empty"}}}}
    ],
    "raw_steps": [
        {{
            "step_id": 0,
            "action": "grab",
            "primary_obj": "beaker_1",
            "secondary_obj": null,
            "grounded_instruction": "Grab the <beaker_1> with liquid on the table."
        }},
        {{
            "step_id": 1,
            "action": "pour",
            "primary_obj": "beaker_1",
            "secondary_obj": "beaker_2",
            "grounded_instruction": "Pour the liquid into the <beaker_2>."
        }}
    ]
}}"""

    max_retries = 2
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                retry_prompt = prompt + f"\n\n## Previous parse failed: {last_error}\nPlease strictly follow the JSON format."
                response = llm.invoke(retry_prompt)
            else:
                response = llm.invoke(prompt)

            content = response.content.strip()
            analysis = _extract_json(content)

            if analysis is None:
                last_error = f"无法解析 JSON: {content[:200]}"
                logger.warning(f"[Analyzer] {last_error}")
                continue

            is_valid, error_msg = _validate_analysis(analysis)
            if not is_valid:
                last_error = error_msg
                logger.warning(f"[Analyzer] 校验失败: {error_msg}")
                continue

            # 推断 task_name（框架注册标识符）
            analysis["task_name"] = _infer_task_name(analysis)

            # 日志输出
            logger.info(f"[Analyzer] ✓ 识别到 {len(analysis['raw_entities'])} 个实体:")
            for e in analysis["raw_entities"]:
                logger.info(f"    - {e['raw_id']}: {e['raw_type']}, attrs={e['semantic_attributes']}")

            logger.info(f"[Analyzer] ✓ 解析到 {len(analysis['raw_steps'])} 个动作:")
            for s in analysis["raw_steps"]:
                logger.info(f"    [{s['step_id']}] {s['action']}: {s['primary_obj']} -> {s.get('secondary_obj')}")

            logger.info("=" * 60 + "\n")

            output = {
                "task_analysis": analysis,
                "messages": state.get("messages", []) + [response],
                "current_stage": "normalizer",
            }
            log_node_output_file("analyzer", state, output)
            return output

        except Exception as e:
            last_error = str(e)
            logger.error(f"[Analyzer] LLM 调用失败: {e}")

    logger.error(f"[Analyzer] ✗ 分析失败: {last_error}")
    return {
            "errors": state.get("errors", []) + [f"任务分析失败: {last_error}"],
    }
