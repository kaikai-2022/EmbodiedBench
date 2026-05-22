"""
Skill Formatter - 技能序列格式化

将 Skill Planner 输出的 global_skill_plan 转为
partial(SkillLib.xxx, ...) 调用序列。

UID 贯穿设计：
  - Skill 参数中的 uid 保持原样，不替换为 spec
  - uid 就是 entity name，框架通过 uid 精确定位实体
  - 同一 spec 的多实例通过 uid 区分（如 beaker_0, beaker_1）
"""

import logging
from typing import Dict, List
import numpy as np

logger = logging.getLogger(__name__)

# VLABench SkillLib 合法技能白名单
VALID_SKILLS = {
    "pick", "place", "drop", "lift", "moveto", "moveto_entity", "pour", "pour_to_entity", "push", "press",
    "flip", "wait", "rotate", "open_gripper", "close_gripper",
    "open_door", "close_door", "open_drawer", "open_laptop",
    "move_offset", "reset", "shake", "insert_to_entity", "stir_entity_with_tool",
    "wait_for",
}


def format_skill_sequence(skill_plan: Dict) -> str:
    """
    将 global_skill_plan 转为 partial(SkillLib.xxx, ...) 代码行。

    uid 直接作为 target_entity_name 使用，不做 uid→spec 替换。
    """
    global_plan = skill_plan.get("global_skill_plan", [])
    if not global_plan:
        # 空 skill_plan 时返回空字符串，模板的 `return skill_sequence` 即 `return []`
        return ""

    lines = []
    for step in global_plan:
        for entry in step.get("atomic_sequence", []):
            skill = entry.get("skill", "")
            if skill not in VALID_SKILLS:
                logger.warning(f"[Skill Formatter] 未知技能: {skill}，跳过")
                continue
            params = entry.get("params", {})
            param_str = _format_params(params, skill)
            if param_str:
                lines.append(f"            partial(SkillLib.{skill}, {param_str}),")
            else:
                lines.append(f"            partial(SkillLib.{skill}),")

    return "\n".join(lines) if lines else "            pass"


def _format_params(params: Dict, skill_name: str = "") -> str:
    """
    将 params 格式化为 Python 参数字符串。

    策略:
      - target_uid → 根据技能映射为正确的参数名:
          place → target_container_name
          pour_to_entity → target_container_name
          insert_to_entity → target_entity_name (目标是对应的父容器如 chemistry_tube_stand)
          其他 → target_entity_name
      - np.array(...) 字符串 → 直接写，不 repr
      - 数值/布尔 → repr
      - list → 递归格式化
      - 字符串 → repr（保留 uid 原样）
    """
    parts = []
    for k, v in params.items():
        # 参数名映射
        if k == "target_uid" or k == "target_container_uid":
            if skill_name in ("place", "pour_to_entity", "stir_entity_with_tool"):
                k = "target_container_name"
            else:
                k = "target_entity_name"

        if isinstance(v, str) and _is_np_expr(v):
            parts.append(f"{k}={v}")
        elif isinstance(v, bool):
            parts.append(f"{k}={v}")
        elif isinstance(v, (int, float)):
            parts.append(f"{k}={repr(v)}")
        elif isinstance(v, list):
            # offset 等数值列表转为 np.array
            if k == "offset" and all(isinstance(x, (int, float)) for x in v):
                parts.append(f"{k}=np.array({_format_list(v)})")
            else:
                parts.append(f"{k}={_format_list(v)}")
        elif isinstance(v, str):
            parts.append(f'{k}="{v}"')
        else:
            parts.append(f"{k}={repr(v)}")

    return ", ".join(parts)


def _is_np_expr(s: str) -> bool:
    return s.startswith("np.") or s.startswith("numpy.")


def _format_list(v: list) -> str:
    """
    将 list 格式化为 Python 列表字符串。
    处理嵌套列表（如 prior_eulers）中的数值字符串（如 "pi", "-pi", "pi/2"）。
    """
    items = []
    for item in v:
        if isinstance(item, list):
            # 递归处理嵌套列表（如 prior_eulers）
            items.append(_format_list(item))
        elif isinstance(item, (int, float)):
            items.append(repr(item))
        elif isinstance(item, str):
            # 处理数值字符串（如 "pi", "-pi", "pi/2"）
            # 尝试用 numpy 解析，如 "pi", "-pi", "pi/2", "-pi/4"
            math_expr = item.strip()
            try:
                val = eval(math_expr, {"np": np, "pi": np.pi})
                if isinstance(val, (int, float)):
                    items.append(repr(float(val)))
                else:
                    items.append(repr(math_expr))
            except:
                # 无法解析的字符串用引号包裹
                items.append(f'"{item}"')
        else:
            items.append(repr(item))
    return "[" + ", ".join(items) + "]"
