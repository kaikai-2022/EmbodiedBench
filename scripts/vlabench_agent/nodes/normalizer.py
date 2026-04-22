"""
Normalizer Node - 语义到物理的网关

按 Normalizer 设计规范文档实现：
  - 工序 1: 资产映射双轨制 (cache + LLM 分类)
  - 工序 2: InitParams 翻译 (semantic_attributes → init_params)
  - 工序 3: UID 生成 (纯序号，不含角色语义)
  - 工序 4: 指令重接地 (<raw_id> → <uid>)

设计原则:
  - 只发指令，不干苦力: 决定资产去哪找，但不执行下载/文件 I/O
  - 只给语义标签，不猜物理数值
  - 不篡改动词，只替换名词
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
from ..tools.asset_cache import load_cache, save_cache, get_cached, set_cached
from .node_logger import log_node_output_file

logger = logging.getLogger(__name__)

# 标准资产库列表 (从 constant.py 中提取的 key)
STANDARD_ASSET_LIBRARY = [
    "beaker", "chemistry_beaker", "tube", "chemistry_tube_stand",
    "flask", "petri_dish", "bunsen_burner", "centrifuge",
    "coverslip", "nametag", "microscope", "scale",
    "plate", "tray", "cabinet", "fridge", "microwave",
    "mug", "cup", "bottle", "bowl", "plate", "knife",
    "fork", "spoon", "chopstick",
    "apple", "banana", "orange", "avocado", "pear",
    "bread", "croissant", "bagel", "cake", "donut",
    "hammer", "cord", "laptop",
]


# Key → 属性匹配规则表（扁平化，易扩展）
_KEY_RULES = [
    # (匹配的 key 集合, [(value_tokens, param_name, param_value), ...])
    (("state", "status", "状态"), [
        (("open", "opened", "打开"),            "is_open", True),
        (("closed", "close", "sealed", "关闭", "密封"), "is_open", False),
        (("on", "powered on", "turned on", "开启", "通电"), "is_powered_on", True),
        (("off", "powered off", "turned off", "关闭电源", "断电"), "is_powered_on", False),
        (("with liquid", "filled", "有液体", "装有", "contains"), "contains_substance", None),
        (("empty", "空"),                       "state", "empty"),
    ]),
    (("open", "openness", "开关"), [
        (("open", "true", "打开"),              "is_open", True),
        (("close", "closed", "false", "关闭"),  "is_open", False),
    ]),
    (("powered", "power", "电源"), [
        (("on", "true", "开", "开启"),          "is_powered_on", True),
        (("off", "false", "关", "关闭"),        "is_powered_on", False),
    ]),
    (("contains", "containing", "holding", "包含"), [
        ((),                                    "contains_substance", None),
    ]),
    (("amount", "quantity", "数量", "体积", "volume", "weight"), [
        ((),                                    "quantity_hint", None),
    ]),
]


def _translate_semantic_attributes(semantic_attrs: Dict, raw_type: str, is_physical: bool) -> Dict:
    """
    将 semantic_attributes 翻译为 InitParams。
    只提取文本中明确提及的状态，未提到则留空。
    """
    init_params = {}

    for key, value in semantic_attrs.items():
        key_lower = str(key).lower()
        value_str = str(value)
        value_lower = value_str.lower()

        for key_set, rules in _KEY_RULES:
            if key_lower not in key_set:
                continue
            for tokens, param_name, param_value in rules:
                if tokens and not any(t in value_lower for t in tokens):
                    continue
                # param_value=None 表示使用原始 value 字符串
                init_params[param_name] = value_str if param_value is None else param_value
            break  # 命中一个 key_set 后不再匹配其他

    if not is_physical:
        init_params.setdefault("contains_substance", raw_type)

    return init_params


def _classify_asset_type(raw_type: str, cache: Dict, llm: Optional) -> tuple:
    """
    对单个 raw_type 进行资产分类。

    Returns:
        (spec, source_type, is_physical)
        - spec: 标准大类名称
        - source_type: "local" | "objaverse" | "non_physical"
        - is_physical: True | False
    """
    # 轨道 1: 缓存查询
    cached = get_cached(raw_type, cache)
    if cached:
        logger.info(f"[Normalizer] 缓存命中: {raw_type} → spec={cached['spec']}, source={cached['source_type']}")
        return cached["spec"], cached["source_type"], cached["is_physical"]

    # 轨道 2: 受限 LLM 分类
    if llm is None:
        # 无 LLM，回退到简单匹配
        logger.warning(f"[Normalizer] 无 LLM，回退到简单 spec 匹配: {raw_type}")
        return _simple_spec_match(raw_type), "local", True

    prompt = f"""You are an asset classification expert. Given an object phrase, decide which route it belongs to.

## Standard Asset Library
Choose the closest match from the following library.
- If it matches a local asset, output: LOCAL: <standard_name>
- If it definitely does not exist locally, output: OBJAVERSE: <search_keyword>
- If it is a non-physical concept entity such as powder, liquid, light, or chemical solution that does not need an independent 3D model, output: NON_PHYSICAL: <substance_name>

Standard library: {STANDARD_ASSET_LIBRARY}

## Object Phrase
"{raw_type}"

## Output Requirement
Output exactly one line starting with LOCAL:, OBJAVERSE:, or NON_PHYSICAL:. Do not output any other text.

Examples:
- Input: "beaker" → LOCAL: beaker
- Input: "centrifuge" → LOCAL: centrifuge
- Input: "CuSO4 solution" → NON_PHYSICAL: CuSO4 solution
- Input: "some strange instrument" → OBJAVERSE: laboratory instrument
"""

    try:
        response = llm.invoke(prompt)
        # 扫描所有行，找第一个分类前缀（LLM 可能返回 thinking 内容）
        lines = response.content.strip().split("\n")
        result = None
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("LOCAL:", "OBJAVERSE:", "NON_PHYSICAL:")):
                result = stripped
                break

        if result is None:
            logger.warning(f"[Normalizer] 未找到分类前缀，使用简单匹配: {raw_type}")
            spec = _simple_spec_match(raw_type)
            source_type = "local"
            is_physical = True
        elif result.startswith("LOCAL:"):
            spec = result.replace("LOCAL:", "").strip()
            source_type = "local"
            is_physical = True
        elif result.startswith("OBJAVERSE:"):
            spec = result.replace("OBJAVERSE:", "").strip().replace(" ", "_")
            source_type = "objaverse"
            is_physical = True
        elif result.startswith("NON_PHYSICAL:"):
            spec = result.replace("NON_PHYSICAL:", "").strip()
            source_type = "local"
            is_physical = False

        # 缓存结果
        set_cached(raw_type, spec, source_type, is_physical, cache)
        save_cache(cache)
        logger.info(f"[Normalizer] LLM 分类: {raw_type} → spec={spec}, source={source_type}, physical={is_physical}")
        return spec, source_type, is_physical

    except Exception as e:
        logger.warning(f"[Normalizer] LLM 分类失败: {e}，回退到简单匹配")
        spec = _simple_spec_match(raw_type)
        return spec, "local", True


def _simple_spec_match(raw_type: str) -> str:
    """无 LLM 时的简单 spec 匹配"""
    raw_lower = raw_type.lower()
    for standard in STANDARD_ASSET_LIBRARY:
        if standard in raw_lower or raw_lower in standard:
            return standard
    # 回退: 直接用 raw_type 作为 spec（转下划线）
    return raw_lower.replace(" ", "_")


def _generate_uid(spec: str, global_counter: Dict) -> str:
    """
    生成全局唯一的 UID。
    格式: spec_序号 (纯序号，不含角色语义)
    """
    counter = global_counter.get(spec, 0)
    uid = f"{spec}_{counter}"
    global_counter[spec] = counter + 1
    return uid


def _reround_instruction(text: str, raw_id_to_uid: Dict) -> str:
    """
    指令重接地: 将 <raw_id> 替换为 <uid>
    """
    for raw_id, uid in raw_id_to_uid.items():
        text = re.sub(rf'<{re.escape(raw_id)}>', f'<{uid}>', text)
    return text


def normalizer_node(state: Dict) -> Dict:
    """
    Normalizer 节点 - 语义到物理的网关

    输入: state["task_analysis"] (raw_entities + raw_steps)
    输出: state["normalized_context"] (instances + steps)
    """
    task_analysis = state.get("task_analysis", {})
    raw_entities = task_analysis.get("raw_entities", [])
    raw_steps = task_analysis.get("raw_steps", [])

    if not raw_entities:
        logger.warning("[Normalizer] raw_entities 为空，跳过")
        return {
            "normalized_context": {"instances": [], "steps": []},
            "current_stage": "asset_manager",
        }

    logger.info("=" * 60)
    logger.info("[Normalizer] 开始实体归一化...")

    # 加载缓存
    cache = load_cache()
    save_cache(cache)  # 确保文件存在

    # 初始化 LLM (可选)
    llm = None
    if ChatAnthropic is not None:
        try:
            llm = ChatAnthropic(**AgentConfig.get_llm_config())
        except Exception as e:
            logger.warning(f"[Normalizer] LLM 初始化失败: {e}，使用简单匹配模式")

    # 全局计数器
    global_counter: Dict[str, int] = {}

    # 工序 1+2: 对每个 raw_entity 进行分类和 InitParams 翻译
    instances = []
    raw_id_to_uid = {}

    for raw_entity in raw_entities:
        raw_id = raw_entity["raw_id"]
        raw_type = raw_entity["raw_type"]
        semantic_attrs = raw_entity.get("semantic_attributes", {})

        # 分类
        spec, source_type, is_physical = _classify_asset_type(raw_type, cache, llm)

        # InitParams 翻译
        init_params = _translate_semantic_attributes(semantic_attrs, raw_type, is_physical)

        # UID 生成
        uid = _generate_uid(spec, global_counter)
        raw_id_to_uid[raw_id] = uid

        instances.append({
            "uid": uid,
            "spec": spec,
            "source_type": source_type,
            "is_physical": is_physical,
            "init_params": init_params,
        })

        logger.info(f"  [{uid}] raw_type={raw_type} → spec={spec}, source={source_type}, "
                    f"physical={is_physical}, init_params={init_params}")

    # 保存缓存
    save_cache(cache)

    # 工序 3+4: 处理 raw_steps (UID 映射 + 指令重接地)
    steps = []
    for raw_step in raw_steps:
        primary_uid = raw_id_to_uid.get(raw_step.get("primary_obj", ""), raw_step.get("primary_obj", ""))
        secondary_uid = raw_id_to_uid.get(raw_step.get("secondary_obj") or "", None)

        rerounded_instruction = _reround_instruction(
            raw_step.get("grounded_instruction", ""),
            raw_id_to_uid
        )

        steps.append({
            "step_id": raw_step["step_id"],
            "action": raw_step["action"],
            "primary_uid": primary_uid,
            "secondary_uid": secondary_uid,
            "grounded_instruction": rerounded_instruction,
        })

    normalized_context = {
        "instances": instances,
        "steps": steps,
    }

    # 日志输出
    logger.info(f"[Normalizer] ✓ 生成 {len(instances)} 个实例:")
    for inst in instances:
        logger.info(f"    {inst['uid']}: spec={inst['spec']}, source={inst['source_type']}, "
                    f"physical={inst['is_physical']}")
    logger.info(f"[Normalizer] ✓ 生成 {len(steps)} 个步骤:")
    for step in steps:
        logger.info(f"    [{step['step_id']}] {step['action']}: {step['primary_uid']} "
                    f"-> {step['secondary_uid']}")
    logger.info("=" * 60 + "\n")

    output = {
        "normalized_context": normalized_context,
        "current_stage": "asset_manager",
    }
    log_node_output_file("normalizer", state, output)
    return output
