"""
化学物质反应表 + 反应解析 + 颜色混合工具。

SOLUTE2RGBA:  物质名 → RGBA 颜色（含初始溶质和反应产物）
REACTIONS:    {物质A, 物质B} → 产物名
alpha_over:   Porter-Duff A-over-B 合成
mix_rgba:     对称 alpha 加权混合（fallback 当无反应时用）
"""
import re
from functools import lru_cache

# ─────────────────────────────────────────────────────────────────────────────
# 物质 → 颜色表（初始溶质 + 常见反应产物）
# ─────────────────────────────────────────────────────────────────────────────
SOLUTE2RGBA = {
    # 初始溶质（9 种有特征色，其余无色透明）
    "CuCl2":   [0.141, 1.0,   0.174, 0.4],
    "CuSO4":   [0.0,   0.45,  1.0,   0.4],
    "FeCl3":   [0.6475, 0.5686, 0.023, 0.4],
    "KMnO4":   [0.5,   0.0,   0.5,   0.4],
    "I2":      [0.3,   0.13,  0.0,   0.4],
    "K2CrO4":  [0.57,  0.12,  0.013, 0.4],
    # 无色溶质（alpha 不同以区分浓度/状态）
    "NaCl":    [1.0,   1.0,   1.0,   0.3],
    "AgNO3":   [1.0,   1.0,   1.0,   0.3],
    "BaCl2":   [1.0,   1.0,   1.0,   0.3],
    "H2SO4":   [1.0,   1.0,   1.0,   0.3],
    "NaOH":    [1.0,   1.0,   1.0,   0.3],
    "Ba(NO3)2":[1.0,   1.0,   1.0,   0.3],
    "Pb(NO3)2":[1.0,   1.0,   1.0,   0.3],
    "Na2CO3":  [1.0,   1.0,   1.0,   0.3],
    "CaCl2":   [1.0,   1.0,   1.0,   0.3],
    "HCl":     [1.0,   1.0,   1.0,   0.3],
    "CaSO4":   [1.0,   1.0,   1.0,   0.7],
    # 常见反应产物
    "Cu(OH)2": [0.0,   0.32,  0.78,  0.5],   # 蓝色絮状沉淀
    "Fe(OH)3": [0.55,  0.27,  0.07,  0.5],   # 红褐色沉淀
    "AgCl":    [1.0,   1.0,   1.0,   0.7],   # 白色沉淀
    "BaSO4":   [1.0,   1.0,   1.0,   0.7],   # 白色沉淀
    "Na2SO4":  [1.0,   1.0,   1.0,   0.3],   # 可溶无色
}

# ─────────────────────────────────────────────────────────────────────────────
# 反应表：键为 frozenset({reactant_a, reactant_b})，值为产物名
# 顺序无关（A+B 和 B+A 命中同一键）
# ─────────────────────────────────────────────────────────────────────────────
REACTIONS = {
    frozenset({"CuSO4",  "NaOH"}):   "Cu(OH)2",
    frozenset({"FeCl3",  "NaOH"}):   "Fe(OH)3",
    frozenset({"AgNO3",  "NaCl"}):   "AgCl",
    frozenset({"BaCl2",  "Na2SO4"}): "BaSO4",
    frozenset({"BaCl2",  "H2SO4"}):  "BaSO4",
    # 后续可在此扩展更多反应 …
}


# ─────────────────────────────────────────────────────────────────────────────
# 反应查表
# ─────────────────────────────────────────────────────────────────────────────

def lookup_reaction(solutes_a, solutes_b):
    """
    给定两个溶质列表，返回 (merged_solutes, product_name)。

    规则：
      1. 遍历 solutes_a × solutes_b，找第一个在 REACTIONS 中的配对
      2. 反应物消耗掉，产物加入幸存列表
      3. 未参与反应的溶质全部保留
      4. 若 a 和 b 有多组可反应物质，依次处理（目前先匹配第一对）

    Returns:
      merged_solutes (list): 合并后的溶质列表（含产物，若有的话）
      product_name   (str|None): 单一产物名；多产物或无产物时为 None
    """
    survivors_a = list(solutes_a)
    survivors_b = list(solutes_b)
    produced = []

    # 依次在交叉乘积中找第一对可反应物质
    for ra in list(survivors_a):
        for rb in list(survivors_b):
            key = frozenset({ra, rb})
            if key in REACTIONS:
                product = REACTIONS[key]
                survivors_a.remove(ra)
                survivors_b.remove(rb)
                produced.append(product)
                # 只匹配第一对，避免链式消耗过于复杂
                # （如需支持多步反应，后续在调用侧循环调用即可）
                break
        else:
            continue
        break

    merged = survivors_a + survivors_b + produced

    if produced:
        # 只取第一个产物名；多产物时 product=None，颜色走混合 fallback
        product_name = produced[0] if len(produced) == 1 else None
    else:
        product_name = None

    return merged, product_name


# ─────────────────────────────────────────────────────────────────────────────
# 颜色合成
# ─────────────────────────────────────────────────────────────────────────────

def alpha_over(rgba_top, rgba_bottom):
    """
    Porter-Duff A-over-B 合成：rgba_top 覆盖在 rgba_bottom 上方。

    Args:
        rgba_top:     [r, g, b, a]  0-1
        rgba_bottom:  [r, g, b, a]  0-1

    Returns:
        RGBA list [r, g, b, a]（各通道 0-1，a 已截断到 [0, 1]）
    """
    r0, g0, b0, a0 = rgba_bottom
    r1, g1, b1, a1 = rgba_top
    a0 = max(0.0, min(1.0, a0))
    a1 = max(0.0, min(1.0, a1))

    a_out = a1 + a0 * (1.0 - a1)
    if a_out < 1e-6:
        return [0.0, 0.0, 0.0, 0.0]

    r = (r1 * a1 + r0 * a0 * (1.0 - a1)) / a_out
    g = (g1 * a1 + g0 * a0 * (1.0 - a1)) / a_out
    b = (b1 * a1 + b0 * a0 * (1.0 - a1)) / a_out
    return [r, g, b, min(a_out, 1.0)]


def mix_rgba(rgba_a, rgba_b):
    """
    对称混合：A 和 B 颜色按各自 alpha 加权平均。
    用于 fallback：当混合不触发任何已知反应时，用此公式产生"混合色"。

    Args:
        rgba_a, rgba_b: [r, g, b, a]，a ∈ [0, 1]

    Returns:
        RGBA list
    """
    a0 = max(0.0, min(1.0, rgba_a[3]))
    a1 = max(0.0, min(1.0, rgba_b[3]))
    total = a0 + a1
    if total < 1e-6:
        return [1.0, 1.0, 1.0, 0.0]

    r = (rgba_a[0] * a0 + rgba_b[0] * a1) / total
    g = (rgba_a[1] * a0 + rgba_b[1] * a1) / total
    b = (rgba_a[2] * a0 + rgba_b[2] * a1) / total
    a = (a0 + a1) / 2.0
    return [r, g, b, a]


def resolve_color_from_solutes(solutes, fallback_rgba=None):
    """
    根据溶质列表返回渲染颜色。

    颜色权威来源（优先级从高到低）：
      1. SOLUTE2RGBA 表：物质名命中表内颜色（即使 LLM 给了错误颜色也以表为准）
      2. fallback_rgba：LLM 明确指定的颜色（仅在物质名不在表中时使用）
      3. 默认无色：[1,1,1,0.3]

    物质名支持后缀清理（如 "NaOH_solution_1" → "NaOH"）。
    """
    if not solutes:
        return [1.0, 1.0, 1.0, 0.0]

    # 对每个溶质：清理后缀，命中 SOLUTE2RGBA 则用表内颜色
    rgba_list = []
    for name in solutes:
        clean = resolve_substance_name(name)
        if clean in SOLUTE2RGBA:
            rgba_list.append(list(SOLUTE2RGBA[clean]))
        elif fallback_rgba is not None and len(solutes) == 1:
            # 仅单未知溶质才用 LLM fallback；多物质混合用默认色
            rgba_list.append(list(fallback_rgba))
        else:
            rgba_list.append([1.0, 1.0, 1.0, 0.3])

    if len(rgba_list) == 1:
        return rgba_list[0]
    # 多物质：alpha-over 混合
    result = rgba_list[0]
    for rgba in rgba_list[1:]:
        result = alpha_over(rgba, result)
    return result


def resolve_substance_name(name: str) -> str:
    """
    从带后缀的物质名中提取纯净物质名。

    例如：
      "NaOH_solution_1" → "NaOH"
      "FeCl3_1"         → "FeCl3"
      "CuSO4"           → "CuSO4"
      "NaOH"            → "NaOH"
    """
    if name in SOLUTE2RGBA:
        return name
    # 去掉常见的 _数字、_solution_N、_liquid_N 等后缀
    cleaned = re.sub(r'(_(solution|liquid|powder|acid|base)_\d+)$', '', name, flags=re.IGNORECASE)
    cleaned = re.sub(r'_\d+$', '', cleaned)  # 任意 _N 后缀：FeCl3_1 → FeCl3，NaOH_solution_1 → NaOH_solution
    return cleaned
