#!/usr/bin/env python3
"""
模型注册脚本 - 将处理好的模型注册到 VLABench 系统

完成两件事：
  1. 根据指定的 class 类型对 XML 进行必要注入
  2. 在 constant.py 的 name2class_xml 中注册

支持实体类:
  - CommonGraspedEntity: 可抓取物体（默认，无需额外注入）
  - CommonContainer: 3D 容器（注入 keypoints + placepoint）
  - FlatContainer: 平面容器（注入 4 角 keypoints + placepoint）
  - ChemistryBeaker: 化学烧杯（注入 solution geom + materials + sites）
  - ChemistryTube: 试管（注入 solution geom + materials + sites）
  - ContainerWithDoor: 带门容器（注入 keypoints + placepoint）

Usage:
    python scripts/register_model.py \\
        --model_dir VLABench/assets/review/flask/flask \\
        --class_name ChemistryBeaker \\
        --name flask
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent

# 允许的类名列表
VALID_CLASSES = {
    "CommonGraspedEntity",
    "CommonContainer",
    "FlatContainer",
    "ChemistryBeaker",
    "ChemistryTube",
    "ContainerWithDoor",
}

# Solution materials（与 SolutionMixin.solution2rgba 对应）
SOLUTION_MATERIALS = {
    "CaSO4":  [1, 1, 1, 0.7],
    "CuCl2":  [0.141, 1.0, 0.174, 0.4],
    "CuSO4":  [0, 0.45, 1, 0.4],
    "FeCl3":  [0.6475, 0.5686, 0.023, 0.4],
    "KMnO4":  [0.5, 0, 0.5, 0.4],
    "I2":     [0.3, 0.13, 0.0, 0.4],
    "K2CrO4": [0.57, 0.12, 0.013, 0.4],
    "NaCl":   [1, 1, 1, 0.3],
}


# ==================== OBJ 几何工具 ====================

def get_obj_bbox(obj_path: str) -> Dict:
    """读取 OBJ 文件的 bounding box。"""
    coords = []
    with open(obj_path) as f:
        for line in f:
            if line.startswith("v ") and not line.startswith("vt ") and not line.startswith("vn "):
                parts = line.split()
                if len(parts) >= 4:
                    coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not coords:
        return {"min": [0, 0, 0], "max": [0, 0, 0], "center": [0, 0, 0], "size": [0, 0, 0]}

    import numpy as np
    arr = np.array(coords)
    mn = arr.min(axis=0)
    mx = arr.max(axis=0)
    return {
        "min": mn.tolist(),
        "max": mx.tolist(),
        "center": ((mn + mx) / 2).tolist(),
        "size": (mx - mn).tolist(),
    }


def get_xml_scale(xml_path: str) -> float:
    """从 XML 读取第一个 mesh 的 scale。"""
    with open(xml_path) as f:
        content = f.read()
    m = re.search(r'<mesh[^>]*scale="([0-9.e+-]+)"', content)
    return float(m.group(1)) if m else 1.0


def get_xml_body_name(xml_path: str) -> Optional[str]:
    """从 XML 读取 body name。"""
    with open(xml_path) as f:
        content = f.read()
    m = re.search(r'<body\s+name="([^"]+)"', content)
    return m.group(1) if m else None


# ==================== XML 注入函数 ====================

def _find_body_close(content: str, body_name: Optional[str] = None) -> str:
    """找到 body 闭合标签前的位置，返回插入点前的 content。"""
    if body_name:
        pattern = rf'(</body(\s+name="{re.escape(body_name)}")?>)'
    else:
        pattern = r'</body>'
    match = re.search(pattern, content)
    if match:
        return match.start()
    # fallback: worldbody close
    match = re.search(r'</worldbody>', content)
    if match:
        return match.start()
    return -1


def ensure_grasppoints(xml_path: str) -> bool:
    """确认 XML 中有 grasppoint，没有则警告。"""
    with open(xml_path) as f:
        content = f.read()
    count = len(re.findall(r'<site class="grasppoint"', content))
    if count > 0:
        logger.info(f"  ✓ 已有 {count} 个 grasppoint")
        return True
    logger.warning(f"  ⚠ 未找到 grasppoint！模型可能无法被抓取")
    return False


def inject_top_bottom_sites(xml_path: str):
    """注入 top_site 和 bottom_site（如果不存在）。"""
    with open(xml_path) as f:
        content = f.read()

    if re.search(r'<site\s+name="top_site"', content):
        logger.info(f"  ✓ 已有 top_site/bottom_site，跳过")
        return

    body_name = get_xml_body_name(xml_path)
    scale = get_xml_scale(xml_path)

    # 找主 OBJ
    xml_dir = Path(xml_path).parent
    obj_match = re.search(r'<mesh file="([^"]+\.obj)"', content)
    if not obj_match:
        logger.warning("  ⚠ 未找到主 OBJ mesh，跳过 site 注入")
        return

    obj_file = xml_dir / obj_match.group(1)
    if not obj_file.exists():
        logger.warning(f"  ⚠ OBJ 文件不存在: {obj_file}")
        return

    bbox = get_obj_bbox(str(obj_file))
    z_min = bbox["min"][2] * scale
    z_max = bbox["max"][2] * scale

    injection = (
        f'      <site name="top_site" pos="0 0 {z_max:.6f}" size="0.01" rgba="1 0 0 0"/>\n'
        f'      <site name="bottom_site" pos="0 0 {z_min:.6f}" size="0.01" rgba="0 1 0 0"/>\n'
    )

    pos = _find_body_close(content, body_name)
    if pos < 0:
        logger.warning("  ⚠ 未找到 body 闭合标签，跳过 site 注入")
        return

    content = content[:pos] + injection + content[pos:]
    with open(xml_path, "w") as f:
        f.write(content)
    logger.info(f"  ✓ 注入 top_site (z={z_max:.4f}) 和 bottom_site (z={z_min:.4f})")


def inject_solution_template(xml_path: str):
    """注入 solution materials + solution geom（化学容器专用）。"""
    with open(xml_path) as f:
        content = f.read()

    body_name = get_xml_body_name(xml_path)
    modified = False

    # 1. 注入 solution materials（如果不存在）
    if not re.search(r'<material name="CaSO4"', content):
        materials_lines = []
        for name, rgba in SOLUTION_MATERIALS.items():
            materials_lines.append(
                f'    <material name="{name}" rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>\n'
            )
        materials_block = "".join(materials_lines)

        # 在 </asset> 之前插入
        asset_close = content.find("</asset>")
        if asset_close > 0:
            content = content[:asset_close] + materials_block + content[asset_close:]
            modified = True
            logger.info(f"  ✓ 注入 {len(SOLUTION_MATERIALS)} 个 solution materials")

    # 2. 注入 solution geom（如果不存在）
    if not re.search(r'<geom\s+name="solution"', content):
        # 计算 solution 圆柱体尺寸
        scale = get_xml_scale(xml_path)
        xml_dir = Path(xml_path).parent
        obj_match = re.search(r'<mesh file="([^"]+\.obj)"', content)
        if obj_match:
            obj_file = xml_dir / obj_match.group(1)
            if obj_file.exists():
                bbox = get_obj_bbox(str(obj_file))
                radius = min(bbox["size"][0], bbox["size"][1]) * scale / 2 * 0.8
                half_h = bbox["size"][2] * scale / 2 * 0.6
                z_center = bbox["center"][2] * scale

                solution_geom = (
                    f'      <geom name="solution" type="cylinder" '
                    f'size="{radius:.6f} {half_h:.6f}" pos="0 0 {z_center:.6f}" '
                    f'material="CaSO4" class="visual"/>\n'
                )

                pos = _find_body_close(content, body_name)
                if pos > 0:
                    content = content[:pos] + solution_geom + content[pos:]
                    modified = True
                    logger.info(f"  ✓ 注入 solution geom (radius={radius:.4f}, half_h={half_h:.4f})")

    if modified:
        with open(xml_path, "w") as f:
            f.write(content)


def inject_container_sites(xml_path: str):
    """注入 keypoints (group=3) + placepoint (group=2)。"""
    with open(xml_path) as f:
        content = f.read()

    body_name = get_xml_body_name(xml_path)
    scale = get_xml_scale(xml_path)

    # 找主 OBJ
    xml_dir = Path(xml_path).parent
    obj_match = re.search(r'<mesh file="([^"]+\.obj)"', content)
    if not obj_match:
        logger.warning("  ⚠ 未找到主 OBJ mesh")
        return

    obj_file = xml_dir / obj_match.group(1)
    if not obj_file.exists():
        logger.warning(f"  ⚠ OBJ 文件不存在: {obj_file}")
        return

    bbox = get_obj_bbox(str(obj_file))
    mn = [v * scale for v in bbox["min"]]
    mx = [v * scale for v in bbox["max"]]
    center = [v * scale for v in bbox["center"]]

    # 确保有 keypoint 和 placepoint default class
    defaults_needed = []
    if not re.search(r'<default class="keypoint"', content):
        defaults_needed.append('      <default class="keypoint"><site type="sphere" size="0.01" group="3" rgba="1 0 0 1"/></default>\n')
    if not re.search(r'<default class="placepoint"', content):
        defaults_needed.append('      <default class="placepoint"><site type="sphere" size="0.01" group="2" rgba="0 0 1 0"/></default>\n')

    if defaults_needed:
        # 在 </default> 之前插入
        default_close = content.find("</default>")
        if default_close > 0:
            for d in defaults_needed:
                content = content[:default_close] + d + content[default_close:]
                default_close += len(d)

    # 两个对角 keypoints + 一个 placepoint
    sites = (
        f'      <site class="keypoint" pos="{mn[0]:.4f} {mn[1]:.4f} {mn[2]:.4f}"/>\n'
        f'      <site class="keypoint" pos="{mx[0]:.4f} {mx[1]:.4f} {mx[2]:.4f}"/>\n'
        f'      <site class="placepoint" pos="{center[0]:.4f} {center[1]:.4f} {mx[2] + 0.01:.4f}"/>\n'
    )

    pos = _find_body_close(content, body_name)
    if pos < 0:
        logger.warning("  ⚠ 未找到 body 闭合标签")
        return

    content = content[:pos] + sites + content[pos:]
    with open(xml_path, "w") as f:
        f.write(content)
    logger.info(f"  ✓ 注入 keypoints + placepoint")


def inject_flat_container_sites(xml_path: str):
    """注入 4 角 keypoints + placepoint（平面容器）。"""
    with open(xml_path) as f:
        content = f.read()

    body_name = get_xml_body_name(xml_path)
    scale = get_xml_scale(xml_path)

    xml_dir = Path(xml_path).parent
    obj_match = re.search(r'<mesh file="([^"]+\.obj)"', content)
    if not obj_match:
        return

    obj_file = xml_dir / obj_match.group(1)
    if not obj_file.exists():
        return

    bbox = get_obj_bbox(str(obj_file))
    mn = [v * scale for v in bbox["min"]]
    mx = [v * scale for v in bbox["max"]]
    center = [v * scale for v in bbox["center"]]

    # 确保有 default class
    defaults_needed = []
    if not re.search(r'<default class="keypoint"', content):
        defaults_needed.append('      <default class="keypoint"><site type="sphere" size="0.01" group="3" rgba="1 0 0 1"/></default>\n')
    if not re.search(r'<default class="placepoint"', content):
        defaults_needed.append('      <default class="placepoint"><site type="sphere" size="0.01" group="2" rgba="0 0 1 0"/></default>\n')

    if defaults_needed:
        default_close = content.find("</default>")
        if default_close > 0:
            for d in defaults_needed:
                content = content[:default_close] + d + content[default_close:]
                default_close += len(d)

    # 4 角 keypoints + 中心 placepoint
    sites = (
        f'      <site class="keypoint" pos="{mn[0]:.4f} {mn[1]:.4f} {mn[2]:.4f}"/>\n'
        f'      <site class="keypoint" pos="{mx[0]:.4f} {mx[1]:.4f} {mn[2]:.4f}"/>\n'
        f'      <site class="keypoint" pos="{mn[0]:.4f} {mx[1]:.4f} {mn[2]:.4f}"/>\n'
        f'      <site class="keypoint" pos="{mx[0]:.4f} {mn[1]:.4f} {mn[2]:.4f}"/>\n'
        f'      <site class="placepoint" pos="{center[0]:.4f} {center[1]:.4f} {mx[2] + 0.01:.4f}"/>\n'
    )

    pos = _find_body_close(content, body_name)
    if pos < 0:
        return

    content = content[:pos] + sites + content[pos:]
    with open(xml_path, "w") as f:
        f.write(content)
    logger.info(f"  ✓ 注入 4 角 keypoints + placepoint")


# ==================== constant.py 注册 ====================

def register_to_constant(name: str, class_name: str, xml_rel_path: str):
    """
    在 constant.py 的 name2class_xml 中注册模型。

    - 如果 name 已存在且 class_name + xml_rel_path 完全一致 → 跳过
    - 如果 name 已存在但 class_name 或路径不同 → 提示用户用新名字（不覆盖）
    - 如果 name 不存在 → 新增条目
    """
    constant_path = _PROJECT_ROOT / "VLABench" / "configs" / "constant.py"
    content = constant_path.read_text(encoding="utf-8")

    # 检查 name2class_xml 中是否已有此 key
    existing_pattern = rf'"{re.escape(name)}"\s*:\s*\[components\.(\w+),\s*"([^"]+)"\]'
    existing = re.search(existing_pattern, content)
    if existing:
        existing_class = existing.group(1)
        existing_path = existing.group(2)
        if existing_class == class_name and existing_path == xml_rel_path:
            logger.info(f"  ✓ {name} 已注册为 {class_name}，跳过")
            return
        # 已存在但内容不同 → 警告，不覆盖
        logger.warning(f"  ⚠ '{name}' 已存在 (class={existing_class}, path={existing_path})")
        logger.warning(f"  ⚠ 不覆盖！请用 --name 指定不同的注册名")
        return

    # 新增条目：在 name2class_xml 字典的 } 之前插入
    new_line = f'    "{name}": [components.{class_name}, "{xml_rel_path}"],\n'

    # 在字典结尾 } 之前插入（} 后面紧跟 additional_dict）
    dict_end_pattern = r'(\n)(}\s*\n\s*additional_dict)'
    if re.search(dict_end_pattern, content):
        content = re.sub(dict_end_pattern, new_line + r'\1\2', content)
    else:
        # fallback: 在最后一个 name2class_xml 条目之后插入
        last_entry = re.findall(r'    "[^"]+"\s*:\s*\[components\.\w+,\s*"[^"]*"\],?\n', content)
        if last_entry:
            target = last_entry[-1]
            content = content.replace(target, target + new_line)
        else:
            logger.warning("  ⚠ 未找到插入点，请手动添加")
            return

    constant_path.write_text(content, encoding="utf-8")
    logger.info(f"  ✓ 注册到 constant.py: {name} → {class_name} ({xml_rel_path})")


# ==================== 主流程 ====================

def find_xml(model_dir: Path) -> Optional[Path]:
    """在模型目录下查找 XML 文件。"""
    xmls = list(model_dir.glob("*.xml"))
    if len(xmls) == 1:
        return xmls[0]
    if len(xmls) > 1:
        # 优先选和目录同名的
        for x in xmls:
            if x.stem == model_dir.name:
                return x
        return xmls[0]
    return None


def compute_xml_rel_path(xml_path: Path) -> str:
    """计算 XML 相对于 assets/ 的路径。"""
    assets_dir = _PROJECT_ROOT / "VLABench" / "assets"
    try:
        return os.path.relpath(str(xml_path), str(assets_dir))
    except ValueError:
        return str(xml_path)


def main():
    parser = argparse.ArgumentParser(description="注册处理好的模型到 VLABench 系统")
    parser.add_argument("--model_dir", required=True, help="已处理模型的目录路径")
    parser.add_argument("--class_name", required=True, choices=sorted(VALID_CLASSES),
                        help="实体类名")
    parser.add_argument("--name", default=None, help="注册名（默认用目录名）")
    parser.add_argument("--xml_file", default=None, help="XML 文件名（默认自动查找）")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    if not model_dir.is_dir():
        logger.error(f"目录不存在: {model_dir}")
        sys.exit(1)

    name = args.name or model_dir.name

    # 查找 XML
    if args.xml_file:
        xml_path = model_dir / args.xml_file
    else:
        xml_path = find_xml(model_dir)

    if xml_path is None or not xml_path.exists():
        logger.error(f"未找到 XML 文件: {model_dir}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info(f"模型注册: {name}")
    logger.info(f"  目录: {model_dir}")
    logger.info(f"  XML:  {xml_path.name}")
    logger.info(f"  类:   {args.class_name}")
    logger.info("=" * 60)

    # Step 1: XML 注入
    logger.info(f"\n[Step 1] XML 注入 ({args.class_name})")

    ensure_grasppoints(str(xml_path))

    if args.class_name in ("CommonContainer", "ContainerWithDoor"):
        inject_container_sites(str(xml_path))

    elif args.class_name == "FlatContainer":
        inject_flat_container_sites(str(xml_path))

    elif args.class_name in ("ChemistryBeaker", "ChemistryTube"):
        inject_solution_template(str(xml_path))
        inject_top_bottom_sites(str(xml_path))

    # Step 2: 注册到 constant.py
    logger.info(f"\n[Step 2] 注册到 constant.py")
    xml_rel_path = compute_xml_rel_path(xml_path)
    register_to_constant(name, args.class_name, xml_rel_path)

    logger.info(f"\n✓ 完成！模型 '{name}' 已注册为 {args.class_name}")
    logger.info(f"  XML 路径: {xml_rel_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
