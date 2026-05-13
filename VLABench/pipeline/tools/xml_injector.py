"""
XML Injector - MuJoCo XML 模型外科手术工具

按 Asset Manager 设计规范文档实现：
  - 场景 A: 为容器类模型注入 top_site / bottom_site
  - 场景 B: 为化学容器注入 solution 占位符 geom

通过 lxml 在内存中修改 XML 结构，保存为 _injected.xml。
"""

import logging
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 已内置 site 的模型（不需要注入）
BUILTIN_SITES = {
    "tube": ["top_site", "bottom_site"],      # tube.xml 有 top_site/bottom_site
    "chemistry_tube_stand": [],                # TubeStand 没有 site
}

# 已内置 solution geom 的模型（不需要注入 solution 占位符）
BUILTIN_SOLUTION = {
    "chemistry_beaker": True,                  # chemistry_beaker.xml 有 solution geom
    "chemistry_beaker_0": True,
    "tube": True,                               # tube.xml 有 solution geom
}


def _compute_bounding_box(mesh_path: Path) -> Tuple[float, float]:
    """
    从 OBJ 文件计算 bounding box 的 z_min 和 z_max。

    Returns:
        (z_min, z_max) 单位为米
    """
    try:
        import trimesh
    except ImportError:
        logger.warning("[XML Injector] trimesh 未安装，无法计算 Bounding Box")
        return (0.0, 0.1)  # 回退默认值

    try:
        mesh = trimesh.load(str(mesh_path), force='mesh')
        bounds = mesh.bounds  # shape (2, 3) = [[xmin, ymin, zmin], [xmax, ymax, zmax]]
        z_min = float(bounds[0][2])
        z_max = float(bounds[1][2])
        logger.debug(f"[XML Injector] Bounding box: z_min={z_min:.4f}, z_max={z_max:.4f}")
        return (z_min, z_max)
    except Exception as e:
        logger.warning(f"[XML Injector] Bounding box 计算失败: {e}")
        return (0.0, 0.1)  # 回退默认值


def _find_body_and_mesh_paths(xml_path: str) -> Tuple[Optional[str], List[str]]:
    """
    解析 XML 文件，查找 body 节点名称和 mesh 文件路径。

    Returns:
        (body_name, mesh_paths)
    """
    # 简单解析: 从字符串中查找
    try:
        with open(xml_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 查找 body name
        body_match = re.search(r'<body\s+name="([^"]+)"', content)
        body_name = body_match.group(1) if body_match else None

        # 查找所有 mesh 文件引用
        mesh_pattern = r'<mesh\s+name="([^"]+)"\s+file="([^"]+)"'
        mesh_paths = []
        for match in re.finditer(mesh_pattern, content):
            mesh_name = match.group(1)
            mesh_file = match.group(2)
            xml_dir = os.path.dirname(xml_path)
            full_path = os.path.normpath(os.path.join(xml_dir, mesh_file))
            mesh_paths.append((mesh_name, full_path))

        return body_name, mesh_paths

    except Exception as e:
        logger.warning(f"[XML Injector] XML 解析失败: {e}")
        return None, []


def _inject_sites(xml_path: str, z_min: float, z_max: float, body_name: str) -> str:
    """
    向 XML 注入 top_site 和 bottom_site。

    在 </body> 之前插入 site 元素。
    """
    with open(xml_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 幂等检查：如果已有 top_site 或 bottom_site，跳过注入
    if re.search(r'<site\s+name="top_site"', content) or re.search(r'<site\s+name="bottom_site"', content):
        logger.info(f"[XML Injector] XML 中已有 top_site/bottom_site，跳过注入")
        return xml_path

    # 构建 site 注入字符串
    site_injection = (
        f'    <site name="top_site" pos="0 0 {z_max}" size="0.01" rgba="1 0 0 0"/>\n'
        f'    <site name="bottom_site" pos="0 0 {z_min}" size="0.01" rgba="0 1 0 0"/>\n'
    )

    # 查找 body 闭合标签
    if body_name:
        # 在 </body> 之前插入
        body_close_pattern = rf'(</body(\s+name="{re.escape(body_name)}")?>)'
    else:
        # 通用 </body>
        body_close_pattern = r'</body>'

    if re.search(body_close_pattern, content):
        new_content = re.sub(body_close_pattern, site_injection + r'\1', content, count=1)
        logger.info(f"[XML Injector] ✓ 已注入 top_site (z={z_max:.4f}) 和 bottom_site (z={z_min:.4f})")
    else:
        # 如果没有找到 </body>，在 </worldbody> 之前插入
        worldbody_close = r'</worldbody>'
        if re.search(worldbody_close, content):
            new_content = re.sub(worldbody_close, '  <body name="' + (body_name or "entity") + '">\n' + site_injection + '  </body>\n' + worldbody_close, content, count=1)
            logger.info(f"[XML Injector] ✓ 已注入 site (worldbody 模式)")
        else:
            logger.warning("[XML Injector] ✗ 无法找到 </body> 或 </worldbody>，跳过 site 注入")
            return xml_path

    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    return xml_path


def _inject_solution_placeholder(xml_path: str, body_name: Optional[str]) -> str:
    """
    向 XML 注入 solution 占位符 geom。

    注入一个简单的透明圆柱体作为 solution 几何体，
    供 SolutionMixin 在运行时修改 rgba 颜色。
    """
    with open(xml_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 检查是否已有 solution geom
    if re.search(r'<geom\s+name="solution"', content):
        logger.info("[XML Injector] XML 中已有 solution geom，跳过注入")
        return xml_path

    # 注入 solution geom（在 </body> 或 </worldbody> 之前）
    solution_injection = (
        '    <geom name="solution" type="cylinder" size="0.02 0.05" pos="0 0 0.05" '
        'rgba="1 1 1 0" class="visual"/>\n'
    )

    if body_name:
        body_close_pattern = rf'(</body(\s+name="{re.escape(body_name)}")?>)'
    else:
        body_close_pattern = r'</body>'

    if re.search(body_close_pattern, content):
        new_content = re.sub(body_close_pattern, solution_injection + r'\1', content, count=1)
        logger.info("[XML Injector] ✓ 已注入 solution 占位符 geom")
    else:
        worldbody_close = r'</worldbody>'
        if re.search(worldbody_close, content):
            new_content = re.sub(worldbody_close, '  <body name="' + (body_name or "entity") + '">\n' + solution_injection + '  </body>\n' + worldbody_close, content, count=1)
            logger.info("[XML Injector] ✓ 已注入 solution 占位符 geom (worldbody 模式)")
        else:
            logger.warning("[XML Injector] ✗ 无法找到 </body>，跳过 solution 注入")
            return xml_path

    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    return xml_path


def inject_xml(xml_path: str, class_name: str, spec: str) -> str:
    """
    对 XML 模型执行外科手术注入。

    Args:
        xml_path: 原始 XML 文件路径
        class_name: VLABench 类名 (ChemistryBeaker, CommonContainer 等)
        spec: 资产 spec 名称

    Returns:
        修改后的 XML 路径 (如果修改了则为原路径，否则不变)
    """
    xml_path = str(Path(xml_path).resolve())
    xml_dir = os.path.dirname(xml_path)
    xml_name = Path(xml_path).stem

    logger.info(f"[XML Injector] 检查注入需求: {xml_path} (class={class_name}, spec={spec})")

    # 判断是否需要注入 site
    needs_site = True
    for builtin_spec in BUILTIN_SITES:
        if spec == builtin_spec or builtin_spec in spec:
            existing = BUILTIN_SITES[builtin_spec]
            if "top_site" in existing or "bottom_site" in existing:
                needs_site = False
                logger.info(f"[XML Injector] spec={spec} 已有内置 site，跳过 site 注入")
                break

    # 判断是否需要注入 solution
    needs_solution = False
    is_chemistry = class_name in ("ChemistryBeaker", "ChemistryTube")
    if is_chemistry:
        builtin_has_solution = any(
            spec == s or s in spec
            for s in BUILTIN_SOLUTION if BUILTIN_SOLUTION.get(s)
        )
        if not builtin_has_solution:
            needs_solution = True
            logger.info(f"[XML Injector] class={class_name} 需要 solution 占位符")

    if not needs_site and not needs_solution:
        logger.info(f"[XML Injector] 无需注入，返回原始路径: {xml_path}")
        return xml_path

    # 查找 body 名称和 mesh 路径
    body_name, mesh_paths = _find_body_and_mesh_paths(xml_path)
    if body_name:
        logger.debug(f"[XML Injector] body_name={body_name}, mesh_paths={len(mesh_paths)}")

    # 计算 bounding box
    z_min, z_max = 0.0, 0.1  # 默认值
    if mesh_paths and needs_site:
        # 使用第一个 mesh 计算 bounding box
        _, first_mesh = mesh_paths[0]
        if os.path.exists(first_mesh):
            z_min, z_max = _compute_bounding_box(Path(first_mesh))
        else:
            logger.warning(f"[XML Injector] mesh 文件不存在: {first_mesh}")

    # 执行注入
    if needs_site:
        _inject_sites(xml_path, z_min, z_max, body_name)

    if needs_solution:
        _inject_solution_placeholder(xml_path, body_name)

    return xml_path
