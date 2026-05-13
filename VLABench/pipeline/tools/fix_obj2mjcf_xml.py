#!/usr/bin/env python3
"""
修复obj2mjcf生成的XML文件，添加必要的MuJoCo属性

功能：
1. OBJ 几何中心归零（平移顶点使 bounding box 中心在原点）
2. 尺寸归一化（在 XML mesh 标签中添加 scale 属性）
3. 朝向检查（启发式判断最长轴，输出警告）
4. 添加 <compiler> 标签
5. 添加 <default> 标签（visual/collision/grasppoint classes）
6. 添加 <inertial> 标签
7. 添加抓取点 (grasppoint sites)
8. 可选移除 freejoint
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


# ==================== OBJ 几何工具函数 ====================

def read_obj_vertices(obj_path: str) -> list:
    """读取 OBJ 文件中的所有顶点坐标"""
    vertices = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return vertices


def get_obj_dimensions(obj_path: str) -> Dict[str, float]:
    """
    获取 OBJ 文件的几何尺寸信息

    Returns:
        dict with keys: center_x, center_y, center_z,
                        x_range, y_range, z_range, max_dim
    """
    vertices = read_obj_vertices(obj_path)
    if not vertices:
        raise ValueError(f"OBJ 文件中没有顶点: {obj_path}")

    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)

    x_range = max_x - min_x
    y_range = max_y - min_y
    z_range = max_z - min_z

    return {
        'center_x': (min_x + max_x) / 2,
        'center_y': (min_y + max_y) / 2,
        'center_z': (min_z + max_z) / 2,
        'min_x': min_x, 'max_x': max_x,
        'min_y': min_y, 'max_y': max_y,
        'min_z': min_z, 'max_z': max_z,
        'x_range': x_range,
        'y_range': y_range,
        'z_range': z_range,
        'max_dim': max(x_range, y_range, z_range),
        'vertex_count': len(vertices),
    }


def center_obj_file(
    obj_path: str,
    threshold: float = 0.01,
    offset: Optional[Tuple[float, float, float]] = None,
) -> Tuple[bool, Tuple[float, float, float]]:
    """
    将 OBJ 文件的顶点按指定偏移量平移。

    如果未提供 offset，则自动计算该文件自身的几何中心作为偏移量（向后兼容）。
    对于 collision mesh，应传入主 OBJ 的偏移量以保持空间一致性。

    Args:
        obj_path: OBJ 文件路径
        threshold: 偏移阈值(m)，低于此值不修改（仅在 offset=None 时生效）
        offset: 外部指定的平移偏移量 (cx, cy, cz)，传入后直接使用

    Returns:
        (was_modified, (offset_x, offset_y, offset_z))
    """
    if offset is None:
        dims = get_obj_dimensions(obj_path)
        cx, cy, cz = dims['center_x'], dims['center_y'], dims['center_z']
        offset_norm = (cx**2 + cy**2 + cz**2) ** 0.5
        if offset_norm < threshold:
            return False, (0, 0, 0)
    else:
        cx, cy, cz = offset

    # 读取整个文件，平移顶点
    lines = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                if len(parts) >= 4:
                    x = float(parts[1]) - cx
                    y = float(parts[2]) - cy
                    z = float(parts[3]) - cz
                    lines.append(f"v {x:.8f} {y:.8f} {z:.8f}\n")
                else:
                    lines.append(line)
            else:
                lines.append(line)

    with open(obj_path, 'w') as f:
        f.writelines(lines)

    logger.info(f"  几何中心修正: ({cx:.4f}, {cy:.4f}, {cz:.4f}) -> (0, 0, 0) [{Path(obj_path).name}]")
    return True, (cx, cy, cz)


def check_orientation(dims: Dict[str, float]) -> Optional[str]:
    """
    启发式检查模型朝向

    Returns:
        警告消息（如果最长轴不是 Z），或 None
    """
    ranges = {
        'X': dims['x_range'],
        'Y': dims['y_range'],
        'Z': dims['z_range'],
    }
    longest_axis = max(ranges, key=ranges.get)
    longest_val = ranges[longest_axis]

    if longest_axis != 'Z' and longest_val > dims['z_range'] * 1.5:
        return (f"模型最长轴为 {longest_axis} ({longest_val:.3f}m)，"
                f"Z轴为 {dims['z_range']:.3f}m，可能需要旋转")
    return None


def compute_surface_grasppoints(
    obj_path: str,
    effective_height: float,
    scale_factor: Optional[float] = None,
    n_points: int = 3,
    band_ratio: float = 0.15,
    min_band: float = 0.02,
) -> list:
    """
    从 OBJ 网格顶部区域采样表面抓取点。

    在模型顶部附近的顶点中，用贪心最远点采样选取分散的表面点，
    确保抓取点在模型表面而非内部。

    Args:
        obj_path: OBJ 文件路径
        effective_height: 缩放后的模型有效高度(m)
        scale_factor: 缩放因子（None 表示 1.0）
        n_points: 采样点数
        band_ratio: 顶部采样带占高度的比例
        min_band: 采样带最小厚度(m)

    Returns:
        [(x, y, z), ...] 在缩放后坐标空间中的抓取点位置
    """
    sf = scale_factor if scale_factor is not None else 1.0
    vertices = read_obj_vertices(obj_path)
    if len(vertices) < 3:
        # 顶点不足，降级
        return [(0, 0, effective_height * 0.8 + i * 0.01) for i in range(n_points)]

    # 缩放顶点到实际坐标
    scaled = [(v[0] * sf, v[1] * sf, v[2] * sf) for v in vertices]

    band_thickness = max(min_band, effective_height * band_ratio)
    z_threshold = effective_height - band_thickness

    # 逐步加宽采样带直到获得足够顶点
    for attempt in range(10):
        top_verts = [v for v in scaled if v[2] >= z_threshold]
        if len(top_verts) >= n_points:
            break
        z_threshold -= band_thickness * 0.2  # 每次下降 20%
    else:
        if len(top_verts) < n_points:
            return [(0, 0, effective_height * 0.8 + i * 0.01) for i in range(n_points)]

    # 贪心最远点采样
    # 第一点：Z 最高的顶点
    selected = [max(top_verts, key=lambda v: v[2])]

    for _ in range(n_points - 1):
        best_vert = None
        best_min_dist = -1
        for v in top_verts:
            min_dist = min(
                ((v[0] - s[0])**2 + (v[1] - s[1])**2 + (v[2] - s[2])**2) ** 0.5
                for s in selected
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_vert = v
        if best_vert is not None:
            selected.append(best_vert)

    return [(round(x, 4), round(y, 4), round(z, 4)) for x, y, z in selected]


def compute_scale_factor(max_dim: float, target_max: float = 0.15,
                         min_threshold: float = 0.02, max_threshold: float = 0.3) -> Optional[float]:
    """
    计算尺寸归一化的缩放因子

    Returns:
        scale_factor（如果需要缩放），或 None（尺寸已在合理范围）
    """
    if min_threshold <= max_dim <= max_threshold:
        return None
    return target_max / max_dim


# ==================== XML 修复函数 ====================

def fix_obj2mjcf_xml(
    input_xml: str,
    output_xml: str = None,
    mass: float = 0.02,
    height: float = 0.1,
    add_grasppoints: bool = True,
    remove_freejoint: bool = False,
    scale_factor: Optional[float] = None,
    obj_path: Optional[str] = None,
):
    """
    修复obj2mjcf生成的XML文件

    Args:
        input_xml: 输入的XML文件路径
        output_xml: 输出的XML文件路径（如果为None，则覆盖原文件）
        mass: 物体质量(kg)
        height: 物体高度(m)，用于计算质心位置（应为缩放后的高度）
        add_grasppoints: 是否添加抓取点
        remove_freejoint: 是否移除 freejoint
        scale_factor: 缩放因子（添加到所有 mesh 标签），None 表示不缩放
    """
    input_path = Path(input_xml)
    if output_xml is None:
        output_path = input_path
    else:
        output_path = Path(output_xml)

    # 解析XML
    tree = ET.parse(input_path)
    root = tree.getroot()

    # 移除现有的compiler和default（如果有）
    for elem in root.findall('compiler'):
        root.remove(elem)
    for elem in root.findall('default'):
        root.remove(elem)

    # 1. 添加<compiler>标签
    compiler = ET.SubElement(root, 'compiler')
    compiler.set('boundmass', str(mass * 0.75))
    compiler.set('boundinertia', str(mass * 0.0005))
    compiler.set('angle', 'radian')

    # 2. 添加<default>标签
    default = ET.SubElement(root, 'default')

    # Visual class
    visual_default = ET.SubElement(default, 'default', {'class': 'visual'})
    ET.SubElement(visual_default, 'geom', {
        'group': '2',
        'type': 'mesh',
        'contype': '0',
        'conaffinity': '0',
        'density': '50'
    })

    # Collision class
    collision_default = ET.SubElement(default, 'default', {'class': 'collision'})
    ET.SubElement(collision_default, 'geom', {
        'group': '3',
        'type': 'mesh',
        'density': '50',
        'friction': '1.5 0.1 0.1',
        'solimp': '0.9 0.95 0.001',
        'solref': '0.02 1'
    })

    # Grasppoint class
    if add_grasppoints:
        grasppoint_default = ET.SubElement(default, 'default', {'class': 'grasppoint'})
        ET.SubElement(grasppoint_default, 'site', {
            'type': 'sphere',
            'size': '0.005',
            'group': '4',
            'rgba': '0 0 1 1'
        })

    # 3. 如果需要缩放，给所有 <mesh> 标签添加 scale 属性
    if scale_factor is not None:
        asset = root.find('asset')
        if asset is not None:
            scale_str = f"{scale_factor} {scale_factor} {scale_factor}"
            for mesh_elem in asset.findall('mesh'):
                mesh_elem.set('scale', scale_str)
            logger.info(f"  尺寸归一化: scale={scale_factor:.6f}")

    # 4. 找到<worldbody><body>并处理
    worldbody = root.find('worldbody')
    if worldbody is None:
        raise ValueError("No <worldbody> found in XML")

    body = worldbody.find('body')
    logger.info(f"  [DEBUG fix_xml] worldbody children BEFORE body-wrap: {[c.tag for c in worldbody]}")
    logger.info(f"  [DEBUG fix_xml] body found = {body is not None}")
    if body is None:
        # obj2mjcf 有时不生成 <body>，geom 直接在 <worldbody> 下
        # 自动创建 <body> 包裹所有 worldbody 的子元素
        logger.info("  [DEBUG fix_xml] Creating <body> wrapper...")
        logger.info("  <body> 不存在，自动创建包裹层")
        model_name = root.get('model', 'object')
        body = ET.SubElement(worldbody, 'body', {'name': model_name})
        children_to_move = list(worldbody)
        for child in children_to_move:
            if child is not body:
                worldbody.remove(child)
                body.append(child)
        logger.info(f"  [DEBUG fix_xml] worldbody children AFTER body-wrap: {[c.tag for c in worldbody]}")
        logger.info(f"  [DEBUG fix_xml] body children: {[c.tag for c in body]}")
    else:
        logger.info(f"  [DEBUG fix_xml] <body> already exists, name={body.get('name')}")

    # 处理 freejoint
    freejoint = body.find('freejoint')
    if remove_freejoint and freejoint is not None:
        body.remove(freejoint)
        logger.info("  已移除 freejoint")
        insert_index = 0
    elif freejoint is not None:
        insert_index = list(body).index(freejoint) + 1
    else:
        insert_index = 0

    # 移除已有的 inertial 和 grasppoint sites（幂等）
    for existing_inertial in body.findall('inertial'):
        body.remove(existing_inertial)
    for existing_site in body.findall('site'):
        if existing_site.get('class') == 'grasppoint':
            body.remove(existing_site)

    # 添加 <inertial>
    inertial = ET.Element('inertial', {
        'pos': f'0 0 {height/2}',
        'mass': str(mass),
        'diaginertia': f'{mass*0.04} {mass*0.04} {mass*0.025}'
    })
    body.insert(insert_index, inertial)

    # 5. 添加抓取点
    if add_grasppoints:
        if obj_path and os.path.exists(obj_path):
            grasp_positions = compute_surface_grasppoints(
                obj_path, effective_height=height, scale_factor=scale_factor
            )
            logger.info(f"  抓取点(表面采样): {grasp_positions}")
        else:
            grasp_positions = [(0, 0, height * 0.8 + i * 0.01) for i in range(3)]
            logger.info(f"  抓取点(中轴降级): {grasp_positions}")
        for gx, gy, gz in grasp_positions:
            ET.SubElement(body, 'site', {
                'class': 'grasppoint',
                'pos': f'{gx:.4f} {gy:.4f} {gz:.4f}'
            })

    # 6. 美化输出
    from xml.dom import minidom
    xml_str = ET.tostring(root, encoding='unicode')
    logger.info(f"  [DEBUG fix_xml] ET.tostring output (first 500 chars):")
    logger.info(f"  {xml_str[:500]}")
    # 检查 <body> 是否在序列化结果中
    if '<body' in xml_str:
        logger.info(f"  [DEBUG fix_xml] ✓ <body> tag found in serialized XML")
    else:
        logger.info(f"  [DEBUG fix_xml] ✗ <body> tag MISSING in serialized XML!")
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent='  ')
    lines = [line for line in pretty_xml.split('\n') if line.strip()]
    pretty_xml = '\n'.join(lines)
    # 再检查美化后的输出
    if '<body' in pretty_xml:
        logger.info(f"  [DEBUG fix_xml] ✓ <body> tag found in pretty XML")
    else:
        logger.info(f"  [DEBUG fix_xml] ✗ <body> tag MISSING in pretty XML!")

    output_path.write_text(pretty_xml, encoding='utf-8')
    logger.info(f"  已修复: {output_path}")
    logger.info(f"  质量: {mass}kg, 高度: {height:.4f}m, 抓取点: {3 if add_grasppoints else 0}个")

    return output_path


# ==================== 完整后处理 Pipeline ====================

def postprocess_model(
    model_dir: str,
    uid: str,
    mass: float = 0.02,
    target_max_dim: float = 0.15,
    remove_freejoint: bool = False,
    add_grasppoints: bool = True,
) -> Dict:
    """
    对 obj2mjcf 转换后的模型执行完整后处理

    包括：几何中心归零、尺寸归一化、朝向检查、XML 物理属性修复

    Args:
        model_dir: 模型目录路径
        uid: 模型 UID
        mass: 物体质量(kg)
        target_max_dim: 目标最大尺寸(m)，超出范围时归一化到此值
        remove_freejoint: 是否移除 freejoint
        add_grasppoints: 是否添加抓取点

    Returns:
        后处理报告 dict
    """
    model_path = Path(model_dir)
    report = {
        'uid': uid,
        'centered': False,
        'scaled': False,
        'scale_factor': None,
        'orientation_warning': None,
        'original_dims': None,
        'final_dims': None,
    }

    main_obj = model_path / f"{uid}.obj"
    target_xml = model_path / f"{uid}.xml"

    if not main_obj.exists():
        raise FileNotFoundError(f"OBJ 文件不存在: {main_obj}")
    if not target_xml.exists():
        raise FileNotFoundError(f"XML 文件不存在: {target_xml}")

    print(f"后处理模型: {uid}")

    # --- Step 1: 几何中心归零 ---
    # 收集所有 OBJ 文件（主 OBJ + collision OBJs）
    obj_files = [main_obj] + list(model_path.glob(f"{uid}/{uid}_collision_*.obj"))

    # 先读取主 OBJ 的原始尺寸
    original_dims = get_obj_dimensions(str(main_obj))
    report['original_dims'] = {
        'center': [original_dims['center_x'], original_dims['center_y'], original_dims['center_z']],
        'ranges': [original_dims['x_range'], original_dims['y_range'], original_dims['z_range']],
        'max_dim': original_dims['max_dim'],
    }

    # 居中所有 OBJ（使用主 OBJ 的中心偏移量）
    cx = original_dims['center_x']
    cy = original_dims['center_y']
    cz = original_dims['center_z']
    offset_norm = (cx**2 + cy**2 + cz**2) ** 0.5

    if offset_norm >= 0.01:
        # 先居中主 OBJ（自动计算偏移量）
        main_offset = (cx, cy, cz)
        center_obj_file(str(main_obj))
        # 对所有 collision OBJ 使用主 OBJ 的偏移量统一平移
        for obj_file in obj_files[1:]:  # 跳过 main_obj
            if obj_file.exists():
                center_obj_file(str(obj_file), offset=main_offset)
        # 同步根目录主 OBJ 到子目录（obj2mjcf 在子目录放了一份拷贝，XML 引用的是子目录版本）
        subdir_visual_obj = model_path / uid / f"{uid}.obj"
        if subdir_visual_obj.exists():
            import shutil
            shutil.copy2(str(main_obj), str(subdir_visual_obj))
        report['centered'] = True
        print(f"  几何中心: 偏移 {offset_norm:.4f}m -> 已归零 ({len(obj_files)} 个 OBJ 文件)")
    else:
        print(f"  几何中心: 已在原点 (偏移 {offset_norm:.4f}m)")

    # --- Step 2: 朝向检查 ---
    orientation_warn = check_orientation(original_dims)
    if orientation_warn:
        report['orientation_warning'] = orientation_warn
        print(f"  朝向警告: {orientation_warn}")

    # --- Step 3: 尺寸归一化 ---
    # 重新读取居中后的尺寸
    centered_dims = get_obj_dimensions(str(main_obj))
    scale_factor = compute_scale_factor(centered_dims['max_dim'], target_max=target_max_dim)

    if scale_factor is not None:
        report['scaled'] = True
        report['scale_factor'] = scale_factor
        final_max = centered_dims['max_dim'] * scale_factor
        print(f"  尺寸归一化: {centered_dims['max_dim']:.4f}m -> {final_max:.4f}m (scale={scale_factor:.6f})")
        effective_height = centered_dims['z_range'] * scale_factor
    else:
        print(f"  尺寸: {centered_dims['max_dim']:.4f}m (在合理范围内，无需缩放)")
        effective_height = centered_dims['z_range']

    report['final_dims'] = {
        'ranges': [
            centered_dims['x_range'] * (scale_factor or 1),
            centered_dims['y_range'] * (scale_factor or 1),
            centered_dims['z_range'] * (scale_factor or 1),
        ],
        'max_dim': centered_dims['max_dim'] * (scale_factor or 1),
    }

    # --- Step 4: 修复 XML ---
    fix_obj2mjcf_xml(
        str(target_xml),
        mass=mass,
        height=effective_height,
        add_grasppoints=add_grasppoints,
        remove_freejoint=remove_freejoint,
        scale_factor=scale_factor,
        obj_path=str(main_obj),
    )

    print(f"  后处理完成!")
    return report


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(
        description='修复obj2mjcf生成的XML文件，支持几何中心归零和尺寸归一化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 完整后处理（推荐）：居中 + 归一化 + 物理属性
  python fix_obj2mjcf_xml.py --model-dir /path/to/model_dir --uid model_uid

  # 仅修复 XML 物理属性（旧模式）
  python fix_obj2mjcf_xml.py input.xml --mass 0.05 --height 0.15

  # 批量修复目录
  python fix_obj2mjcf_xml.py --dir /path/to/models

  # 完整后处理 + 移除 freejoint
  python fix_obj2mjcf_xml.py --model-dir /path/to/model --uid xxx --remove-freejoint
        """
    )

    parser.add_argument('input', nargs='?', help='输入XML文件路径（仅 XML 修复模式）')
    parser.add_argument('-o', '--output', help='输出XML文件路径（默认覆盖原文件）')
    parser.add_argument('--mass', type=float, default=0.02, help='物体质量(kg)，默认0.02')
    parser.add_argument('--height', type=float, default=0.1, help='物体高度(m)，默认0.1')
    parser.add_argument('--no-grasppoints', action='store_true', help='不添加抓取点')
    parser.add_argument('--dir', help='批量处理目录（修复目录中所有.xml文件）')
    parser.add_argument('--remove-freejoint', action='store_true', help='移除 freejoint')

    # 完整后处理模式
    parser.add_argument('--model-dir', help='模型目录路径（完整后处理模式）')
    parser.add_argument('--uid', help='模型 UID（完整后处理模式）')
    parser.add_argument('--target-max-dim', type=float, default=0.15,
                        help='目标最大尺寸(m)，默认0.15')

    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # 完整后处理模式
    if args.model_dir and args.uid:
        try:
            report = postprocess_model(
                model_dir=args.model_dir,
                uid=args.uid,
                mass=args.mass,
                target_max_dim=args.target_max_dim,
                remove_freejoint=args.remove_freejoint,
                add_grasppoints=not args.no_grasppoints,
            )
            import json
            print("\n后处理报告:")
            print(json.dumps(report, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"错误: {e}")
            return 1

    elif args.dir:
        # 批量处理目录
        dir_path = Path(args.dir)
        xml_files = list(dir_path.rglob('*.xml'))

        if not xml_files:
            print(f"错误: 在 {dir_path} 中没有找到XML文件")
            return 1

        print(f"找到 {len(xml_files)} 个XML文件")
        print(f"批量处理参数: mass={args.mass}kg, height={args.height}m")
        print()

        for xml_file in xml_files:
            try:
                fix_obj2mjcf_xml(
                    str(xml_file),
                    mass=args.mass,
                    height=args.height,
                    add_grasppoints=not args.no_grasppoints,
                    remove_freejoint=args.remove_freejoint,
                )
            except Exception as e:
                print(f"  失败: {xml_file}")
                print(f"  错误: {e}")

        print(f"\n批量处理完成!")

    elif args.input:
        # 处理单个文件
        try:
            fix_obj2mjcf_xml(
                args.input,
                args.output,
                mass=args.mass,
                height=args.height,
                add_grasppoints=not args.no_grasppoints,
                remove_freejoint=args.remove_freejoint,
            )
        except Exception as e:
            print(f"错误: {e}")
            return 1
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
