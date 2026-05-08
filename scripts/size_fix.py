#!/usr/bin/env python3
"""
模型尺寸自动修正模块

利用 LLM 常识知识判断物体的合理真实世界尺寸，
重新计算 XML 中的 scale 属性，使模型在仿真场景中比例正确。

流程: ask_llm_size → rescale_xml → re-validate
"""

import base64
import json
import logging
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def ask_llm_size(object_name: str) -> dict:
    """
    用 LLM 常识知识判断物体的合理物理尺寸。

    Args:
        object_name: 物体名称（如 "microscope"）

    Returns:
        {"height_m": float, "width_m": float}
    """
    import anthropic

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from vlabench_agent.config import AgentConfig

    client = anthropic.Anthropic(
        api_key=AgentConfig.ANTHROPIC_API_KEY,
        base_url=AgentConfig.BASE_URL,
    )

    prompt_text = (
        f'你是一个3D物理仿真专家。请告诉我以下物品的合理物理尺寸。\n\n'
        f'物品名称: "{object_name}"\n\n'
        f'请返回该物品正立放置时的合理高度（米）和宽度（米）。\n'
        f'考虑这是用于机器人桌面操纵任务的物品，应为常见的标准尺寸。\n\n'
        f'参考信息：\n'
        f'- Franka Emika Panda 机械臂总高约 1.1m\n'
        f'- 桌面高度约 0.75m，物品放在桌面上\n'
        f'- 机械臂夹爪最大张开宽度约 0.08m\n'
        f'- 常见实验室烧杯高约 0.10-0.15m\n'
        f'- 常见实验室锥形瓶高约 0.15-0.25m\n'
        f'- 常见实验室显微镜高约 0.30-0.40m\n\n'
        f'请仅返回 JSON: {{"height_m": 0.35, "width_m": 0.15}}\n'
        f'仅输出 JSON，不要其他文字。'
    )

    response = client.messages.create(
        model=AgentConfig.MODEL_NAME,
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": prompt_text,
            }
        ],
    )

    raw = response.content
    if isinstance(raw, list):
        text_block = next((b for b in raw if hasattr(b, 'text')), None)
        response_text = text_block.text.strip() if text_block else ''
    else:
        response_text = raw.strip()
    logger.info(f"  LLM 尺寸判断原始回复: {response_text}")

    # 提取 JSON
    if "```" in response_text:
        import re
        json_match = re.search(r"```(?:json)?\s*(.*?)```", response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1).strip()

    result = json.loads(response_text)
    return result


def rescale_xml(xml_path: str, obj_path: str, target_height_m: float) -> dict:
    """
    修改 XML 中的 scale 属性，使模型达到目标高度。

    同时更新 inertial 和 grasppoint 位置。

    Args:
        xml_path: XML 文件路径
        obj_path: 主 OBJ 文件路径（用于读取原始几何尺寸）
        target_height_m: 目标高度（米）

    Returns:
        修正报告 dict
    """
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from fix_obj2mjcf_xml import get_obj_dimensions

    # 读取 OBJ 原始尺寸（未缩放的顶点坐标）
    dims = get_obj_dimensions(obj_path)
    z_range = dims['z_range']  # 朝向修正后 Z 应为高度轴
    x_range = dims['x_range']
    y_range = dims['y_range']

    if z_range <= 0:
        logger.warning(f"  OBJ z_range={z_range}，无法计算缩放")
        return {"action": "skipped", "reason": "invalid_z_range"}

    # 计算新 scale: target_height = z_range * new_scale
    new_scale = target_height_m / z_range

    # 读取当前 XML 中的 scale
    tree = ET.parse(xml_path)
    root = tree.getroot()
    asset = root.find('asset')

    old_scale = None
    if asset is not None:
        first_mesh = asset.find('mesh')
        if first_mesh is not None and first_mesh.get('scale'):
            old_scale_str = first_mesh.get('scale')
            old_scale = float(old_scale_str.split()[0])

    old_height = z_range * (old_scale or 1)
    new_height = z_range * new_scale

    logger.info(f"  尺寸修正: {old_height:.4f}m -> {new_height:.4f}m (scale: {old_scale} -> {new_scale:.8f})")

    # 1. 更新所有 <mesh> 的 scale
    if asset is not None:
        scale_str = f"{new_scale} {new_scale} {new_scale}"
        for mesh_elem in asset.findall('mesh'):
            mesh_elem.set('scale', scale_str)

    # 2. 更新 <inertial>
    body = root.find('.//body')
    if body is not None:
        inertial = body.find('inertial')
        if inertial is not None:
            # 质心在高度中点
            inertial.set('pos', f"0 0 {new_height / 2:.4f}")
            # 重新计算惯性张量（近似为长方体）
            new_width = x_range * new_scale
            new_depth = y_range * new_scale
            mass = float(inertial.get('mass', '0.02'))
            ixx = mass / 12 * (new_depth**2 + new_height**2)
            iyy = mass / 12 * (new_width**2 + new_height**2)
            izz = mass / 12 * (new_width**2 + new_depth**2)
            inertial.set('diaginertia', f"{ixx:.6f} {iyy:.6f} {izz:.6f}")

        # 3. 更新 grasppoint sites
        grasppoints = [s for s in body.findall('site') if s.get('class') == 'grasppoint']
        if grasppoints:
            # 重新分布抓取点在高度的 80%-95%
            n = len(grasppoints)
            for i, site in enumerate(grasppoints):
                frac = 0.80 + 0.15 * i / max(n - 1, 1)
                site.set('pos', f"0 0 {new_height * frac:.4f}")

    # 写回 XML
    tree.write(xml_path, xml_declaration=True, encoding='unicode')

    # 美化格式
    import xml.dom.minidom
    with open(xml_path, 'r') as f:
        dom = xml.dom.minidom.parseString(f.read())
    pretty = dom.toprettyxml(indent='  ')
    # 去除多余空行
    lines = [line for line in pretty.split('\n') if line.strip()]
    with open(xml_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    return {
        "action": "rescaled",
        "old_scale": old_scale,
        "new_scale": new_scale,
        "old_height_m": old_height,
        "new_height_m": new_height,
        "target_height_m": target_height_m,
        "obj_z_range": z_range,
    }


def _shift_obj_z(obj_path: Path, z_offset: float):
    """将 OBJ 文件中所有顶点在 Z 方向平移 z_offset。"""
    lines = []
    with open(obj_path, "r") as f:
        for line in f:
            if line.startswith("v ") and not line.startswith("vt ") and not line.startswith("vn "):
                parts = line.split()
                if len(parts) >= 4:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    lines.append(f"v {x:.8f} {y:.8f} {z + z_offset:.8f}\n")
                else:
                    lines.append(line)
            else:
                lines.append(line)
    with open(obj_path, "w") as f:
        f.writelines(lines)


def _bottom_align_obj_files(model_dir: str, uid: str) -> float:
    """
    将模型所有 OBJ 顶点在 Z 方向上移，使底部对齐到 Z=0。

    这样 set_pose(position=[x, y, z]) 中的 z 就表示模型底部的世界高度，
    与 VLABench 默认的 position=[0, 0, 0.8] 约定兼容。

    Args:
        model_dir: 模型目录路径
        uid: 模型 UID

    Returns:
        上移量（原始坐标空间中的半高度值）
    """
    model_path = Path(model_dir)

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from fix_obj2mjcf_xml import get_obj_dimensions

    main_obj = model_path / f"{uid}.obj"
    dims = get_obj_dimensions(str(main_obj))
    half_height = dims['z_range'] / 2  # 当前居中后 Z 范围 [-h/2, +h/2]

    # 收集所有 OBJ 文件：根目录主 OBJ + 子目录碰撞 OBJ
    obj_files = [main_obj]
    collision_dir = model_path / uid
    if collision_dir.is_dir():
        # 只处理碰撞 OBJ（文件名含 _collision_），跳过子目录中的视觉 OBJ 拷贝
        obj_files.extend(
            f for f in collision_dir.glob("*.obj") if "_collision_" in f.name
        )

    for obj_file in obj_files:
        _shift_obj_z(obj_file, half_height)

    # 同步根目录主 OBJ 到子目录（XML 引用的是子目录的视觉 OBJ）
    subdir_visual_obj = collision_dir / f"{uid}.obj"
    if collision_dir.is_dir() and subdir_visual_obj.exists():
        import shutil
        shutil.copy2(main_obj, subdir_visual_obj)
        logger.info(f"  底部对齐: 同步视觉 OBJ 到子目录")

    logger.info(f"  底部对齐: 所有 OBJ 上移 {half_height:.4f} (Z: [-{half_height:.4f}, +{half_height:.4f}] -> [0, {half_height*2:.4f}])")
    return half_height


def fix_size(
    model_dir: str,
    uid: str,
    object_name: str,
    target_xml: str,
) -> dict:
    """
    尺寸修正编排函数。

    流程: LLM 判断合理尺寸 → 修改 XML scale → 重新验证

    Args:
        model_dir: 模型目录路径
        uid: 模型 UID
        object_name: 物体名称
        target_xml: 目标 XML 路径

    Returns:
        修正报告 dict
    """
    model_path = Path(model_dir)
    obj_path = model_path / f"{uid}.obj"

    if not obj_path.exists():
        logger.warning(f"  未找到 OBJ 文件: {obj_path}")
        return {"action": "skipped", "reason": "no_obj_file"}

    # Step 1: 请求 LLM 判断合理尺寸
    logger.info(f"  尺寸检查: 请求 LLM 判断 '{object_name}' 的合理尺寸...")
    try:
        size_info = ask_llm_size(object_name)
    except Exception as e:
        logger.warning(f"  LLM 尺寸判断失败: {e}")
        return {"action": "skipped", "reason": f"llm_error: {e}"}

    target_height = size_info.get("height_m")
    if not target_height or target_height <= 0:
        logger.warning(f"  LLM 返回无效高度: {size_info}")
        return {"action": "skipped", "reason": "invalid_height", "llm_response": size_info}

    logger.info(f"  LLM 建议尺寸: 高度={target_height}m, 宽度={size_info.get('width_m', 'N/A')}m")

    # Step 2: 修改 XML scale
    logger.info(f"  尺寸修正: 重新缩放 XML...")
    rescale_report = rescale_xml(target_xml, str(obj_path), target_height)

    if rescale_report.get("action") != "rescaled":
        return rescale_report

    # Step 3: 底部对齐（上移 OBJ 顶点使底部在 Z=0）
    logger.info(f"  尺寸修正: 底部对齐...")
    _bottom_align_obj_files(model_dir, uid)

    # Step 4: 重新验证（生成新的 validation.mp4）
    logger.info(f"  尺寸修正: 重新验证...")
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    try:
        from validate_asset import validate_asset
        validate_asset(target_xml, render_preview=False, render_video=True)
    except Exception as e:
        logger.warning(f"  重新验证失败: {e}")

    report = {
        "action": "rescaled",
        "llm_suggestion": size_info,
        **rescale_report,
    }

    logger.info(f"  尺寸修正完成: {rescale_report['old_height_m']:.4f}m -> {rescale_report['new_height_m']:.4f}m")
    return report
