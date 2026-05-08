#!/usr/bin/env python3
"""
资产朝向自动修正模块

利用 LLM 多模态视觉能力判断 3D 模型渲染图中物体的朝向，
如果物体倒置或侧躺，自动旋转 OBJ 顶点并重新后处理。

流程: extract_frame → ask_llm → rotate_obj → re-postprocess → re-validate → 二次确认
"""

import base64
import io
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def extract_frame_from_video(video_path: str, frame_index: int = 0) -> bytes:
    """
    从验证视频中提取指定帧，返回 PNG 格式的 bytes。

    Args:
        video_path: validation.mp4 的路径
        frame_index: 要提取的帧索引（默认第 0 帧）

    Returns:
        PNG 格式的图片 bytes
    """
    import imageio.v3 as iio
    from PIL import Image

    frames = iio.imread(video_path)
    if frame_index >= len(frames):
        frame_index = len(frames) - 1
    frame_array = frames[frame_index]

    img = Image.fromarray(frame_array)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def ask_llm_orientation(image_bytes: bytes, object_name: str) -> dict:
    """
    发送渲染图给 LLM，请求判断物体朝向。

    Args:
        image_bytes: PNG 格式的图片 bytes
        object_name: 物体名称（如 "microscope"）

    Returns:
        {"is_upright": bool, "rotation": {"axis": "x"/"y"/"z", "degrees": 90/180/270} | None}
    """
    import anthropic

    # 获取 API 配置
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from vlabench_agent.config import AgentConfig

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    client = anthropic.Anthropic(
        api_key=AgentConfig.ANTHROPIC_API_KEY,
        base_url=AgentConfig.BASE_URL,
    )

    prompt_text = (
        f'这是一个3D模型"{object_name}"的MuJoCo仿真渲染图。'
        f"图中有一个灰色地面和待检查的物体模型。\n\n"
        f"请判断该物体的朝向是否正确：\n"
        f"1. 物体是否正立放置？（例如：瓶子/量筒开口朝上，显微镜目镜朝上，椅子腿朝下等）\n"
        f"2. 如果物体倒置或侧躺，需要绕哪个轴旋转多少度才能正立？\n\n"
        f"坐标系说明：X轴向右，Y轴向前，Z轴向上。\n"
        f"旋转规则：绕X轴旋转会改变Y-Z平面的朝向，绕Y轴旋转会改变X-Z平面的朝向。\n\n"
        f'请用JSON格式回答：\n'
        f'{{"is_upright": true/false, "rotation": {{"axis": "x"或"y", "degrees": 90或180或270}} 或 null}}\n'
        f"仅输出JSON，不要其他文字。"
    )

    response = client.messages.create(
        model=AgentConfig.MODEL_NAME,
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                ],
            }
        ],
    )

    raw = response.content
    if isinstance(raw, list):
        text_block = next((b for b in raw if hasattr(b, 'text')), None)
        response_text = text_block.text.strip() if text_block else ''
    else:
        response_text = raw.strip()
    logger.info(f"  LLM 朝向判断原始回复: {response_text}")

    # 提取 JSON（处理可能的 markdown 代码块包裹）
    if "```" in response_text:
        import re
        json_match = re.search(r"```(?:json)?\s*(.*?)```", response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1).strip()

    result = json.loads(response_text)
    return result


def _build_rotation_matrix(axis: str, degrees: int) -> np.ndarray:
    """构建绕指定轴旋转的 3x3 旋转矩阵。"""
    rad = math.radians(degrees)
    c, s = math.cos(rad), math.sin(rad)

    if axis.lower() == "x":
        return np.array([
            [1, 0,  0],
            [0, c, -s],
            [0, s,  c],
        ])
    elif axis.lower() == "y":
        return np.array([
            [ c, 0, s],
            [ 0, 1, 0],
            [-s, 0, c],
        ])
    elif axis.lower() == "z":
        return np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1],
        ])
    else:
        raise ValueError(f"未知旋转轴: {axis}")


def rotate_obj_files(model_dir: str, uid: str, axis: str, degrees: int):
    """
    旋转模型目录下所有 OBJ 文件的顶点和法线。

    Args:
        model_dir: 模型目录路径
        uid: 模型 UID
        axis: 旋转轴 ("x", "y", "z")
        degrees: 旋转角度 (90, 180, 270)
    """
    rot = _build_rotation_matrix(axis, degrees)
    model_path = Path(model_dir)

    # 收集所有需要旋转的 OBJ 文件
    obj_files: List[Path] = []

    # 主 OBJ
    main_obj = model_path / f"{uid}.obj"
    if main_obj.exists():
        obj_files.append(main_obj)

    # collision OBJ（在 uid 子目录下）
    collision_dir = model_path / uid
    if collision_dir.is_dir():
        for f in collision_dir.glob("*.obj"):
            obj_files.append(f)

    logger.info(f"  旋转 {len(obj_files)} 个 OBJ 文件 (轴={axis}, 角度={degrees}°)")

    for obj_path in obj_files:
        _rotate_single_obj(obj_path, rot)


def _rotate_single_obj(obj_path: Path, rot: np.ndarray):
    """旋转单个 OBJ 文件中的顶点和法线。"""
    lines = []
    with open(obj_path, "r") as f:
        for line in f:
            if line.startswith("v ") and not line.startswith("vt ") and not line.startswith("vn "):
                # 顶点坐标
                parts = line.split()
                if len(parts) >= 4:
                    v = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                    v_rot = rot @ v
                    lines.append(f"v {v_rot[0]:.8f} {v_rot[1]:.8f} {v_rot[2]:.8f}\n")
                else:
                    lines.append(line)
            elif line.startswith("vn "):
                # 法线向量
                parts = line.split()
                if len(parts) >= 4:
                    n = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                    n_rot = rot @ n
                    lines.append(f"vn {n_rot[0]:.8f} {n_rot[1]:.8f} {n_rot[2]:.8f}\n")
                else:
                    lines.append(line)
            else:
                lines.append(line)

    with open(obj_path, "w") as f:
        f.writelines(lines)

    logger.info(f"    已旋转: {obj_path.name}")


def fix_orientation(
    model_dir: str,
    uid: str,
    object_name: str,
    target_xml: str,
    mass: float = 0.02,
    target_max_dim: float = 0.15,
) -> dict:
    """
    朝向修正编排函数。

    流程: 提取首帧 → LLM 判断 → 旋转 OBJ → 重新后处理 → 重新验证 → 二次确认

    Args:
        model_dir: 模型目录路径
        uid: 模型 UID
        object_name: 物体名称
        target_xml: 目标 XML 路径
        mass: 物体质量 (kg)
        target_max_dim: 目标最大尺寸 (m)

    Returns:
        修正报告 dict
    """
    model_path = Path(model_dir)
    video_path = model_path / "validation.mp4"

    if not video_path.exists():
        logger.warning(f"  未找到 validation.mp4，跳过朝向检查")
        return {"action": "skipped", "reason": "no_validation_video"}

    # Step 1: 提取首帧
    logger.info(f"  朝向检查: 提取视频首帧...")
    image_bytes = extract_frame_from_video(str(video_path))

    # Step 2: 请求 LLM 判断
    logger.info(f"  朝向检查: 请求 LLM 判断...")
    try:
        judgment = ask_llm_orientation(image_bytes, object_name)
    except Exception as e:
        logger.warning(f"  LLM 朝向判断失败: {e}")
        return {"action": "skipped", "reason": f"llm_error: {e}"}

    if judgment.get("is_upright", True):
        logger.info(f"  朝向正确，无需旋转")
        return {"action": "no_change", "judgment": judgment}

    rotation = judgment.get("rotation")
    if not rotation:
        logger.warning(f"  LLM 判断需要旋转但未提供旋转参数")
        return {"action": "skipped", "reason": "no_rotation_params", "judgment": judgment}

    axis = rotation["axis"]
    degrees = rotation["degrees"]
    logger.info(f"  朝向修正: 绕 {axis} 轴旋转 {degrees}°")

    # Step 3: 旋转 OBJ 文件
    rotate_obj_files(model_dir, uid, axis, degrees)

    # Step 4: 重新后处理（居中 + 缩放 + 物理属性）
    logger.info(f"  朝向修正: 重新后处理...")
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from fix_obj2mjcf_xml import postprocess_model as _postprocess

    _postprocess(
        model_dir=model_dir,
        uid=uid,
        mass=mass,
        target_max_dim=target_max_dim,
        remove_freejoint=True,
        add_grasppoints=True,
    )

    # Step 5: 重新验证（生成新的 validation.mp4）
    logger.info(f"  朝向修正: 重新验证...")
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from validate_asset import validate_asset

    validate_asset(target_xml, render_preview=False, render_video=True)

    # Step 6: 二次确认
    logger.info(f"  朝向修正: 二次确认...")
    try:
        image_bytes_2 = extract_frame_from_video(str(video_path))
        judgment_2 = ask_llm_orientation(image_bytes_2, object_name)
        verified = judgment_2.get("is_upright", False)
    except Exception as e:
        logger.warning(f"  二次确认失败: {e}")
        verified = None

    report = {
        "action": "rotated",
        "rotation": {"axis": axis, "degrees": degrees},
        "initial_judgment": judgment,
        "verification_judgment": judgment_2 if verified is not None else None,
        "verified": verified,
    }

    if verified:
        logger.info(f"  朝向修正成功（二次确认通过）")
    elif verified is False:
        logger.warning(f"  朝向修正后二次确认未通过，可能需要人工检查")
    else:
        logger.warning(f"  二次确认异常，无法判断修正结果")

    return report
