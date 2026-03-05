"""
Asset Manager Node - 资产管理智能体

检查所需模型资产,自动调用 get_assets.py 下载缺失模型
"""

import logging
import os
from pathlib import Path
from typing import Dict

from ..tools.asset_tools import check_asset_exists, download_asset

logger = logging.getLogger(__name__)


def asset_manager_node(state: Dict) -> Dict:
    """
    资产管理节点 - 检查并下载所需模型

    Args:
        state: VLABenchAgentState

    Returns:
        更新后的状态字典
    """
    logger.info("=" * 60)
    logger.info("[Asset Manager] 开始检查资产...")

    task_analysis = state.get('task_analysis', {})
    required_objects = task_analysis.get('objects', [])

    if not required_objects:
        logger.error("[Asset Manager] ✗ 未找到需要的物体列表")
        return {
            "current_stage": "error",
            "errors": state.get("errors", []) + ["任务分析结果中缺少 objects 字段"]
        }

    logger.info(f"[Asset Manager] 需要检查 {len(required_objects)} 个物体: {required_objects}")

    asset_status = {}
    missing_objects = []

    # 1. 检查所有物体资产
    for obj in required_objects:
        logger.info(f"[Asset Manager] 检查物体: {obj}")
        status = check_asset_exists(obj)
        asset_status[obj] = status

        if not status['found']:
            missing_objects.append(obj)

    # 2. 下载缺失的资产
    errors = []
    if missing_objects:
        logger.info(f"[Asset Manager] 缺失资产: {missing_objects}")
        logger.info(f"[Asset Manager] 开始下载...")

        for obj in missing_objects:
            download_result = download_asset(obj, max_downloads=5)

            if download_result['success'] and download_result['assets']:
                # 更新资产状态 - 使用相对路径
                abs_path = download_result['assets'][0]
                vlabench_root = os.environ.get('VLABENCH_ROOT')

                # VLABench 期望路径相对于 VLABENCH_ROOT/assets/ 目录
                # 例如: /path/to/VLABench/assets/review/beaker/xxx.xml -> review/beaker/xxx.xml
                assets_dir = Path(vlabench_root) / "assets"
                try:
                    rel_path = os.path.relpath(abs_path, assets_dir)
                except ValueError:
                    # 不同驱动器，使用绝对路径
                    rel_path = abs_path

                # 根据物体名称推断合适的类
                # 对于下载的通用物体，使用 CommonGraspedEntity (所有物体的基类)
                obj_class = "CommonGraspedEntity"

                asset_status[obj] = {
                    "found": True,
                    "xml_path": rel_path,
                    "class": obj_class,
                    "newly_downloaded": True
                }
                logger.info(f"[Asset Manager] ✓ {obj} 下载成功")
            else:
                # 下载失败,记录错误
                error_msg = f"无法下载资产: {obj} - {download_result.get('error', 'Unknown')}"
                errors.append(error_msg)
                logger.error(f"[Asset Manager] ✗ {error_msg}")

    # 3. 输出统计信息
    found_count = sum(1 for s in asset_status.values() if s['found'])
    logger.info(f"[Asset Manager] 资产状态: {found_count}/{len(required_objects)} 可用")

    for obj, status in asset_status.items():
        if status['found']:
            logger.info(f"  ✓ {obj}: {status['xml_path']}")
        else:
            logger.info(f"  ✗ {obj}: 未找到")

    logger.info("=" * 60 + "\n")

    # 4. 返回结果
    if errors:
        return {
            "asset_status": asset_status,
            "current_stage": "error",
            "errors": state.get("errors", []) + errors
        }

    return {
        "asset_status": asset_status,
        "current_stage": "task_creation"
    }
