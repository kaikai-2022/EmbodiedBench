"""
Asset Manager Node - 物理实例化总监

按 Asset Manager 设计规范文档实现：
  - 工序 1: 物理过滤与资产获取 (Fetch & Route)
  - 工序 2: 动态类名推断 (Dynamic Class Inference)
  - 工序 3: XML 外科手术与注入 (XML Surgical Injection)
  - 工序 4: 打包输出 (Pack Status)

设计原则:
  - 语义隔离: 不引入 LLM，只根据 init_params 行事
  - 快速失败: AssetNotFoundError 替代静默失败
  - 数据源唯一: solution 物质名称标签由 init_params["contains_substance"] 决定
"""

import logging
import os
from pathlib import Path
from typing import Dict

from ..tools.asset_tools import check_asset_exists, download_asset, _register_downloaded_asset
from ..tools.xml_injector import inject_xml
from .node_logger import log_node_output_file

logger = logging.getLogger(__name__)


class AssetNotFoundError(Exception):
    """资产未找到或下载失败时抛出"""
    pass


def _infer_class_name(spec: str, init_params: Dict) -> str:
    """
    基于 init_params 的特征工厂模式，动态推断 VLABench 底层类名。

    判断优先级:
    1. contains_substance → ChemistryBeaker / ChemistryTube
    2. is_container → CommonContainer / ContainerWithDoor
    3. name2class_xml registry 默认类名
    4. 兜底 → CommonGraspedEntity
    """
    if init_params.get("contains_substance"):
        if spec in ("tube", "test_tube"):
            logger.info(f"  → 类推断: {spec} + contains_substance → ChemistryTube")
            return "ChemistryTube"
        logger.info(f"  → 类推断: {spec} + contains_substance → ChemistryBeaker")
        return "ChemistryBeaker"

    if init_params.get("is_container"):
        if init_params.get("has_door"):
            logger.info(f"  → 类推断: {spec} + has_door → ContainerWithDoor")
            return "ContainerWithDoor"
        logger.info(f"  → 类推断: {spec} + is_container → CommonContainer")
        return "CommonContainer"

    try:
        from VLABench.configs.constant import name2class_xml
        if spec in name2class_xml:
            class_name = name2class_xml[spec][0].__name__
            logger.info(f"  → 类推断: {spec} 命中 name2class_xml → {class_name}")
            return class_name
    except Exception as e:
        logger.warning(f"  → name2class_xml 查找失败: {e}")

    logger.info(f"  → 类推断: {spec} → CommonGraspedEntity (兜底)")
    return "CommonGraspedEntity"


def _fetch_asset(spec: str, source_type: str) -> Dict:
    """
    工序 1: 获取资产

    Returns:
        {"xml_path": str, "is_objaverse": bool}
    """
    if source_type == "local":
        # 本地查找
        status = check_asset_exists(spec)
        if status.get("found"):
            return {
                "xml_path": status["xml_path"],
                "is_objaverse": False,
            }
        else:
            raise AssetNotFoundError(f"本地资产不存在: {spec}")

    elif source_type == "objaverse":
        # 云端下载
        logger.info(f"  → 资产来源: objaverse，尝试下载...")
        result = download_asset(spec, max_downloads=3)
        if result["success"] and result["assets"]:
            abs_path = result["assets"][0]
            vlabench_root = os.environ.get("VLABENCH_ROOT")
            assets_dir = Path(vlabench_root) / "assets"
            try:
                rel_path = os.path.relpath(abs_path, assets_dir)
            except ValueError:
                rel_path = abs_path
            return {
                "xml_path": rel_path,
                "is_objaverse": True,
            }
        else:
            raise AssetNotFoundError(f"下载失败: {spec} - {result.get('error', 'Unknown')}")

    else:
        raise AssetNotFoundError(f"未知 source_type: {source_type}")


def asset_manager_node(state: Dict) -> Dict:
    """
    Asset Manager 节点 - 物理实例化总监

    输入: state["normalized_context"]["instances"]
    输出: state["asset_status"]
    """
    normalized_context = state.get("normalized_context", {})
    instances = normalized_context.get("instances", [])

    if not instances:
        logger.warning("[Asset Manager] instances 为空")
        return {
            "asset_status": {},
            "current_stage": "skill_planner",
        }

    logger.info("=" * 60)
    logger.info(f"[Asset Manager] 开始处理 {len(instances)} 个实例...")

    asset_status: Dict[str, Dict] = {}

    for inst in instances:
        uid = inst["uid"]
        spec = inst["spec"]
        source_type = inst.get("source_type", "local")
        is_physical = inst.get("is_physical", True)
        init_params = inst.get("init_params", {})

        logger.info(f"  [{uid}] spec={spec}, source={source_type}")

        # ---------- 工序 1: 物理过滤 ----------
        if not is_physical:
            logger.info(f"    → is_physical=False，跳过（无独立 3D 模型）")
            continue

        # ---------- 工序 2: 动态类推断 ----------
        class_name = _infer_class_name(spec, init_params)

        # ---------- 工序 3: 获取资产 ----------
        # 文档要求 Fail-Fast: AssetNotFoundError 向上抛出，不吞掉
        fetch_result = _fetch_asset(spec, source_type)
        xml_path = fetch_result["xml_path"]
        is_objaverse = fetch_result["is_objaverse"]

        # Objaverse 下载的资产注册到 registry
        if is_objaverse:
            _register_downloaded_asset(spec, xml_path)

        # ---------- 工序 4: XML 注入 ----------
        vlabench_root = os.environ.get("VLABENCH_ROOT", "")
        if vlabench_root:
            abs_xml_path = os.path.join(vlabench_root, "assets", xml_path)
        else:
            abs_xml_path = xml_path

        if os.path.exists(abs_xml_path):
            xml_path = inject_xml(abs_xml_path, class_name, spec)

        # ---------- 工序 5: 打包 properties ----------
        properties = dict(init_params)
        if "contains_substance" in init_params:
            properties["solution"] = init_params["contains_substance"]
            if class_name in ("ChemistryBeaker", "ChemistryTube", "CommonContainer"):
                properties["is_container"] = True

        asset_status[uid] = {
            "xml_path": xml_path,
            "class_name": class_name,
            "properties": properties,
        }
        logger.info(f"    → {uid}: xml={xml_path}, class={class_name}, props={properties}")

    logger.info("=" * 60 + "\n")

    output = {
        "asset_status": asset_status,
        "current_stage": "skill_planner",
    }
    log_node_output_file("asset_manager", state, output)
    return output
