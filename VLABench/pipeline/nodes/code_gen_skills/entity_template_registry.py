"""
Entity Template Registry - Subentity 父子映射与依赖注入

保留内容:
  - ENTITY_SUBENTITY_PARENTS: 硬编码父子映射
  - infer_dependencies: 自动依赖注入
  - inject_entity_status: 为注入实体填充 asset_status

注: 旧的模板函数（_builtin_template, _liquid_entity_template 等）
已迁移到 entity_loader.py，本文件不再包含。
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# =============================================================================
# 硬编码的 Subentity 父子映射
# 无法从继承链推断，新增实体时需要在此处维护
# =============================================================================

ENTITY_SUBENTITY_PARENTS = {
    "ChemistryTube": "chemistry_tube_stand",
    "Poker": "card_holder",
}


# =============================================================================
# 依赖注入：当检测到某类实体时，自动注入其必需的父容器实体
# =============================================================================

def infer_dependencies(entity_class_name: str, asset_status: Dict) -> List[str]:
    """
    当检测到某类实体时，返回需要自动注入的额外实体 canonical name 列表。

    Args:
        entity_class_name: 实体类名（从 asset_status["class_name"] 获取）
        asset_status: 资产状态字典

    Returns:
        需要注入的额外实体 canonical name 列表
    """
    injected = []

    if entity_class_name == "ChemistryTube":
        if "chemistry_tube_stand" not in asset_status:
            injected.append("chemistry_tube_stand")
    elif entity_class_name == "Poker":
        if "card_holder" not in asset_status:
            injected.append("card_holder")

    return injected


def inject_entity_status(entity_name: str, asset_status: Dict) -> Dict:
    """
    为自动注入的实体填充 asset_status 条目。

    Args:
        entity_name: 实体 canonical name
        asset_status: 现有资产状态字典

    Returns:
        填充后的 asset_status 字典（在原字典上原地更新）
    """
    if entity_name not in asset_status:
        if entity_name == "chemistry_tube_stand":
            asset_status[entity_name] = {
                "class_name": "TubeStand",
                "xml_path": "obj/meshes/tube/tube_container/tube_stand.xml",
                "properties": {},
            }
        elif entity_name == "card_holder":
            asset_status[entity_name] = {
                "class_name": "CardHolder",
                "xml_path": None,
                "properties": {},
            }
    return asset_status
