"""
Entity Loader - 结构化实体加载代码生成

简化设计：不再区分 container 和 object，所有实体统一放到 load_objects。
只有两种特殊情况：
  1. ChemistryTube → subentity 模式，自动注入 tube_stand 父容器
  2. 带 solution → liquid 模式
  3. 其余 → plain 模式

设计原则:
  - 一个 uid 只出现一次，不会多实例冲突
  - name2class_xml 查询用 spec（lookup key），entity name 用 uid
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# 试管 subentity 布局常量
TUBE_COL_POS = [-0.16, -0.08, 0, 0.08, 0.16]
TUBE_ROW_POS = [-0.05, 0.05]

# 多实体位置分散策略：确保同场景多个实体不会重叠
# 每个实体的位置范围之间有足够的间隔（至少0.1m）
ENTITY_POSITION_RANGES = [
    ([0.05, 0.15], [-0.15, -0.05]),   # 玻璃棒位置
    ([0.35, 0.45], [-0.05, 0.05]),    # 容器位置（与玻璃棒间隔>0.2m）
    ([0.15, 0.25], [0.05, 0.15]),     # 备用位置1
    ([0.25, 0.35], [0.05, 0.15]),     # 备用位置2
    ([0.05, 0.15], [-0.05, 0.05]),    # 备用位置3
    ([0.35, 0.45], [-0.15, -0.05]),   # 备用位置4
]


@dataclass
class EntityLoadPlan:
    """描述一个实体应该如何被加载"""
    uid: str
    spec: str
    class_name: str
    load_mode: str          # "plain" | "liquid" | "subentity"
    method_name: str         # "load_objects" | "load_init_containers"
    properties: Dict = field(default_factory=dict)
    parent_spec: Optional[str] = None
    position_index: int = 0  # 用于位置分散


def plan_entity_loading(
    instances: List[Dict],
    asset_status: Dict,
) -> List[EntityLoadPlan]:
    """
    为每个物理实体生成加载计划。

    简化规则：
      1. 跳过 is_physical=False 的实体
      2. ChemistryTube → subentity 模式，自动注入 tube_stand 父容器
      3. 带 solution → liquid 模式
      4. 其余 → plain 模式
      所有实体统一放到 load_objects（除 TubeStand 放到 load_init_containers）
    """
    plans: List[EntityLoadPlan] = []
    has_init_container = False
    plain_entity_counter = 0  # 用于位置分散

    # 第一遍扫描：找出已有的 tube_stand uid
    existing_tube_stand_uid = None
    for inst in instances:
        if not inst.get("is_physical", True):
            continue
        uid = inst["uid"]
        info = asset_status.get(uid, {})
        if info.get("class_name") == "TubeStand" or inst.get("spec") == "chemistry_tube_stand":
            existing_tube_stand_uid = uid

    for inst in instances:
        if not inst.get("is_physical", True):
            continue

        uid = inst["uid"]
        spec = inst["spec"]
        info = asset_status.get(uid, {})
        class_name = info.get("class_name", "CommonGraspedEntity")
        properties = info.get("properties", {})

        # SubEntity: ChemistryTube
        if class_name == "ChemistryTube":
            parent_spec = "chemistry_tube_stand"
            if existing_tube_stand_uid:
                parent_uid = existing_tube_stand_uid
            else:
                parent_uid = parent_spec
                if parent_uid not in asset_status:
                    _inject_tube_stand(asset_status)
            if not has_init_container:
                plans.append(EntityLoadPlan(
                    uid=parent_uid, spec=parent_spec,
                    class_name="TubeStand",
                    load_mode="plain",
                    method_name="load_init_containers",
                ))
                has_init_container = True

            plans.append(EntityLoadPlan(
                uid=uid, spec="tube", class_name="ChemistryTube",
                load_mode="subentity", method_name="load_objects",
                properties=properties, parent_spec=parent_spec,
            ))

        # 跳过已被 ChemistryTube 作为父容器使用的 TubeStand
        elif uid == existing_tube_stand_uid and has_init_container:
            continue

        # Liquid: 带 solution
        elif "solution" in properties:
            plans.append(EntityLoadPlan(
                uid=uid, spec=spec, class_name=class_name,
                load_mode="liquid", method_name="load_objects",
                properties=properties,
                position_index=plain_entity_counter,
            ))
            plain_entity_counter += 1

        # Plain: 所有其他实体（统一走 load_objects）
        else:
            plans.append(EntityLoadPlan(
                uid=uid, spec=spec, class_name=class_name,
                load_mode="plain", method_name="load_objects",
                properties=properties,
                position_index=plain_entity_counter,
            ))
            plain_entity_counter += 1

    return plans


def _inject_tube_stand(asset_status: Dict) -> None:
    asset_status["chemistry_tube_stand"] = {
        "xml_path": "obj/meshes/tube/tube_container/tube_stand.xml",
        "class_name": "TubeStand",
        "properties": {},
    }


def generate_load_methods(plans: List[EntityLoadPlan], asset_status: Dict) -> tuple:
    """
    从加载计划生成 load 方法代码。

    Returns:
        (load_methods_code: str, extra_imports: str|None, extra_constants: str|None)
    """
    # 按 method_name 分组
    methods: Dict[str, List[EntityLoadPlan]] = {}
    for plan in plans:
        methods.setdefault(plan.method_name, []).append(plan)

    code_parts = []
    needs_name2class_xml = False
    needs_tube_constants = False

    # 只处理 load_init_containers 和 load_objects（不再生成 load_containers）
    for method_name in ["load_init_containers", "load_objects"]:
        method_plans = methods.get(method_name, [])
        if not method_plans:
            continue
        code, flags = _generate_method(method_name, method_plans)
        code_parts.append(code)
        needs_name2class_xml = needs_name2class_xml or flags.get("needs_name2class_xml", False)
        needs_tube_constants = needs_tube_constants or flags.get("needs_tube_constants", False)

    load_methods_code = "\n".join(code_parts)

    extra_imports = None
    if needs_name2class_xml:
        extra_imports = "from VLABench.configs.constant import name2class_xml"

    extra_constants = None
    if needs_tube_constants:
        extra_constants = (
            f"relative_col_pos = {TUBE_COL_POS}\n"
            f"relative_row_pos = {TUBE_ROW_POS}\n"
        )

    return load_methods_code, extra_imports, extra_constants


def _generate_method(method_name: str, plans: List[EntityLoadPlan]) -> tuple:
    """生成单个 load 方法"""
    flags = {"needs_name2class_xml": False, "needs_tube_constants": False}

    if method_name == "load_init_containers":
        return _gen_init_containers(plans, flags)
    return _gen_objects(plans, flags)


# ── load_init_containers ──────────────────────────────────────────────────

def _gen_init_containers(plans: List[EntityLoadPlan], flags: Dict) -> str:
    """
    生成 load_init_containers 方法。
    仅用于 ChemistryTube 的父容器（TubeStand）。
    """
    DEFAULT_PARENT_CONTAINERS = {
        "ChemistryTube": ("chemistry_tube_stand", "TubeStand"),
    }

    lines = ["    def load_init_containers(self, init_container):"]

    default_parents = {}
    for plan in plans:
        parent_info = DEFAULT_PARENT_CONTAINERS.get(
            plan.class_name,
            (plan.spec, plan.class_name)
        )
        default_parents[parent_info] = (*parent_info, plan.uid)

    for (parent_spec, parent_class), (_, _, parent_uid) in default_parents.items():
        triggered_str = ""
        for child_class, (p_spec, p_class) in DEFAULT_PARENT_CONTAINERS.items():
            if p_spec == parent_spec and p_class == parent_class:
                triggered_str = child_class
                break
        lines += [
            f"        if init_container is None or init_container == \"{parent_spec}\":",
            f"            container_config = dict(",
            f"                name=\"{parent_uid}\",",
            f"                xml_path=name2class_xml[\"{parent_spec}\"][-1],",
            f"                position=[random.uniform(-0.15, -0.05), random.uniform(0.05, 0.15), 0.8],",
            f"            )",
            f"            container_config[\"class\"] = \"{parent_class}\"",
            f"            self.config[\"task\"][\"components\"].append(container_config)",
        ]

    lines.append("")
    flags["needs_name2class_xml"] = True
    return "\n".join(lines), flags


# ── load_objects ─────────────────────────────────────────────────────────

def _gen_objects(plans: List[EntityLoadPlan], flags: Dict) -> str:
    """生成 load_objects 方法，支持 plain / liquid / subentity"""
    lines = ["    def load_objects(self, target_entity):"]

    for plan in plans:
        if plan.load_mode == "subentity":
            lines += _code_subentity(plan, flags)
        elif plan.load_mode == "liquid":
            lines += _code_liquid(plan, flags)
        else:
            lines += _code_plain(plan, flags)

    flags["needs_name2class_xml"] = True
    return "\n".join(lines), flags


def _code_plain(plan: EntityLoadPlan, flags: Dict) -> List[str]:
    pos_range = ENTITY_POSITION_RANGES[plan.position_index % len(ENTITY_POSITION_RANGES)]
    return [
        f'        obj_config = dict(',
        f'            name="{plan.uid}",',
        f'            xml_path=name2class_xml["{plan.spec}"][-1],',
        f'            position=[random.uniform({pos_range[0][0]}, {pos_range[0][1]}), random.uniform({pos_range[1][0]}, {pos_range[1][1]}), 0.8],',
        f'        )',
        f'        obj_config["class"] = "{plan.class_name}"',
        f'        obj_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0.05])',
        f'        self.config["task"]["components"].append(obj_config)',
        "",
    ]


def _code_liquid(plan: EntityLoadPlan, flags: Dict) -> List[str]:
    solution = plan.properties.get("solution", plan.uid)
    pos_range = ENTITY_POSITION_RANGES[plan.position_index % len(ENTITY_POSITION_RANGES)]
    return [
        f'        obj_config = dict(',
        f'            name="{plan.uid}",',
        f'            xml_path=name2class_xml["{plan.spec}"][-1],',
        f'            position=[random.uniform({pos_range[0][0]}, {pos_range[0][1]}), random.uniform({pos_range[1][0]}, {pos_range[1][1]}), 0.8],',
        f'            solution="{solution}",',
        f'        )',
        f'        obj_config["class"] = "{plan.class_name}"',
        f'        obj_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0.05])',
        f'        self.config["task"]["components"].append(obj_config)',
        "",
    ]


def _code_subentity(plan: EntityLoadPlan, flags: Dict) -> List[str]:
    solution = plan.properties.get("solution", plan.uid)
    flags["needs_tube_constants"] = True
    return [
        "        col_pos = random.choice(relative_col_pos)",
        "        row_pos = random.choice(relative_row_pos)",
        "        pos = [col_pos, row_pos, 0.05]",
        '        init_container_config = self.config["task"]["components"][-1]',
        '        if "subentities" not in init_container_config:',
        '            init_container_config["subentities"] = []',
        f'        obj_config = dict(',
        f'            name="{plan.uid}",',
        f'            solution="{solution}",',
        f'            xml_path=name2class_xml["tube"][-1],',
        f'            position=pos,',
        f'        )',
        f'        obj_config["class"] = "{plan.class_name}"',
        '        init_container_config["subentities"].append(obj_config)',
        "",
    ]
