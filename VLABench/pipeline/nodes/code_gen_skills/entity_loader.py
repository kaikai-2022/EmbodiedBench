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
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# 试管 subentity 布局常量
TUBE_COL_POS = [-0.16, -0.08, 0, 0.08, 0.16]
TUBE_ROW_POS = [-0.05, 0.05]

# 移液枪架 subentity 布局常量 (4 段中点)
PIPETTE_STAND_COL_POS = [0]
PIPETTE_STAND_ROW_POS = [-0.1088, -0.0363, 0.0363, 0.1088]

# Funnel subentity 位置：铁环 placepoint 高度 (z=0.35)，向竖直铁杆偏移 -0.03m
FUNNEL_SUBENTITY_POSITION = [-0.02, 0.008, 0.30]

# Funnel 支持的父容器
FUNNEL_PARENT_CONTAINERS = {
    "funnel": ("funnel_support", "FunnelSupport"),
}

# 默认父容器映射（class_name -> (spec, class)）
DEFAULT_PARENT_CONTAINERS = {
    "ChemistryTube": ("chemistry_tube_stand", "TubeStand"),
    "Funnel": ("funnel_support", "FunnelSupport"),
}

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

# ── 大件/小件分类与朝向控制 ───────────────────────────────────────────────

# 大件白名单：远离机械臂 + 需要门/把手朝向机械臂
FIXTURE_WITH_HANDLE_CLASSES = {
    "DryingBoxWithButton",
    "ContainerWithDrawer",   # 覆盖 cabinet 和 drawer
}

# 大件朝向：绕 z 轴旋转，使门/把手朝向机械臂（world -y 方向）
# DryingBoxWithButton:  yaw=π/2（90°），让按钮/门把手正对机器人
# ContainerWithDrawer:  原始方向已对，无需旋转
FIXTURE_HANDLE_YAW = {
    "DryingBoxWithButton": np.pi / 2,
    "ContainerWithDrawer": 0.0,
}

# 小件位置区间：靠近机械臂
# x: 全宽 [-0.30, +0.30] 随机（左右分布）
# y: [-0.05, 0.25]，偏近端（robot base 在 y=-0.4）
# 与大件之间留 0.15m 缓冲带（[0.25, 0.40]），防止物理碰撞
SMALL_LABWARE_RANGES = [
    ([-0.30, -0.15], [-0.05, 0.10]),   # 左侧偏前
    ([-0.15,  0.00], [-0.05, 0.10]),   # 中左偏前
    ([ 0.00,  0.15], [-0.05, 0.10]),   # 中右偏前
    ([ 0.15,  0.30], [-0.05, 0.10]),   # 右侧偏前
    ([-0.30, -0.15], [0.10, 0.25]),    # 左侧偏后
    ([-0.15,  0.00], [0.10, 0.25]),    # 中左偏后
    ([ 0.00,  0.15], [0.10, 0.25]),    # 中右偏后
    ([ 0.15,  0.30], [0.10, 0.25]),    # 右侧偏后
]

# 大件位置区间：远离机械臂
# x: 全宽 [-0.30, +0.30] 随机（左右分布）
# y: [0.40, 0.45]，远端（与缓冲带 [0.25, 0.40] 隔开）
FIXTURE_RANGES = [
    ([-0.30, -0.15], [0.40, 0.45]),   # 左侧远端
    ([-0.15,  0.00], [0.40, 0.45]),   # 中左远端
    ([ 0.00,  0.15], [0.40, 0.45]),   # 中右远端
    ([ 0.15,  0.30], [0.40, 0.45]),   # 右侧远端
]

# 固定位置表：key 为 class_name，value 为 (x, y)，z 始终为 0.8
# 用于需要精确定位的物体（如 drying_box 由视觉标定过）
FIXED_POSITIONS = {
    "DryingBoxWithButton": (0.30, 0.45),
    "ContainerWithDrawer": (0.0, 0.4),
}

# 容器类需要被固定到 arena 才能完成任务的清单
# （单手机械臂无法在被自由放置的物体上完成拧/插/按等需要底座稳定的操作）
ATTACH_TO_ARENA_CLASSES = {
    # 暂时禁用 attach_to_arena 以测试 drawer 问题
    "ContainerWithCap",
    "DryingBoxWithButton",
    "ContainerWithDrawer",
}


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
    orientation_yaw: float = 0.0  # 仅大件使用，绕 z 轴 rad；小件默认为 0
    attach_to_arena: bool = False  # 加载后是否焊死到 arena
    subentity_position: list = field(default_factory=list)  # 子实体相对位置


def _append_plain_like_plan(
    plans: List[EntityLoadPlan],
    uid: str,
    spec: str,
    class_name: str,
    properties: Dict,
    small_counter: int,
    fixture_counter: int,
    attach_to_arena: bool = False,
) -> None:
    """
    为 plain/liquid 实体构建 EntityLoadPlan，并根据 class_name 选取对应的位置表和朝向。

    大件走 FIXTURE_RANGES（远离机械臂）并设置 orientation_yaw；
    小件走 SMALL_LABWARE_RANGES（靠近机械臂），无朝向。
    """
    if class_name in FIXTURE_WITH_HANDLE_CLASSES:
        pos_index = fixture_counter
        yaw = FIXTURE_HANDLE_YAW[class_name]
    else:
        pos_index = small_counter
        yaw = 0.0

    plans.append(EntityLoadPlan(
        uid=uid,
        spec=spec,
        class_name=class_name,
        load_mode="liquid" if "solution" in properties else "plain",
        method_name="load_objects",
        properties=properties,
        position_index=pos_index,
        orientation_yaw=yaw,
        attach_to_arena=attach_to_arena,
    ))


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
    small_counter = 0    # 小件位置序号（靠近机械臂）
    fixture_counter = 0  # 大件位置序号（远离机械臂）

    # 第一遍扫描：找出已有的 tube_stand / funnel_support uid
    existing_tube_stand_uid = None
    existing_funnel_support_uid = None
    for inst in instances:
        if not inst.get("is_physical", True):
            continue
        uid = inst["uid"]
        info = asset_status.get(uid, {})
        if info.get("class_name") in ("TubeStand", "MediumTubeStand") or inst.get("spec") in ("chemistry_tube_stand", "chemistry_tube_rack"):
            existing_tube_stand_uid = uid
        if info.get("class_name") == "FunnelSupport" or inst.get("spec") == "funnel_support":
            existing_funnel_support_uid = uid

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
            parent_spec = "chemistry_tube_rack" if spec == "chemistry_tube" else "chemistry_tube_stand"
            parent_class = "MediumTubeStand" if parent_spec == "chemistry_tube_rack" else "TubeStand"
            if existing_tube_stand_uid:
                parent_uid = existing_tube_stand_uid
            else:
                parent_uid = parent_spec
                if parent_uid not in asset_status:
                    _inject_tube_stand(asset_status, parent_spec)
            if not has_init_container:
                plans.append(EntityLoadPlan(
                    uid=parent_uid, spec=parent_spec,
                    class_name=parent_class,
                    load_mode="plain",
                    method_name="load_init_containers",
                ))
                has_init_container = True

            plans.append(EntityLoadPlan(
                uid=uid, spec=spec, class_name="ChemistryTube",
                load_mode="subentity", method_name="load_objects",
                properties=properties, parent_spec=parent_spec,
            ))

        # SubEntity: pipette (放在 tube_stand 试管架上)
        elif spec == "pipette":
            parent_spec = "chemistry_tube_stand"
            parent_class = "TubeStand"
            if existing_tube_stand_uid:
                parent_uid = existing_tube_stand_uid
            else:
                parent_uid = parent_spec
                if parent_uid not in asset_status:
                    _inject_tube_stand(asset_status, parent_spec)
            if not has_init_container:
                plans.append(EntityLoadPlan(
                    uid=parent_uid, spec=parent_spec,
                    class_name=parent_class,
                    load_mode="plain",
                    method_name="load_init_containers",
                ))
                has_init_container = True

            plans.append(EntityLoadPlan(
                uid=uid, spec=spec, class_name=class_name,
                load_mode="subentity", method_name="load_objects",
                properties=properties, parent_spec=parent_spec,
            ))

        # SubEntity: mechanical_pipette (放在 pipettes_stand 顶部)
        elif spec == "mechanical_pipette":
            parent_spec = "pipettes_stand"
            parent_class = "PipetteStand"
            parent_uid = parent_spec
            if parent_uid not in asset_status:
                _inject_pipettes_stand(asset_status)
            if not has_init_container:
                plans.append(EntityLoadPlan(
                    uid=parent_uid, spec=parent_spec,
                    class_name=parent_class,
                    load_mode="plain",
                    method_name="load_init_containers",
                ))
                has_init_container = True

            plans.append(EntityLoadPlan(
                uid=uid, spec=spec, class_name=class_name,
                load_mode="subentity", method_name="load_objects",
                properties=properties, parent_spec=parent_spec,
            ))

        # SubEntity: Funnel (漏斗放在铁架台的铁环上)
        # 注意：funnel 注册的 class 是 CommonGraspedEntity，所以用 spec 判断
        elif spec == "funnel":
            parent_spec = "funnel_support"
            parent_class = "FunnelSupport"
            if existing_funnel_support_uid:
                parent_uid = existing_funnel_support_uid
            else:
                parent_uid = parent_spec
                if parent_uid not in asset_status:
                    _inject_funnel_support(asset_status)
            if not has_init_container:
                plans.append(EntityLoadPlan(
                    uid=parent_uid, spec=parent_spec,
                    class_name=parent_class,
                    load_mode="plain",
                    method_name="load_init_containers",
                ))
                has_init_container = True

            plans.append(EntityLoadPlan(
                uid=uid, spec=spec, class_name=class_name,
                load_mode="subentity", method_name="load_objects",
                properties=properties, parent_spec=parent_spec,
                subentity_position=FUNNEL_SUBENTITY_POSITION,
            ))

        # 跳过已被作为父容器使用的 TubeStand / FunnelSupport
        elif (uid == existing_tube_stand_uid or spec == "funnel_support") and has_init_container:
            continue

        # Liquid: 带 solution
        elif "solution" in properties:
            _append_plain_like_plan(
                plans, uid, spec, class_name, properties,
                small_counter, fixture_counter, attach_to_arena=(class_name in ATTACH_TO_ARENA_CLASSES),
            )
            if class_name in FIXTURE_WITH_HANDLE_CLASSES:
                fixture_counter += 1
            else:
                small_counter += 1

        # Plain: 所有其他实体（统一走 load_objects）
        else:
            _append_plain_like_plan(
                plans, uid, spec, class_name, properties,
                small_counter, fixture_counter, attach_to_arena=(class_name in ATTACH_TO_ARENA_CLASSES),
            )
            if class_name in FIXTURE_WITH_HANDLE_CLASSES:
                fixture_counter += 1
            else:
                small_counter += 1

    return plans


def _inject_tube_stand(asset_status: Dict, parent_spec: str = "chemistry_tube_stand") -> None:
    xml_paths = {
        "chemistry_tube_stand": "obj/meshes/tube/tube_container/tube_stand.xml",
        "chemistry_tube_rack": "review/chemistry_tube_rack/chemistry_tube_rack/chemistry_tube_rack.xml",
    }
    class_names = {
        "chemistry_tube_stand": "TubeStand",
        "chemistry_tube_rack": "MediumTubeStand",
    }
    asset_status[parent_spec] = {
        "xml_path": xml_paths.get(parent_spec, xml_paths["chemistry_tube_stand"]),
        "class_name": class_names.get(parent_spec, "TubeStand"),
        "properties": {},
    }


def _inject_funnel_support(asset_status: Dict, parent_spec: str = "funnel_support") -> None:
    """注入 funnel_support 到 asset_status（如果不在 name2class_xml 中）"""
    asset_status[parent_spec] = {
        "xml_path": "review/universal_support/universal_support/universal_support/universal_support.xml",
        "class_name": "FunnelSupport",
        "properties": {},
    }


def _inject_pipettes_stand(asset_status: Dict, parent_spec: str = "pipettes_stand") -> None:
    """注入 pipettes_stand 到 asset_status（如果不在 name2class_xml 中）"""
    asset_status[parent_spec] = {
        "xml_path": "review/pipettes_stand/pipettes_stand-ver-/pipettes_stand-ver-.xml",
        "class_name": "PipetteStand",
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
    needs_pipette_stand_constants = False

    # 只处理 load_init_containers 和 load_objects（不再生成 load_containers）
    for method_name in ["load_init_containers", "load_objects"]:
        method_plans = methods.get(method_name, [])
        if not method_plans:
            continue
        code, flags = _generate_method(method_name, method_plans)
        code_parts.append(code)
        needs_name2class_xml = needs_name2class_xml or flags.get("needs_name2class_xml", False)
        needs_tube_constants = needs_tube_constants or flags.get("needs_tube_constants", False)
        needs_pipette_stand_constants = needs_pipette_stand_constants or flags.get("needs_pipette_stand_constants", False)

    load_methods_code = "\n".join(code_parts)

    extra_imports = None
    if needs_name2class_xml:
        extra_imports = "from VLABench.configs.constant import name2class_xml"

    extra_constants_lines = []
    if needs_tube_constants:
        extra_constants_lines.append(f"relative_col_pos = {TUBE_COL_POS}")
        extra_constants_lines.append(f"relative_row_pos = {TUBE_ROW_POS}")
    if needs_pipette_stand_constants:
        extra_constants_lines.append(f"relative_pipette_stand_col_pos = {PIPETTE_STAND_COL_POS}")
        extra_constants_lines.append(f"relative_pipette_stand_row_pos = {PIPETTE_STAND_ROW_POS}")
    extra_constants = "\n".join(extra_constants_lines) + "\n" if extra_constants_lines else None

    return load_methods_code, extra_imports, extra_constants


def _generate_method(method_name: str, plans: List[EntityLoadPlan]) -> tuple:
    """生成单个 load 方法"""
    flags = {"needs_name2class_xml": False, "needs_tube_constants": False, "needs_pipette_stand_constants": False}

    if method_name == "load_init_containers":
        return _gen_init_containers(plans, flags)
    return _gen_objects(plans, flags)


# ── load_init_containers ──────────────────────────────────────────────────

def _gen_init_containers(plans: List[EntityLoadPlan], flags: Dict) -> str:
    """
    生成 load_init_containers 方法。
    用于 ChemistryTube 和 Funnel 的父容器（TubeStand / FunnelSupport）。
    """
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

    # 覆盖 target_entity 为第一个 load_objects 计划的 uid，
    # 确保 self.target_entity 与 self.entities 中的 key 一致
    first_uid = next((p.uid for p in plans), None)
    if first_uid:
        lines.append(f'        self.target_entity = "{first_uid}"')
        lines.append("")

    flags["needs_name2class_xml"] = True
    return "\n".join(lines), flags


def _code_plain(plan: EntityLoadPlan, flags: Dict) -> List[str]:
    # 固定位置优先（由视觉标定过的物体）
    if plan.class_name in FIXED_POSITIONS:
        fx, fy = FIXED_POSITIONS[plan.class_name]
        position_line = f'            position=[{fx}, {fy}, 0.8],'
    else:
        if plan.class_name in FIXTURE_WITH_HANDLE_CLASSES:
            pos_table = FIXTURE_RANGES
        else:
            pos_table = SMALL_LABWARE_RANGES
        pos_range = pos_table[plan.position_index % len(pos_table)]
        position_line = (
            f'            position=[random.uniform({pos_range[0][0]}, {pos_range[0][1]}), '
            f'random.uniform({pos_range[1][0]}, {pos_range[1][1]}), 0.8],'
        )

    lines = [
        f'        obj_config = dict(',
        f'            name="{plan.uid}",',
        f'            xml_path=name2class_xml["{plan.spec}"][-1],',
        position_line,
        f'        )',
        f'        obj_config["class"] = "{plan.class_name}"',
    ]
    # 大件：门/把手朝向机械臂（绕 z 轴旋转，dict 构造完之后追加）
    if plan.orientation_yaw != 0.0:
        lines.append(f'        obj_config["orientation"] = [0, 0, {plan.orientation_yaw:.4f}]')
    lines.append(f'        obj_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0.05])')
    if plan.attach_to_arena:
        lines.append(f'        obj_config["attach_to_arena"] = True')
    lines += [
        f'        self.config["task"]["components"].append(obj_config)',
        "",
    ]
    return lines


def _code_liquid(plan: EntityLoadPlan, flags: Dict) -> List[str]:
    solution_rgba = plan.properties.get("solution_rgba")
    solution = plan.properties.get("solution", plan.uid)
    if plan.class_name in FIXTURE_WITH_HANDLE_CLASSES:
        pos_table = FIXTURE_RANGES
    else:
        pos_table = SMALL_LABWARE_RANGES
    pos_range = pos_table[plan.position_index % len(pos_table)]
    lines = [
        f'        obj_config = dict(',
        f'            name="{plan.uid}",',
        f'            xml_path=name2class_xml["{plan.spec}"][-1],',
        f'            position=[random.uniform({pos_range[0][0]}, {pos_range[0][1]}), random.uniform({pos_range[1][0]}, {pos_range[1][1]}), 0.8],',
        f'            solution="{solution}",',
    ]
    if solution_rgba:
        lines.append(f'            solution_rgba={solution_rgba},')
    lines.append(f'        )')
    lines.append(f'        obj_config["class"] = "{plan.class_name}"')
    # 大件：门/把手朝向机械臂（绕 z 轴旋转，单独写赋值语句避免破坏 dict literal）
    if plan.orientation_yaw != 0.0:
        lines.append(f'        obj_config["orientation"] = [0, 0, {plan.orientation_yaw:.4f}]')
    lines.append(f'        obj_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0.05])')
    if plan.attach_to_arena:
        lines.append(f'        obj_config["attach_to_arena"] = True')
    lines += [
        f'        self.config["task"]["components"].append(obj_config)',
        "",
    ]
    return lines


def _code_subentity(plan: EntityLoadPlan, flags: Dict) -> List[str]:
    """生成子实体代码，支持 ChemistryTube 和 Funnel"""
    solution_rgba = plan.properties.get("solution_rgba")
    solution = plan.properties.get("solution", plan.uid)

    # Funnel 使用固定的铁环位置
    if plan.spec == "funnel":
        pos = plan.subentity_position if plan.subentity_position else FUNNEL_SUBENTITY_POSITION
        return [
            f'        init_container_config = self.config["task"]["components"][-1]',
            '        if "subentities" not in init_container_config:',
            '            init_container_config["subentities"] = []',
            f'        funnel_config = dict(',
            f'            name="{plan.uid}",',
            f'            xml_path=name2class_xml["{plan.spec}"][-1],',
            f'            position={pos},',
            f'        )',
            f'        funnel_config["class"] = "{plan.class_name}"',
            '        init_container_config["subentities"].append(funnel_config)',
            "",
        ]

    # ChemistryTube 使用试管架孔位布局
    if solution_rgba:
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
            f'            solution_rgba={solution_rgba},',
            f'            xml_path=name2class_xml["{plan.spec}"][-1],',
            f'            position=pos,',
            f'        )',
            f'        obj_config["class"] = "{plan.class_name}"',
            '        init_container_config["subentities"].append(obj_config)',
            "",
        ]
    elif plan.spec in ("pipette", "mechanical_pipette"):
        if plan.spec == "mechanical_pipette":
            # mechanical_pipette uses pipette_stand's 1x4 slot layout
            pos_lines = [
                "        col_pos = random.choice(relative_pipette_stand_col_pos)",
                "        row_pos = random.choice(relative_pipette_stand_row_pos)",
            ]
            flags["needs_pipette_stand_constants"] = True
            # mechanical_pipette: position relative to stand body origin
            # Pipette body sits in the stand's top groove at Z=0.077 (local stand frame)
            # X offset 0.1 accounts for the pipette tilt direction
            # entity.py adds parent's init_pos, so world Z = 0.077 + stand_init_z (≈0.8) = 0.877 (groove)
            pos_line_value = "[0.1, -0.0363, 0.077]"
        else:
            pos_lines = [
                "        col_pos = random.choice(relative_col_pos)",
                "        row_pos = random.choice(relative_row_pos)",
            ]
            pos_line_value = "[col_pos, row_pos, 0.05]"
        return pos_lines + [
            f"        pos = {pos_line_value}",
            '        init_container_config = self.config["task"]["components"][-1]',
            '        if "subentities" not in init_container_config:',
            '            init_container_config["subentities"] = []',
            f'        obj_config = dict(',
            f'            name="{plan.uid}",',
            f'            xml_path=name2class_xml["{plan.spec}"][-1],',
            f'            position=pos,',
            f'        )',
            f'        obj_config["class"] = "{plan.class_name}"',
            '        init_container_config["subentities"].append(obj_config)',
            "",
        ]
    else:
        # ChemistryTube (no solution)
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
            f'            xml_path=name2class_xml["{plan.spec}"][-1],',
            f'            position=pos,',
            f'        )',
            f'        obj_config["class"] = "{plan.class_name}"',
            '        init_container_config["subentities"].append(obj_config)',
            "",
        ]
