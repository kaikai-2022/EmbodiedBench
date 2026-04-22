"""
Entity Loader - 结构化实体加载代码生成

按 Code Generator 重构设计规范实现：
  - 字段名统一读 class_name（与 asset_manager 输出对齐）
  - SubEntity 模式自动注入父容器（ChemistryTube → chemistry_tube_stand）
  - 所有模板使用手动 dict 构造，不调用 get_entity_config
  - UID 贯穿：entity name = uid（全局唯一标识符）

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


@dataclass
class EntityLoadPlan:
    """描述一个实体应该如何被加载"""
    uid: str
    spec: str
    class_name: str
    load_mode: str          # "plain" | "liquid" | "subentity"
    method_name: str         # load_objects / load_containers / load_init_containers
    properties: Dict = field(default_factory=dict)
    parent_spec: Optional[str] = None


def plan_entity_loading(
    instances: List[Dict],
    asset_status: Dict,
) -> List[EntityLoadPlan]:
    """
    为每个物理实体生成加载计划。

    规则:
      1. 跳过 is_physical=False 的实体
      2. ChemistryTube → subentity 模式，自动注入 tube_stand 父容器
      3. 带 solution → liquid 模式
      4. 其余 → plain 模式
    """
    plans: List[EntityLoadPlan] = []
    has_init_container = False
    has_container = False

    # 第一遍扫描：找出哪些 uid 已经在 instances 中（避免重复注入）
    existing_uids = set()
    existing_tube_stand_uid = None
    for inst in instances:
        if not inst.get("is_physical", True):
            continue
        uid = inst["uid"]
        existing_uids.add(uid)
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
            # 如果 instances 中已有 tube_stand，复用它作为父容器
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
            ))

        # Container: 容器类实体 → 始终用 load_containers（确保在 tube_stand 之前加载）
        # 这样 components 顺序是: [table, beaker, tube_stand, ...tubes]
        # 当 load_objects 中的 [-1] 访问时，指向 tube_stand
        elif _is_container_class(class_name, properties):
            plans.append(EntityLoadPlan(
                uid=uid, spec=spec, class_name=class_name,
                load_mode="plain", method_name="load_containers",
                properties=properties,
            ))

        # Plain: 普通可操作实体
        else:
            plans.append(EntityLoadPlan(
                uid=uid, spec=spec, class_name=class_name,
                load_mode="plain", method_name="load_objects",
                properties=properties,
            ))

    return plans


def _is_container_class(class_name: str, properties: Dict) -> bool:
    """判断是否为容器类"""
    container_classes = {
        "CommonContainer", "ContainerWithDoor", "ContainerWithDrawer",
        "FlatContainer", "Fridge", "Microwave", "Shelf",
        "Vase", "Plate", "Mug",
        # Chemistry containers
        "ChemistryBeaker", "ChemistryFlask", "ChemistryBottle",
        "Beaker", "Flask", "Bottle",
    }
    return class_name in container_classes or bool(properties.get("is_container"))


def _inject_tube_stand(asset_status: Dict) -> None:
    asset_status["chemistry_tube_stand"] = {
        "xml_path": "obj/meshes/tube/tube_container/tube_stand.xml",
        "class_name": "TubeStand",
        "properties": {},
    }


def generate_load_methods(plans: List[EntityLoadPlan], asset_status: Dict) -> tuple:
    """
    从加载计划生成 load 方法代码。

    框架调用顺序: load_containers → load_init_containers → load_objects
    所以生成顺序必须与之匹配，确保 components 数组中顺序正确：
    [table, beaker, tube_stand, ...tubes]

    Returns:
        (load_methods_code: str, extra_imports: str|None, extra_constants: str|None)
    """
    # 按 method_name 分组
    methods: Dict[str, List[EntityLoadPlan]] = {}
    for plan in plans:
        methods.setdefault(plan.method_name, []).append(plan)

    # 按框架调用顺序生成方法
    METHOD_ORDER = ["load_containers", "load_init_containers", "load_objects"]

    code_parts = []
    needs_name2class_xml = False
    needs_tube_constants = False

    for method_name in METHOD_ORDER:
        method_plans = methods.get(method_name, [])
        if not method_plans:
            continue
        code, flags = _generate_method(method_name, method_plans, asset_status)
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


def _generate_method(method_name: str, plans: List[EntityLoadPlan], asset_status: Dict) -> tuple:
    """生成单个 load 方法"""
    flags = {"needs_name2class_xml": False, "needs_tube_constants": False}

    if method_name == "load_init_containers":
        return _gen_init_containers(plans, flags)
    if method_name == "load_containers":
        return _gen_containers(plans, flags)
    return _gen_objects(plans, flags)


# ── load_init_containers ──────────────────────────────────────────────────

def _gen_init_containers(plans: List[EntityLoadPlan], flags: Dict) -> str:
    """
    生成 load_init_containers 方法。

    此方法用于创建子实体（如 ChemistryTube）的父容器（如 TubeStand）。
    只要有 ChemistryTube 实体需要加载，就必须创建对应的父容器，
    不能因为 init_container=None 就提前返回。

    未来扩展：如果需要支持其他 subentity 类型（如 ChemistryFlask -> FlaskRack），
    需要在此处根据 plan.class_name 推断对应的父容器类型，
    并在 DEFAULT_PARENT_CONTAINERS 映射表中添加新的映射关系。
    """
    # 默认父容器映射：子实体类名 -> (父容器spec, 父容器class_name)
    # TODO(扩展): 添加其他 subentity 类型的默认父容器，如：
    #   "ChemistryFlask": ("flask_rack", "FlaskRack")
    DEFAULT_PARENT_CONTAINERS = {
        "ChemistryTube": ("chemistry_tube_stand", "TubeStand"),
    }

    lines = ["    def load_init_containers(self, init_container):"]

    # 收集需要创建的默认父容器类型（去重）
    default_parents = set()
    for plan in plans:
        parent_info = DEFAULT_PARENT_CONTAINERS.get(
            plan.class_name,
            (plan.spec, plan.class_name)
        )
        default_parents.add(parent_info)

    for parent_spec, parent_class in default_parents:
        # 从 DEFAULT_PARENT_CONTAINERS 反查触发的子实体类名
        triggered_str = ""
        for child_class, (p_spec, p_class) in DEFAULT_PARENT_CONTAINERS.items():
            if p_spec == parent_spec and p_class == parent_class:
                triggered_str = child_class
                break
        lines += [
            f"        if init_container is None or init_container == \"{parent_spec}\":",
            f"            # 创建默认父容器: {parent_class} for {triggered_str}",
            f"            container_config = dict(",
            f"                name=\"{parent_spec}\",",
            f"                xml_path=name2class_xml[\"{parent_spec}\"][-1],",
            f"                position=[random.uniform(-0.15, -0.05), random.uniform(0.05, 0.15), 0.8],",
            f"            )",
            f"            container_config[\"class\"] = \"{parent_class}\"",
            f"            self.config[\"task\"][\"components\"].append(container_config)",
        ]

    lines.append("")
    flags["needs_name2class_xml"] = True
    return "\n".join(lines), flags


# ── load_containers ─────────────────────────────────────────────────────

def _gen_containers(plans: List[EntityLoadPlan], flags: Dict) -> str:
    """生成 load_containers 方法"""
    lines = [
        "    def load_containers(self, target_container):",
        "        if target_container is None:",
        "            return",
    ]
    for plan in plans:
        lines += [
            f'        container_config = dict(',
            f'            name="{plan.uid}",',
            f'            xml_path=name2class_xml["{plan.spec}"][-1],',
            f'            position=[random.uniform(0.2, 0.28), random.uniform(-0.1, 0.0), 0.8],',
            f'        )',
            f'        container_config["class"] = "{plan.class_name}"',
            f'        self.config["task"]["components"].append(container_config)',
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

    # plain/liquid/subentity 都用了 name2class_xml 查找
    flags["needs_name2class_xml"] = True

    return "\n".join(lines), flags


def _code_plain(plan: EntityLoadPlan, flags: Dict) -> List[str]:
    return [
        f'        obj_config = dict(',
        f'            name="{plan.uid}",',
        f'            xml_path=name2class_xml["{plan.spec}"][-1],',
        f'            position=[random.uniform(-0.3, 0.3), random.uniform(-0.2, 0.2), 0.8],',
        f'        )',
        f'        obj_config["class"] = "{plan.class_name}"',
        f'        obj_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0.05])',
        f'        self.config["task"]["components"].append(obj_config)',
        "",
    ]


def _code_liquid(plan: EntityLoadPlan, flags: Dict) -> List[str]:
    solution = plan.properties.get("solution", plan.uid)
    return [
        f'        obj_config = dict(',
        f'            name="{plan.uid}",',
        f'            xml_path=name2class_xml["{plan.spec}"][-1],',
        f'            position=[random.uniform(-0.3, 0.3), random.uniform(-0.2, 0.2), 0.8],',
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


def _get_asset_info(uid: str, spec: str, asset_status: Dict) -> Dict:
    """从 asset_status 获取实体的 xml_path（优先）或 name2class_xml 回退"""
    info = asset_status.get(uid, {})
    xml_path = info.get("xml_path")
    if xml_path:
        return {"xml_path": xml_path}
    return {"spec": spec}
