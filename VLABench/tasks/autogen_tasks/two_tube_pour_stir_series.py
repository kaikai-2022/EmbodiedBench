import random
import numpy as np
from functools import partial

from VLABench.tasks.dm_task import *
from VLABench.tasks.autogen_tasks.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
from VLABench.configs.constant import name2class_xml

relative_col_pos = [-0.16, -0.08, 0, 0.08, 0.16]
relative_row_pos = [-0.05, 0.05]


@register.add_config_manager("two_tube_pour_stir")
class TwoTubePourStirConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=[1, 1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.config["task"]["n_distractor"] = 0

    def load_init_containers(self, init_container):
        if init_container is None or init_container == "chemistry_tube_stand":
            container_config = dict(
                name="chemistry_tube_stand_0",
                xml_path=name2class_xml["chemistry_tube_stand"][-1],
                position=[random.uniform(-0.20, -0.10), random.uniform(0.05, 0.10), 0.8],
            )
            container_config["class"] = "TubeStand"
            self.config["task"]["components"].append(container_config)

    def load_objects(self, target_entity):
        init_container_config = self.config["task"]["components"][-1]
        if "subentities" not in init_container_config:
            init_container_config["subentities"] = []

        cuso4_col = random.choice(relative_col_pos)
        cuso4_row = random.choice(relative_row_pos)
        obj_config = dict(
            name="tube_cuso4_0",
            solution_rgba=[0.0, 0.45, 1.0, 0.4],
            xml_path=name2class_xml["tube"][-1],
            position=[cuso4_col, cuso4_row, 0.05],
        )
        obj_config["class"] = "ChemistryTube"
        init_container_config["subentities"].append(obj_config)

        fecl2_col = random.choice([c for c in relative_col_pos if c != cuso4_col])
        fecl2_row = random.choice(relative_row_pos)
        obj_config = dict(
            name="tube_fecl2_0",
            solution_rgba=[0.2, 0.8, 0.3, 0.4],
            xml_path=name2class_xml["tube"][-1],
            position=[fecl2_col, fecl2_row, 0.05],
        )
        obj_config["class"] = "ChemistryTube"
        init_container_config["subentities"].append(obj_config)

        obj_config = dict(
            name="large_beaker_0",
            xml_path=name2class_xml["large_beaker"][-1],
            position=[random.uniform(0.20, 0.25), random.uniform(0.08, 0.12), 0.8],
        )
        obj_config["class"] = "ChemistryBeaker"
        obj_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0.05])
        self.config["task"]["components"].append(obj_config)

        obj_config = dict(
            name="glass_stirring_rod_0",
            xml_path=name2class_xml["glass_stirring_rod"][-1],
            position=[random.uniform(0.30, 0.40), random.uniform(-0.05, 0.05), 0.8],
        )
        obj_config["class"] = "CommonGraspedEntity"
        obj_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0.05])
        self.config["task"]["components"].append(obj_config)

        self.target_entity = "tube_cuso4_0"

    def get_instruction(self, target_entity, init_container, **kwargs):
        self.config["task"]["instructions"] = [
            "pick the <tube_cuso4_0> which contains <CuSO4_0> and pour <CuSO4_0> from <tube_cuso4_0> into <large_beaker_0>, "
            "then insert <tube_cuso4_0> back to the <chemistry_tube_stand_0>. "
            "After that, pick the <tube_fecl2_0> which contains <FeCl2_0> and pour <FeCl2_0> from <tube_fecl2_0> into <large_beaker_0>, "
            "then insert <tube_fecl2_0> back to the <chemistry_tube_stand_0>. "
            "Finally, pick the <glass_stirring_rod_0> and stir the solution in the <large_beaker_0>."
        ]

    def get_condition_config(self, target_entity, init_container, **kwargs):
        conditions_config = [
            dict(is_grasped=dict(entities=['tube_cuso4_0'], robot='robot')),
            dict(pour_into=dict(
                target_entity='tube_cuso4_0',
                receiver_container='large_beaker_0',
                robot='robot',
                tilt_threshold=0,
                z_clearance=0.01,
            )),
            dict(contain=dict(container='chemistry_tube_stand_0', entities=['tube_cuso4_0'])),
            dict(is_grasped=dict(entities=['tube_fecl2_0'], robot='robot')),
            dict(pour_into=dict(
                target_entity='tube_fecl2_0',
                receiver_container='large_beaker_0',
                robot='robot',
                tilt_threshold=0,
                z_clearance=0.01,
            )),
            dict(contain=dict(container='chemistry_tube_stand_0', entities=['tube_fecl2_0'])),
            dict(stir=dict(
                entities=['glass_stirring_rod_0'],
                container='large_beaker_0',
                robot='robot',
            )),
        ]
        self.config["task"]["conditions"] = conditions_config


@register.add_task("two_tube_pour_stir")
class TwoTubePourStirTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name="tube_cuso4_0"),
            partial(SkillLib.lift, lift_height=0.15),
            partial(SkillLib.pour_to_entity, target_container_name="large_beaker_0", tilt_angle=2.1, wait_time=10),
            partial(SkillLib.insert_to_entity, target_entity_name="chemistry_tube_stand_0", insert_depth=0.05),
            partial(SkillLib.pick, target_entity_name="tube_fecl2_0"),
            partial(SkillLib.lift, lift_height=0.15),
            partial(SkillLib.pour_to_entity, target_container_name="large_beaker_0", tilt_angle=2.1, wait_time=10),
            partial(SkillLib.insert_to_entity, target_entity_name="chemistry_tube_stand_0", insert_depth=0.05),
            partial(SkillLib.pick, target_entity_name="glass_stirring_rod_0"),
            partial(SkillLib.lift, lift_height=0.15),
            partial(SkillLib.stir_entity_with_tool, target_container_name="large_beaker_0", stir_radius=0.02, stir_duration=5, insert_ratio=0.6666666666666666),
            partial(SkillLib.drop),
        ]
        return skill_sequence
