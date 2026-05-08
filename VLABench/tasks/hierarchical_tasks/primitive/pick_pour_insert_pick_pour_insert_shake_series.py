import random
import numpy as np
from functools import partial

from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.primitive.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
from VLABench.configs.constant import name2class_xml

relative_col_pos = [-0.16, -0.08, 0, 0.08, 0.16]
relative_row_pos = [-0.05, 0.05]

@register.add_config_manager("pick_pour_insert_pick_pour_insert_shake")
class PickPourInsertPickPourInsertShakeConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=[1, 1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)

    def load_containers(self, target_container):
        if target_container is None:
            return
        container_config = dict(
            name="beaker_0",
            xml_path=name2class_xml["beaker"][-1],
            position=[random.uniform(0.2, 0.28), random.uniform(-0.1, 0.0), 0.8],
        )
        container_config["class"] = "ChemistryBeaker"
        self.config["task"]["components"].append(container_config)

    def load_init_containers(self, init_container):
        if init_container is None or init_container == "chemistry_tube_stand":
            # 创建默认父容器: TubeStand for ChemistryTube
            container_config = dict(
                name="chemistry_tube_stand_0",
                xml_path=name2class_xml["chemistry_tube_stand"][-1],
                position=[random.uniform(-0.15, -0.05), random.uniform(0.05, 0.15), 0.8],
            )
            container_config["class"] = "TubeStand"
            self.config["task"]["components"].append(container_config)

    def load_objects(self, target_entity):
        col_pos = random.choice(relative_col_pos)
        row_pos = random.choice(relative_row_pos)
        pos = [col_pos, row_pos, 0.05]
        init_container_config = self.config["task"]["components"][-1]
        if "subentities" not in init_container_config:
            init_container_config["subentities"] = []
        obj_config = dict(
            name="tube_0",
            solution="CuSO4",
            xml_path=name2class_xml["tube"][-1],
            position=pos,
        )
        obj_config["class"] = "ChemistryTube"
        init_container_config["subentities"].append(obj_config)

        col_pos = random.choice(relative_col_pos)
        row_pos = random.choice(relative_row_pos)
        pos = [col_pos, row_pos, 0.05]
        init_container_config = self.config["task"]["components"][-1]
        if "subentities" not in init_container_config:
            init_container_config["subentities"] = []
        obj_config = dict(
            name="tube_1",
            solution="FeCl3",
            xml_path=name2class_xml["tube"][-1],
            position=pos,
        )
        obj_config["class"] = "ChemistryTube"
        init_container_config["subentities"].append(obj_config)

    def get_instruction(self, target_entity, target_container, init_container, **kwargs):
        self.config["task"]["instructions"] = ["Pick <tube_0> from <chemistry_tube_stand_0>."]

    def get_condition_config(self, target_entity, target_container, init_container, **kwargs):
        # 执行完即成功
        pass

    def get_target_entity(self):
        return "tube_0"


@register.add_task("pick_pour_insert_pick_pour_insert_shake")
class PickPourInsertPickPourInsertShakeTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name="tube_0", prior_eulers=[[-3.14159, 0, 0]]),
            partial(SkillLib.lift, lift_height=0.15),
            partial(SkillLib.pour_to_entity, target_container_name="beaker_0", tilt_angle=1.5708, wait_time=10),
            partial(SkillLib.insert_to_entity, target_entity_name="chemistry_tube_stand_0", insert_depth=0.05),
            partial(SkillLib.pick, target_entity_name="tube_1", prior_eulers=[[-3.14159, 0, 0]]),
            partial(SkillLib.lift, lift_height=0.15),
            partial(SkillLib.pour_to_entity, target_container_name="beaker_0", tilt_angle=1.5708, wait_time=10),
            partial(SkillLib.insert_to_entity, target_entity_name="chemistry_tube_stand_0", insert_depth=0.05),
            partial(SkillLib.pick, target_entity_name="beaker_0", prior_eulers=[[-3.14159, 0, 0]]),
            partial(SkillLib.lift, lift_height=0.15),
            partial(SkillLib.rotate, rotation_angle=0.7854),
            partial(SkillLib.rotate, rotation_angle=-0.7854),
            partial(SkillLib.rotate, rotation_angle=0.7854),
            partial(SkillLib.rotate, rotation_angle=-0.7854),
            partial(SkillLib.rotate, rotation_angle=0.7854),
            partial(SkillLib.rotate, rotation_angle=-0.7854),
            partial(SkillLib.wait, wait_time=50),
            partial(SkillLib.open_gripper),
        ]
        return skill_sequence
