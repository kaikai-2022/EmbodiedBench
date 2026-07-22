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

@register.add_config_manager("pick_tube_shake_tube")
class PickTubeShakeTubeConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=[1, 1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.config["task"]["n_distractor"] = 0

    def load_init_containers(self, init_container):
        if init_container is None or init_container == "chemistry_tube_stand":
            container_config = dict(
                name="chemistry_tube_stand",
                xml_path=name2class_xml["chemistry_tube_stand"][-1],
                position=[random.uniform(-0.1, 0.01), random.uniform(0.05, 0.15), 0.8],
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
            solution="tube_0",
            xml_path=name2class_xml["tube"][-1],
            position=pos,
        )
        obj_config["class"] = "ChemistryTube"
        init_container_config["subentities"].append(obj_config)

        self.target_entity = "tube_0"

    def get_instruction(self, target_entity, init_container, **kwargs):
        self.config["task"]["instructions"] = ["Pick the <tube_0>."]

    def get_condition_config(self, target_entity, init_container, **kwargs):
        conditions_config = [
            dict(is_grasped=dict(entities=['tube_0'], robot='robot')),
            dict(shake=dict(
            entities=['tube_0'],
            robot='robot',
            min_direction_changes=3,
            min_angle_threshold=0.1,
            check_axis=1,
        )),
        ]
        self.config["task"]["conditions"] = conditions_config


@register.add_task("pick_tube_shake_tube")
class PickTubeShakeTubeTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name="tube_0", prior_eulers=[[-3.14159, 0, 0]]),
            partial(SkillLib.lift, lift_height=0.15, gripper_state=[0, 0]),
            partial(SkillLib.shake, n_shakes=3, shake_angle=0.7, steps_per_swing=5),
            partial(SkillLib.insert_to_entity, target_entity_name="chemistry_tube_stand", insert_depth=0.05),
        ]
        return skill_sequence
