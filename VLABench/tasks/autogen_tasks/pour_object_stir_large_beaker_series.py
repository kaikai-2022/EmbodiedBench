import random
import numpy as np
from functools import partial

from VLABench.tasks.dm_task import *
from VLABench.tasks.autogen_tasks.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
from VLABench.configs.constant import name2class_xml

@register.add_config_manager("pour_object_stir_large_beaker")
class PourObjectStirLargeBeakerConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=[1, 1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.config["task"]["n_distractor"] = 0

    def load_objects(self, target_entity):
        obj_config = dict(
            name="cylinder_mid_0",
            xml_path=name2class_xml["cylinder_mid"][-1],
            position=[random.uniform(-0.3, -0.2), random.uniform(-0.05, 0.1), 0.8],
            solution="CuSO4_solution_1",
            solution_rgba=[0.0, 0.45, 1.0, 0.4],
        )
        obj_config["class"] = "ChemistryBeaker"
        obj_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0.05])
        self.config["task"]["components"].append(obj_config)

        obj_config = dict(
            name="large_beaker_0",
            xml_path=name2class_xml["large_beaker"][-1],
            position=[random.uniform(-0.1, 0.05), random.uniform(-0.05, 0.1), 0.8],
            solution="NaOH_solution_1",
            solution_rgba=[1.0, 1.0, 1.0, 0.1],
        )
        obj_config["class"] = "ChemistryBeaker"
        obj_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0.05])
        self.config["task"]["components"].append(obj_config)

        obj_config = dict(
            name="glass_stirring_rod_0",
            xml_path=name2class_xml["glass_stirring_rod"][-1],
            position=[random.uniform(0.1, 0.25), random.uniform(-0.05, 0.1), 0.8],
        )
        obj_config["class"] = "CommonGraspedEntity"
        obj_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0.05])
        self.config["task"]["components"].append(obj_config)

        self.target_entity = "cylinder_mid_0"

    def get_instruction(self, target_entity, **kwargs):
        self.config["task"]["instructions"] = ["Pour the CuSO4 solution_0 in the cylinder_mid_0 into the large_beaker_0 which contains NaOH solution_0."]

    def get_condition_config(self, target_entity, **kwargs):
        conditions_config = [
            dict(pour_into=dict(
            target_entity='cylinder_mid_0',
            receiver_container='large_beaker_0',
            robot='robot',
        )),
            dict(stir=dict(
            entities=['glass_stirring_rod_0'],
            container='large_beaker_0',
            robot='robot',
        )),
        ]
        self.config["task"]["conditions"] = conditions_config


@register.add_task("pour_object_stir_large_beaker")
class PourObjectStirLargeBeakerTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name="cylinder_mid_0", prior_eulers=[[-3.141592653589793, 0, 0]]),
            partial(SkillLib.lift, lift_height=0.15, gripper_state=[0, 0]),
            partial(SkillLib.pour_to_entity, target_container_name="large_beaker_0", tilt_angle=1.8, wait_time=10),
            partial(SkillLib.drop),
            partial(SkillLib.pick, target_entity_name="glass_stirring_rod_0", prior_eulers=[[-3.141592653589793, 0, 0]]),
            partial(SkillLib.lift, lift_height=0.15, gripper_state=[0, 0]),
            partial(SkillLib.stir_entity_with_tool, target_container_name="large_beaker_0"),
            partial(SkillLib.drop),
        ]
        return skill_sequence
