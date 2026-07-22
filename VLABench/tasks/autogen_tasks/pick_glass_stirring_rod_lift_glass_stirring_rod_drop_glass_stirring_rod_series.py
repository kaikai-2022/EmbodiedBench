import random
import numpy as np
from functools import partial

from VLABench.tasks.dm_task import *
from VLABench.tasks.autogen_tasks.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
from VLABench.configs.constant import name2class_xml

@register.add_config_manager("pick_glass_stirring_rod_lift_glass_stirring_rod_drop_glass_stirring_rod")
class PickGlassStirringRodLiftGlassStirringRodDropGlassStirringRodConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=[1, 1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.config["task"]["n_distractor"] = 0

    def load_objects(self, target_entity):
        obj_config = dict(
            name="glass_stirring_rod_0",
            xml_path=name2class_xml["glass_stirring_rod"][-1],
            position=[random.uniform(-0.3, -0.15), random.uniform(-0.05, 0.1), 0.8],
        )
        obj_config["class"] = "CommonGraspedEntity"
        obj_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0.05])
        self.config["task"]["components"].append(obj_config)

        self.target_entity = "glass_stirring_rod_0"

    def get_instruction(self, target_entity, **kwargs):
        self.config["task"]["instructions"] = ["Pick the <glass_stirring_rod_0>."]

    def get_condition_config(self, target_entity, **kwargs):
        conditions_config = [
            dict(is_grasped=dict(entities=['glass_stirring_rod_0'], robot='robot')),
            dict(lift=dict(entities=['glass_stirring_rod_0'], lift_height=0.15)),
        ]
        self.config["task"]["conditions"] = conditions_config


@register.add_task("pick_glass_stirring_rod_lift_glass_stirring_rod_drop_glass_stirring_rod")
class PickGlassStirringRodLiftGlassStirringRodDropGlassStirringRodTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name="glass_stirring_rod_0", prior_eulers=[[-3.14159, 0, 0]]),
            partial(SkillLib.lift, lift_height=0.15, gripper_state=[0, 0]),
            partial(SkillLib.drop),
        ]
        return skill_sequence
