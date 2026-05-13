import random
import numpy as np
from functools import partial

from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.primitive.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
from VLABench.configs.constant import name2class_xml

@register.add_config_manager("pick_stir")
class PickStirConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=[1, 1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.config["task"]["n_distractor"] = 0

    def load_objects(self, target_entity):
        obj_config = dict(
            name="glass_stirring_rod_0",
            xml_path=name2class_xml["glass_stirring_rod"][-1],
            position=[random.uniform(0.05, 0.15), random.uniform(-0.15, -0.05), 0.8],
        )
        obj_config["class"] = "CommonGraspedEntity"
        obj_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0.05])
        self.config["task"]["components"].append(obj_config)

        obj_config = dict(
            name="large_beaker_0",
            xml_path=name2class_xml["large_beaker"][-1],
            position=[random.uniform(0.35, 0.45), random.uniform(-0.05, 0.05), 0.8],
        )
        obj_config["class"] = "ChemistryBeaker"
        obj_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0.05])
        self.config["task"]["components"].append(obj_config)

    def get_instruction(self, target_entity, **kwargs):
        self.config["task"]["instructions"] = ["Pick the <glass_stirring_rod_0>."]

    def get_condition_config(self, target_entity, **kwargs):
        # 执行完即成功
        pass

    def get_target_entity(self):
        return "glass_stirring_rod_0"


@register.add_task("pick_stir")
class PickStirTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name="glass_stirring_rod_0"),
            partial(SkillLib.lift, lift_height=0.05),
            partial(SkillLib.stir_entity_with_tool, target_container_name="large_beaker_0", stir_radius=0.02, stir_duration=5, insert_ratio=0.6667),
        ]
        return skill_sequence
