import random
import numpy as np
from functools import partial

from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.primitive.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
from VLABench.configs.constant import name2class_xml

@register.add_config_manager("wait_for_object_lift_beaker")
class WaitForObjectLiftBeakerConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=[1, 1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.config["task"]["n_distractor"] = 0

    def load_objects(self, target_entity):
        obj_config = dict(
            name=target_entity,
            xml_path=name2class_xml["small_beaker"][-1],
            position=[random.uniform(0.05, 0.15), random.uniform(-0.15, -0.05), 0.8],
        )
        obj_config["class"] = "ChemistryBeaker"
        obj_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0.05])
        self.config["task"]["components"].append(obj_config)

    def get_instruction(self, target_entity, **kwargs):
        self.config["task"]["instructions"] = [f"Wait for human to add <CuSO4_0> in <{target_entity}>."]

    def get_condition_config(self, target_entity, **kwargs):
        conditions_config = dict(
            wait_for=dict(entity=target_entity, wait_duration=2.0, change_type="add_solution", solution="CuSO4"),
            lift=dict(entities=[target_entity], lift_height=0.15)
        )
        self.config["task"]["conditions"] = conditions_config


@register.add_task("wait_for_object_lift_beaker")
class WaitForObjectLiftBeakerTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.wait_for, wait_duration=3.0, entity_name=self.target_entity, change_type="add_solution", solution="CuSO4"),
            partial(SkillLib.pick, target_entity_name=self.target_entity, prior_eulers=[[-3.14159, 0, 0]]),
            partial(SkillLib.lift, lift_height=0.15),
        ]
        return skill_sequence
