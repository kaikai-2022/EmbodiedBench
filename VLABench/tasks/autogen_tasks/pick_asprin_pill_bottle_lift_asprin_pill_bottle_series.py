import random
import numpy as np
from functools import partial

from VLABench.tasks.dm_task import *
from VLABench.tasks.autogen_tasks.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
from VLABench.configs.constant import name2class_xml

@register.add_config_manager("pick_asprin_pill_bottle_lift_asprin_pill_bottle")
class PickAsprinPillBottleLiftAsprinPillBottleConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=[1, 1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.config["task"]["n_distractor"] = 0

    def load_objects(self, target_entity):
        obj_config = dict(
            name="aspirin_pill_bottle_0",
            xml_path=name2class_xml["aspirin_pill_bottle"][-1],
            position=[random.uniform(0.05, 0.15), random.uniform(-0.15, -0.05), 0.8],
        )
        obj_config["class"] = "ContainerWithCap"
        obj_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0.05])
        obj_config["attach_to_arena"] = True
        self.config["task"]["components"].append(obj_config)

        self.target_entity = "aspirin_pill_bottle_0"

    def get_instruction(self, target_entity, **kwargs):
        self.config["task"]["instructions"] = ["pick the <aspirin_pill_bottle_0>."]

    def get_condition_config(self, target_entity, **kwargs):
        conditions_config = [
            dict(is_grasped=dict(entities=['aspirin_pill_bottle_0'], robot='robot')),
            dict(lift=dict(entities=['aspirin_pill_bottle_0'], lift_height=0.15)),
        ]
        self.config["task"]["conditions"] = conditions_config


@register.add_task("pick_asprin_pill_bottle_lift_asprin_pill_bottle")
class PickAsprinPillBottleLiftAsprinPillBottleTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name="aspirin_pill_bottle_0"),
            partial(SkillLib.lift, lift_height=0.15),
        ]
        return skill_sequence
