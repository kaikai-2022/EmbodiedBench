import random
import numpy as np
from functools import partial

from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.primitive.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register

@register.add_config_manager("lift_petri_dish")
class LiftPetriDishConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=[1, 1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)


    def get_instruction(self, target_entity, **kwargs):
        instruction = ["Lift the {target_entity}"]
        self.config["task"]["instructions"] = instruction

    def get_condition_config(self, target_entity, **kwargs):
        conditions_config = dict(
            lift=dict(entities=["petri_dish"], target_height=0.9)
        )
        self.config["task"]["conditions"] = conditions_config

    def get_target_entity(self):
        return "petri_dish"



@register.add_task("lift_petri_dish")
class LiftPetriDishTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name=self.target_entity, prior_eulers=[[-3.14159, 0, 0]]),
            partial(SkillLib.lift, lift_height=0.2, gripper_state=np.zeros(2)),        ]
        return skill_sequence
