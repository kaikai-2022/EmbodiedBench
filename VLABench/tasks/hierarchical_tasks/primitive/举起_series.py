import random
import numpy as np
from functools import partial

from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.primitive.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register

@register.add_config_manager("举起")
class 举起ConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=[1, 1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)

        pass
    def get_instruction(self, target_entity, **kwargs):
        instruction = ["举起 <beaker_0>。"]
        self.config["task"]["instructions"] = instruction

    def get_condition_config(self, target_entity, **kwargs):
        # 执行完即成功
        pass

    def get_target_entity(self):
        return "target_entity"


@register.add_task("举起")
class 举起Task(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.pick, target_uid='beaker_0', prior_eulers=[[-3.14159, 0, 0]]),
            partial(SkillLib.lift, lift_height=0.15, gripper_state=[0, 0]),
        ]
        return skill_sequence
