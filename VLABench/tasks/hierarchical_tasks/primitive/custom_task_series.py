import random
import numpy as np
from functools import partial

from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.primitive.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register

@register.add_config_manager("custom_task")
class CustomTaskConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=[1, 1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)


    def get_instruction(self, target_entity, **kwargs):
        self.config["task"]["instructions"] = ["Perform the task"]

    def get_condition_config(self, target_entity, **kwargs):
        # 执行完即成功
        pass

    def get_target_entity(self):
        return ""


@register.add_task("custom_task")
class CustomTaskTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        skill_sequence = [

        ]
        return skill_sequence
