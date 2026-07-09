import random
import numpy as np
from functools import partial

from VLABench.tasks.dm_task import *
from VLABench.tasks.autogen_tasks.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
from VLABench.configs.constant import name2class_xml

@register.add_config_manager("open_drawer")
class OpenDrawerConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=[1, 1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.config["task"]["n_distractor"] = 0

    def load_objects(self, target_entity):
        obj_config = dict(
            name="drawer_0",
            xml_path=name2class_xml["drawer"][-1],
            position=[random.uniform(0.30, 0.32), random.uniform(0.10, 0.12), 0.8],
        )
        obj_config["class"] = "ContainerWithDrawer"
        obj_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0.05])
        self.config["task"]["components"].append(obj_config)

        self.target_entity = "drawer_0"

    def get_instruction(self, target_entity, **kwargs):
        self.config["task"]["instructions"] = ["open the drawer_0"]

    def get_condition_config(self, target_entity, **kwargs):
        conditions_config = [
            dict(drawer_open=dict(entities=['drawer_0'])),
        ]
        self.config["task"]["conditions"] = conditions_config


@register.add_task("open_drawer")
class OpenDrawerTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.open_drawer, target_container_name="drawer_0", drawer_id=0),
        ]
        return skill_sequence
