import random
import numpy as np
from functools import partial

from VLABench.tasks.dm_task import *
from VLABench.tasks.autogen_tasks.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
from VLABench.configs.constant import name2class_xml

@register.add_config_manager("open_drying_box")
class OpenDryingBoxConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=[1, 1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.config["task"]["n_distractor"] = 0

    def load_objects(self, target_entity):
        obj_config = dict(
            name="drying_box_0",
            xml_path=name2class_xml["drying_box"][-1],
            position=[0.2, 0.5, 0.8],
        )
        obj_config["class"] = "DryingBoxWithButton"
        obj_config["orientation"] = [0, 0, 1.5708]
        obj_config["attach_to_arena"] = True
        self.config["task"]["components"].append(obj_config)

        self.target_entity = "drying_box_0"

    def get_instruction(self, target_entity, **kwargs):
        self.config["task"]["instructions"] = ["Open the <drying_box_0>."]

    def get_condition_config(self, target_entity, **kwargs):
        conditions_config = [
            dict(press_button=dict(target_button='drying_box_0')),
        ]
        self.config["task"]["conditions"] = conditions_config


@register.add_task("open_drying_box")
class OpenDryingBoxTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.open_door, target_container_name="drying_box_0"),
        ]
        return skill_sequence
