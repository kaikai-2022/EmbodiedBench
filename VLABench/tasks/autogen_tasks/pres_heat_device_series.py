import random
import numpy as np
from functools import partial

from VLABench.tasks.dm_task import *
from VLABench.tasks.autogen_tasks.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
from VLABench.configs.constant import name2class_xml

@register.add_config_manager("pres_heat_device")
class PresHeatDeviceConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=[1, 1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.config["task"]["n_distractor"] = 0

    def load_objects(self, target_entity):
        obj_config = dict(
            name="heat_device_0",
            xml_path=name2class_xml["heat_device"][-1],
            position=[random.uniform(-0.3, -0.15), random.uniform(-0.05, 0.1), 0.8],
        )
        obj_config["class"] = "HeatDevice"
        obj_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0.05])
        self.config["task"]["components"].append(obj_config)

        self.target_entity = "heat_device_0"

    def get_instruction(self, target_entity, **kwargs):
        self.config["task"]["instructions"] = ["press the <heat_device_0>"]

    def get_condition_config(self, target_entity, **kwargs):
        conditions_config = [
            dict(press_button=dict(target_button='heat_device_0')),
        ]
        self.config["task"]["conditions"] = conditions_config


@register.add_task("pres_heat_device")
class PresHeatDeviceTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.press, target_pos="heat_device_0"),
        ]
        return skill_sequence
