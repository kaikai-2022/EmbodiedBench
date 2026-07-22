import random
import numpy as np
from functools import partial

from VLABench.tasks.dm_task import *
from VLABench.tasks.autogen_tasks.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
from VLABench.configs.constant import name2class_xml

@register.add_config_manager("pres_button")
class PresButtonConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=[1, 1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.config["task"]["n_distractor"] = 0

    def load_objects(self, target_entity):
        obj_config = dict(
            name="button_0",
            xml_path=name2class_xml["button"][-1],
            position=[random.uniform(0.05, 0.15), random.uniform(-0.15, -0.05), 0.8],
        )
        obj_config["class"] = "Button"
        obj_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0.05])
        self.config["task"]["components"].append(obj_config)

        obj_config = dict(
            name="drying_box_0",
            xml_path=name2class_xml["drying_box"][-1],
            position=[random.uniform(0.35, 0.45), random.uniform(-0.05, 0.05), 0.8],
        )
        obj_config["class"] = "DryingBoxWithButton"
        obj_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0.05])
        self.config["task"]["components"].append(obj_config)

        self.target_entity = "button_0"

    def get_instruction(self, target_entity, **kwargs):
        self.config["task"]["instructions"] = ["Press the <button_0> on the <drying_box_0>."]

    def get_condition_config(self, target_entity, **kwargs):
        conditions_config = [
            dict(press_button=dict(target_button='button_0')),
        ]
        self.config["task"]["conditions"] = conditions_config


@register.add_task("pres_button")
class PresButtonTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.moveto_entity, target_entity_name="button_0", offset=np.array([0, 0, 0.05])),
            partial(SkillLib.press, target_pos="button_0"),
        ]
        return skill_sequence
