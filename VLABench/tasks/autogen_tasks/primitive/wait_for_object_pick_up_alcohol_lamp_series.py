import random
import numpy as np
from functools import partial

from VLABench.tasks.dm_task import *
from VLABench.tasks.autogen_tasks.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
from VLABench.configs.constant import name2class_xml

@register.add_config_manager("wait_for_object_pick_up_alcohol_lamp")
class WaitForObjectPickUpAlcoholLampConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=[1, 1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.config["task"]["n_distractor"] = 0

    def load_objects(self, target_entity):
        obj_config = dict(
            name="alcohol_lamp_0",
            xml_path=name2class_xml["alcohol_lamp"][-1],
            position=[random.uniform(0.05, 0.15), random.uniform(-0.15, -0.05), 0.8],
        )
        obj_config["class"] = "AlcoholLamp"
        obj_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0.05])
        self.config["task"]["components"].append(obj_config)

        self.target_entity = "alcohol_lamp_0"

    def get_instruction(self, target_entity, **kwargs):
        self.config["task"]["instructions"] = ["Wait for the alcohol lamp to be lit up."]

    def get_condition_config(self, target_entity, **kwargs):
        conditions_config = [
            dict(is_grasped=dict(entities=['alcohol_lamp_0'], robot='robot')),
        ]
        self.config["task"]["conditions"] = conditions_config


@register.add_task("wait_for_object_pick_up_alcohol_lamp")
class WaitForObjectPickUpAlcoholLampTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.wait_for, wait_duration=2.0, entity_name="alcohol_lamp_0", change_type="Light_the_alcohol_lamp"),
            partial(SkillLib.pick, target_entity_name="alcohol_lamp_0", prior_eulers=[[-3.141592653589793, 0, 0]]),
        ]
        return skill_sequence
