import random
import numpy as np
from functools import partial

from VLABench.tasks.dm_task import *
from VLABench.tasks.autogen_tasks.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
from VLABench.configs.constant import name2class_xml

@register.add_config_manager("place_beaker_on_the_mat")
class PlaceBeakerOnTheMatConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=[1, 1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.config["task"]["n_distractor"] = 0

    def load_objects(self, target_entity):
        obj_config = dict(
            name="small_beaker_0",
            xml_path=name2class_xml["small_beaker"][-1],
            position=[random.uniform(0.05, 0.25), random.uniform(-0.15, 0.3), 0.8],
        )
        obj_config["class"] = "ChemistryBeaker"
        obj_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 3.14])
        self.config["task"]["components"].append(obj_config)

        obj_config = dict(
            name="square_mat_0",
            xml_path=name2class_xml["square_mat"][-1],
            position=[random.uniform(-0.25, -0.05), random.uniform(-0.15, 0.3), 0.8],
        )
        obj_config["class"] = "FlatContainer"
        obj_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0.05])
        self.config["task"]["components"].append(obj_config)

        self.target_entity = "small_beaker_0"

    def get_instruction(self, target_entity, **kwargs):
        self.config["task"]["instructions"] = ["pick the <small_beaker_0>."]

    def get_condition_config(self, target_entity, **kwargs):
        conditions_config = [
            dict(is_grasped=dict(entities=['small_beaker_0'], robot='robot')),
            dict(on=dict(entities=['small_beaker_0'], container='square_mat_0')),
        ]
        self.config["task"]["conditions"] = conditions_config


@register.add_task("place_beaker_on_the_mat")
class PlaceBeakerOnTheMatTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name="small_beaker_0"),
            partial(SkillLib.place, target_container_name="square_mat_0"),
        ]
        return skill_sequence
