import random
import numpy as np
from functools import partial

from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.primitive.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
from VLABench.configs.constant import name2class_xml

@register.add_config_manager("pick_pour_place")
class PickPourPlaceConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=[1, 1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.config["task"]["n_distractor"] = 0

    def load_objects(self, target_entity):
        obj_config = dict(
            name="flask_0",
            xml_path=name2class_xml["flask"][-1],
            position=[random.uniform(0.15, 0.25), random.uniform(-0.15, -0.05), 0.8],
        )
        obj_config["class"] = "ChemistryBeaker"
        obj_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0.05])
        self.config["task"]["components"].append(obj_config)

        obj_config = dict(
            name="small_beaker_0",
            xml_path=name2class_xml["small_beaker"][-1],
            position=[random.uniform(0.25, 0.35), random.uniform(-0.15, -0.05), 0.8],
        )
        obj_config["class"] = "ChemistryBeaker"
        obj_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0.05])
        self.config["task"]["components"].append(obj_config)

        obj_config = dict(
            name="table_0",
            xml_path=name2class_xml["table"][-1],
            position=[random.uniform(0.15, 0.25), random.uniform(0.05, 0.15), 0.8],
        )
        obj_config["class"] = "CommonContainer"
        obj_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0.05])
        self.config["task"]["components"].append(obj_config)

    def get_instruction(self, target_entity, **kwargs):
        self.config["task"]["instructions"] = ["Pick the <flask_0>."]

    def get_condition_config(self, target_entity, **kwargs):
        # 执行完即成功
        pass

    def get_target_entity(self):
        return "flask_0"


@register.add_task("pick_pour_place")
class PickPourPlaceTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name="flask_0", prior_eulers=[[-3.14159, 0, 0]]),
            partial(SkillLib.lift, lift_height=0.15),
            partial(SkillLib.pour_to_entity, target_container_name="small_beaker_0", tilt_angle=1.8, wait_time=10),
            partial(SkillLib.place, target_container_name="table_0"),
            partial(SkillLib.open_gripper),
        ]
        return skill_sequence
