import random
import numpy as np
from functools import partial

from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.primitive.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
from VLABench.configs.constant import name2class_xml

@register.add_config_manager("pour")
class PourConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=[1, 1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)

    def load_containers(self, target_container):
        if target_container is None:
            return
        container_config = dict(
            name="beaker_1",
            xml_path=name2class_xml["beaker"][-1],
            position=[random.uniform(0.2, 0.28), random.uniform(-0.1, 0.0), 0.8],
        )
        container_config["class"] = "ChemistryBeaker"
        self.config["task"]["components"].append(container_config)

    def load_objects(self, target_entity):
        obj_config = dict(
            name="beaker_0",
            xml_path=name2class_xml["beaker"][-1],
            position=[random.uniform(-0.3, 0.3), random.uniform(-0.2, 0.2), 0.8],
            solution="CuSO4 solution",
        )
        obj_config["class"] = "ChemistryBeaker"
        obj_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0.05])
        self.config["task"]["components"].append(obj_config)

    def get_instruction(self, target_entity, target_container, **kwargs):
        self.config["task"]["instructions"] = ["Pour <liquid_0> from <beaker_0> into <beaker_1>."]

    def get_condition_config(self, target_entity, target_container, **kwargs):
        # 执行完即成功
        pass

    def get_target_entity(self):
        return "beaker_0"


@register.add_task("pour")
class PourTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name="beaker_0"),
            partial(SkillLib.lift, lift_height=0.15),
            partial(SkillLib.pour_to_entity, target_container_name="beaker_1", tilt_angle=1.5707963, wait_time=10),
            partial(SkillLib.open_gripper),
        ]
        return skill_sequence
