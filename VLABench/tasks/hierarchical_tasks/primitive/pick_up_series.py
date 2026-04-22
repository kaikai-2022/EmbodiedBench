import random
import numpy as np
from functools import partial

from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.primitive.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
from VLABench.configs.constant import name2class_xml

@register.add_config_manager("pick_up")
class PickUpConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=[1, 1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)

    def load_containers(self, target_container):
        if target_container is None:
            return
        container_config = dict(
            name="beaker_0",
            xml_path=name2class_xml["beaker"][-1],
            position=[random.uniform(0.2, 0.28), random.uniform(-0.1, 0.0), 0.8],
        )
        container_config["class"] = "ChemistryBeaker"
        self.config["task"]["components"].append(container_config)

    def load_objects(self, target_entity):
        obj_config = dict(
            name="pipettes_stand_0",
            xml_path=name2class_xml["pipettes_stand"][-1],
            position=[random.uniform(-0.3, 0.3), random.uniform(-0.2, 0.2), 0.8],
        )
        obj_config["class"] = "CommonGraspedEntity"
        obj_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0.05])
        self.config["task"]["components"].append(obj_config)

    def get_instruction(self, target_entity, target_container, **kwargs):
        self.config["task"]["instructions"] = ["Pick up the <beaker_0>."]

    def get_condition_config(self, target_entity, target_container, **kwargs):
        # 执行完即成功
        pass

    def get_target_entity(self):
        return "pipettes_stand_0"


@register.add_task("pick_up")
class PickUpTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name="beaker_0", prior_eulers=[[-3.14159, 0, 0]]),
        ]
        return skill_sequence
