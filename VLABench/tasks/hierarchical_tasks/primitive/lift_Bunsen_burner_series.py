import random
import numpy as np
from functools import partial

from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.primitive.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register

@register.add_config_manager("lift_Bunsen_burner")
class LiftBunsenBurnerConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=[1, 1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)

    def load_init_containers(self, init_container):
        pass

    def load_objects(self, target_entity):
        object_config = dict(
            name=target_entity,
            xml_path="review/bunsen_burner/7532be8f501d435194e3feec33a3addf/7532be8f501d435194e3feec33a3addf.xml",
            position=[0.0, 0.0, 0.8],
            orientation=[0, 0, 0],
        )
        object_config["class"] = "CommonGraspedEntity"
        object_config["randomness"] = dict(pos=[0.05, 0.05, 0], quat=[0, 0, 0.1])
        self.config["task"]["components"].append(object_config)

    def get_instruction(self, target_entity, **kwargs):
        instruction = ["Lift the {target_entity}"]
        self.config["task"]["instructions"] = instruction

    def get_condition_config(self, target_entity, **kwargs):
        conditions_config = dict(
            lift=dict(entities=["Bunsen_burner"], target_height=0.9)
        )
        self.config["task"]["conditions"] = conditions_config

    def get_target_entity(self):
        return "Bunsen_burner"

    def get_task_config(self, **kwargs):
        self.target_entity = "Bunsen_burner"
        self.target_container = None
        self.config["task"]["target_entity"] = "Bunsen_burner"
        self.load_objects("Bunsen_burner")
        self.get_instruction("Bunsen_burner")
        self.get_condition_config("Bunsen_burner")
        return self.config


@register.add_task("lift_Bunsen_burner")
class LiftBunsenBurnerTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name=self.target_entity, prior_eulers=[[-3.14159, 0, 0]]),
            partial(SkillLib.lift, lift_height=0.15, gripper_state=np.zeros(2)),        ]
        return skill_sequence
