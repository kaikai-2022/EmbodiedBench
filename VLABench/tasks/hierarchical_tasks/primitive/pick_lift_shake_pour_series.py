import random
import numpy as np
from functools import partial

from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.primitive.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
from VLABench.configs.constant import name2class_xml

@register.add_config_manager("pick_lift_shake_pour")
class PickLiftShakePourConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=[1, 1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.config["task"]["n_distractor"] = 0

    def load_containers(self, target_container):
        if target_container is None:
            return
        container_config = dict(
            name="flask_0",
            xml_path=name2class_xml["flask"][-1],
            position=[random.uniform(0.2, 0.28), random.uniform(-0.1, 0.0), 0.8],
        )
        container_config["class"] = "ChemistryBeaker"
        self.config["task"]["components"].append(container_config)
        container_config = dict(
            name="beaker_0",
            xml_path=name2class_xml["beaker"][-1],
            position=[random.uniform(0.2, 0.28), random.uniform(-0.1, 0.0), 0.8],
        )
        container_config["class"] = "ChemistryBeaker"
        self.config["task"]["components"].append(container_config)

    def get_instruction(self, target_entity, target_container, **kwargs):
        self.config["task"]["instructions"] = ["Pick the <flask_0>."]

    def get_condition_config(self, target_entity, target_container, **kwargs):
        # 执行完即成功
        pass

    def get_target_entity(self):
        return "flask_0"


@register.add_task("pick_lift_shake_pour")
class PickLiftShakePourTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name="flask_0", prior_eulers=[[-3.14159, 0, 0]]),
            partial(SkillLib.lift, lift_height=0.15, gripper_state=[0, 0]),
            partial(SkillLib.rotate, rotation_angle=1.5708),
            partial(SkillLib.rotate, rotation_angle=-1.5708),
            partial(SkillLib.rotate, rotation_angle=1.5708),
            partial(SkillLib.rotate, rotation_angle=-1.5708),
            partial(SkillLib.wait, wait_time=20),
            partial(SkillLib.pour_to_entity, target_container_name="beaker_0", tilt_angle=1.8, wait_time=10),
            partial(SkillLib.open_gripper),
        ]
        return skill_sequence
