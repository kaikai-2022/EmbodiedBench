import random
import numpy as np
from functools import partial

from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.primitive.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
from VLABench.configs.constant import name2class_xml

@register.add_config_manager("pick_shake")
class PickShakeConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=[1, 1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.config["task"]["n_distractor"] = 0

    def load_objects(self, target_entity):
        obj_config = dict(
            name="small_beaker_0",
            xml_path=name2class_xml["small_beaker"][-1],
            position=[random.uniform(0.15, 0.25), random.uniform(-0.15, -0.05), 0.8],
        )
        obj_config["class"] = "ChemistryBeaker"
        obj_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0.05])
        self.config["task"]["components"].append(obj_config)

    def get_instruction(self, target_entity, **kwargs):
        self.config["task"]["instructions"] = ["Pick the <small_beaker_0>."]

    def get_condition_config(self, target_entity, **kwargs):
        # 执行完即成功
        pass

    def get_target_entity(self):
        return "small_beaker_0"


@register.add_task("pick_shake")
class PickShakeTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name="small_beaker_0", prior_eulers=[[-3.14159, 0, 0]]),
            partial(SkillLib.lift, lift_height=0.15, gripper_state=[0, 0]),
            partial(SkillLib.shake, n_shakes=3, shake_angle=0.5, steps_per_swing=5),
            partial(SkillLib.wait, wait_time=50),
            partial(SkillLib.place, target_container_name="table"),
            partial(SkillLib.open_gripper),
        ]
        return skill_sequence
