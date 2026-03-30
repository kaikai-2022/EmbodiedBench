import random
import numpy as np
from functools import partial
from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.primitive.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register

@register.add_config_manager("place_test_tube_into_rack")
class PlaceTestTubeIntoRackConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=[2, 2], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)

    def load_init_containers(self, init_container):
        pass

    def load_containers(self, target_container):
        # 使用 canonical_name "chemistry_tube_stand"
        container_config = self.get_entity_config(
            "chemistry_tube_stand",
            position=[0.0, 0.0, 0.8],
            randomness=dict(pos=[0.05, 0.05, 0], quat=[0, 0, 0.1])
        )
        self.config["task"]["components"].append(container_config)

    def load_objects(self, target_entity):
        # 使用 canonical_name "tube" 并添加必需的 solution 参数
        object_config = self.get_entity_config(
            "tube",
            position=[0.15, -0.15, 0.8],
            randomness=dict(pos=[0.03, 0.03, 0], quat=[0, 0, 0.05])
        )
        # ChemistryTube 必须设置 solution 参数
        object_config["solution"] = "NaCl"
        self.config["task"]["components"].append(object_config)

    def get_instruction(self, target_entity, target_container, **kwargs):
        instruction = ["place the test tube into the test tube rack"]
        self.config["task"]["instructions"] = instruction

    def get_condition_config(self, target_entity, target_container, **kwargs):
        conditions_config = dict(
            contain=dict(
                container=target_container,
                entities=[target_entity]
            )
        )
        self.config["task"]["conditions"] = conditions_config

@register.add_task("place_test_tube_into_rack")
class PlaceTestTubeIntoRackTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        # 获取试管架的位置，计算放置目标位置（试管架上方）
        rack_pos = np.array(self.entities[self.target_container].get_xpos(physics))
        target_pos = rack_pos + np.array([0, 0, 0.15])
        
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name=self.target_entity, prior_eulers=[[-np.pi, 0, 0]]),
            partial(SkillLib.lift, lift_height=0.15, gripper_state=np.zeros(2)),
            partial(SkillLib.place, target_container_name=self.target_container, target_pos=target_pos),
        ]
        return skill_sequence