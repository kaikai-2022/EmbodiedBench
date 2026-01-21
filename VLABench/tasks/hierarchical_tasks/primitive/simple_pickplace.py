
import random

from VLABench.tasks.dm_task import SkillLib
from VLABench.tasks.hierarchical_tasks.primitive.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
from functools import partial
import numpy as np

@register.add_config_manager("simple_pickplace")
class SimplePickPlaceConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=1, **kwargs):
        super().__init__(task_name, num_objects=num_objects, **kwargs)

    def load_containers(self, target_container="pan_seen", **kwargs):
        # 放置锅在合适位置
        container_config = self.get_entity_config(
            target_container,
            position=[random.uniform(-0.2, -0.1), random.uniform(-0.2, 0.0), 0.84],
            randomness=None
        )
        # 加 dishes 子实体（官方 add_condiment 有，稳定）
        container_config["subentities"] = [self.get_entity_config("dishes", position=[0, 0.03, 0.0], randomness=None)]
        self.config["task"]["components"].append(container_config)

    def load_objects(self, target_entity="salt", **kwargs):
        # 只放置盐罐（固定一个，无随机其他物）
        object_config = self.get_entity_config(
            target_entity,
            position=[random.uniform(0.15, 0.2), random.uniform(-0.15, 0.15), 0.85],
            randomness=None
        )
        self.config["task"]["components"].append(object_config)

    def get_instruction(self, **kwargs):
        instruction = ["Put the salt into the pan."]
        self.config["task"]["instructions"] = instruction

    def get_condition_config(self, **kwargs):
        # 用 above（稳定，避开 contain 可能的 keypoint 问题）
        condition_config = dict(
            above=dict(
                target_entity="salt",
                platform="pan_seen"
            )
        )
        self.config["task"]["conditions"] = condition_config


@register.add_task("simple_pickplace")
class SimplePickPlaceTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    # 可选：简单专家轨迹（pick + move + place）
    def get_expert_skill_sequence(self, physics):
        target_pos = np.array(self.entities["pan_seen"].get_xpos(physics)) + np.array([0, 0, 0.2])
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name="salt", prior_eulers=[[-np.pi/2, -np.pi/2, np.pi/2]]),
            partial(SkillLib.lift, lift_height=0.2),
            partial(SkillLib.moveto, target_pos=target_pos),
            partial(SkillLib.release)  # 或 place/pour
        ]
        return skill_sequence

@register.add_task("simple_pickplace")
class SimplePickPlaceTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics, prior_eulers=[[ -np.pi/2, 0, 0 ]]):
        # 专家轨迹：pick -> move above bowl -> place（可选，如果你想测试专家策略）
        bowl_pos = np.array(self.entities[self.target_container].get_xpos(physics)) + np.array([0, 0, 0.25])  # bowl 上方 25cm
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name=self.target_object, prior_eulers=prior_eulers),
            partial(SkillLib.lift, lift_height=0.2),  # 先抬高一点
            partial(SkillLib.moveto, target_pos=bowl_pos),
            partial(SkillLib.place)  # 如果有 place skill；否则用 release 或 pour
        ]
        return skill_sequence