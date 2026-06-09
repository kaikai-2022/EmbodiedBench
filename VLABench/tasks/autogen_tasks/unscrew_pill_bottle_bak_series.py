import random
import numpy as np
from functools import partial

from VLABench.tasks.dm_task import *
from VLABench.tasks.autogen_tasks.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
from VLABench.configs.constant import name2class_xml


@register.add_config_manager("unscrew_pill_bottle")
class UnscrewPillBottleConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=[1, 1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.config["task"]["n_distractor"] = 0

    def load_objects(self, target_entity):
        obj_config = dict(
            name="pill_bottle_0",
            xml_path=name2class_xml["pill_bottle"][-1],
            position=[random.uniform(0.05, 0.15), random.uniform(-0.15, -0.05), 0.8],
        )
        obj_config["class"] = "ContainerWithCap"
        obj_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0])
        self.config["task"]["components"].append(obj_config)

        self.target_entity = "pill_bottle_0"

    def get_instruction(self, target_entity, **kwargs):
        self.config["task"]["instructions"] = ["Unscrew the cap of the <pill_bottle_0>."]

    def get_condition_config(self, target_entity, **kwargs):
        conditions_config = [
            dict(cap_open=dict(entities=['pill_bottle_0'])),
        ]
        self.config["task"]["conditions"] = conditions_config


@register.add_task("unscrew_pill_bottle")
class UnscrewPillBottleTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def build_from_config(self, eval=False, **kwargs):
        super().build_from_config(eval, **kwargs)
        for key, entity in self.entities.items():
            if "pill_bottle" in key:
                entity.detach()
                self._arena.attach(entity)

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)

    def after_substep(self, physics, random_state):
        # 瓶身已通过 arena.attach() 焊死，无需 after_substep 重置
        pass

    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.unscrew_cap, target_entity_name="pill_bottle_0"),
        ]
        return skill_sequence
