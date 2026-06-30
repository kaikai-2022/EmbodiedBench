import random
import numpy as np
from functools import partial

from VLABench.tasks.dm_task import *
from VLABench.tasks.autogen_tasks.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
from VLABench.configs.constant import name2class_xml

@register.add_config_manager("pick_funnel")
class PickFunnelConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=[1, 1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.config["task"]["n_distractor"] = 0

    def load_init_containers(self, init_container):
        if init_container is None or init_container == "funnel_support":
            container_config = dict(
                name="funnel_support",
                xml_path=name2class_xml["funnel_support"][-1],
                position=[random.uniform(-0.15, -0.05), random.uniform(0.05, 0.15), 0.8],
            )
            container_config["class"] = "FunnelSupport"
            self.config["task"]["components"].append(container_config)

    def load_objects(self, target_entity):
        init_container_config = self.config["task"]["components"][-1]
        if "subentities" not in init_container_config:
            init_container_config["subentities"] = []
        funnel_config = dict(
            name="funnel_0",
            xml_path=name2class_xml["funnel"][-1],
            position=[-0.02, 0.007, 0.30],
        )
        funnel_config["class"] = "CommonGraspedEntity"
        init_container_config["subentities"].append(funnel_config)

        self.target_entity = "funnel_0"

    def get_instruction(self, target_entity, init_container, **kwargs):
        self.config["task"]["instructions"] = ["Pick the <funnel_0>."]

    def get_condition_config(self, target_entity, init_container, **kwargs):
        conditions_config = [
            dict(is_grasped=dict(entities=['funnel_0'], robot='robot')),
        ]
        self.config["task"]["conditions"] = conditions_config


@register.add_task("pick_funnel")
class PickFunnelTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name="funnel_0"),
        ]
        return skill_sequence
