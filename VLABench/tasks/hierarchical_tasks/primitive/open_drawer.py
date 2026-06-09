"""
open_drawer task: minimal demo of SkillLib.open_drawer.

Scene: 1 table + 1 cabinet (with drawers) → open the top drawer (drawer_id=0).
"""
import random
from functools import partial

import numpy as np

from VLABench.tasks.dm_task import LM4ManipBaseTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
from VLABench.utils.skill_lib import SkillLib


@register.add_config_manager("open_drawer")
class OpenDrawerConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=[1, 1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)

    def get_seen_task_config(self):
        init_container = random.choice(self.seen_init_container) if self.seen_init_container else None
        return self.get_task_config(init_container=init_container)

    def get_unseen_task_config(self):
        init_container = random.choice(self.unseen_init_container) if self.unseen_init_container else None
        return self.get_task_config(init_container=init_container)

    def get_task_config(self, init_container, **kwargs):
        self.init_container = init_container
        # Provide a placeholder for trajectory_generation.py / metadata writers
        # that read config_manager.target_entity. Not used by the task itself.
        self.target_entity = init_container
        self.load_init_containers(init_container)
        self.get_condition_config(init_container=init_container)
        self.get_instruction()
        return self.config

    def load_init_containers(self, init_container):
        # Force the short_cabinet variant: its drawers slide along the X axis
        # with a visible handle protruding from the side, which the gripper
        # can reach from the front of the cabinet. (wooden_cabinet's handles
        # are recessed inside the top face — unreachable for our pick pose.)
        from VLABench.configs.constant import name2class_xml
        short_cabinet_xml = "obj/meshes/containers/cabinets/short_cabinet/short_cabinet_fix.xml"
        name2class_xml["cabinet"][1] = [short_cabinet_xml]

        # All three cabinet variants have the drawer handle on the -Y side of
        # the cabinet body. The Franka base is at y=-0.7; to face the handle
        # toward the gripper we place the cabinet in front of the base and
        # rotate it 180° around Z (so the cabinet's -Y aligns with +Y global).
        container_config = self.get_entity_config(
            init_container,
            position=[random.uniform(-0.1, 0.1), random.uniform(0.0, 0.1), 0.78],
            orientation=[0, 0, 0],
        )
        container_config["subentities"] = list()
        container_config["randomness"] = None
        self.config["task"]["components"].append(container_config)
        self.config["task"]["random_ignored_entities"].append(init_container)

    def get_condition_config(self, init_container, **kwargs):
        conditions_config = dict(
            drawer_open=dict(
                entities=[f"{init_container}"],
                open_threshold=0.05,
            )
        )
        self.config["task"]["conditions"] = conditions_config

    def get_instruction(self, **kwargs):
        self.config["task"]["instructions"] = ["open the drawer"]


@register.add_task("open_drawer")
class OpenDrawerTask(LM4ManipBaseTask):
    def __init__(self, task_name, robot, random_init=False, **kwargs):
        super().__init__(task_name, robot=robot, random_init=random_init, **kwargs)

    def get_expert_skill_sequence(self, physics):
        return [
            partial(
                SkillLib.open_drawer,
                target_container_name=self.init_container,
                pick_prior_eulers=[[-np.pi, 0, 0]],
                drawer_id=0,
            )
        ]
