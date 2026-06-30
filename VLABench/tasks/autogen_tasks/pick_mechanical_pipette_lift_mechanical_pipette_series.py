import random
import numpy as np
from functools import partial

from VLABench.tasks.dm_task import *
from VLABench.tasks.autogen_tasks.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
from VLABench.configs.constant import name2class_xml

relative_pipette_stand_col_pos = [0]
relative_pipette_stand_row_pos = [-0.1088, -0.0363, 0.0363, 0.1088]

@register.add_config_manager("pick_mechanical_pipette_lift_mechanical_pipette")
class PickMechanicalPipetteLiftMechanicalPipetteConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=[1, 1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.config["task"]["n_distractor"] = 0

    def load_init_containers(self, init_container):
        if init_container is None or init_container == "pipettes_stand":
            container_config = dict(
                name="pipettes_stand",
                xml_path=name2class_xml["pipettes_stand"][-1],
                position=[random.uniform(-0.15, -0.05), random.uniform(0.05, 0.15), 0.8],
            )
            container_config["class"] = "PipetteStand"
            self.config["task"]["components"].append(container_config)

    def load_objects(self, target_entity):
        col_pos = random.choice(relative_pipette_stand_col_pos)
        row_pos = random.choice(relative_pipette_stand_row_pos)
        pos = [0.1, -0.0363, 0.077]
        init_container_config = self.config["task"]["components"][-1]
        if "subentities" not in init_container_config:
            init_container_config["subentities"] = []
        obj_config = dict(
            name="mechanical_pipette_0",
            xml_path=name2class_xml["mechanical_pipette"][-1],
            position=pos,
        )
        obj_config["class"] = "CommonGraspedEntity"
        init_container_config["subentities"].append(obj_config)

        self.target_entity = "mechanical_pipette_0"

    def get_instruction(self, target_entity, init_container, **kwargs):
        self.config["task"]["instructions"] = ["pick the <mechanical_pipette_0>."]

    def get_condition_config(self, target_entity, init_container, **kwargs):
        conditions_config = [
            dict(is_grasped=dict(entities=['mechanical_pipette_0'], robot='robot')),
            dict(lift=dict(entities=['mechanical_pipette_0'], lift_height=0.15)),
        ]
        self.config["task"]["conditions"] = conditions_config


@register.add_task("pick_mechanical_pipette_lift_mechanical_pipette")
class PickMechanicalPipetteLiftMechanicalPipetteTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.gently_pick, target_entity_name="mechanical_pipette_0", prior_eulers=[[-3.141592653589793, 0, 0]], extra_close_ratio=0.2, n_close_steps=20, contact_dist_threshold=0.005, hold_steps=5),
            partial(SkillLib.lift, lift_height=0.15),
        ]
        return skill_sequence
