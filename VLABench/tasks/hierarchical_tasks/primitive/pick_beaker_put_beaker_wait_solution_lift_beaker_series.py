import random
import numpy as np
from functools import partial

from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.primitive.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
from VLABench.configs.constant import name2class_xml

@register.add_config_manager("pick_beaker_put_beaker_wait_solution_lift_beaker")
class PickBeakerPutBeakerWaitSolutionLiftBeakerConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=[1, 1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.config["task"]["n_distractor"] = 0

    def load_objects(self, target_entity):
        obj_config = dict(
            name="small_beaker_0",
            xml_path=name2class_xml["small_beaker"][-1],
            position=[random.uniform(0.05, 0.15), random.uniform(-0.15, -0.05), 0.8],
            solution_rgba=[0.0, 0.45, 1.0, 0.4],
        )
        obj_config["class"] = "ChemistryBeaker"
        obj_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0.05])
        self.config["task"]["components"].append(obj_config)

        obj_config = dict(
            name="hot_plate_0",
            xml_path=name2class_xml["hot_plate"][-1],
            position=[random.uniform(0.35, 0.45), random.uniform(-0.05, 0.05), 0.8],
        )
        obj_config["class"] = "FlatContainer"
        obj_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0.05])
        self.config["task"]["components"].append(obj_config)

    def get_instruction(self, target_entity, **kwargs):
        self.config["task"]["instructions"] = ["pick the <small_beaker_0> which contains <CuSO4_0>"]

    def get_condition_config(self, target_entity, **kwargs):
        # 执行完即成功
        pass

    def get_target_entity(self):
        return "small_beaker_0"


@register.add_task("pick_beaker_put_beaker_wait_solution_lift_beaker")
class PickBeakerPutBeakerWaitSolutionLiftBeakerTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name="small_beaker_0", prior_eulers=[[-3.141592653589793, 0, 0]]),
            partial(SkillLib.moveto_entity, target_entity_name="hot_plate_0"),
            partial(SkillLib.place, target_container_name="hot_plate_0"),
            partial(SkillLib.wait_for, wait_duration=2.0, entity_name="small_beaker_0", change_type="solution_change_color", color=[1, 0, 0, 0.4]),
            partial(SkillLib.moveto_entity, target_entity_name="small_beaker_0"),
            partial(SkillLib.pick, target_entity_name="small_beaker_0", prior_eulers=[[-3.141592653589793, 0, 0]]),
            partial(SkillLib.lift, lift_height=0.15),
        ]
        return skill_sequence
