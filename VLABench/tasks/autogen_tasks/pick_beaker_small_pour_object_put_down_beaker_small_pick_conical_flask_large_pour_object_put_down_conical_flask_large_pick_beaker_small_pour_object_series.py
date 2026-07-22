import random
import numpy as np
from functools import partial

from VLABench.tasks.dm_task import *
from VLABench.tasks.autogen_tasks.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
from VLABench.configs.constant import name2class_xml

@register.add_config_manager("pick_beaker_small_pour_object_put_down_beaker_small_pick_conical_flask_large_pour_object_put_down_conical_flask_large_pick_beaker_small_pour_object")
class PickBeakerSmallPourObjectPutDownBeakerSmallPickConicalFlaskLargePourObjectPutDownConicalFlaskLargePickBeakerSmallPourObjectConfigManager(BenchTaskConfigManager):
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
            name="conical_flask_large_0",
            xml_path=name2class_xml["conical_flask_large"][-1],
            position=[random.uniform(0.35, 0.45), random.uniform(-0.05, 0.05), 0.8],
            solution_rgba=[0.0, 0.45, 1.0, 0.4],
        )
        obj_config["class"] = "ChemistryBeaker"
        obj_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0.05])
        self.config["task"]["components"].append(obj_config)

        obj_config = dict(
            name="cylinder_small_0",
            xml_path=name2class_xml["cylinder_small"][-1],
            position=[random.uniform(0.15, 0.25), random.uniform(0.05, 0.15), 0.8],
        )
        obj_config["class"] = "ChemistryBeaker"
        obj_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0.05])
        self.config["task"]["components"].append(obj_config)

        self.target_entity = "small_beaker_0"

    def get_instruction(self, target_entity, **kwargs):
        self.config["task"]["instructions"] = ["Pick the <small_beaker_0> which contains <CuSO4 solution_0>."]

    def get_condition_config(self, target_entity, **kwargs):
        # 执行完即成功
        pass



@register.add_task("pick_beaker_small_pour_object_put_down_beaker_small_pick_conical_flask_large_pour_object_put_down_conical_flask_large_pick_beaker_small_pour_object")
class PickBeakerSmallPourObjectPutDownBeakerSmallPickConicalFlaskLargePourObjectPutDownConicalFlaskLargePickBeakerSmallPourObjectTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name="small_beaker_0"),
            partial(SkillLib.lift),
            partial(SkillLib.pour_to_entity, target_container_name="conical_flask_large_0"),
            partial(SkillLib.open_gripper),
            partial(SkillLib.pick, target_entity_name="conical_flask_large_0"),
            partial(SkillLib.lift),
            partial(SkillLib.pour_to_entity, target_container_name="small_beaker_0"),
            partial(SkillLib.open_gripper),
            partial(SkillLib.pick, target_entity_name="small_beaker_0"),
            partial(SkillLib.lift),
            partial(SkillLib.pour_to_entity, target_container_name="cylinder_small_0"),
            partial(SkillLib.open_gripper),
        ]
        return skill_sequence
