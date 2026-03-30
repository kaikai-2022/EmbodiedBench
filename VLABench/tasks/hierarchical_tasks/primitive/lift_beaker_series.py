import random
from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.primitive.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register


@register.add_config_manager("lift_beaker")
class LiftBeakerConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=[1, 1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)

    def load_objects(self, target_entity):
        object_config = dict(
            name=target_entity,
            xml_path="obj/meshes/lab_equipment/beaker/beaker_0/beaker/beaker.xml",
            position=[0.0, 0.0, 0.8],
            orientation=[0, 0, 0],
        )
        object_config["class"] = "CommonGraspedEntity"
        object_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0.05])
        self.config["task"]["components"].append(object_config)

    def get_instruction(self, target_entity, **kwargs):
        instruction = [f"lift the {target_entity}"]
        self.config["task"]["instructions"] = instruction

    def get_condition_config(self, target_entity, **kwargs):
        conditions_config = dict(
            lift=dict(
                entities=[target_entity],
                target_height=0.9
            )
        )
        self.config["task"]["conditions"] = conditions_config


@register.add_task("lift_beaker")
class LiftBeakerTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name=self.target_entity),
            partial(SkillLib.lift, lift_height=0.15, gripper_state=np.zeros(2)),
        ]
        return skill_sequence