import random
from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.primitive.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register


@register.add_config_manager("move_microscope")
class MoveMicroscopeConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=[1, 2], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)

    def load_objects(self, target_entity):
        object_config = dict(
            name=target_entity,
            xml_path="review/microscope/9562226299a045bab13c3d66f8593208/9562226299a045bab13c3d66f8593208.xml",
            position=[random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), 0.8],
            orientation=[0, 0, 0],
        )
        object_config["class"] = "CommonGraspedEntity"
        object_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0.05])
        self.config["task"]["components"].append(object_config)

    def get_instruction(self, target_entity, **kwargs):
        instruction = [f"Move the microscope"]
        self.config["task"]["instructions"] = instruction

    def get_condition_config(self, target_entity, **kwargs):
        conditions_config = dict(
            is_grasped=dict(
                entities=[target_entity],
                robot="robot"
            )
        )
        self.config["task"]["conditions"] = conditions_config


@register.add_task("move_microscope")
class MoveMicroscopeTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            partial(SkillLib.pick, target_entity_name=self.target_entity, prior_eulers=[[-np.pi, 0, 0]]),
        ]
        return skill_sequence