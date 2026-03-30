import random
import numpy as np
from functools import partial

from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.primitive.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
from VLABench.utils.utils import euler_to_quaternion
from VLABench.configs.constant import name2class_xml

# 试管架上的试管位置（本地坐标，传给 subentity）
relative_col_pos = [-0.16, -0.08, 0, 0.08, 0.16]
relative_row_pos = [-0.05, 0.05]


@register.add_config_manager("pour_liquid_from_test_tube_into_beaker")
class PourLiquidFromTestTubeIntoBeakerConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=[1, 1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)

    def load_init_containers(self, init_container):
        """加载试管架，试管作为其子实体，放在左侧"""
        init_container_config = self.get_entity_config(
            init_container,
            position=[random.uniform(-0.15, -0.05), random.uniform(0.05, 0.15), 0.8],
            randomness=None
        )
        self.config["task"]["components"].append(init_container_config)

    def load_containers(self, target_container):
        """加载烧杯，放在右侧，与试管架充分拉开距离"""
        container_config = self.get_entity_config(
            target_container,
            position=[random.uniform(0.2, 0.28), random.uniform(-0.1, 0.0), 0.8],
        )
        self.config["task"]["components"].append(container_config)

    def load_objects(self, target_entity):
        """加载试管作为试管架的子实体，传 solution 参数"""
        col_pos = random.choice(relative_col_pos)
        row_pos = random.choice(relative_row_pos)
        pos = [col_pos, row_pos, 0.05]

        init_container_config = self.config["task"]["components"][-1]
        init_container_config["subentities"] = []
        obj_config = dict(
            name=target_entity,
            solution=target_entity,
            xml_path=name2class_xml["tube"][-1],
            position=pos,
        )
        obj_config["class"] = name2class_xml["tube"][0]
        init_container_config["subentities"].append(obj_config)

    def get_instruction(self, target_entity, target_container, **kwargs):
        instruction = [f"Pick up the test tube from the rack and pour the liquid into the {target_container}"]
        self.config["task"]["instructions"] = instruction

    def get_condition_config(self, target_entity, target_container, **kwargs):
        conditions_config = dict(
            pour=dict(target_entity=target_entity)
        )
        self.config["task"]["conditions"] = conditions_config

    def get_task_config(self, target_entity, target_container, init_container, **kwargs):
        self.target_entity = target_entity
        self.target_container = target_container
        self.init_container = init_container
        self.config["task"]["target_entity"] = target_entity
        self.config["task"]["target_container"] = target_container
        self.load_containers(target_container=target_container)
        self.load_init_containers(init_container=init_container)
        self.load_objects(target_entity=target_entity)
        self.get_instruction(target_entity, target_container)
        self.get_condition_config(target_entity, target_container)
        return self.config


@register.add_task("pour_liquid_from_test_tube_into_beaker")
class PourLiquidFromTestTubeIntoBeakerTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def build_from_config(self, eval=False, **kwargs):
        """固定试管架，增大试管摩擦力防止倾倒时滑脱"""
        super().build_from_config(eval, **kwargs)
        for key, entity in self.entities.items():
            if "tube_stand" in key:
                entity.detach()
                self._arena.attach(entity)
        # 增大试管的摩擦力，防止倾倒时从夹爪滑脱
        target = self.entities[self.target_entity]
        for geom in target.mjcf_model.worldbody.find_all("geom"):
            if geom.group == 3:  # collision geoms only
                geom.friction = [5, 5, 0.01]
                geom.condim = 4

    def get_expert_skill_sequence(self, physics):
        # 45度倾斜抓取试管（参考 shake_tube）
        target = self.entities[self.target_entity]
        grasppoint = target.get_grasped_keypoints(physics)[0]
        grasp_pos = grasppoint + np.array([0, 0, 0.02])
        grasp_quat = euler_to_quaternion(-np.pi, 5*np.pi/12, -np.pi/2)

        # 烧杯位置
        beaker_xpos = np.array(self.entities[self.target_container].get_xpos(physics))
        # 先移到烧杯右侧上方（保持高度避免碰撞）
        approach_pos = beaker_xpos + np.array([0.08, 0, 0.25])
        # 再略微下降，使试管口靠近烧杯口上方
        pour_pos = beaker_xpos + np.array([0.06, 0, 0.2])

        skill_sequence = [
            # 1. 45度角抓取试管
            partial(SkillLib.pick, target_entity_name=self.target_entity,
                    target_pos=grasp_pos, target_quat=grasp_quat),
            # 2. 抬高，确保试管完全离开试管架再横移
            partial(SkillLib.lift, lift_height=0.25, gripper_state=np.zeros(2)),
            # 3. 移到烧杯右侧上方（高位接近，避免碰撞）
            partial(SkillLib.moveto, target_pos=approach_pos, gripper_state=np.zeros(2)),
            # 4. 下降到倾倒位置
            partial(SkillLib.moveto, target_pos=pour_pos, gripper_state=np.zeros(2)),
            # 5. 反向倾倒（从右侧向左倒，负方向旋转，低速防滑）
            partial(SkillLib.pour, target_delta_qpos=-(np.pi/2+np.pi/4), target_q_velocity=-np.pi/80, n_repeat_step=6),
            # 6. 等待液体流出
            partial(SkillLib.wait, wait_time=10),
        ]
        return skill_sequence
