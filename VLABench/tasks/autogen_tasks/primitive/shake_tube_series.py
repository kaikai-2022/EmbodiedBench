import random
import numpy as np
from functools import partial
from VLABench.tasks.dm_task import *
from VLABench.tasks.autogen_tasks.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
from VLABench.utils.utils import euler_to_quaternion
from VLABench.configs.constant import name2class_xml


# 真实孔位的相对坐标（世界坐标系，已考虑试管架90度旋转）
# 从 select_chemistry_tube 任务中实测得到
HOLE_REL_X = [-0.165, -0.08, 0, 0.08, 0.165]
HOLE_REL_Y = [-0.05, 0.05]

# 试管架上的试管位置（本地坐标，传给 subentity）
relative_col_pos = [-0.16, -0.08, 0, 0.08, 0.16]
relative_row_pos = [-0.05, 0.05]


@register.add_config_manager("shake_tube")
class ShakeTubeConfigManager(BenchTaskConfigManager):
    """
    摇晃试管后插回试管架。
    场景：3根试管插在试管架上，取出目标试管摇晃后插回。
    复用 select_chemistry_tube 的 subentity 模式加载试管。
    """
    def __init__(self, task_name, num_objects=[3], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.target_tube_hole = None  # 记录目标试管的孔位，用于插回

    def load_init_containers(self, init_container):
        """加载试管架，试管作为其子实体"""
        init_container_config = self.get_entity_config(
            init_container,
            position=[random.uniform(-0.1, 0.1), random.uniform(0.05, 0.15), 0.8],
            randomness=None
        )
        self.config["task"]["components"].append(init_container_config)

    def load_objects(self, target_entity):
        """
        加载试管作为试管架的子实体（subentity）。
        复用 select_chemistry_tube 的方式，传 solution 参数。
        """
        objects = [target_entity]

        # 选取干扰试管
        other_objects = self.seen_object.copy() + self.unseen_object.copy()
        for item in self.seen_object + self.unseen_object:
            if isinstance(item, list) and target_entity in item:
                other_objects.remove(item)
            elif isinstance(item, str) and target_entity == item:
                other_objects.remove(item)
        other_flat = []
        for item in other_objects:
            if isinstance(item, list):
                other_flat.extend(item)
            else:
                other_flat.append(item)
        n_distractors = min(self.num_object - 1, len(other_flat))
        objects.extend(random.sample(other_flat, n_distractors))

        # 随机分配孔位
        col_positions = random.sample(relative_col_pos, len(objects))
        target_poses = [[col, random.choice(relative_row_pos), 0.05] for col in col_positions]

        # 记录目标试管的孔位（用于插回）
        self.target_tube_hole = target_poses[0]

        # 作为试管架的 subentities 添加
        init_container_config = self.config["task"]["components"][-1]
        init_container_config["subentities"] = []
        for obj, pos in zip(objects, target_poses):
            obj_config = dict(
                name=obj,
                solution=obj,
                xml_path=name2class_xml["tube"][-1],
                position=pos,
            )
            obj_config["class"] = name2class_xml["tube"][0]
            init_container_config["subentities"].append(obj_config)

    def get_instruction(self, target_entity, init_container, **kwargs):
        instruction = f"Pick up the {target_entity} tube from the stand, shake it, then put it back."
        self.config["task"]["instructions"] = instruction

    def get_condition_config(self, target_entity, init_container, **kwargs):
        """成功条件：试管回到试管架内"""
        conditions_config = dict(
            contain=dict(
                container=init_container,
                entities=[target_entity]
            )
        )
        self.config["task"]["conditions"] = conditions_config


@register.add_task("shake_tube")
class ShakeTubeTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def build_from_config(self, eval=False, **kwargs):
        """固定试管架"""
        super().build_from_config(eval, **kwargs)
        for key, entity in self.entities.items():
            if "tube_stand" in key:
                entity.detach()
                self._arena.attach(entity)

    def should_terminate_episode(self, physics):
        """不提前终止，让技能序列完整执行"""
        return False

    def get_expert_skill_sequence(self, physics):
        """
        技能序列：
        1. pick  — 倾斜45度从试管架抓取试管
        2. lift  — 抬起
        3. shake — 摇晃试管
        4. moveto — 移到原孔位上方
        5. lift(-0.2) — 下降插回
        """
        # 获取试管架
        stand = None
        for key, entity in self.entities.items():
            if "tube_stand" in key:
                stand = entity
                break
        stand_pos = stand.get_xpos(physics)

        # 获取目标试管的抓取点和倾斜45度姿态
        target = self.entities[self.target_entity]
        grasppoint = target.get_grasped_keypoints(physics)[0]
        grasp_pos = grasppoint + np.array([0, 0, 0.02])
        grasp_quat = euler_to_quaternion(-np.pi, np.pi/4, -np.pi/2)

        # 插回目标：孔位上方足够高的安全位置（借鉴 insert_tube，高于障碍物让 RRT 能规划）
        stand_top_z = stand_pos[2] + 0.10
        insert_above = np.array([grasp_pos[0], grasp_pos[1], stand_top_z + 0.15])

        skill_sequence = [
            # 1. 倾斜45度抓取试管
            partial(SkillLib.pick, target_entity_name=self.target_entity,
                    target_pos=grasp_pos, target_quat=grasp_quat),
            # 2. 抬起
            partial(SkillLib.lift, lift_height=0.25),
            # 3. 摇晃试管
            partial(SkillLib.shake, n_shakes=3, shake_angle=0.5),
            # 4. 移到孔位正上方（足够高，避开试管架障碍物）
            partial(SkillLib.moveto, target_pos=insert_above, target_quat=grasp_quat),
            # 5. 直线下降插回孔中（lift 不走 RRT，直线垂直下降）
            partial(SkillLib.lift, lift_height=-0.1),
            # 6. 松开抓夹
            partial(SkillLib.open_gripper),
        ]
        return skill_sequence
