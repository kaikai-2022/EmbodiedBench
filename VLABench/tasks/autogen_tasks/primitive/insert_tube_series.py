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
# rel_x 可选值: -0.165, -0.08, 0, 0.08, 0.165（对应5列）
# rel_y 可选值: -0.05, 0.05（对应2行）
HOLE_REL_X = [-0.165, -0.08, 0, 0.08, 0.165]
HOLE_REL_Y = [-0.05, 0.05]


@register.add_config_manager("insert_tube")
class InsertTubeConfigManager(BenchTaskConfigManager):
    """
    将桌面上水平放置的试管插入试管架。
    场景：3根不同溶液的试管水平放在桌面前方，试管架放在桌面后方。
    """
    def __init__(self, task_name, num_objects=[3], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)

    def load_containers(self, target_container):
        """加载试管架作为 target_container，放在桌面后方"""
        container_config = self.get_entity_config(
            "chemistry_tube_stand",
            position=[random.uniform(-0.2, 0.2), random.uniform(0.1, 0.2), 0.8],
            randomness=None
        )
        self.config["task"]["components"].append(container_config)

    def load_objects(self, target_entity):
        """
        加载试管，水平放在桌面前方。
        ChemistryTube 需要 solution 参数来决定溶液颜色。
        试管以 orientation=[np.pi/2, 0, 0] 横躺放置。
        """
        objects = [target_entity]

        # 从其他溶液中选取干扰试管
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

        # 在桌面前方水平排列试管
        x_positions = [-0.2, 0, 0.2]
        random.shuffle(x_positions)
        for i, obj in enumerate(objects):
            obj_config = dict(
                name=obj,
                solution=obj,
                xml_path=name2class_xml["tube"][-1],
                position=[x_positions[i], random.uniform(-0.15, -0.05), 0.82],
                orientation=[-np.pi/2, 0, 0],  # 横躺（试管口朝外），无随机偏差
            )
            obj_config["class"] = name2class_xml["tube"][0]
            self.config["task"]["components"].append(obj_config)

    def get_instruction(self, target_entity, target_container, **kwargs):
        instruction = f"Pick up the {target_entity} tube and insert it into the tube stand."
        self.config["task"]["instructions"] = instruction

    def get_condition_config(self, target_entity, target_container, **kwargs):
        """成功条件：试管在试管架的 AABB 包围盒内"""
        conditions_config = dict(
            contain=dict(
                container=target_container,
                entities=[target_entity]
            )
        )
        self.config["task"]["conditions"] = conditions_config


@register.add_task("insert_tube")
class InsertTubeTask(PrimitiveTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def build_from_config(self, eval=False, **kwargs):
        """固定试管架，防止碰撞时被推倒"""
        super().build_from_config(eval, **kwargs)
        for key, entity in self.entities.items():
            if "tube_stand" in key:
                entity.detach()
                self._arena.attach(entity)

    def should_terminate_episode(self, physics):
        """
        重写终止条件：除了 ContainCondition 之外，
        还要求试管 z 坐标低于试管架顶部，确保试管真正插入而非从上方经过。
        """
        # 先检查原始条件
        if not super().should_terminate_episode(physics):
            return False
        # 额外检查：试管必须低于试管架顶部才算真正插入
        target = self.entities[self.target_entity]
        stand = self.entities[self.target_container]
        tube_z = physics.bind(target.mjcf_model.worldbody).xpos[2]
        stand_top_z = stand.get_xpos(physics)[2] + 0.05  # 严格阈值：试管必须深入架内
        if tube_z > stand_top_z:
            return False  # 试管还在试管架上方，不算插入
        return True

    def get_expert_skill_sequence(self, physics):
        """
        技能序列（借鉴 insert_flower）：
        1. pick  — 水平抓取桌面上的试管
        2. lift  — 抬起并旋转90度，让试管竖直
        3. moveto — 移到试管架某个孔位正上方
        4. lift(-0.15) — 下降，将试管插入孔中
        """
        # 获取试管架实体
        stand = self.entities[self.target_container]
        stand_pos = stand.get_xpos(physics)

        # 选择一个空孔位（中间列，前排）
        hole_x = stand_pos[0] + HOLE_REL_X[2]  # 中间列, rel_x=0
        hole_y = stand_pos[1] + HOLE_REL_Y[0]  # 前排, rel_y=-0.05

        # 孔位上方的目标位置，z 足够高以便对准后下降
        # 试管架顶部约 z=0.096(本地) + stand_pos[2]
        stand_top_z = stand_pos[2] + 0.10
        insert_target = np.array([hole_x, hole_y, stand_top_z + 0.15])

        # 竖直姿态（和 insert_flower 相同）
        vertical_quat = euler_to_quaternion(-np.pi/2, np.pi/2, 0)

        skill_sequence = [
            # 1. 水平抓取试管
            partial(SkillLib.pick, target_entity_name=self.target_entity),
            # 2. 抬起并旋转为竖直
            partial(SkillLib.lift, target_quat=vertical_quat),
            # 3. 移到孔位正上方
            partial(SkillLib.moveto, target_pos=insert_target, target_quat=vertical_quat),
            # 4. 负高度 lift 下降插入孔中（和 insert_flower 相同方式）
            partial(SkillLib.lift, lift_height=-0.2),
        ]
        return skill_sequence
