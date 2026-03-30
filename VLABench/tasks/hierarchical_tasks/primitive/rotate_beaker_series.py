"""
Rotate Beaker Task Series

任务：拾取烧杯并旋转90度
操作序列：pick -> lift -> rotate -> moveto -> open_gripper
"""

import random
import numpy as np
from functools import partial

from VLABench.tasks.dm_task import SkillLib
from VLABench.tasks.hierarchical_tasks.primitive.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register


@register.add_config_manager("rotate_beaker")
class RotateBeakerConfigManager(BenchTaskConfigManager):
    """配置管理器：旋转烧杯任务"""

    def __init__(self, task_name, num_objects=1, **kwargs):
        super().__init__(task_name, num_objects=num_objects, **kwargs)
        # 设置目标实体为beaker
        self.target_entity = "beaker"

    def load_containers(self, target_container=None, **kwargs):
        """不需要容器"""
        pass

    def load_init_containers(self, init_container=None, **kwargs):
        """不需要初始容器"""
        pass

    def load_objects(self, target_entity, **kwargs):
        """加载烧杯 - 初始位置在桌面中心"""
        self.init_position = [
            random.uniform(-0.05, 0.05),   # X: 中心附近
            random.uniform(-0.05, 0.05),   # Y: 中心附近
            0.761  # Z: 桌面顶部0.76 + 1mm
        ]
        object_config = self.get_entity_config(
            target_entity,
            position=self.init_position,
            orientation=[0, 0, 0],
            randomness=None
        )
        self.config["task"]["components"].append(object_config)

    def get_instruction(self, target_entity, **kwargs):
        """生成任务指令"""
        instruction = [f"Rotate the {target_entity} 90 degrees counterclockwise"]
        self.config["task"]["instructions"] = instruction

    def get_condition_config(self, target_entity, **kwargs):
        """生成评测条件：检查物体是否被旋转且放回桌面"""
        # 条件1：朝向正确（yaw旋转了约90度）
        # 条件2：烧杯放回初始位置附近（使用实际初始位置，而非硬编码）
        # ConditionSet要求所有条件同时满足，动作序列全部执行完才可能同时满足
        condition_config = {
            "on_orientation": {
                "entities": [target_entity],
                "orientations": [[0, 0, np.pi/2], [0, 0, -np.pi/2]],
                "tolerance_angle": np.pi/6
            },
            "on_position": {
                "entities": [target_entity],
                "positions": [self.init_position],
                "tolerance_distance": 0.08,
                "dimension": 3
            }
        }
        self.config["task"]["conditions"] = condition_config


@register.add_task("rotate_beaker")
class RotateBeakerTask(PrimitiveTask):
    """旋转烧杯任务"""

    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        """
        专家技能序列：
        1. pick: 从上往下抓取烧杯
        2. lift: 抬起烧杯
        3. rotate: 旋转90度
        4. moveto: 放回原位置
        5. open_gripper: 释放烧杯
        """
        # 获取烧杯的初始位置
        beaker_entity = self.entities[self.config_manager.target_entity]
        beaker_pos = beaker_entity.get_xpos(physics).copy()

        # 目标旋转角度: 90度 (π/2弧度)
        rotation_angle = np.pi / 2

        return [
            # 1. 从上往下抓取烧杯
            partial(
                SkillLib.pick,
                target_entity_name=self.config_manager.target_entity,
                prior_eulers=[[-np.pi, 0, 0]]  # 垂直朝下抓取
            ),
            # 2. 抬起到安全高度
            partial(
                SkillLib.lift,
                lift_height=0.1,  # 抬高10cm
                gripper_state=np.zeros(2)  # 保持夹爪闭合
            ),
            # 3. 旋转90度
            partial(
                SkillLib.rotate,
                rotation_angle=rotation_angle,  # 旋转90度
                gripper_state=np.zeros(2),  # 保持夹爪闭合
                target_q_velocity=np.pi/40  # 旋转速度
            ),
            # 4. 放下烧杯（目标位置要抬高到抓取点高度）
            partial(
                SkillLib.moveto,
                target_pos=beaker_pos + np.array([0, 0, 0.07]),  # 抬高7cm（抓取点高度）
                gripper_state=np.zeros(2)  # 保持夹爪闭合
            ),
            # 5. 打开夹爪
            partial(SkillLib.open_gripper)
        ]
