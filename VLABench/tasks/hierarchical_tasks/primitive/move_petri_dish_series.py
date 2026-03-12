"""
Move Petri Dish Task Series

任务：拾取培养皿并移动到桌面右侧指定位置
操作序列：pick -> move_to_target -> release
"""

import random
import numpy as np
from functools import partial

from VLABench.tasks.dm_task import SkillLib
from VLABench.tasks.hierarchical_tasks.primitive.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register


@register.add_config_manager("move_petri_dish")
class MovePetriDishConfigManager(BenchTaskConfigManager):
    """配置管理器：移动培养皿任务"""

    def __init__(self, task_name, num_objects=1, **kwargs):
        super().__init__(task_name, num_objects=num_objects, **kwargs)

    def load_containers(self, target_container=None, **kwargs):
        """不需要容器"""
        pass

    def load_init_containers(self, init_container=None, **kwargs):
        """不需要初始容器"""
        pass

    def load_objects(self, target_entity, **kwargs):
        """加载培养皿 - 初始位置在桌面左侧"""
        object_config = self.get_entity_config(
            target_entity,
            position=[
                random.uniform(-0.25, -0.15),  # X: 左侧
                random.uniform(-0.1, 0.1),      # Y: 中间区域
                0.761  # Z: 桌面顶部0.76 + 1mm
            ],
            orientation=[0, 0, 0],
            randomness=None
        )
        self.config["task"]["components"].append(object_config)

    def get_instruction(self, target_entity, **kwargs):
        """生成任务指令"""
        instruction = [f"Move the {target_entity} to the right side of the table"]
        self.config["task"]["instructions"] = instruction

    def get_condition_config(self, target_entity, **kwargs):
        """生成评测条件：检查物体是否在目标位置"""
        # 使用 on_position 条件 - 检查物体是否接近目标位置
        target_position = [0.2, 0.0, 0.76]  # 目标位置中心点
        condition_config = {
            "on_position": {
                "entities": [target_entity],
                "positions": [target_position],
                "tolerance_distance": 0.15,  # 允许15cm误差
                "dimension": 2  # 只检查XY平面
            }
        }
        self.config["task"]["conditions"] = condition_config


@register.add_task("move_petri_dish")
class MovePetriDishTask(PrimitiveTask):
    """移动培养皿任务"""

    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        """
        专家技能序列：
        1. pick: 从上往下抓取培养皿
        2. moveto: 移动到桌面右侧
        3. open_gripper: 释放培养皿
        """
        # 计算目标放置位置（桌面右侧）
        target_pos = np.array([
            random.uniform(0.15, 0.25),  # X: 右侧
            random.uniform(-0.1, 0.1),    # Y: 中间
            0.85  # Z: 桌面上方适当高度(末端执行器高度)
        ])

        skill_sequence = [
            # 1. 从上往下抓取培养皿
            partial(
                SkillLib.pick,
                target_entity_name=self.target_entity,
                prior_eulers=[[-np.pi, 0, 0]]  # 从上往下
            ),
            # 2. 移动到目标位置
            partial(
                SkillLib.moveto,
                target_pos=target_pos,
                gripper_state=np.zeros(2)  # 保持夹爪闭合
            ),
            # 3. 打开夹爪释放
            partial(SkillLib.open_gripper)
        ]

        return skill_sequence
