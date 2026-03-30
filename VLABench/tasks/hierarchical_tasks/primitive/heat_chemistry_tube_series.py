import random
import numpy as np
from VLABench.tasks.dm_task import *
from VLABench.tasks.hierarchical_tasks.primitive.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
from VLABench.configs.constant import name2class_xml


# 试管在试管架上的相对位置（列方向偏移）
relative_col_pos = [-0.16, -0.08, 0, 0.08, 0.16]
relative_row_pos = [-0.05, 0.05]


@register.add_config_manager("heat_chemistry_tube")
class HeatChemistryTubeConfigManager(BenchTaskConfigManager):
    """
    加热试管任务的配置管理器。
    场景：试管架上有多个试管（不同溶液），旁边有本生灯。
    目标：从试管架上取出目标溶液的试管，移到本生灯上方加热。
    """
    def __init__(self,
                 task_name,
                 num_objects=[2, 3],
                 **kwargs):
        super().__init__(task_name, num_objects, **kwargs)

    def load_containers(self, target_container):
        """
        加载本生灯作为 target_container（加热的目标位置）。
        放在桌面右侧区域，与试管架拉开距离。
        """
        if target_container is not None:
            container_config = self.get_entity_config(
                target_container,
                position=[random.uniform(0.15, 0.25), random.uniform(0.0, 0.15), 0.8],
                randomness=None
            )
            self.config["task"]["components"].append(container_config)

    def load_init_containers(self, init_container):
        """
        加载试管架作为 init_container（试管的初始容器）。
        放在桌面左侧区域。
        """
        if init_container is not None:
            init_container_config = self.get_entity_config(
                init_container,
                position=[random.uniform(-0.25, -0.1), random.uniform(0.0, 0.15), 0.8],
                randomness=None
            )
            self.config["task"]["components"].append(init_container_config)

    def load_objects(self, target_entity):
        """
        在试管架上创建多个试管作为 subentity。
        每个试管是 ChemistryTube 类型，需要 solution 参数来决定颜色。
        参考 select_chemistry_tube_series.py 的做法。
        """
        objects = []
        objects.append(target_entity)

        # 从 seen/unseen object 列表中选取其他溶液作为干扰项
        other_objects = self.seen_object.copy() + self.unseen_object.copy()
        for similar_objects in self.seen_object + self.unseen_object:
            if isinstance(similar_objects, list) and target_entity in similar_objects:
                other_objects.remove(similar_objects)
            elif isinstance(similar_objects, str) and target_entity == similar_objects:
                other_objects.remove(similar_objects)
        other_objects_flatten = []
        for similar_objects in other_objects:
            if isinstance(similar_objects, list):
                other_objects_flatten.extend(similar_objects)
            else:
                other_objects_flatten.append(similar_objects)
        objects.extend(random.sample(other_objects_flatten, self.num_object - 1))

        # 随机选取列位置，放置试管
        target_col_poses = random.sample(relative_col_pos, self.num_object)
        target_poses = [[pos, random.choice(relative_row_pos), 0.05] for pos in target_col_poses]

        # 将试管作为试管架的 subentity 添加
        # config["task"]["components"] 中最后一个是试管架（load_init_containers 刚加入的）
        init_container_config = self.config["task"]["components"][-1]
        init_container_config["subentities"] = []
        for obj, pos in zip(objects, target_poses):
            object_config = dict(
                name=obj,
                solution=obj,  # ChemistryTube 需要 solution 参数来决定溶液颜色
                xml_path=name2class_xml["tube"][-1],
                position=pos,
            )
            object_config["class"] = name2class_xml["tube"][0]
            init_container_config["subentities"].append(object_config)

    def get_condition_config(self, target_entity, target_container, **kwargs):
        """
        使用 heated 条件：目标试管在本生灯上方累计停留 duration 秒即成功。
        """
        conditions_config = dict(
            heated=dict(
                target_entity=f"{target_entity}",
                heat_source=f"{target_container}",
                duration=5.0,
                xy_tolerance=0.08
            )
        )
        self.config["task"]["conditions"] = conditions_config
        return conditions_config

    def get_instruction(self, target_entity, target_container=None, **kwargs):
        instruction = [f"Pick the {target_entity} solution from the tube rack and heat it over the bunsen burner for 5 seconds."]
        self.config["task"]["instructions"] = instruction
        return instruction


@register.add_task("heat_chemistry_tube")
class HeatChemistryTubeTask(PrimitiveTask):
    """
    加热试管任务。
    流程：从试管架上取出目标试管 → 抬起 → 移动到本生灯上方 → 保持加热。
    """
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def build_from_config(self, eval=False, **kwargs):
        """
        构建场景后，需要将试管架从默认父级 detach 并 attach 到 arena，
        使试管架和试管正确出现在场景中。
        参考 select_chemistry_tube_series.py 的做法。
        """
        super().build_from_config(eval, **kwargs)
        self.random_ignored_entities = ["table"]
        for key in list(self.entities.keys()):
            if "tube_stand" in key:
                tube_stand = self.entities[key]
                tube_stand.detach()
                self._arena.attach(tube_stand)

    def get_expert_skill_sequence(self, physics):
        """
        专家技能序列：pick → lift → moveto（本生灯上方）→ wait（等待加热完成）
        抓取姿态参考 take_chemistry_experiment：倾斜 45 度避免夹爪卡住试管架，
        抓取点上移 2cm 避免夹到试管底部。

        moveto 目标高度计算：
          夹爪目标 z = 本生灯顶部 z + 试管底部到抓取点的距离 + 加热间距
          这样试管底部刚好悬在本生灯顶部上方一小段距离。
        """
        from VLABench.utils.utils import euler_to_quaternion
        # 获取试管的抓取点，上移 2cm
        grasppoint = np.array(self.entities[self.target_entity].get_grasped_keypoints(physics)[-1])
        grasp_pos = grasppoint + np.array([0, 0, 0.02])
        grasp_quat = euler_to_quaternion(-np.pi, np.pi/4, -np.pi/2)

        # 计算试管底部到抓取点的距离（即试管在夹爪下方悬挂的长度）
        tube_bottom_z = self.entities[self.target_entity].get_xpos(physics)[2]
        tube_hang_length = grasppoint[2] - tube_bottom_z  # 抓取点 z - 试管底部 z

        # 获取本生灯顶部 z（遍历所有 geom 取最高点）
        burner = self.entities[self.target_container]
        burner_geoms = burner.mjcf_model.find_all('geom')
        burner_top_z = max(physics.bind(g).xpos[2] for g in burner_geoms)

        # 夹爪目标 z = 本生灯顶部 + 试管悬挂长度 + 8cm 加热间距
        heating_gap = 0.08
        target_z = burner_top_z + tube_hang_length + heating_gap

        above_burner = burner.get_xpos(physics).copy()
        above_burner[2] = target_z

        skill_sequence = [
            partial(SkillLib.pick, target_entity_name=self.target_entity,
                    target_pos=grasp_pos, target_quat=grasp_quat),
            partial(SkillLib.lift, gripper_state=np.zeros(2), lift_height=0.25),
            partial(SkillLib.moveto, target_pos=above_burner, gripper_state=np.zeros(2)),
            partial(SkillLib.wait, wait_time=50),
        ]
        return skill_sequence
