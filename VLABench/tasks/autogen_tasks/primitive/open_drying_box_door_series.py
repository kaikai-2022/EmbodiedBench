import random
import numpy as np
import mujoco
from functools import partial

from VLABench.tasks.dm_task import *
from VLABench.tasks.autogen_tasks.base import PrimitiveTask
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
from VLABench.configs.constant import name2class_xml


@register.add_config_manager("open_drying_box_door")
class OpenDryingBoxDoorConfigManager(BenchTaskConfigManager):
    def __init__(self, task_name, num_objects=[1, 1], **kwargs):
        super().__init__(task_name, num_objects, **kwargs)
        self.config["task"]["n_distractor"] = 0

    def load_objects(self, target_entity):
        obj_config = dict(
            name="drying_box_0",
            xml_path=name2class_xml["drying_box"][-1],
            position=[random.uniform(0.15, 0.15), random.uniform(0.4, 0.4), 0.8],
            # 绕Z轴旋转90度，让柜门朝向机械臂
            quat=[0, 0, 0.7071, 0.7071],  # 90度绕Z轴的四元数
        )
        obj_config["class"] = "ContainerWithDoor"
        obj_config["randomness"] = dict(pos=[0.02, 0.02, 0], quat=[0, 0, 0.05])
        self.config["task"]["components"].append(obj_config)

        self.target_entity = "drying_box_0"

    def get_instruction(self, target_entity, **kwargs):
        self.config["task"]["instructions"] = ["open the drying_box door"]

    def get_condition_config(self, target_entity, **kwargs):
        # 使用恒为false的条件，只为了测试模型是否能正常工作
        conditions_config = [
            dict(always_false=dict())
        ]
        self.config["task"]["conditions"] = conditions_config


@register.add_task("open_drying_box_door")
class OpenDryingBoxDoorTask(PrimitiveTask):
    def __init__(self, task_name="open_drying_box_door_series", robot=None, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

    def get_expert_skill_sequence(self, physics):
        skill_sequence = [
            # 先抓取门把手
            partial(SkillLib.pick, target_entity_name="drying_box_0"),
            # 然后使用 open_door 技能打开门
            partial(self._custom_open_door, target_container_name="drying_box_0"),
        ]
        return skill_sequence

    def _custom_open_door(self, env, target_container_name):
        """
        自定义开门技能：直接控制门关节打开
        """
        from VLABench.utils.utils import quaternion_to_euler
        target_container = env.task.entities[target_container_name]
        door_joint = target_container.door_joint

        # 获取门的当前角度
        current_qpos = float(env.physics.bind(door_joint).qpos[0])
        target_qpos = float(env.physics.bind(door_joint).range[0])  # 目标是打开到 -1.8

        print(f"[custom_open_door] 当前门角度: {current_qpos:.4f}, 目标角度: {target_qpos:.4f}")

        # 保持机械臂位置不变，只旋转门关节
        observations = [env.get_observation()]
        waypoints = []
        task_success = False
        stage_success = False

        # 获取当前机械臂位置
        current_qpos_arm = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        gripper_state = SkillLib._get_gripper_state(env)

        # 逐步旋转门
        num_steps = 20
        for i in range(num_steps):
            # 计算当前步骤的目标角度
            t = (i + 1) / num_steps
            intermediate_qpos = current_qpos + (target_qpos - current_qpos) * t

            # 设置门关节位置
            env.physics.bind(door_joint).qpos[0] = intermediate_qpos
            mujoco.mj_forward(env.physics.model._model, env.physics.data._data)

            # 保持机械臂位置
            action = np.concatenate([current_qpos_arm, gripper_state])
            timestep = env.step(action)
            if timestep.last():
                task_success = True
                break

            obs = env.get_observation()
            waypoint = np.concatenate([
                env.robot.get_end_effector_pos(env.physics),
                quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                gripper_state
            ])
            observations.append(obs)
            waypoints.append(waypoint)

        observations.pop(-1)

        # 检查门是否打开
        final_qpos = float(env.physics.bind(door_joint).qpos[0])
        if abs(final_qpos - current_qpos) > 0.1:
            stage_success = True
            print(f"[custom_open_door] ✓ 门已打开: {current_qpos:.4f} -> {final_qpos:.4f}")
        else:
            print(f"[custom_open_door] ✗ 门未打开: {current_qpos:.4f} -> {final_qpos:.4f}")

        return observations, waypoints, stage_success, task_success