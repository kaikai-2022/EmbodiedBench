"""
Skill Library for data generation.
"""
import logging
import numpy as np
import random
import time as _time
import mujoco
from VLABench.utils.utils import find_keypoint_and_prepare_grasp, distance, quaternion_to_euler, euler_to_quaternion, quaternion_from_axis_angle, quaternion_multiply
from VLABench.algorithms.motion_planning.rrt import rrt_motion_planning
from VLABench.algorithms.utils import interpolate_path, qauternion_slerp

logger = logging.getLogger(__name__)


# ========== Grasp Lock: Quaternion Helpers (shared with dm_env.py) ==========
def _quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def _quat_rotate(q, v):
    qv = np.array([0, v[0], v[1], v[2]])
    return _quat_mul(_quat_mul(q, qv), _quat_conjugate(q))[1:]

PRIOR_EULERS = [[np.pi, 0, -np.pi/2], # face down, horizontal
                [np.pi, 0, 0], # face down, vertical
                [-np.pi/2, -np.pi/2, 0], # face forward, horizontal
                [-np.pi/2, 0, 0], # face forward, vertical
            ]

class SkillLib:
    @staticmethod
    def _get_gripper_state(env, gripper_state=None):
        """获取正确的 gripper_state：lock 模式下保持当前宽度，否则使用传入值"""
        if gripper_state is not None:
            return gripper_state
        if hasattr(env, "_lock_gripper_state") and env._lock_gripper_state is not None:
            return env._lock_gripper_state
        # 默认关闭
        return np.zeros(2)

    @staticmethod
    def step_trajectory(env,
                        points,
                        quats,
                        gripper_state,
                        max_n_substep=1,
                        tolerance=0.02):
        """
        Universal step function for data generation.
        Input:
            env: LM4ManipEnv, for success detection
            points: np.array (n, 3), target positions
            quates: np.array(n, 4), target quaternions
            gripper_state: np.array(2), gripper state
            max_n_step: int, max number of substep for each step
            tolerance: float, tolerance for the qpos error between the target and the current qpos
        Return:
            observations: list of observations
            waypoints: list of waypoints
            stage_success: bool, whether the stage is successful
            task_success: bool, whether the task is successful
        """
        observations = []
        waypoints = []
        stage_success = False
        task_success = False
        last_executed_waypoint = -1

        for i, (point, quat) in enumerate(zip(points, quats)):
            success, action = env.robot.get_qpos_from_ee_pos(physics=env.physics, pos=point, quat=quat)
            action = np.concatenate([action, gripper_state])
            waypoint = np.concatenate([point, quaternion_to_euler(quat), gripper_state])

            for substep_idx in range(max_n_substep):
                timestep = env.step(action)

                if timestep.last():
                    task_success = True
                    break

                current_qpos = np.array(env.task.robot.get_qpos(env.physics)).reshape(-1)
                qpos_max_error = np.max(current_qpos - np.array(action[:7]))
                qpos_min_error = np.min(current_qpos - np.array(action[:7]))

                if qpos_max_error < tolerance and qpos_min_error > -tolerance:
                    break

            last_executed_waypoint = i

            if task_success:
                break

            obs = env.get_observation()
            observations.append(obs)
            waypoints.append(waypoint)

        if len(points) > 0:
            final_ee_pos = env.robot.get_end_effector_pos(env.physics)
            final_distance = distance(points[-1], final_ee_pos)
            if final_distance < tolerance:
                stage_success = True

        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, stage_success, task_success

    @staticmethod
    def moveto_entity(env, target_entity_name, offset=None, gripper_state=None, **kwargs):
        """移动到指定实体上方（自动从实体位置计算目标位置）"""
        if offset is None:
            offset = np.array([0, 0, 0.2])
        entity = env.task.entities[target_entity_name]
        target_pos = np.array(entity.get_xpos(env.physics)) + offset
        return SkillLib.moveto(env, target_pos, gripper_state=gripper_state, **kwargs)

    @staticmethod
    def moveto(env,
               target_pos,
               target_quat=None,
               target_velocity=0.05,
               gripper_state=None,
               **kwargs
               ):
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(env.physics)

        # DEBUG: 打印 moveto 开始信息
        observations = [env.get_observation()]
        waypoints = []
        task_success = False
        gripper_closed = env.robot.get_ee_open_state(env.physics)
        if gripper_state is None:
            gripper_state = SkillLib._get_gripper_state(env)

        obstacle_pcd = np.asarray(env.get_obstacle_pcd().points)

        if target_quat is None:
            target_quat = start_quat

        motion_planning_path = rrt_motion_planning(tuple(start_pos),
                                                tuple(target_pos),
                                                obstacle_pcd)
        if motion_planning_path is None:
            motion_planning_path = [start_pos, target_pos]

        quats_in_path = []
        for t in np.linspace(0, 1, len(motion_planning_path), endpoint=False):
            quats_in_path.append(qauternion_slerp(start_quat, target_quat, t))

        interplate_path, interplate_quat = interpolate_path(np.array(motion_planning_path),
                                                            np.array(quats_in_path),
                                                            target_velocity)

        new_obs, new_waypoints, stage_success, task_success = SkillLib.step_trajectory(env,
                                                                   interplate_path,
                                                                   interplate_quat,
                                                                   gripper_state,
                                                                   **kwargs)
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        observations.pop(-1)

        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, stage_success, task_success
    
    @staticmethod
    def pick(env, 
             target_entity_name, 
             target_pos=None, 
             target_quat=None, 
             prepare_distance=-0.1, 
             prepare_quat=None,
             prior_eulers=PRIOR_EULERS, 
             specific_keypoint=None,
             target_velocity=0.05,
             motion_planning_kwargs=dict(),
             **kwargs):
        """
        general pick function for data generation
        param:
            env: LM4manipEnv object
            target_entity_name: str, target entity name
            target_pos: np.array, target position. If None, will propose a target position automatically
            target_quat: np.array, target quaternion. If None, will propose a target quaternion automatically
            prepare_distance: float, distance to grasp point
            specific_keypoint: int, specific keypoint id, for multi keypoints such as drawer
        return: 
            observations: list of obs
            waypoints: list of waypoints
            key_frames: list of key action such as move to prepare point, grasp
        """
        target_entity = env.task.entities[target_entity_name]

        # DEBUG: 打印实体位置信息
        print(f"\n{'='*60}")
        print(f"DEBUG [pick]: 目标实体 = {target_entity_name}")
        print(f"实体 worldbody 位置: {target_entity.get_xpos(env.physics)}")
        print(f"实体抓取点: {target_entity.get_grasped_keypoints(env.physics)}")
        print(f"机械臂起始位置: {env.robot.get_end_effector_pos(env.physics)}")
        print(f"{'='*60}\n")

        if target_pos is None or target_quat is None:
            key_pos, prepare_key_pos, key_quat = find_keypoint_and_prepare_grasp(env, target_entity, prior_eulers, specific_keypoint_id=specific_keypoint, move_vector=prepare_quat)
            if key_pos is None or prepare_key_pos is None:
                print("DEBUG [pick]: can not find valid keypoint and prepare point, reset the env")
                return None
            else:
                print(f"DEBUG [pick]: 找到有效抓取点!")
                print(f"  抓取点位置 (key_pos): {key_pos}")
                print(f"  准备点位置 (prepare_pos): {prepare_key_pos}")
                print(f"  抓取姿态 (key_quat): {key_quat}")
        else:
            key_pos, key_quat = target_pos, target_quat
            if prepare_quat is None:
                gripper_pcd, move_quat = env.robot.gripper_pcd(key_pos, key_quat)
            else:
                move_quat = prepare_quat
            prepare_key_pos = key_pos + move_quat * prepare_distance
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(env.physics)
        # env_pcd = env.get_observation()["masked_point_cloud"]
        # obstacle_pcd = np.asarray(env_pcd.points)

        obstacle_pcd = np.asarray(env.get_obstacle_pcd().points)

        start_pos, start_quat, key_quat, prepare_pos, key_pos = np.array(start_pos), np.array(start_quat), np.array(key_quat), np.array(prepare_key_pos), np.array(key_pos)

        print(f"DEBUG [pick]: 运动规划路径")
        print(f"  起始位置: {start_pos}")
        print(f"  准备位置: {prepare_pos}")
        print(f"  抓取位置: {key_pos}")

        # motion planning -> path(start, prepare_point) & path(prepare_point, key_point)
        init2prepare_path = rrt_motion_planning(tuple(start_pos),
                                                tuple(prepare_pos),
                                                obstacle_pcd,
                                                **motion_planning_kwargs)
        if init2prepare_path is None:
            init2prepare_path = [start_pos, prepare_pos]
            print(f"DEBUG [pick]: RRT规划失败，使用直线路径")
        else:
            print(f"DEBUG [pick]: RRT规划成功，路径点数: {len(init2prepare_path)}")
        quats_in_path = [start_quat for _ in range(len(init2prepare_path)-1)]
        quats_in_path.append(key_quat)
        # quats_in_path = []
        # for t in np.linspace(0, 1, len(init2prepare_path), endpoint=False):
        #     quats_in_path.append(qauternion_slerp(start_quat, key_quat, t))
        init2prepare_path.append(tuple(key_pos))
        path = np.array(init2prepare_path)
        quats_in_path.append(key_quat)
        
        interplate_path, interplate_quat = interpolate_path(path, quats_in_path, target_velocity)

        print(f"DEBUG [pick]: interpolated path")
        print(f"  path points: {len(interplate_path)}")
        print(f"  last 5 points:")
        for i, p in enumerate(interplate_path[-5:]):
            print(f"    [{len(interplate_path)-5+i}]: {p}")

        # DEBUG: print finger midpoint vs beaker center
        import mujoco as mj
        raw_m = env.physics.model._model
        raw_d = env.physics.data._data
        # 用 gripper_geoms 获取手指 geom 的世界坐标
        gripper_geoms = env.robot.gripper_geoms
        gripper_positions = []
        for geom in gripper_geoms:
            eid = env.physics.bind(geom).element_id
            gname = mj.mj_id2name(raw_m, mj.mjtObj.mjOBJ_GEOM, eid) or ''
            gpos = raw_d.geom_xpos[eid]
            gripper_positions.append((gname, gpos, eid))
            print(f"DEBUG [pick]: gripper geom '{gname}' id={eid}: ({gpos[0]:.4f}, {gpos[1]:.4f}, {gpos[2]:.4f})")
        # 取所有 pad geom 的中点
        pad_positions = [pos for name, pos, _ in gripper_positions if 'pad' in name.lower()]
        if len(pad_positions) >= 2:
            finger_mid = sum(pad_positions) / len(pad_positions)
        else:
            finger_mid = sum([pos for _, pos, _ in gripper_positions]) / len(gripper_positions)
        beaker_pos = np.array(target_entity.get_xpos(env.physics))
        print(f"DEBUG [pick]: === BEFORE GRASP ===")
        print(f"  finger_mid: ({finger_mid[0]:.4f}, {finger_mid[1]:.4f}, {finger_mid[2]:.4f})")
        print(f"  beaker:     ({beaker_pos[0]:.4f}, {beaker_pos[1]:.4f}, {beaker_pos[2]:.4f})")
        print(f"  XY_diff_mm: ({abs(finger_mid[0]-beaker_pos[0])*1000:.1f}, {abs(finger_mid[1]-beaker_pos[1])*1000:.1f})")

        waypoints = []
        stage_success = False
        task_success = False
        observations = [env.get_observation()]
        # move along the interplated path
        gripper_state = np.ones(2) * 0.04

        new_obs, new_waypoints, _, task_success = SkillLib.step_trajectory(env,
                                                                   interplate_path,
                                                                   interplate_quat,
                                                                   gripper_state,
                                                                   **kwargs)
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        # 无论 step_trajectory 返回什么，都要执行 close_gripper 完成抓取
        # grasp
        new_obs, new_waypoints, _, _ = SkillLib.close_gripper(env)
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)

        observations.pop(-1)
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        if env.task.entities[target_entity_name].is_grasped(env.physics, env.robot):
            stage_success = True
            print(f"DEBUG [pick]: ✓ 抓取成功! stage_success=True")

            # ========== close_gripper 完成后立即设置 lock ==========
            # 关键：立即启用同步，让 step() 中的同步逻辑生效
            # 不再等待手指"完全稳定"，因为那时物体可能已经开始滑动
            if hasattr(env, "_grasp_lock_mode") and env._grasp_lock_mode > 0:
                raw_m = env.physics.model._model
                raw_d = env.physics.data._data
                hand_id = mujoco.mj_name2id(raw_m, mujoco.mjtObj.mjOBJ_BODY, "franka/hand")
                if hand_id >= 0:
                    hand_pos = raw_d.xpos[hand_id].copy()
                    hand_quat = raw_d.xquat[hand_id].copy()
                    obj_pos = np.array(env.task.entities[target_entity_name].get_xpos(env.physics))
                    obj_quat = np.array(env.task.entities[target_entity_name].get_xqaut(env.physics))
                    rel_pos = obj_pos - hand_pos
                    rel_quat = _quat_mul(_quat_conjugate(hand_quat), obj_quat)
                    env._grasped_entity_info = {
                        "name": target_entity_name,
                        "rel_pos_local": _quat_rotate(_quat_conjugate(hand_quat), rel_pos),
                        "rel_quat_local": rel_quat,
                    }
                    # 读取当前手指关节的实际 qpos，lock 模式下保持这个宽度不再合紧
                    finger1_qpos = raw_d.qpos[raw_m.jnt_qposadr[mujoco.mj_name2id(raw_m, mujoco.mjtObj.mjOBJ_JOINT, "franka/finger_joint1")]]
                    finger2_qpos = raw_d.qpos[raw_m.jnt_qposadr[mujoco.mj_name2id(raw_m, mujoco.mjtObj.mjOBJ_JOINT, "franka/finger_joint2")]]
                    env._lock_gripper_state = np.array([finger1_qpos, finger2_qpos])
                    print(f"DEBUG [pick]: ✓ lock 模式 {env._grasp_lock_mode} 已启用, 手指保持宽度: [{finger1_qpos:.4f}, {finger2_qpos:.4f}]")

                    # Mode 2: 创建 weld 约束
                    if env._grasp_lock_mode == 2 and hasattr(env, "_setup_weld_constraint"):
                        env._setup_weld_constraint()

                # 等几帧让 lock 同步稳定
                for wait_frame in range(5):
                    arm_qpos = np.array(env.robot.get_qpos(env.physics))
                    action = np.concatenate([arm_qpos, env._lock_gripper_state])
                    env.step(action)
                    mujoco.mj_forward(raw_m, raw_d)
        else:
            print(f"DEBUG [pick]: ✗ 抓取失败! is_grasped=False")
            print(f"  末端执行器最终位置: {env.robot.get_end_effector_pos(env.physics)}")
            print(f"  目标抓取点位置: {key_pos}")
            print(f"  位置误差: {np.linalg.norm(env.robot.get_end_effector_pos(env.physics) - key_pos)}")
        # pick 只是中间步骤，不是任务完成，所以 task_success 始终为 False
        return observations, waypoints, stage_success, False

    @staticmethod
    def gently_pick(env,
                    target_entity_name,
                    target_pos=None,
                    target_quat=None,
                    prepare_distance=-0.1,
                    prepare_quat=None,
                    prior_eulers=PRIOR_EULERS,
                    specific_keypoint=None,
                    target_velocity=0.05,
                    extra_close_ratio=0.2,
                    n_close_steps=20,
                    contact_dist_threshold=0.005,
                    hold_steps=5,
                    motion_planning_kwargs=dict(),
                    **kwargs):
        """
        柔性抓取：移动到抓取点后，逐步闭合夹爪；当左右两侧手指均与目标物体
        产生接触时，仅再额外合上 extra_close_ratio 比例的开度（即 0.04 的
        extra_close_ratio），然后**主动发控制信号让夹爪保持在该宽度**，
        不会再继续合紧。无需开启 grasp lock。

        与 pick() 的关键区别：
        - pick() 直接调用 close_gripper 合到 0，可能压坏物体；
        - gently_pick() 在两侧触碰物体时停止主动合拢，仅做轻压。

        保持宽度的实现：franka 夹爪是 position-controlled actuator，
        持续给 `gripper_state = [w, w]` 就会把手指位置伺服到 w 并停在那里。
        物理接触会让手指停在物体表面而不是继续往下合——但为了避免伺服
        持续施力压坏物体，我们用 hold_steps 步后的**实际手指 qpos**作为
        目标宽度，这样 servo 的目标就是"当前在哪停在哪"。

        param:
            env: LM4manipEnv object
            target_entity_name: str, target entity name
            target_pos / target_quat: 可选的手动抓取点；为 None 时由 keypoint 算法生成
            prepare_distance: 准备点距抓取点的距离（沿 move_vector 方向）
            prepare_quat: 准备姿态
            prior_eulers: 候选欧拉角
            specific_keypoint: 指定 keypoint id
            target_velocity: 路径插值速度
            extra_close_ratio: 两侧接触后再额外合上的开度比例（相对 0.04），默认 0.2
            n_close_steps: 闭合阶段最多步数
            contact_dist_threshold: 判定手指-物体接触的距离阈值（m）
            hold_steps: 轻压后稳定步数；用这段时间内的实际 qpos 作为最终保持目标
            motion_planning_kwargs: 传给 RRT 的额外参数
        return:
            observations, waypoints, stage_success, False
        """
        target_entity = env.task.entities[target_entity_name]

        if target_pos is None or target_quat is None:
            key_pos, prepare_key_pos, key_quat = find_keypoint_and_prepare_grasp(
                env, target_entity, prior_eulers, specific_keypoint_id=specific_keypoint, move_vector=prepare_quat)
            if key_pos is None or prepare_key_pos is None:
                print("DEBUG [gently_pick]: can not find valid keypoint and prepare point, reset the env")
                return None
        else:
            key_pos, key_quat = target_pos, target_quat
            if prepare_quat is None:
                gripper_pcd, move_quat = env.robot.gripper_pcd(key_pos, key_quat)
            else:
                move_quat = prepare_quat
            prepare_key_pos = key_pos + move_quat * prepare_distance

        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(env.physics)
        obstacle_pcd = np.asarray(env.get_obstacle_pcd().points)
        start_pos, start_quat, key_quat, prepare_pos, key_pos = (
            np.array(start_pos), np.array(start_quat), np.array(key_quat),
            np.array(prepare_key_pos), np.array(key_pos)
        )

        # 1) RRT 规划：start -> prepare -> key
        init2prepare_path = rrt_motion_planning(tuple(start_pos),
                                                tuple(prepare_pos),
                                                obstacle_pcd,
                                                **motion_planning_kwargs)
        if init2prepare_path is None:
            init2prepare_path = [start_pos, prepare_pos]
        quats_in_path = [start_quat for _ in range(len(init2prepare_path) - 1)]
        quats_in_path.append(key_quat)
        init2prepare_path.append(tuple(key_pos))
        path = np.array(init2prepare_path)
        quats_in_path.append(key_quat)

        interplate_path, interplate_quat = interpolate_path(path, quats_in_path, target_velocity)

        waypoints = []
        observations = [env.get_observation()]
        stage_success = False
        task_success = False

        # 2) 张开夹爪移动到抓取点
        gripper_state = np.ones(2) * 0.04
        new_obs, new_waypoints, _, task_success = SkillLib.step_trajectory(
            env, interplate_path, interplate_quat, gripper_state, **kwargs)
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)

        # 3) 逐步闭合夹爪，检测两侧接触
        import mujoco as mj
        raw_m = env.physics.model._model
        raw_d = env.physics.data._data

        left_geoms = env.robot.mjcf_model.find("body", "left_finger").find_all("geom")
        right_geoms = env.robot.mjcf_model.find("body", "right_finger").find_all("geom")
        left_geom_ids = {env.physics.bind(g).element_id for g in left_geoms}
        right_geom_ids = {env.physics.bind(g).element_id for g in right_geoms}
        target_geom_ids = {env.physics.bind(g).element_id for g in target_entity.geoms}

        # 获取两侧 pad geom 的世界坐标中点
        def _get_finger_pad_centers():
            left_pad_pos = np.zeros(3)
            right_pad_pos = np.zeros(3)
            for g in left_geoms:
                eid = env.physics.bind(g).element_id
                gname = mj.mj_id2name(raw_m, mj.mjtObj.mjOBJ_GEOM, eid) or ''
                if 'pad' in gname.lower():
                    left_pad_pos = raw_d.geom_xpos[eid].copy()
                    break
            for g in right_geoms:
                eid = env.physics.bind(g).element_id
                gname = mj.mj_id2name(raw_m, mj.mjtObj.mjOBJ_GEOM, eid) or ''
                if 'pad' in gname.lower():
                    right_pad_pos = raw_d.geom_xpos[eid].copy()
                    break
            return left_pad_pos, right_pad_pos

        def _side_has_contact(side_ids):
            for c in raw_d.contact:
                if c.dist > contact_dist_threshold:
                    continue
                if (c.geom1 in side_ids and c.geom2 in target_geom_ids) or \
                   (c.geom2 in side_ids and c.geom1 in target_geom_ids):
                    return True
            return False

        # 关键修复：闭合开始时记录 arm qpos，整个闭合过程保持不变
        # 原因：像 close_gripper 一样，避免每次循环读取当前 qpos 导致微小扰动累积
        arm_qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        both_touched = False
        touch_pad_dist = None  # 双侧接触瞬间两指 pad 的实际距离
        for i in range(n_close_steps):
            # 从 0.04 线性合到 0
            target = 0.04 * (1.0 - (i + 1) / n_close_steps)
            gripper_state = np.ones(2) * target
            # 使用固定的 arm_qpos，而不是每次重新读取
            action = np.concatenate([arm_qpos, gripper_state])
            timestep = env.step(action)
            if timestep.last():
                task_success = True
                break

            waypoint = np.concatenate([
                env.robot.get_end_effector_pos(env.physics),
                quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                gripper_state,
            ])
            observations.append(env.get_observation())
            waypoints.append(waypoint)

            if _side_has_contact(left_geom_ids) and _side_has_contact(right_geom_ids):
                both_touched = True
                # 记录双侧接触瞬间两指 pad 的实际距离（而非 finger joint 的目标值）
                left_pad_pos, right_pad_pos = _get_finger_pad_centers()
                touch_pad_dist = np.linalg.norm(left_pad_pos - right_pad_pos)
                print(f"DEBUG [gently_pick]: 两侧均已接触目标 @ step {i}, "
                      f"pad 间距={touch_pad_dist:.4f}m, target={target:.4f}")
                break

        if not both_touched:
            # 走完所有步仍未两侧接触 → 退化：直接读取两指 pad 距离作为 fallback
            print("DEBUG [gently_pick]: 警告：未检测到双侧接触，使用当前 pad 距离作为 fallback")
            left_pad_pos, right_pad_pos = _get_finger_pad_centers()
            touch_pad_dist = np.linalg.norm(left_pad_pos - right_pad_pos)

        # 4) 基于 pad 实际距离轻压 extra_close_ratio 比例
        # 逻辑：双侧接触时两指 pad 距离为 touch_pad_dist；
        #      再缩小该距离的 extra_close_ratio 比例（默认 20%）
        # 最终 pad 目标距离 = touch_pad_dist * (1 - extra_close_ratio)
        # 然后把这个距离转换为每侧 finger joint 的目标位置（除以 2）
        extra_pad_dist = max(touch_pad_dist * (1 - extra_close_ratio), 0.0)
        # gripper_state 是单侧 finger joint 的位移（0~0.04），pad 距离 ≈ 2 * gripper_state
        extra_target = extra_pad_dist / 2.0
        print(f"DEBUG [gently_pick]: 轻压 pad 距离: {touch_pad_dist:.4f} -> {extra_pad_dist:.4f} "
              f"(gripper_state: {touch_pad_dist/2:.4f} -> {extra_target:.4f})")

        # 轻压开始前重新读取 arm_qpos，避免闭合循环中的微小累积误差
        arm_qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        touch_gripper = touch_pad_dist / 2.0  # 双侧接触时的单侧 gripper 宽度

        # 短时间内逐步逼近 extra_target，避免瞬时跳变
        for j in range(3):
            interp = touch_gripper + (extra_target - touch_gripper) * (j + 1) / 3.0
            gripper_state = np.ones(2) * interp
            # 使用重新读取的 arm_qpos
            action = np.concatenate([arm_qpos, gripper_state])
            timestep = env.step(action)
            if timestep.last():
                task_success = True
                break
            waypoint = np.concatenate([
                env.robot.get_end_effector_pos(env.physics),
                quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                gripper_state,
            ])
            observations.append(env.get_observation())
            waypoints.append(waypoint)

        # 5) 稳定几步，继续发 extra_target 目标
        # 原因：轻压阶段已经基于 pad 距离计算了 extra_target
        # 直接用 extra_target 作为 gripper_state，避免读取"被物体撑开后的实际宽度"
        # 对于轻小物体（如滴管），读取实际宽度会导致过松而滑落
        print(f"DEBUG [gently_pick]: 稳定 {hold_steps} 步，gripper_state 固定为 {extra_target:.4f}")
        for _ in range(hold_steps):
            arm_qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
            gripper_state = np.ones(2) * extra_target  # 使用目标宽度，不是实际宽度
            action = np.concatenate([arm_qpos, gripper_state])
            timestep = env.step(action)
            if timestep.last():
                task_success = True
                break
            waypoint = np.concatenate([
                env.robot.get_end_effector_pos(env.physics),
                quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                gripper_state,
            ])
            observations.append(env.get_observation())
            waypoints.append(waypoint)

        # 最终保持的目标宽度
        final_hold_width = np.ones(2) * extra_target
        print(f"DEBUG [gently_pick]: 最终保持宽度: {final_hold_width} (基于 extra_target {extra_target:.4f})")
        # 设置 _lock_gripper_state 让后续 moveto/lift/place 技能继续用这个宽度
        env._lock_gripper_state = final_hold_width
        print(f"DEBUG [gently_pick]: ✓ _lock_gripper_state 已设为 {final_hold_width}")

        observations.pop(-1)
        assert len(observations) == len(waypoints), \
            f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        if env.task.entities[target_entity_name].is_grasped(env.physics, env.robot):
            stage_success = True
            print(f"DEBUG [gently_pick]: ✓ 轻抓成功! stage_success=True")
        else:
            print(f"DEBUG [gently_pick]: ✗ 轻抓未通过 is_grasped 判定")
        return observations, waypoints, stage_success, False

    @staticmethod
    def place(env,
              target_container_name,
              target_pos=None,
              target_quat=None,
              motion_planning_kwargs=dict()):
        """
        将抓取的物体精确放置到目标容器内部。

        与 drop() 的区别：
        - place() 用于精确放置到容器内部（如烧杯、盒子），使用容器的 place_point
        - place() 不使用 ee_offset，只做物体高度补偿
        - drop() 用于放到桌面，使用 ee_offset 补偿夹爪几何

        param:
            env: LM4manipEnv object
            target_container_name: str, 目标容器名称
            target_pos: np.array, 目标位置。如果为 None，使用容器的 place_point。
            target_quat: np.array, 目标姿态。如果为 None，使用当前姿态。
        return:
            observations: list of obs
            waypoints: list of actions
            stage_success: bool, 技能是否成功
            task_success: bool, 任务是否完成
        """
        target_container = env.task.entities[target_container_name]
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(env.physics)
        if target_pos is None:
            place_points = target_container.get_place_point(env.physics)
            if not place_points:
                print("can not find valid place point, reset the env")
                return None
            if isinstance(place_points, list):
                place_point = random.choice(place_points)
            target_pos = place_point
        if target_quat is None:
            target_quat = env.robot.get_end_effector_quat(env.physics)

        print(f"DEBUG [place]: === 开始 place 技能 ===")
        print(f"DEBUG [place]: 目标容器: {target_container_name}")
        print(f"DEBUG [place]: 目标 place_point (世界坐标): ({target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f})")
        print(f"DEBUG [place]: 容器 xpos: {np.array(target_container.get_xpos(env.physics))}")
        print(f"DEBUG [place]: 机械臂起始位置 EE: ({start_pos[0]:.4f}, {start_pos[1]:.4f}, {start_pos[2]:.4f})")

        obstacle_pcd = np.asarray(env.get_obstacle_pcd().points)
        # np.save("obstacle_pcd.npy", obstacle_pcd)
        
        start_pos, start_quat, target_pos, target_quat = np.array(start_pos), np.array(start_quat), np.array(target_pos), np.array(target_quat)

        # 两阶段路径规划：先到目标正上方，再垂直下降
        # 这样可以避免从侧面碰撞容器
        safe_height = max(start_pos[2], target_pos[2] + 0.1)  # 至少比目标高 10cm
        above_target = np.array([target_pos[0], target_pos[1], safe_height])

        # 阶段1：从当前位置到目标正上方（RRT 避障）
        init2above_path = rrt_motion_planning(tuple(start_pos),
                                               tuple(above_target),
                                               obstacle_pcd,
                                               **motion_planning_kwargs)
        if init2above_path is None:
            print("can not find a path to above target, use default")
            init2above_path = [start_pos, above_target]

        # 阶段2：从正上方垂直下降到目标位置
        init2above_path.append(tuple(target_pos))
        path = np.array(init2above_path)
        print(f"DEBUG [place]: RRT 路径点: {len(path)} 个")
        print(f"DEBUG [place]: RRT 路径首尾点: 起点({path[0][0]:.4f}, {path[0][1]:.4f}, {path[0][2]:.4f}) -> 终点({path[-1][0]:.4f}, {path[-1][1]:.4f}, {path[-1][2]:.4f})")

        # 补偿被抓取物体的高度：
        # place() 用于精确放置到容器内部，使用容器的 place_point 作为目标
        # 物体底部可能在 end_effector_move 下方（被夹爪夹在中上部）
        # 需要把路径抬高，让物体底部刚好在 target_pos Z 高度
        grasped_names, grasped_entities = env.get_grasped_entity()
        if grasped_entities:
            grasped_entity = grasped_entities[0]
            obj_bottom_z = np.array(grasped_entity.get_xpos(env.physics))[2]
            ee_move_site = env.robot._mjcf_model.find("site", "end_effector_move")
            ee_move_z = env.physics.bind(ee_move_site).xpos[2]
            z_offset = obj_bottom_z - ee_move_z
            print(f"DEBUG [place]: === 高度补偿计算 ===")
            print(f"DEBUG [place]: 物体位置 (xpos): ({obj_bottom_z:.4f})")
            print(f"DEBUG [place]: ee_move_site Z: {ee_move_z:.4f}")
            print(f"DEBUG [place]: z_offset = obj_bottom_z - ee_move_z = {z_offset:.4f}")
            if z_offset < 0:
                path[:, 2] += (-z_offset)
                print(f"DEBUG [place]: 抬高路径 {max(0, -z_offset):.4f}m")
            else:
                print(f"DEBUG [place]: 无需抬高 (z_offset >= 0)")
            print(f"DEBUG [place]: 补偿后路径终点 Z: {path[-1][2]:.4f}")

        path_point_len = len(path)
        quats = [start_quat for _ in range(path_point_len)]
        quats[-1] = target_quat
        interplate_path, interplate_quat = interpolate_path(path, quats)   
        observations= [env.get_observation()]
        waypoints = []
        stage_success = True
        task_success = False
        new_obs, new_waypoints, _, task_success = SkillLib.step_trajectory(env,
                                                                interplate_path,
                                                                interplate_quat,
                                                                SkillLib._get_gripper_state(env))
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        if task_success:
            observations.pop(-1)
            assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
            return observations, waypoints, True, task_success
        # 松开夹爪
        new_obs, new_waypoints, _, task_success = SkillLib.open_gripper(env)
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)

        # 松开后向上抬升，避免碰撞刚放下的物体
        new_obs, new_waypoints, _, _ = SkillLib.lift(env, lift_height=0.05, gripper_state=np.ones(2) * 0.04)
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)

        observations.pop(-1)
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        for entity in env.task.entities.values():
            if hasattr(entity, "is_grasped") and entity.is_grasped(env.physics, env.robot):
                stage_success = False
        return observations, waypoints, stage_success, task_success

    @staticmethod
    def drop(env,
             target_surface_pos=None,
             target_quat=None,
             drop_height=0.05,
             motion_planning_kwargs=dict()):
        """
        将抓取的物体安全地放到桌面上。

        与 place() 的区别：
        - drop() 使用 ee_offset 来补偿夹爪几何，确保物体不会与桌面碰撞
        - drop() 适用于放到桌面/平台等平坦表面
        - place() 用于精确放置到容器内部（不需要 ee_offset）

        param:
            env: LM4manipEnv object
            target_surface_pos: np.array, 目标表面位置。如果为 None，使用当前位置正下方的桌面位置。
            target_quat: np.array, 目标姿态。如果为 None，使用当前姿态。
            drop_height: float, 物体底部距离表面的高度（m），默认 0.05m。
            motion_planning_kwargs: dict, 运动规划参数。
        return:
            observations: list of obs
            waypoints: list of actions
            stage_success: bool, 技能是否成功
            task_success: bool, 任务是否完成
        """
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(env.physics)

        if target_surface_pos is None:
            # 默认：物体正下方的桌面位置
            grasped_names, grasped_entities = env.get_grasped_entity()
            if grasped_entities:
                obj_pos = np.array(grasped_entities[0].get_xpos(env.physics))
                target_surface_pos = np.array([obj_pos[0], obj_pos[1], obj_pos[2] - drop_height])
            else:
                target_surface_pos = np.array([start_pos[0], start_pos[1], start_pos[2] - 0.1])

        if target_quat is None:
            target_quat = start_quat

        obstacle_pcd = np.asarray(env.get_obstacle_pcd().points)
        start_pos, start_quat, target_surface_pos, target_quat = np.array(start_pos), np.array(start_quat), np.array(target_surface_pos), np.array(target_quat)

        # 运动规划
        init2target_path = rrt_motion_planning(tuple(start_pos),
                                               tuple(target_surface_pos),
                                               obstacle_pcd,
                                               **motion_planning_kwargs)
        offset = env.robot.ee_offset(env.physics)  # 夹爪几何补偿，防止物体撞桌面
        if init2target_path is None:
            print("can not find a path to target position, use default lift")
            if start_pos[2] <= target_surface_pos[2]:
                mid_point = np.array([start_pos[0], start_pos[1], target_surface_pos[2]])
            else:
                mid_point = np.array([target_surface_pos[0], target_surface_pos[1], start_pos[2]])
            init2target_path = [start_pos, mid_point, target_surface_pos]

        path = np.array(init2target_path)
        path += offset  # ee_offset 补偿，确保物体底部距离桌面有足够间隙

        # 额外高度补偿：如果物体底部会低于目标位置，抬高路径
        grasped_names, grasped_entities = env.get_grasped_entity()
        if grasped_entities:
            grasped_entity = grasped_entities[0]
            obj_bottom_z = np.array(grasped_entity.get_xpos(env.physics))[2]
            ee_move_site = env.robot._mjcf_model.find("site", "end_effector_move")
            ee_move_z = env.physics.bind(ee_move_site).xpos[2]
            z_offset = obj_bottom_z - ee_move_z
            if z_offset < 0:
                path[:, 2] += (-z_offset)
            print(f"DEBUG [drop]: 物体高度补偿")
            print(f"  物体底部 Z: {obj_bottom_z:.4f}, ee_move Z: {ee_move_z:.4f}")
            print(f"  z_offset: {z_offset:.4f}m, 抬高量: {max(0, -z_offset):.4f}m")

        path_point_len = len(init2target_path)
        quats = [start_quat for _ in range(path_point_len)]
        quats[-1] = target_quat
        interplate_path, interplate_quat = interpolate_path(path, quats)

        observations = [env.get_observation()]
        waypoints = []
        stage_success = True
        task_success = False

        new_obs, new_waypoints, _, task_success = SkillLib.step_trajectory(
            env, interplate_path, interplate_quat, SkillLib._get_gripper_state(env)
        )
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)

        if task_success:
            observations.pop(-1)
            assert len(observations) == len(waypoints)
            return observations, waypoints, True, task_success

        # 释放夹爪
        new_obs, new_waypoints, _, task_success = SkillLib.open_gripper(env)
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)

        observations.pop(-1)
        assert len(observations) == len(waypoints)

        # 检查是否仍有物体被抓取
        for entity in env.task.entities.values():
            if hasattr(entity, "is_grasped") and entity.is_grasped(env.physics, env.robot):
                stage_success = False
        return observations, waypoints, stage_success, task_success

    @staticmethod
    def open_door(env,
                  target_container_name):
        """
        Open the door of the target container
        Input:
            env: LM4manipEnv object
            target_container_name: str, target container name
        Return:
            observations: list of observations
            waypoints: list of waypoints
            trajectory: list of trajectory
            trajectory_quats: list of trajectory quaternions
        """
        target_container = env.task.entities[target_container_name]
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(env.physics)
        
        trajectory = target_container.get_open_trajectory(env.physics)
        trajectory_quats = []
        door_joint = target_container.door_joint
        rotation_axis = env.physics.bind(door_joint).xaxis
        # rotation_anchor = env.physics.bind(door_joint).xanchor
        observations = [env.get_observation()]
        waypoints = []
        stage_success = False
        task_success = False
        for i in range(len(trajectory)):
            rot_quat = quaternion_from_axis_angle(rotation_axis, -0.1*(i+1))
            new_quat = quaternion_multiply(start_quat, rot_quat)
            trajectory_quats.append(new_quat)
        # init_qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        interplate_path, interplate_quat = interpolate_path(trajectory, trajectory_quats)
        new_obs, new_waypoints, _, task_success = SkillLib.step_trajectory(env,
                                                            interplate_path, 
                                                            interplate_quat, 
                                                            np.zeros(2))
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        pos, quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(env.physics)
        for _ in range(10):
            action = np.concatenate([qpos, np.ones(2)*(0.04/10)*(i+1)])
            timestep = env.step(action)
            if timestep.last():
                task_success = True
                break
            obs = env.get_observation()
            observations.append(obs)
            waypoints.append(np.concatenate([pos, quaternion_to_euler(quat), np.ones(2)*0.04]))
        observations.pop(-1)
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        if env.task.entities[target_container_name].is_open(env.physics):
            stage_success = True
        return observations, waypoints, stage_success, task_success
    
    @staticmethod
    def close_door(env, target_container_name, gripper_state=np.zeros(2)):
        target_container = env.task.entities[target_container_name]
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(env.physics)
        
        trajectory = target_container.get_close_trajectory(env.physics)
        trajectory_quats = [start_quat for _ in range(len(trajectory))]
        
        observations = [env.get_observation()]
        waypoints = []
        stage_success = False
        task_success = False
        if len(trajectory) == 0:
             return observations, waypoints, True, task_success
        interplate_path, interplate_quat = interpolate_path(trajectory, trajectory_quats)
        obs, new_waypoints, _, task_success = SkillLib.step_trajectory(env,
                                                            interplate_path, 
                                                            interplate_quat, 
                                                            gripper_state)
        observations.extend(obs)
        waypoints.extend(new_waypoints)
        observations.pop(-1)
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        if env.task.entities[target_container_name].is_closed(env.physics):
            stage_success = True
        return observations, waypoints, stage_success, task_success
    
    @staticmethod
    def open_drawer(env, 
                    target_container_name, 
                    pick_prior_eulers=[[-np.pi/2, 0, 0]],
                    drawer_id=0):
        """
        common open drawer function
        Input:
            drawer_id: 0-2 means top to bottom drawer.
            pick_prioer_euelrs: list of list, prior eulers for pick
        """
        observations = [env.get_observation()]
        waypoints = []
        stage_success = False
        task_success = False
        target_container = env.task.entities[target_container_name]
        # grasp handle
        new_obs, new_waypoints, _, success_ = SkillLib.pick(env, target_container_name, prior_eulers=pick_prior_eulers, specific_keypoint=drawer_id)
        
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        task_success = task_success or success_
        # open drawer   
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(env.physics)
        trajectory = target_container.get_drawer_open_trajectory(env.physics, drawer_id)
        trajectory_quats = [start_quat for _ in range(len(trajectory))]
        trajectory, trajectory_quats = interpolate_path(trajectory, trajectory_quats)
        new_obs, new_waypoints, _, success_ = SkillLib.step_trajectory(env, trajectory, trajectory_quats, np.zeros(2))
        
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        task_success = task_success or success_
        observations.pop(-1)
        
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        # TODO check the drawer state
        return observations, waypoints, True, task_success
    
    @staticmethod
    def press(env, target_pos, target_quat=None, move_vector=[0, 0, 0.1], max_n_substep=100): #TODO move vector to determine the press direction
        prepare_pos = target_pos + np.array(move_vector) if move_vector is not None else target_pos
        observations, waypoints, _, _ = SkillLib.moveto(env, 
                                                     prepare_pos, 
                                                     target_quat,
                                                     max_n_substep=max_n_substep)
        # close gripper
        qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        for i in range(10):
            gripper_state = np.ones(2) * (0.04 - i/10 * 0.04)
            action = np.concatenate([qpos, gripper_state])
            timestep = env.step(action)
            if timestep.last():
                break
            obs = env.get_observation()
            observations.append(obs)
            waypoints.append(np.concatenate([env.robot.get_end_effector_pos(env.physics),
                                             quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                                             gripper_state]))
        new_obs, new_waypoints, stage_success, task_success = SkillLib.moveto(env, target_pos, target_quat, max_n_substep=max_n_substep)
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, stage_success, task_success
    
    @staticmethod
    def pull(env, target_pos=None, target_quat=None, gripper_state=None, pull_distance=0.3):
        """
        Common pull function.
        """
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(env.physics)
        if target_pos is None: target_pos = np.array(start_pos) + np.array([0, -pull_distance, 0])
        if target_quat is None: target_quat = start_quat
        interplate_path, interplate_quat = interpolate_path([start_pos, target_pos], [np.array(start_quat), np.array(target_quat)])
        observations = [env.get_observation()]
        waypoints = []
        task_success = False
        if gripper_state is None:
            gripper_state = SkillLib._get_gripper_state(env)
        new_obs, new_waypoints, stage_success, task_success = SkillLib.step_trajectory(env, 
                                                           interplate_path, 
                                                           interplate_quat, 
                                                           gripper_state)
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        observations.pop(-1)
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, stage_success, task_success
    
    @staticmethod
    def push(env, target_pos=None, target_quat=None, gripper_state=None, push_distance=0.3):
        obs, waypoints, stage_success, task_success = SkillLib.pull(env, target_pos, target_quat, gripper_state, -push_distance)
        return obs, waypoints, stage_success, task_success
    
    @staticmethod
    def pour(env, target_delta_qpos=np.pi, target_q_velocity=np.pi/40, n_repeat_step=2, tolerance=0.01):
        """
        Common pour function.
        """
        waypoints = []
        stage_success = False
        task_success = False
        observations = [env.get_observation()]
        
        init_qpos = np.array(env.robot.get_qpos(env.physics))
        gripper_closed = env.robot.get_ee_open_state(env.physics)
        if gripper_closed: gripper_state = SkillLib._get_gripper_state(env)
        else: gripper_state = np.ones(2) * 0.04
        timesteps = int(target_delta_qpos / target_q_velocity)
        for i in range(timesteps):
            action = np.array(init_qpos).reshape(-1)
            action[-1] += target_q_velocity * i
            action = np.concatenate([action, gripper_state])
            for _ in range(n_repeat_step):
                timestep = env.step(action)
                if timestep.last():
                    task_success = True
                    break
                current_qpos = np.array(env.task.robot.get_qpos(env.physics)).reshape(-1)
                if np.max(current_qpos - np.array(action[:7])) < tolerance \
                    and np.min(current_qpos - np.array(action[:7])) > -tolerance:
                  break
            waypoint = np.concatenate([env.robot.get_end_effector_pos(env.physics), 
                                       quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)), 
                                       gripper_state])
            obs = env.get_observation()
            observations.append(obs)
            waypoints.append(waypoint)
            if task_success:
                break
        observations.pop(-1)
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, True, task_success

    @staticmethod
    def pour_to_entity(env, target_container_name, tilt_angle=np.pi*2/3, tilt_velocity=np.pi/80,
                       n_repeat_step=6, lift_before=0.2, wait_time=10):
        """
        倾倒到指定容器上方。

        策略：正上方抓取后，��过 IK 求解逐步倾斜末端执行器，
        同时用 IK 保持末端位置不变（试管口不位移）。
        先抬高避免碰撞，倾倒后等待液体流出。

        Args:
            target_container_name: 目标容器实体名
            tilt_angle: 总倾斜角度（弧度，默认 π/2 = 90°）
            tilt_velocity: 每步倾斜角速度
            n_repeat_step: 每个动作步重复次数
            lift_before: 倾倒前抬高距离（m）
            wait_time: 倾倒后等待步数
        """
        observations = [env.get_observation()]
        waypoints = []
        task_success = False

        # 获取容器位置
        container_pos = np.array(env.task.entities[target_container_name].get_xpos(env.physics))
        pour_target_pos = container_pos + np.array([0, 0, 0.3])

        # 1. 抬高到倾倒高度
        start_pos = np.array(env.robot.get_end_effector_pos(env.physics))
        start_quat = np.array(env.robot.get_end_effector_quat(env.physics))
        lift_pos = np.array([start_pos[0], start_pos[1], start_pos[2] + lift_before])
        _gs = SkillLib._get_gripper_state(env)
        obs, wp, stage_success, _ = SkillLib.moveto(env, target_pos=lift_pos, gripper_state=_gs)
        observations.extend(obs)
        waypoints.extend(wp)
        if not stage_success:
            return observations, waypoints, False, task_success

        # 2. 移到容器上方
        obs, wp, stage_success, _ = SkillLib.moveto(env, target_pos=pour_target_pos, gripper_state=_gs)
        observations.extend(obs)
        waypoints.extend(wp)
        if not stage_success:
            return observations, waypoints, False, task_success

        # 3. IK 补偿逐步倾倒（记录每步 qpos，用于逆向回放恢复）
        current_quat = np.array(env.robot.get_end_effector_quat(env.physics))
        current_euler = quaternion_to_euler(current_quat)
        pre_tilt_quat = current_quat
        n_steps = int(abs(tilt_angle / tilt_velocity))
        tilt_sign = 1 if tilt_angle > 0 else -1
        tilt_qpos_history = []  # 记录倾倒过程中每一步的 qpos

        for i in range(1, n_steps + 1):
            step_euler = np.array(current_euler)
            step_euler[1] += tilt_sign * tilt_velocity * i
            step_quat = euler_to_quaternion(step_euler[0], step_euler[1], step_euler[2])

            success, target_qpos = env.robot.get_qpos_from_ee_pos(
                env.physics, pour_target_pos, step_quat)

            if not success:
                continue

            tilt_qpos_history.append(np.array(target_qpos))  # 记录 IK 解

            gripper_state = SkillLib._get_gripper_state(env)  # 保持 lock 模式的手指宽度
            action = np.concatenate([target_qpos, gripper_state])
            for _ in range(n_repeat_step):
                timestep = env.step(action)
                if timestep.last():
                    task_success = True
                    break

            waypoint = np.concatenate([
                env.robot.get_end_effector_pos(env.physics),
                quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                gripper_state
            ])
            observations.append(env.get_observation())
            waypoints.append(waypoint)
            if task_success:
                break

        # 4. 等待液体流出
        if wait_time > 0:
            obs, wp, _, _ = SkillLib.wait(env, wait_time=wait_time)
            observations.extend(obs)
            waypoints.extend(wp)

        # 5. 逆向回放倾倒过程的 qpos（完全对称，无 IK 跳变）
        _gs = SkillLib._get_gripper_state(env)
        for qpos in reversed(tilt_qpos_history):
            action = np.concatenate([qpos, _gs])
            for _ in range(n_repeat_step):
                timestep = env.step(action)
                if timestep.last():
                    task_success = True
                    break

            waypoint = np.concatenate([
                env.robot.get_end_effector_pos(env.physics),
                quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                _gs
            ])
            observations.append(env.get_observation())
            waypoints.append(waypoint)
            if task_success:
                break

        return observations, waypoints, True, task_success

    @staticmethod
    def lift(env, target_pos=None, target_quat=None, gripper_state=None, lift_height=0.3):
        """
        Common lift function.
        """
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(env.physics)

        # DEBUG: 打印 lift 前后烧杯位置
        grasped_names, grasped_entities = env.get_grasped_entity()
        beaker_before = None
        beaker_name = None
        if grasped_entities:
            for name, entity in zip(grasped_names, grasped_entities):
                beaker_before = np.array(entity.get_xpos(env.physics))
                beaker_name = name
                print(f"\n{'='*60}")
                print(f"DEBUG [lift]: LIFT 前 - {name}")
                print(f"  beaker pos: ({beaker_before[0]:.4f}, {beaker_before[1]:.4f}, {beaker_before[2]:.4f})")
                print(f"  finger_mid Z: {env.robot.get_end_effector_pos(env.physics)[2]:.4f}")
                print(f"{'='*60}\n")

        if target_pos is None:
            target_pos = np.array(start_pos) + np.array([0, 0, lift_height])
        if target_quat is None:
            target_quat = start_quat
        interplate_path, interplate_quat = interpolate_path([start_pos, target_pos], [np.array(start_quat), np.array(target_quat)])
        observations = [env.get_observation()]
        waypoints = []
        if gripper_state is None:
            gripper_state = SkillLib._get_gripper_state(env)
        obs, new_waypoints, stage_success, task_success = SkillLib.step_trajectory(env,
                                                           interplate_path,
                                                           interplate_quat,
                                                           gripper_state)
        observations.extend(obs)
        waypoints.extend(new_waypoints)
        observations.pop(-1)

        # DEBUG: 打印 lift 后烧杯位置
        if beaker_before is not None and beaker_name is not None:
            grasped_names, grasped_entities = env.get_grasped_entity()
            beaker_after = None
            for name, entity in zip(grasped_names, grasped_entities):
                if name == beaker_name:
                    beaker_after = np.array(entity.get_xpos(env.physics))
                    break
            if beaker_after is not None:
                ee_pos = env.robot.get_end_effector_pos(env.physics)
                print(f"\n{'='*60}")
                print(f"DEBUG [lift]: LIFT 后 - {beaker_name}")
                print(f"  beaker pos: ({beaker_after[0]:.4f}, {beaker_after[1]:.4f}, {beaker_after[2]:.4f})")
                print(f"  finger_mid Z: {ee_pos[2]:.4f}")
                print(f"  beaker lift 高度: {beaker_after[2] - beaker_before[2]:.4f}m")
                print(f"  ee lift 高度: {ee_pos[2] - start_pos[2]:.4f}m")
                print(f"  是否被提起: {'✓ YES' if abs(beaker_after[2] - beaker_before[2]) > 0.05 else '✗ NO (可能滑落)'}")
                print(f"{'='*60}\n")

        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, stage_success, task_success
    
    @staticmethod
    def reset(env, max_n_substep=200, tolerance=0.01):
        init_qpos = env.task.robot.default_qpos
        observations = [env.get_observation()]
        waypoints = []
        for _ in range(max_n_substep):
            action = np.array(init_qpos)
            env.step(action)
            obs = env.get_observation()
            pos, euler = np.array(env.robot.get_end_effector_pos(env.physics)), quaternion_to_euler(env.robot.get_end_effector_quat(env.physics))
            waypoint = np.concatenate([pos, euler, np.ones(2)*0.04])
            observations.append(obs)
            waypoints.append(waypoint)
            current_qpos = np.array(env.task.robot.get_qpos(env.physics)).reshape(-1)
            if np.max(current_qpos - np.array(action[:7])) < tolerance \
                and np.min(current_qpos - np.array(action[:7])) > -tolerance:
                break
        observations.pop(-1)
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, True, False
        
    
    @staticmethod
    def close_gripper(env, repeat=1):
        qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        observations = [env.get_observation()]
        waypoints = []
        success = False

        # DEBUG: 获取夹爪和被抓物体信息
        import mujoco as mj
        raw_m = env.physics.model._model
        raw_d = env.physics.data._data
        gripper_geoms = env.robot.gripper_geoms

        # 收集 pad geom ids
        pad_info = []
        for geom in gripper_geoms:
            eid = env.physics.bind(geom).element_id
            gname = mj.mj_id2name(raw_m, mj.mjtObj.mjOBJ_GEOM, eid) or ''
            if 'pad' in gname.lower():
                pad_info.append((gname, eid))

        print(f"DEBUG [close_gripper]: 开始关闭夹爪")
        print(f"  手指关节初始位置: finger1={raw_d.qpos[7]:.4f}, finger2={raw_d.qpos[8]:.4f}")
        print(f"  夹爪控制目标 qpos: {qpos}")

        # 获取夹爪 pad 的世界坐标
        def get_pad_info():
            pads = []
            for gname, eid in pad_info:
                pos = raw_d.geom_xpos[eid]
                pads.append((gname, pos))
            return pads

        # 打印初始 pad 位置
        initial_pads = get_pad_info()
        for gname, pos in initial_pads:
            print(f"  pad '{gname}': ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
        if len(initial_pads) >= 2:
            pad_dist = np.linalg.norm(initial_pads[0][1][:2] - initial_pads[1][1][:2])
            print(f"  pad XY间距: {pad_dist:.4f}m")

        for i in range(10):
            gripper_target = np.ones(2) * (0.04 - i * 0.04/10)
            action = np.concatenate([qpos, gripper_target])
            for _ in range(repeat):
                timestep = env.step(action)
                if timestep.last():
                    success = True
                    obs = env.get_observation()
                    observations.append(obs)
                    waypoints.append(np.concatenate([env.robot.get_end_effector_pos(env.physics),
                                             quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                                             gripper_target]))
                    break

            # DEBUG: 每步打印手指位置和接触力
            f1_actual = raw_d.qpos[7]
            f2_actual = raw_d.qpos[8]
            ctrl = raw_d.ctrl[-2:]  # 夹爪控制信号
            ctrlerr = gripper_target - np.array([f1_actual, f2_actual])

            # 检查接触力
            contacts = raw_d.contact
            contact_count = 0
            finger_tube_contact = False
            for c in contacts:
                if c.dist < 0.01:  # 距离小于1cm的接触
                    contact_count += 1
                    g1 = mj.mj_id2name(raw_m, mj.mjtObj.mjOBJ_GEOM, c.geom1) or ''
                    g2 = mj.mj_id2name(raw_m, mj.mjtObj.mjOBJ_GEOM, c.geom2) or ''
                    if ('finger' in g1.lower() or 'finger' in g2.lower()) and \
                       ('tube' in g1.lower() or 'tube' in g2.lower()):
                        finger_tube_contact = True

            # 获取 pad 世界坐标
            current_pads = get_pad_info()
            if len(current_pads) >= 2:
                pad_dist = np.linalg.norm(current_pads[0][1][:2] - current_pads[1][1][:2])

            print(f"  step {i}: target={gripper_target[0]:.4f}, actual f1={f1_actual:.4f} f2={f2_actual:.4f}, "
                  f"ctrl_err={ctrlerr[0]:.4f}/{ctrlerr[1]:.4f}, contacts={contact_count}, "
                  f"finger_tube={finger_tube_contact}, pad_dist={pad_dist:.4f}m")

            obs = env.get_observation()
            observations.append(obs)
            waypoints.append(np.concatenate([env.robot.get_end_effector_pos(env.physics),
                                             quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                                             gripper_target]))

        # DEBUG: 打印关闭后的手指状态和接触情况
        f1_final = raw_d.qpos[7]
        f2_final = raw_d.qpos[8]
        print(f"  关闭后: finger1={f1_final:.4f}, finger2={f2_final:.4f}")
        print(f"  夹爪闭合程度: {1 - f1_final/0.04:.1%} / {1 - f2_final/0.04:.1%}")

        observations.pop(-1)
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, True, success
    
    @staticmethod
    def open_gripper(env, repeat=1):
        qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        observations = [env.get_observation()]
        waypoints = []
        task_success = False
        stage_success = False
        for i in range(10):
            gripper_state = np.ones(2) * (i+1)/10 * 0.04
            action = np.concatenate([qpos, gripper_state])
            for _ in range(repeat):
                timestep = env.step(action)
                if timestep.last():
                    task_success = True
                    obs = env.get_observation()
                    observations.append(obs)
                    waypoints.append(np.concatenate([env.robot.get_end_effector_pos(env.physics),
                                             quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                                             gripper_state]))
            obs = env.get_observation()
            observations.append(obs)
            waypoints.append(np.concatenate([env.robot.get_end_effector_pos(env.physics),
                                             quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                                             gripper_state]))
            if timestep.last():
                task_success = True
                break
        observations.pop(-1)
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        # get_ee_open_state 实际返回"夹爪是否关闭"（与函数名相反）
        # open_gripper 成功 = 夹爪不再关闭 = 返回 False
        if not env.robot.get_ee_open_state(env.physics):
            stage_success = True
            # ========== Grasp Lock: 夹爪打开后清除抓取状态 ==========
            if hasattr(env, "_grasped_entity_info"):
                env._grasped_entity_info = None
            if hasattr(env, "_lock_gripper_state"):
                env._lock_gripper_state = None
            # Mode 2: 移除 weld 约束
            if hasattr(env, "_grasp_lock_mode") and env._grasp_lock_mode == 2:
                if hasattr(env, "_remove_weld_constraint"):
                    env._remove_weld_constraint()
        return observations, waypoints, stage_success, task_success
    
    @staticmethod
    def flip(env, gripper_state=None, target_q_velocity=np.pi/40, max_n_substep=30, tolerance=0.01):
        qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        observations = [env.get_observation()]
        waypoints = []
        if gripper_state is None:
            gripper_state = SkillLib._get_gripper_state(env)
        timestep = int(np.pi / target_q_velocity)
        success = False
        for i in range(timestep):
            action = np.array(qpos).copy()            
            action[-1] += target_q_velocity * i
            action = np.concatenate([action, gripper_state])
            for _ in range(max_n_substep):
                timestep = env.step(action)
                if timestep.last():
                    success = True
                    break
                current_qpos = np.array(env.task.robot.get_qpos(env.physics)).reshape(-1)
                if np.max(current_qpos - np.array(action[:7])) < tolerance \
                    and np.min(current_qpos - np.array(action[:7])) > -tolerance:
                  break
            if success:
                break
            waypoint = np.concatenate([env.robot.get_end_effector_pos(env.physics), 
                                       quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)), 
                                       gripper_state])
            obs = env.get_observation()
            observations.append(obs)
            waypoints.append(waypoint)
        observations.pop(-1)
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, success

    @staticmethod
    def rotate(env, rotation_angle=np.pi/2, gripper_state=None, target_q_velocity=np.pi/40, max_n_substep=30, tolerance=0.01):
        """
        Rotate the grasped object by rotating the wrist joint.

        Args:
            env: LM4manipEnv object
            rotation_angle: rotation angle in radians (default: pi/2 = 90 degrees)
                           positive for counter-clockwise, negative for clockwise
            gripper_state: gripper state (None for auto-detect)
            target_q_velocity: angular velocity for rotation
            max_n_substep: max substeps per timestep
            tolerance: tolerance for qpos error

        Returns:
            observations: list of observations
            waypoints: list of waypoints
            stage_success: whether the rotation succeeded
            task_success: whether the task is complete
        """
        qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        observations = [env.get_observation()]
        waypoints = []
        if gripper_state is None:
            gripper_state = SkillLib._get_gripper_state(env)

        # Calculate number of timesteps based on rotation angle
        timesteps = int(abs(rotation_angle) / target_q_velocity)
        task_success = False
        stage_success = False

        for i in range(timesteps):
            action = np.array(qpos).copy()
            # Rotate the last joint (wrist) by the specified angle
            action[-1] += target_q_velocity * (i + 1) * np.sign(rotation_angle)
            action = np.concatenate([action, gripper_state])

            for _ in range(max_n_substep):
                timestep = env.step(action)
                if timestep.last():
                    task_success = True
                    break
                current_qpos = np.array(env.task.robot.get_qpos(env.physics)).reshape(-1)
                if np.max(current_qpos - np.array(action[:7])) < tolerance \
                    and np.min(current_qpos - np.array(action[:7])) > -tolerance:
                    break

            if task_success:
                break

            waypoint = np.concatenate([env.robot.get_end_effector_pos(env.physics),
                                       quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                                       gripper_state])
            obs = env.get_observation()
            observations.append(obs)
            waypoints.append(waypoint)

        observations.pop(-1)
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"

        # Check if rotation was successful by comparing final angle
        final_qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        angle_diff = abs(final_qpos[-1] - (qpos[-1] + rotation_angle))
        if angle_diff < tolerance * 10:  # Allow some tolerance
            stage_success = True

        return observations, waypoints, stage_success, task_success

    @staticmethod
    def open_laptop(env, target_entity_name):
        laptop = env.task.entities[target_entity_name]
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(env.physics)
        
        trajectory = laptop.get_open_trajectory(env.physics)
        trajectory_quats = []
        screen_joint = laptop.screen_joint
        rotation_axis = env.physics.bind(screen_joint).xaxis
        # rotation_anchor = env.physics.bind(door_joint).xanchor
        observations = [env.get_observation()]
        waypoints = []
        stage_success = False
        task_success = False
        for i in range(len(trajectory)):
            rot_quat = quaternion_from_axis_angle(rotation_axis, 0.04*(i+1))
            new_quat = quaternion_multiply(start_quat, rot_quat)
            trajectory_quats.append(new_quat)
        # init_qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        interplate_path, interplate_quat = interpolate_path(trajectory, trajectory_quats)
        new_obs, new_waypoints, _, task_success = SkillLib.step_trajectory(env,
                                                            interplate_path, 
                                                            interplate_quat, 
                                                            np.zeros(2))
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        if task_success:
            observations.pop(-1)
            assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
            return observations, waypoints, True, task_success
        
        new_obs, new_waypoints, _, task_success = SkillLib.open_gripper(env)
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        observations.pop(-1)
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        if env.task.entities[target_entity_name].is_open(env.physics):
            stage_success = True
        return observations, waypoints, stage_success, task_success
    
    @staticmethod
    def wait(env, wait_time=100, gripper_state=None):
        print(f"wait for {wait_time} steps")
        current_qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        if gripper_state is None:
            gripper_state = SkillLib._get_gripper_state(env)
        observations = [env.get_observation()]
        waypoints = []
        task_success = False
        for _ in range(wait_time):
            action = np.concatenate([current_qpos, gripper_state])
            timestep = env.step(action)
            if timestep.last():
                task_success = True
                break
            obs = env.get_observation()
            observations.append(obs)
            waypoints.append(np.concatenate([env.robot.get_end_effector_pos(env.physics),
                                             quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                                             gripper_state]))
        observations.pop(-1)
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, True, task_success

    @staticmethod
    def wait_for(env,
                 wait_duration=2.0,
                 entity_name=None,
                 change_type=None,
                 solution=None,
                 color=None,
                 gripper_state=None):
        """
        等待外部状态变化的技能。

        机械臂保持不动，等待指定时间后自动应用环境变化。
        用于人机协同场景，如"等待人工添加液体"。

        Args:
            wait_duration: 等待时间（秒），默认 2.0 秒
            entity_name: 要改变的实体名称，如 "beaker_0"
            change_type: 变化类型：
                - "add_solution": 添加溶液（需要 solution 参数）
                - "change_color": 改变颜色（需要 color 参数）
                - "solution_change_color": 改变溶液颜色（需要 color 参数）
                - "Light_the_alcohol_lamp": 点燃酒精灯火焰（需要 entity_name，不需额外参数）
            solution: 溶液名称，如 "CuSO4", "FeCl3", "KMnO4" 等
            color: RGBA 颜色值，如 [1, 0, 0, 1] 表示红色
            gripper_state: 夹爪状态，默认保持当前状态

        Returns:
            (observations, waypoints, stage_success, task_success)
        """
        logger.info(f"wait_for: 等待 {wait_duration}s, change_type={change_type}, entity={entity_name}")

        current_qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        if gripper_state is None:
            gripper_state = SkillLib._get_gripper_state(env)

        # 机械臂保持不动，等待一段时间
        observations = [env.get_observation()]
        waypoints = []

        # 计算需要等待的步数（假设 10 fps）
        steps_per_second = 10
        wait_steps = int(wait_duration * steps_per_second)

        for _ in range(wait_steps):
            action = np.concatenate([current_qpos, gripper_state])
            timestep = env.step(action)
            obs = env.get_observation()
            observations.append(obs)
            waypoints.append(np.concatenate([
                env.robot.get_end_effector_pos(env.physics),
                quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                gripper_state
            ]))

        # 应用环境变化
        if entity_name and change_type:
            try:
                entity = env.task.entities.get(entity_name)
                if entity is None:
                    logger.warning(f"wait_for: 实体 {entity_name} 不存在，可用实体: {list(env.task.entities.keys())}")
                elif change_type == "add_solution" and solution:
                    if hasattr(entity, 'set_solution_rgba'):
                        entity.set_solution_rgba(env.physics, solution)
                        logger.info(f"wait_for: 已添加溶液 {solution} 到 {entity_name}")
                    else:
                        logger.warning(f"wait_for: 实体 {entity_name} 不支持 set_solution_rgba")
                elif change_type in ("change_color", "solution_change_color") and color:
                    if hasattr(entity, 'set_solution_rgba'):
                        entity.set_solution_rgba(env.physics, target_rgba=color)
                        logger.info(f"wait_for: 已设置 {entity_name} 溶液颜色为 {color}")
                    elif hasattr(entity, 'mjcf_model'):
                        for geom in entity.mjcf_model.find_all('geom'):
                            env.physics.bind(geom).rgba = color
                        logger.info(f"wait_for: 已设置 {entity_name} 颜色为 {color}")
                elif change_type == "Light_the_alcohol_lamp":
                    if hasattr(entity, 'set_flame_state'):
                        entity.set_flame_state(env.physics, lit=True)
                        logger.info(f"wait_for: 已点燃 {entity_name} 酒精灯")
                    else:
                        logger.warning(f"wait_for: 实体 {entity_name} 不支持 set_flame_state")
            except Exception as e:
                logger.warning(f"wait_for: 应用状态变化失败 - {e}")

        observations.pop(-1)
        return observations, waypoints, True, False  # Don't return task_success, let loop continue

    @staticmethod
    def shake(env, n_shakes=3, shake_angle=0.5, steps_per_swing=5, gripper_state=None):
        """
        摇晃操作：在当前位置保持不动，通过快速交替偏转末端姿态实现摇摆。

        参数：
            n_shakes: 摇摆次数（一次 = 正→负 完整往返）
            shake_angle: 摇摆幅度（弧度），默认0.5（约30度）
            steps_per_swing: 每段摇摆的插值步数，默认5
            gripper_state: 夹爪状态，默认保持当前状态
        """
        start_pos = np.array(env.robot.get_end_effector_pos(env.physics))
        start_quat = np.array(env.robot.get_end_effector_quat(env.physics))

        if gripper_state is None:
            gripper_state = SkillLib._get_gripper_state(env)
        observations = [env.get_observation()]
        waypoints = []
        task_success = False

        # 生成摇摆的目标四元数序列
        quat_positive = quaternion_multiply(
            quaternion_from_axis_angle(np.array([0, 1, 0]), shake_angle),
            start_quat
        )
        quat_negative = quaternion_multiply(
            quaternion_from_axis_angle(np.array([0, 1, 0]), -shake_angle),
            start_quat
        )

        # 摇摆序列：+angle, -angle, ..., 回正
        shake_targets = []
        for i in range(n_shakes):
            shake_targets.append(quat_positive)
            shake_targets.append(quat_negative)
        shake_targets.append(start_quat)

        current_quat = start_quat
        for target_quat in shake_targets:
            # 手动生成插值点（interpolate_path 对 distance=0 只返回1个点）
            interp_positions = []
            interp_quats = []
            for t in np.linspace(0, 1, steps_per_swing, endpoint=True):
                interp_positions.append(start_pos.copy())
                interp_quats.append(qauternion_slerp(current_quat, target_quat, t))

            obs, new_waypoints, stage_success, ts = SkillLib.step_trajectory(
                env, interp_positions, interp_quats, gripper_state
            )
            observations.extend(obs)
            waypoints.extend(new_waypoints)

            if ts:
                task_success = True
                break

            current_quat = target_quat

        observations.pop(-1)
        assert len(observations) == len(waypoints), \
            f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        return observations, waypoints, True, task_success

    @staticmethod
    def move_offset(env, offset, target_quat=None, gripper_state=None):
        start_pos, start_quat = env.robot.get_end_effector_pos(env.physics), env.robot.get_end_effector_quat(env.physics)
        target_pos = np.array(start_pos) + np.array(offset)
        if target_quat is None: target_quat = start_quat
        observations, waypoints, stage_success, task_success = SkillLib.moveto(env, target_pos, target_quat, gripper_state=gripper_state)
        return observations, waypoints, stage_success, task_success

    @staticmethod
    def insert_to_entity(env, target_entity_name, insert_depth=0.05, gripper_state=None):
        """
        将抓取的物体插入目标实体的孔位。

        使用目标实体的 place point（group=2 site）作为插入点，
        自动选择空闲孔位，用 RRT 规划路径，最后松开夹爪。

        Args:
            target_entity_name: 目标实体名（如 chemistry_tube_stand）
            insert_depth: 插入深度（m），默认 5cm
            gripper_state: 夹爪状态，默认保持当前状态
        """
        task_success = False
        entity = env.task.entities[target_entity_name]
        place_points = entity.get_place_point(env.physics)

        if not place_points:
            print(f"[insert_to_entity] {target_entity_name} 没有 place point")
            return [env.get_observation()], [], False, False

        if gripper_state is None:
            gripper_state = SkillLib._get_gripper_state(env)

        # 选择空闲孔位：找离当前夹持物体 XY 最近的空闲孔位
        ee_pos = np.array(env.robot.get_end_effector_pos(env.physics))

        # 获取当前夹持的物体位置（更准确的插入目标）
        grasped_obj_names, grasped_objs = env.get_grasped_entity()
        if grasped_obj_names:
            # 使用夹持物体的位置来选择最近的孔位
            grasped_obj_pos = np.array(grasped_objs[0].get_xpos(env.physics))
            ref_pos = grasped_obj_pos
            print(f"[insert_to_entity] 夹持物体位置: {np.round(grasped_obj_pos, 3)}")
        else:
            # 没有夹持物体时回退到 EE 位置
            ref_pos = ee_pos

        # 按孔位与参考位置的 XY 距离排序
        stand_xpos = np.array(entity.get_xpos(env.physics))
        sorted_points = sorted(place_points, key=lambda p: np.linalg.norm(np.array(p)[:2] - ref_pos[:2]))
        insert_point = None
        for pp in sorted_points:
            pp = np.array(pp)
            # 检查该孔位附近是否已被其他物体占据（排除当前夹持的物体）
            occupied = False
            for name, other_entity in env.task.entities.items():
                if name == target_entity_name:
                    continue
                other_pos = np.array(other_entity.get_xpos(env.physics))
                # 孔位 XY 范围 4cm 内有物体且 Z 低于孔位，认为已占据
                # 排除距离末端 15cm 内的物体（当前夹持的试管）
                if (np.linalg.norm(other_pos[:2] - pp[:2]) < 0.04
                        and other_pos[2] < pp[2] + 0.05
                        and np.linalg.norm(other_pos - ee_pos) > 0.15):
                    occupied = True
                    break
            if not occupied:
                insert_point = pp
                break

        if insert_point is None:
            insert_point = np.array(sorted_points[-1])

        print(f"[insert_to_entity] EE 当前位置: {np.round(ee_pos, 3)}")
        print(f"[insert_to_entity] 试管架中心: {np.round(stand_xpos, 3)}")
        print(f"[insert_to_entity] 选中孔位: {np.round(insert_point, 3)}  offset={np.round(insert_point - stand_xpos, 3)}")

        # 1. RRT 移动到孔位上方（偏移 12cm），指定竖直向下姿态
        # 降低 hover 高度避免试管底部怼到桌面
        hover_pos = insert_point + np.array([0, 0, 0.12])
        vertical_quat = euler_to_quaternion(-np.pi, 0, 0)  # 末端竖直向下
        # 先初始化 observations，再调用 moveto，保证帧顺序正确
        observations = [env.get_observation()]
        waypoints = []
        obs, wp, stage_success, _ = SkillLib.moveto(
            env, target_pos=hover_pos, target_quat=vertical_quat, gripper_state=gripper_state
        )
        observations.extend(obs)
        waypoints.extend(wp)

        if not stage_success:
            return observations, waypoints, False, False

        # 2. 用 lift 负值从 hover_pos 直接下降到插入位置
        # hover_pos = insert_point + [0,0,0.12]
        # 需要下降到 insert_point - [0,0,insert_depth]
        # 总下降距离 = 0.12 + insert_depth
        descend = -(0.12 + insert_depth)
        obs, wp, _, _ = SkillLib.lift(
            env, lift_height=descend, gripper_state=gripper_state)
        observations.extend(obs)
        waypoints.extend(wp)

        # 下降执行完毕即尝试松开（不依赖 lift 的位置误差判定）
        insert_pos = insert_point - np.array([0, 0, insert_depth])
        final_pos = np.array(env.robot.get_end_effector_pos(env.physics))
        dist = np.linalg.norm(final_pos - insert_pos)
        print(f"[insert_to_entity] 插入目标: {np.round(insert_pos,3)}  实际EE: {np.round(final_pos,3)}  误差: {dist:.4f}")

        # 3. 松开夹爪
        obs, wp, _, _ = SkillLib.open_gripper(env)
        observations.extend(obs)
        waypoints.extend(wp)

        # 4. 抬回安全高度，避免后续技能受低位关节构型影响
        obs, wp, _, _ = SkillLib.lift(env, lift_height=0.15, gripper_state=np.ones(2) * 0.04)
        observations.extend(obs)
        waypoints.extend(wp)

        return observations, waypoints, True, task_success

    @staticmethod
    def stir_entity_with_tool(env, target_container_name, stir_radius=0.01, stir_duration=5, insert_ratio=2/3):
        """
        使用搅拌工具搅动容器内的液体。

        假设当前夹爪已抓取搅拌工具，执行以下步骤：
        1. 获取容器的 place_point 作为圆心
        2. 移动到 place_point 正上方 25cm 处
        3. 下降到容器内部 insert_ratio 深度处
        4. 以 place_point XY 为圆心，做半径 stir_radius 的圆周运动

        Args:
            target_container_name: 目标容器实体名
            stir_radius: 圆周运动半径（m），默认 1cm
            stir_duration: 搅拌持续时间（s），默认 5s
            insert_ratio: 插入深度比例（容器高度的倍数），默认 2/3
        """
        gripper_state = SkillLib._get_gripper_state(env)
        observations = [env.get_observation()]
        waypoints = []

        container = env.task.entities[target_container_name]
        place_points = container.get_place_point(env.physics)
        if not place_points:
            return [env.get_observation()], [], False, False
        place_point = np.array(place_points[0]) if isinstance(place_points, list) else np.array(place_points)

        top_site = container.mjcf_model.find("site", "top_site")
        bottom_site = container.mjcf_model.find("site", "bottom_site")
        bottom_z = env.physics.bind(bottom_site).xpos[2] if bottom_site else 0
        top_z = env.physics.bind(top_site).xpos[2] if top_site else (place_point[2] + 0.126)
        internal_height = top_z - bottom_z
        insert_target_z = bottom_z + internal_height * insert_ratio

        grasped_names, grasped_entities = env.get_grasped_entity()
        tool_offset_z = 0.0
        if grasped_entities:
            grasp_keypoints = grasped_entities[0].get_grasped_keypoints(env.physics)
            if grasp_keypoints:
                tool_bottom_site = grasped_entities[0].mjcf_model.find("site", "bottom_site")
                tool_bottom_z = env.physics.bind(tool_bottom_site).xpos[2] if tool_bottom_site else 0
                tool_offset_z = grasp_keypoints[0][2] - tool_bottom_z

        vertical_quat = euler_to_quaternion(-np.pi, 0, 0)
        hover_pos = np.array([place_point[0], place_point[1], place_point[2] + 0.25 + tool_offset_z])

        print(f"[stir_entity_with_tool] moving to hover: {hover_pos}")
        obs, wp, success, _ = SkillLib.moveto(env, target_pos=hover_pos, target_quat=vertical_quat, gripper_state=gripper_state)
        print(f"[stir_entity_with_tool] hover moveto result: success={success}, obs={len(obs)}, wp={len(wp)}")
        if not success:
            return observations, waypoints, False, False
        observations.extend(obs)
        waypoints.extend(wp)

        descend_pos = np.array([place_point[0], place_point[1], insert_target_z + tool_offset_z])
        print(f"[stir_entity_with_tool] descending to: {descend_pos}")
        # 用 lift 做纯垂直下降，避免 moveto+RRT 产生水平漂移导致撞壁
        descend_lift = descend_pos[2] - hover_pos[2]
        obs, wp, _, _ = SkillLib.lift(env, lift_height=descend_lift, gripper_state=gripper_state)
        print(f"[stir_entity_with_tool] descend lift result: obs={len(obs)}, wp={len(wp)}")
        observations.extend(obs)
        waypoints.extend(wp)

        current_quat = np.array(env.robot.get_end_effector_quat(env.physics))
        # 每圈 36 步（每步 10°），转 stir_duration 圈
        steps_per_circle = 36
        angle_step = 2 * np.pi / steps_per_circle
        total_circles = stir_duration  # stir_duration 作为圈数
        total_steps = int(total_circles * steps_per_circle)
        circle_z = insert_target_z + tool_offset_z

        for i in range(total_steps):
            angle = angle_step * (i + 1)
            circle_x = place_point[0] + stir_radius * np.cos(angle)
            circle_y = place_point[1] + stir_radius * np.sin(angle)
            success, target_qpos = env.robot.get_qpos_from_ee_pos(env.physics, [circle_x, circle_y, circle_z], current_quat)
            if success:
                action = np.concatenate([target_qpos, gripper_state])
                # 每步重复执行直到机器人到达目标位置
                for _ in range(10):
                    timestep = env.step(action)
                    if timestep.last():
                        break
                    current_qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
                    if np.max(np.abs(current_qpos - action[:7])) < 0.02:
                        break
            waypoints.append(np.concatenate([env.robot.get_end_effector_pos(env.physics),
                                              quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                                              gripper_state]))
            observations.append(env.get_observation())

        observations.pop(-1)
        return observations, waypoints, True, False

    @staticmethod
    def unscrew_cap(env, target_entity_name, rotation_angle=-4*np.pi,
                    target_q_velocity=np.pi/40, max_n_substep=30, tolerance=0.01,
                    lift_height=0.03):
        """
        拧开瓶盖技能：抓取瓶盖 → 旋转腕关节 → 释放

        核心思路：参考 SkillLib.rotate 的实现。
        不再用 IK 规划末端在空间中的姿态变化（会导致手腕奇异位姿）。
        而是直接修改腕关节 qpos[-1] 旋转指定角度，瓶盖 hinge joint 会被动跟随。
        slide joint 上升则由物理仿真器在瓶盖与瓶身脱离约束时自动产生。

        Args:
            target_entity_name: 目标容器实体名（如 pill_bottle_0）
            rotation_angle: 总旋转角度（弧度），默认 4π（两圈，逆时针为正）
            target_q_velocity: 角速度（弧度/步），默认 π/40
            max_n_substep: 每步最大子步数
            tolerance: qpos 容差
            lift_height: 拧完后末端额外上提距离（m），让 slide joint 上升
        """
        target_entity = env.task.entities[target_entity_name]

        # 0. 记录初始 slide 关节位置（用于拧开后的位移判定）
        if hasattr(target_entity, 'slide_joint') and target_entity.slide_joint is not None:
            target_entity._unscrew_initial_slide = float(env.physics.bind(target_entity.slide_joint).qpos)
            print(f"[unscrew_cap] initial slide qpos: {target_entity._unscrew_initial_slide:.6f}")
        print(f"[unscrew_cap] rotation_angle={rotation_angle:.4f} ({rotation_angle/np.pi:.2f}*pi)")

        # 1. 抓取瓶盖
        observations = [env.get_observation()]
        waypoints = []
        task_success = False

        new_obs, new_waypoints, pick_success, _ = SkillLib.pick(
            env, target_entity_name,
            prior_eulers=[[np.pi, 0, 0]])
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)

        if not pick_success:
            print("[unscrew_cap] pick failed, aborting")
            observations.pop(-1)
            return observations, waypoints, False, False

        # 2. 旋转腕关节 qpos[-1]，带动瓶盖 hinge joint
        gripper_state = SkillLib._get_gripper_state(env)
        qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        timesteps = int(abs(rotation_angle) / target_q_velocity)
        rot_sign = 1 if rotation_angle > 0 else -1

        for i in range(timesteps):
            action = np.array(qpos).copy()
            action[-1] += target_q_velocity * (i + 1) * rot_sign
            action = np.concatenate([action, gripper_state])

            for _ in range(max_n_substep):
                timestep = env.step(action)
                if timestep.last():
                    task_success = True
                    break
                current_qpos = np.array(env.task.robot.get_qpos(env.physics)).reshape(-1)
                if np.max(current_qpos - np.array(action[:7])) < tolerance \
                    and np.min(current_qpos - np.array(action[:7])) > -tolerance:
                    break

            if task_success:
                break

            waypoint = np.concatenate([env.robot.get_end_effector_pos(env.physics),
                                       quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                                       gripper_state])
            obs = env.get_observation()
            observations.append(obs)
            waypoints.append(waypoint)

        # 3. 拧完后向上提一下，让 slide joint 上升。
        # 不用 SkillLib.lift：它会检查抓取 entity 是否被提起，
        # 但本任务中瓶身被 attach 固定，body 永远不会动，会一直返回失败。
        if lift_height > 0:
            start_pos = env.robot.get_end_effector_pos(env.physics)
            start_quat = env.robot.get_end_effector_quat(env.physics)
            target_pos = np.array(start_pos) + np.array([0, 0, lift_height])
            interplate_path, interplate_quat = interpolate_path(
                [start_pos, target_pos],
                [np.array(start_quat), np.array(start_quat)])
            obs, wp, _, _ = SkillLib.step_trajectory(
                env, interplate_path, interplate_quat, gripper_state)
            observations.extend(obs)
            waypoints.extend(wp)

        # 4. 释放瓶盖
        new_obs, new_waypoints, _, task_success = SkillLib.open_gripper(env)
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)

        observations.pop(-1)
        assert len(observations) == len(waypoints), \
            f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"

        # 用 slide 关节相对位移判定拧开成功。
        # 不用 is_cap_separated：物理接触力始终把 cap 压在 body 上，slide 关节
        # 物理上无法真正离开 body，contact-based 判定不可靠。
        stage_success = False
        if hasattr(target_entity, 'slide_joint') and target_entity.slide_joint is not None:
            current_slide = float(env.physics.bind(target_entity.slide_joint).qpos)
            initial_slide = float(getattr(target_entity, '_unscrew_initial_slide', 0.0))
            stage_success = (current_slide - initial_slide) > 0.003
            print(f"[unscrew_cap] final slide qpos: {current_slide:.6f}, delta: {current_slide - initial_slide:.6f}, stage_success: {stage_success}")
        return observations, waypoints, stage_success, task_success

