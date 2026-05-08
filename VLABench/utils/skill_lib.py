"""
Skill Library for data generation.
"""
import numpy as np
import random
import time as _time
import mujoco
from VLABench.utils.utils import find_keypoint_and_prepare_grasp, distance, quaternion_to_euler, euler_to_quaternion, quaternion_from_axis_angle, quaternion_multiply
from VLABench.algorithms.motion_planning.rrt import rrt_motion_planning
from VLABench.algorithms.utils import interpolate_path, qauternion_slerp


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
        _t_start = _time.time()
        print(f"\n[STEP] step_trajectory 开始: {len(points)} 个路径点, max_substep={max_n_substep}")

        observations = []
        waypoints = []
        stage_success = False
        task_success = False
        last_executed_waypoint = -1

        # DEBUG: 只在 pick 的路径上打印最后5个点的实际手指位置
        import mujoco as mj
        raw_m = env.physics.model._model
        raw_d = env.physics.data._data
        gripper_geoms = env.robot.gripper_geoms
        pad_ids = []
        for geom in gripper_geoms:
            eid = env.physics.bind(geom).element_id
            gname = mj.mj_id2name(raw_m, mj.mjtObj.mjOBJ_GEOM, eid) or ''
            if 'pad' in gname.lower():
                pad_ids.append(eid)

        # DEBUG: 只在最后几个点启用 IK 调试
        debug_ik_enabled = False

        for i, (point, quat) in enumerate(zip(points, quats)):
            # 在最后 5 个路径点启用 IK 调试
            if i >= len(points) - 5:
                if not debug_ik_enabled:
                    debug_ik_enabled = True
                    env.robot._debug_ik = True
            else:
                env.robot._debug_ik = False

            success, action = env.robot.get_qpos_from_ee_pos(physics=env.physics, pos=point, quat=quat)
            action = np.concatenate([action, gripper_state])
            waypoint = np.concatenate([point, quaternion_to_euler(quat), gripper_state])

            for substep_idx in range(max_n_substep):
                timestep = env.step(action)

                if timestep.last():
                    print(f"[STEP] 路径点 {i}/{len(points)} 处 timestep.last()=True")
                    task_success = True
                    break

                current_qpos = np.array(env.task.robot.get_qpos(env.physics)).reshape(-1)
                qpos_max_error = np.max(current_qpos - np.array(action[:7]))
                qpos_min_error = np.min(current_qpos - np.array(action[:7]))

                if qpos_max_error < tolerance and qpos_min_error > -tolerance:
                    break

            last_executed_waypoint = i

            # DEBUG: 打印最后5个路径点时的手指 pad 世界坐标
            if i >= len(points) - 5 and len(pad_ids) >= 2:
                pad1 = raw_d.geom_xpos[pad_ids[0]]
                pad2 = raw_d.geom_xpos[pad_ids[1]]
                finger_mid = (pad1 + pad2) / 2
                print(f"[STEP] path_idx={i}: target=({point[0]:.4f},{point[1]:.4f},{point[2]:.4f}), finger_mid=({finger_mid[0]:.4f},{finger_mid[1]:.4f},{finger_mid[2]:.4f})")

            if task_success:
                break

            obs = env.get_observation()
            observations.append(obs)
            waypoints.append(waypoint)

            # 每50步或最后一步打印进度
            if i % 50 == 0 or i == len(points) - 1:
                print(f"[STEP] 进度 {i+1}/{len(points)}, 耗时 {_time.time()-_t_start:.1f}s")

        # 最终检查
        if len(points) > 0:
            final_ee_pos = env.robot.get_end_effector_pos(env.physics)
            final_distance = distance(points[-1], final_ee_pos)
            if final_distance < tolerance:
                stage_success = True
            print(f"[STEP] step_trajectory 完成: {last_executed_waypoint+1}/{len(points)} 点, "
                  f"final_dist={final_distance:.4f}, stage={stage_success}, task={task_success}, "
                  f"耗时 {_time.time()-_t_start:.1f}s")
        else:
            print(f"[STEP] step_trajectory 完成: 空路径, 耗时 {_time.time()-_t_start:.1f}s")

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
        print(f"\n{'='*60}")
        print(f"DEBUG [moveto]: 开始移动")
        print(f"  起始位置: {start_pos}")
        print(f"  目标位置: {target_pos}")
        print(f"  移动距离: {np.linalg.norm(target_pos - start_pos):.4f}m")
        print(f"{'='*60}\n")

        observations = [env.get_observation()]
        waypoints = []
        task_success = False
        gripper_closed = env.robot.get_ee_open_state(env.physics)
        if gripper_state is None:
            gripper_state = SkillLib._get_gripper_state(env)
        # env_pcd = observations[0]["masked_point_cloud"]
        # obstacle_pcd = np.asarray(env_pcd.points)

        obstacle_pcd = np.asarray(env.get_obstacle_pcd().points)
        print(f"DEBUG [moveto]: 障碍物点云数量: {len(obstacle_pcd)}")

        if target_quat is None:
            target_quat = start_quat

        print(f"DEBUG [moveto]: 开始 RRT 运动规划...")
        motion_planning_path = rrt_motion_planning(tuple(start_pos),
                                                tuple(target_pos),
                                                obstacle_pcd)
        if motion_planning_path is None:
            print(f"DEBUG [moveto]: ⚠️  RRT 规划失败，使用直线路径")
            motion_planning_path = [start_pos, target_pos]
        else:
            print(f"DEBUG [moveto]: ✓ RRT 规划成功，路径点数: {len(motion_planning_path)}")

        quats_in_path = []
        for t in np.linspace(0, 1, len(motion_planning_path), endpoint=False):
            quats_in_path.append(qauternion_slerp(start_quat, target_quat, t))

        print(f"DEBUG [moveto]: 插值路径...")
        interplate_path, interplate_quat = interpolate_path(np.array(motion_planning_path),
                                                            np.array(quats_in_path),
                                                            target_velocity)
        print(f"DEBUG [moveto]: 插值后路径点数: {len(interplate_path)}")

        print(f"DEBUG [moveto]: 执行轨迹...")
        new_obs, new_waypoints, stage_success, task_success = SkillLib.step_trajectory(env,
                                                                   interplate_path,
                                                                   interplate_quat,
                                                                   gripper_state,
                                                                   **kwargs)
        # if new_obs is None:
            # return None, None, False, False
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        observations.pop(-1)

        # DEBUG: 打印执行结果
        final_pos = env.robot.get_end_effector_pos(env.physics)
        pos_error = np.linalg.norm(final_pos - target_pos)
        print(f"\nDEBUG [moveto]: 执行完成")
        print(f"  最终位置: {final_pos}")
        print(f"  目标位置: {target_pos}")
        print(f"  位置误差: {pos_error:.4f}m")
        print(f"  Stage success: {stage_success}")
        print(f"  Task success: {task_success}")
        print(f"{'='*60}\n")

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

        print(f"DEBUG [pick]: step_trajectory 完成")
        print(f"  执行了 {len(new_waypoints)} 个路径点")
        print(f"  当前末端位置: {env.robot.get_end_effector_pos(env.physics)}")
        print(f"  目标位置: {key_pos}")
        print(f"  XY误差: {np.linalg.norm(env.robot.get_end_effector_pos(env.physics)[:2] - key_pos[:2])}")
        print(f"  Z误差: {abs(env.robot.get_end_effector_pos(env.physics)[2] - key_pos[2])}")
        # DEBUG: 打印 step_trajectory 完成后手指中点和烧杯中心
        import mujoco as mj
        raw_m = env.physics.model._model
        raw_d = env.physics.data._data
        gripper_geoms = env.robot.gripper_geoms
        pad_positions = []
        for geom in gripper_geoms:
            eid = env.physics.bind(geom).element_id
            gname = mj.mj_id2name(raw_m, mj.mjtObj.mjOBJ_GEOM, eid) or ''
            if 'pad' in gname.lower():
                pad_positions.append(raw_d.geom_xpos[eid])
        if len(pad_positions) >= 2:
            finger_mid = sum(pad_positions) / len(pad_positions)
            beaker_pos = np.array(target_entity.get_xpos(env.physics))
            print(f"DEBUG [pick]: === AFTER STEP_TRAJECTORY ===")
            print(f"  finger1: ({pad_positions[0][0]:.4f}, {pad_positions[0][1]:.4f}, {pad_positions[0][2]:.4f})")
            print(f"  finger2: ({pad_positions[1][0]:.4f}, {pad_positions[1][1]:.4f}, {pad_positions[1][2]:.4f})")
            print(f"  finger_mid: ({finger_mid[0]:.4f}, {finger_mid[1]:.4f}, {finger_mid[2]:.4f})")
            print(f"  beaker:     ({beaker_pos[0]:.4f}, {beaker_pos[1]:.4f}, {beaker_pos[2]:.4f})")
            print(f"  XY_diff_mm: ({abs(finger_mid[0]-beaker_pos[0])*1000:.1f}, {abs(finger_mid[1]-beaker_pos[1])*1000:.1f})")
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
    def place(env, 
              target_container_name, 
              target_pos=None, 
              target_quat=None,
              motion_planning_kwargs=dict()):
        """
        general place function for data generation
        param:
            env: LM4manipEnv object
            target_entity_name: str, target entity name
            target_pos: np.array, target position. If None, will propose a target position automatically
            target_quat: np.array, target quaternion. If None, will propose a target quaternion automatically
        return: 
            observations: list of obs
            waypoints: list of actions
            key_frame: list of key action such as move to prepare point, grasp
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
            # if no target_quat is provided, use the default quat
            target_quat = env.robot.get_end_effector_quat(env.physics)
        
        
        obstacle_pcd = np.asarray(env.get_obstacle_pcd().points)
        # np.save("obstacle_pcd.npy", obstacle_pcd)
        
        start_pos, start_quat, target_pos, target_quat = np.array(start_pos), np.array(start_quat), np.array(target_pos), np.array(target_quat)
        #FIXME if can not find a path, consider change another algorithm
        #FIXME optimize the path with min margin to obstacles for safer moving
        init2target_path = rrt_motion_planning(tuple(start_pos), 
                                                tuple(target_pos), 
                                                obstacle_pcd,
                                                **motion_planning_kwargs)
        offset = env.robot.ee_offset(env.physics) # for avoid the collision
        if init2target_path is None:
            print("can not find a path to target position, use default lift")
            # default solution is lifting
            if start_pos[2] <= target_pos[2]:
                mid_point = np.array([start_pos[0], start_pos[1], target_pos[2]])
            else:
                mid_point = np.array([target_pos[0], target_pos[1], start_pos[2]])
        
            init2target_path = [start_pos, mid_point, target_pos]
        path = np.array(init2target_path)
        path += offset

        # 补偿被抓取物体的高度：
        # path += offset 后，end_effector_move 会到达 target_pos（桌面高度）
        # 但物体底部可能在 end_effector_move 下方（被夹爪夹在中上部）
        # 需要把路径抬高，让物体底部刚好在 target_pos Z 高度
        grasped_names, grasped_entities = env.get_grasped_entity()
        if grasped_entities:
            grasped_entity = grasped_entities[0]
            obj_bottom_z = np.array(grasped_entity.get_xpos(env.physics))[2]
            ee_move_site = env.robot._mjcf_model.find("site", "end_effector_move")
            ee_move_z = env.physics.bind(ee_move_site).xpos[2]
            z_offset = obj_bottom_z - ee_move_z
            if z_offset < 0:
                path[:, 2] += (-z_offset)
            print(f"DEBUG [place]: 物体高度补偿")
            print(f"  物体底部 Z: {obj_bottom_z:.4f}, ee_move Z: {ee_move_z:.4f}")
            print(f"  z_offset: {z_offset:.4f}m, 抬高量: {max(0, -z_offset):.4f}m")

        path_point_len = len(init2target_path)
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
        # grasp
        new_obs, new_waypoints, _, task_success = SkillLib.open_gripper(env)
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        
        observations.pop(-1)   
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
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
    def pour_to_entity(env, target_container_name, tilt_angle=np.pi/2, tilt_velocity=np.pi/80,
                       n_repeat_step=6, lift_before=0.1, wait_time=10):
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