"""
Skill Library for data generation.
"""
import numpy as np
import random
import time as _time
from VLABench.utils.utils import find_keypoint_and_prepare_grasp, distance, quaternion_to_euler, euler_to_quaternion, quaternion_from_axis_angle, quaternion_multiply
from VLABench.algorithms.motion_planning.rrt import rrt_motion_planning
from VLABench.algorithms.utils import interpolate_path, qauternion_slerp

PRIOR_EULERS = [[np.pi, 0, -np.pi/2], # face down, horizontal
                [np.pi, 0, 0], # face down, vertical
                [-np.pi/2, -np.pi/2, 0], # face forward, horizontal
                [-np.pi/2, 0, 0], # face forward, vertical
            ]

class SkillLib:
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

        for i, (point, quat) in enumerate(zip(points, quats)):
            success, action = env.robot.get_qpos_from_ee_pos(physics=env.physics, pos=point, quat=quat)
            # if not success: # a wrong action beyond the embodied limit
            #     return None, None, False, False
            action = np.concatenate([action, gripper_state])
            waypoint = np.concatenate([point, quaternion_to_euler(quat), gripper_state])

            for substep_idx in range(max_n_substep):
                timestep = env.step(action)

                # 检查任务是否完成
                if timestep.last():
                    print(f"[STEP] 路径点 {i}/{len(points)} 处 timestep.last()=True")
                    task_success = True
                    break

                # 检查关节位置收敛
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
            if gripper_closed: gripper_state = np.zeros(2)
            else: gripper_state = np.ones(2) * 0.04
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

        print(f"DEBUG [pick]: 插值后的路径")
        print(f"  路径点数: {len(interplate_path)}")
        print(f"  最后5个路径点:")
        for i, p in enumerate(interplate_path[-5:]):
            print(f"    [{len(interplate_path)-5+i}]: {p}")

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
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        if task_success:
            observations.pop(-1)
            return observations, waypoints, True, task_success
        # grasp
        new_obs, new_waypoints, _, task_success = SkillLib.close_gripper(env)
        observations.extend(new_obs)
        waypoints.extend(new_waypoints)
        
        observations.pop(-1)
        assert len(observations) == len(waypoints), f"observations and waypoints should have the same length, {len(observations)} and {len(waypoints)}"
        if env.task.entities[target_entity_name].is_grasped(env.physics, env.robot):
            stage_success = True
            print(f"DEBUG [pick]: ✓ 抓取成功! stage_success=True")
        else:
            print(f"DEBUG [pick]: ✗ 抓取失败! is_grasped=False")
            print(f"  末端执行器最终位置: {env.robot.get_end_effector_pos(env.physics)}")
            print(f"  目标抓取点位置: {key_pos}")
            print(f"  位置误差: {np.linalg.norm(env.robot.get_end_effector_pos(env.physics) - key_pos)}")
        return observations, waypoints, stage_success, task_success
    
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
                                                                np.zeros(2))
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
            gripper_closed = env.robot.get_ee_open_state(env.physics)
            if gripper_closed: gripper_state = np.zeros(2)
            else: gripper_state = np.ones(2) * 0.04
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
        if gripper_closed: gripper_state = np.zeros(2)
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
        obs, wp, stage_success, _ = SkillLib.moveto(env, target_pos=lift_pos, gripper_state=np.zeros(2))
        observations.extend(obs)
        waypoints.extend(wp)
        if not stage_success:
            return observations, waypoints, False, task_success

        # 2. 移到容器上方
        obs, wp, stage_success, _ = SkillLib.moveto(env, target_pos=pour_target_pos, gripper_state=np.zeros(2))
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

            gripper_state = np.zeros(2)
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
        for qpos in reversed(tilt_qpos_history):
            action = np.concatenate([qpos, np.zeros(2)])
            for _ in range(n_repeat_step):
                timestep = env.step(action)
                if timestep.last():
                    task_success = True
                    break

            waypoint = np.concatenate([
                env.robot.get_end_effector_pos(env.physics),
                quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                np.zeros(2)
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
        if target_pos is None: 
            target_pos = np.array(start_pos) + np.array([0, 0, lift_height])
        if target_quat is None: 
            target_quat = start_quat
        interplate_path, interplate_quat = interpolate_path([start_pos, target_pos], [np.array(start_quat), np.array(target_quat)])
        observations = [env.get_observation()]
        waypoints = []
        if gripper_state is None:
            gripper_closed = env.robot.get_ee_open_state(env.physics)
            if gripper_closed: gripper_state = np.zeros(2)
            else: gripper_state = np.ones(2) * 0.04
        obs, new_waypoints, stage_success, task_success = SkillLib.step_trajectory(env, 
                                                           interplate_path, 
                                                           interplate_quat, 
                                                           gripper_state)
        observations.extend(obs)
        waypoints.extend(new_waypoints)
        observations.pop(-1)
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
        for i in range(10):
            gripper_state = np.ones(2) * (0.04 - i * 0.04/10)
            action = np.concatenate([qpos, gripper_state])
            for _ in range(repeat):
                timestep = env.step(action)
                if timestep.last():
                    success = True
                    obs = env.get_observation()
                    observations.append(obs)
                    waypoints.append(np.concatenate([env.robot.get_end_effector_pos(env.physics),
                                             quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                                             gripper_state]))
                    break
            obs = env.get_observation()
            observations.append(obs)
            waypoints.append(np.concatenate([env.robot.get_end_effector_pos(env.physics),
                                             quaternion_to_euler(env.robot.get_end_effector_quat(env.physics)),
                                             gripper_state]))
            
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
        return observations, waypoints, stage_success, task_success
    
    @staticmethod
    def flip(env, gripper_state=None, target_q_velocity=np.pi/40, max_n_substep=30, tolerance=0.01):
        qpos = np.array(env.robot.get_qpos(env.physics)).reshape(-1)
        observations = [env.get_observation()]
        waypoints = []
        if gripper_state is None:
            gripper_closed = env.robot.get_ee_open_state(env.physics)
            if gripper_closed: gripper_state = np.zeros(2)
            else: gripper_state = np.ones(2) * 0.04
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
            gripper_closed = env.robot.get_ee_open_state(env.physics)
            if gripper_closed: gripper_state = np.zeros(2)
            else: gripper_state = np.ones(2) * 0.04

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
            gripper_closed = env.robot.get_ee_open_state(env.physics)
            if gripper_closed: gripper_state = np.zeros(2)
            else: gripper_state = np.ones(2) * 0.04
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
            gripper_closed = env.robot.get_ee_open_state(env.physics)
            if gripper_closed:
                gripper_state = np.zeros(2)
            else:
                gripper_state = np.ones(2) * 0.04

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
            gripper_state = np.zeros(2)

        # 选择空闲孔位：找离当前夹持物体（末端执行器）XY 最近的空闲孔位
        ee_pos = np.array(env.robot.get_end_effector_pos(env.physics))
        # 按孔位与末端的 XY 距离排序——倾倒完后末端在烧杯上方，
        # 但夹持的试管在末端正下方，所以按 XY 排序仍能找到正确孔列
        # 更可靠的方式：按孔位与试管架中心的距离从近到远排序，
        # 优先选择离末端 XY 最近的空闲孔
        stand_xpos = np.array(entity.get_xpos(env.physics))
        sorted_points = sorted(place_points, key=lambda p: np.linalg.norm(np.array(p)[:2] - ee_pos[:2]))
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

        # 1. RRT 移动到孔位上方（偏移 15cm），指定竖直向下姿态
        # 参考 insert_tube_series：hover 高度需足够让试管底部高于孔口
        hover_pos = insert_point + np.array([0, 0, 0.15])
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
        # hover_pos = insert_point + [0,0,0.15]，只下降 0.12m，让试管口进入孔位即可
        descend = -0.12
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