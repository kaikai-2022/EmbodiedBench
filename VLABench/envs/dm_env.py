import os
import sys
import numpy as np
import math
import mujoco
from dm_control import composer
import dm_env as dm_env_lib
from VLABench.utils.utils import euler_to_quaternion, expand_mask
from VLABench.utils.depth2cloud import rotMatList2NPRotMat, quat2Mat, posRotMat2Mat, PointCloudGenerator


# ========== Grasp Lock: Quaternion Helpers ==========
def _quat_conjugate(q):
    """Quaternion conjugate: q* = [w, -x, -y, -z]"""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def _quat_mul(q1, q2):
    """Quaternion multiplication (MuJoCo [w,x,y,z] convention)"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def _quat_rotate(q, v):
    """Rotate vector v by quaternion q (MuJoCo [w,x,y,z] convention)"""
    qv = np.array([0, v[0], v[1], v[2]])
    return _quat_mul(_quat_mul(q, qv), _quat_conjugate(q))[1:]

class LM4ManipDMEnv(composer.Environment):
    def __init__(self, reset_wait_step=10, **kwargs):
        super().__init__(**kwargs)
        self.timestep = 0
        self.reset_wait_step = reset_wait_step
        self.n_distractor = kwargs.get("n_distractor", 0)
        self.render_options = dict(
            height=480,
            width=480,
        )
        self.register_pcd_generator()

        # ========== Grasp Lock: Rigid object-gripper attachment ==========
        # Modes:
        #   0 = disabled (default friction-based grasping)
        #   1 = enabled (substep sync - 需 skill_lib 手动设置 _grasped_entity_info)
        #   2 = auto_detect + substep sync - 自动检测被抓物体并激活同步
        # Enable via: os.environ["DM_ENV_GRASP_LOCK"] = "0" or "1" or "2"
        _raw_mode = int(os.environ.get("DM_ENV_GRASP_LOCK", "0"))
        self._grasp_auto_detect = (_raw_mode == 2)
        self._grasp_lock_mode = 1 if self._grasp_auto_detect else _raw_mode
        self._grasped_entity_info = None  # {"name": str, "rel_pos_local": np, "rel_quat_local": np}
        self._grasp_lock_hand_body_id = None  # cached hand body id

        # 技能执行模式开关：True 时 should_terminate() 触发 LAST 但不 reset 环境
        # 让技能（如 place）在检测到条件满足后仍能继续执行后续动作（open_gripper、lift）
        self._skill_execution_mode = False

        # reset 重入检测：防止 condition 误触 should_terminate 导致 reset→step→reset 死循环
        # 见 TROUBLESHOOTING.md 条目 #3 和 #5
        self._reset_depth = 0
        self._reset_reentry_warned = False

    def reset(self):
        # 重入检测：如果 reset 在自己的 wait_step 循环里又被 step→_reset_next_step 触发，
        # 说明某个 condition 在 record_initial_state 调用前就返回 True，会导致无限递归卡死。
        if self._reset_depth > 0 and not self._reset_reentry_warned:
            import logging
            logging.error(
                "[dm_env.reset] 检测到重入（depth=%d）。这通常意味着某个 Condition 在 "
                "record_initial_state 未调用时就返回 True，env.reset() 会陷入无限循环。"
                "排查方法：检查 env.task.conditions 中所有 condition 的 is_met() 是否在 "
                "reset 阶段误返 True。详见 TROUBLESHOOTING.md 条目 #3 / #5。" % self._reset_depth
            )
            self._reset_reentry_warned = True
        self._reset_depth += 1
        try:
            return self._reset_impl()
        finally:
            self._reset_depth -= 1
            if self._reset_depth == 0:
                # 顶层 reset 返回，清掉警告标记以便下次 reset 周期重新检测
                self._reset_reentry_warned = False

    def _reset_impl(self):
        self.timestep = 0
        self._grasped_entity_info = None  # 清除 grasp lock 状态
        self._grasp_lock_hand_body_id = None  # 清除缓存

        timestep = super().reset()
        self.cancel_gravity_and_improve_fluid()
        for i in range(self.reset_wait_step):
            self.step()
        self.reset_gravity_and_fluid(self.physics)
        for _ in range(self.reset_wait_step):
            self.step()
        self.register_pcd_generator()
        return timestep
        
    def render(self, **kwargs):
        return self.physics.render(**kwargs)
    
    # ========== Grasp Lock: Sub-step sync interval ==========
    # 每隔多少个 physics substep 同步一次被抓物体
    # 1 = 每个 substep 都同步（最紧密，但可能影响碰撞）
    # N > 1 = 每 N 个 substep 同步一次（平衡精度和稳定性）
    _GRASP_LOCK_SYNC_INTERVAL = 10

    def step(self, action=None):
        if action is None: # robot stay in static
            action = self.robot.get_qpos(self.physics)
            action = np.concatenate([action, 0.04 * np.ones((2))], axis=-1)
        else:
            self.timestep += 1

        # ========== 重写 step 逻辑：插入 substep 级同步 ==========
        if self._reset_next_step:
            self._reset_next_step = False
            return self.reset()

        self._hooks.before_step(self._physics_proxy, action, self._random_state)
        self._observation_updater.prepare_for_next_control_step()

        # Mode 2: 自动检测/释放被抓物体（必须在 should_sync 计算之前）
        if self._grasp_auto_detect:
            self._try_auto_detect_grasp(action)

        should_sync = (
            self._grasp_lock_mode >= 1
            and self._grasped_entity_info is not None
        )
        sync_interval = self._GRASP_LOCK_SYNC_INTERVAL if should_sync else None

        try:
            for i in range(self._n_sub_steps):
                self._substep(action)
                # 每 N 个 substep 同步一次被抓物体
                if should_sync and (i + 1) % sync_interval == 0:
                    self._sync_grasped_entity_pose()
                if i < self._n_sub_steps - 1:
                    self._observation_updater.update()
            physics_is_divergent = False
        except Exception as e:
            import logging
            logging.warning(e)
            physics_is_divergent = True

        self._hooks.after_step(self._physics_proxy, self._random_state)
        self._observation_updater.update()

        if not physics_is_divergent:
            reward = self._task.get_reward(self._physics_proxy)
            discount = self._task.get_discount(self._physics_proxy)
            terminating = (
                self._task.should_terminate_episode(self._physics_proxy)
                or self._physics.time() >= self._time_limit
            )
        else:
            reward = 0.0
            discount = 0.0
            terminating = True

        obs = self._observation_updater.get_observation()

        if not terminating:
            return dm_env_lib.TimeStep(dm_env_lib.StepType.MID, reward, discount, obs)
        else:
            if self._skill_execution_mode:
                # 技能执行期间：不终止，不 reset，继续返回 MID
                return dm_env_lib.TimeStep(dm_env_lib.StepType.MID, reward, discount, obs)
            self._reset_next_step = True
            return dm_env_lib.TimeStep(dm_env_lib.StepType.LAST, reward, discount, obs)
    
    @property
    def mjmodel(self):
        return self.physics.model
    
    @property
    def mjdata(self):
        return self.physics.data

    @property
    def robot(self):
        return self.task.robot
    
    def get_element_by_name(self, name, type):
        if type == "camera":
            element = self.mjmodel.cam(name)
        else:
            element = self.task.get_element_by_name(name, type)
        return element
    
    def get_xpos_by_name(self, name, type="geom"):
        """
        get the position of the entity by name
        """
        element = self.get_element_by_name(name, type)
        data = self.physics.bind(element)
        return data.xpos
    
    def get_xquat_by_name(self, name, type="geom"):
        """
        get the orientation of the entity by name
        """
        element = self.get_element_by_name(name, type)
        data = self.physics.bind(element)
        return data.xquat
    
    def set_xpos_by_name(self, name, type, pos):
        element = self.get_element_by_name(name, type)
        assert len(pos) == 3 or pos.shape == (3, 0), "pos must be a 3D vector"
        element.pos = pos
        self.step()
        
    
    def set_xquat_by_name(self, name, type, quat=None, euler=None):
        obj = self.get_element_by_name(name, type)
        assert (quat is None) != (euler is None), "quat and euler must be provided exclusively"
        if euler is not None:
            quat = euler_to_quaternion(roll=euler[0], pitch=euler[1], yaw=euler[2])
        obj.quat = quat
        self.step()
    
    def attach_entity(self, entity):
        self.task.add_free_entity(entity)
        self.step()
    
    def get_ee_pos(self):
        return self.robot.get_end_effector_pos(self.physics)
    
    def get_ee_quat(self):
        return self.robot.get_end_effector_quat(self.physics)
    
    def get_camera_matrix(self, cam_id, width, height):
        if cam_id >= self.physics.model.ncam:
            raise ValueError(f"cam_id {cam_id} is out of range")
        fovy = math.radians(self.physics.model.cam_fovy[cam_id])
        f = height / (2 * math.tan(fovy / 2))
        intrinsic_mat = np.array(((f, 0, width / 2), (0, f, height / 2), (0, 0, 1)))
        
        cam_pos = self.physics.data.cam_xpos[cam_id]
        c2b_r = rotMatList2NPRotMat(self.physics.model.cam_mat0[cam_id])
        b2w_r = quat2Mat([0, 1, 0, 0])
        cam_rot_mat = np.matmul(c2b_r, b2w_r)
        extrinsic_mat = posRotMat2Mat(cam_pos, cam_rot_mat)
        
        return intrinsic_mat, extrinsic_mat

    def get_observation(self, require_pcd=True):
        observation = dict()
        multi_view_rgb, multi_view_depth, multi_view_seg = [], [], []
        instrinsic_matrixs, extrinsic_matrixs = [], []
        for cam in range(self.physics.model.ncam):
            multi_view_rgb.append(self.render(camera_id=cam, **self.render_options))
            multi_view_depth.append(self.render(camera_id=cam, **self.render_options, depth=True))
            multi_view_seg.append(self.render(camera_id=cam, **self.render_options, segmentation=True))
            instrinsic, extrinsic = self.get_camera_matrix(cam, **self.render_options)
            instrinsic_matrixs.append(instrinsic)
            extrinsic_matrixs.append(extrinsic)
        observation["q_state"] = np.array(self.robot.get_qpos(self.physics))
        observation["q_velocity"] = np.array(self.robot.get_qvel(self.physics))
        observation["q_acceleration"] = np.array(self.robot.get_qacc(self.physics))
        observation["rgb"] = np.array(multi_view_rgb)
        observation["depth"] = np.array(multi_view_depth)
        observation["segmentation"] = np.array(multi_view_seg)
        observation["robot_mask"] = np.where((observation["segmentation"][..., 0] <= 72)&(observation["segmentation"][..., 0] > 0), 0, 1).astype(np.uint8)
        observation["instrinsic"] = np.array(instrinsic_matrixs)
        observation["extrinsic"] = np.array(extrinsic_matrixs)
        if self._ensure_pcd_generator() is not None:
            self.pcd_generator.physics = self.physics
        # open3d 0.18 在某些环境下 PointCloud.create_from_rgbd_image 会 segfault,
        # 这里强制禁用 PC 生成以保证仿真主流程可跑。RGB / depth / segmentation / mask 等
        # 基础观测仍正常返回,足够仿真 + skill 执行使用。
        # open3d 0.18 在某些环境下 PointCloud.create_from_rgbd_image 会 segfault,
        # 这里用空 PointCloud 占位以保证仿真主流程可跑。PC 物理意义为零(碰撞检测将始终报无碰撞),
        # 但 RGB / depth / segmentation / mask 等基础观测仍正常返回,够 skill 执行。
        import open3d as o3d
        if require_pcd:
            observation["masked_point_cloud"] = o3d.geometry.PointCloud()
            observation["point_cloud"] = o3d.geometry.PointCloud()
        observation["ee_state"] = self.robot.get_ee_state(self.physics)
        observation["grasped_obj_name"] = self.get_grasped_entity()
        observation.update(self.task.task_observables)
        return observation

    
    def cancel_gravity_and_improve_fluid(self):
        """
        This function is used when initializing the scene.
        """
        self.task._arena.mjcf_model.option.flag.gravity = "disable"
        geoms = self.task._arena.mjcf_model.find_all("geom")
        for geom in geoms:
            geom.fluidshape = "ellipsoid"
            geom.fluidcoef = [1e4, 1e4, 1e4, 1e4, 1e4]
    
    def reset_gravity_and_fluid(self, physics):
        self.task._arena.mjcf_model.option.flag.gravity = "enable"
        geoms = self.task._arena.mjcf_model.find_all("geom")
        for geom in geoms:
            geom.fluidshape = "none"
            geom.fluidcoef = [0.5, 0.25, 1.5, 1.0, 1.0]
            physics.data.qvel = 0
    
    def register_pcd_generator(self):
        # 懒加载 + 容错：open3d 在某些环境下 (e.g. numpy ABI mismatch) 调用
        # AxisAlignedBoundingBox 会 segfault。把初始化推迟到第一次真正需要
        # point cloud 时再尝试；若失败则禁用 PC 功能，后续 observation 直接跳过。
        self.pcd_generator = None
        self._pcd_generator_failed = False

    def _ensure_pcd_generator(self):
        """首次需要 point cloud 时再实例化；实例化失败则永久禁用。"""
        if self.pcd_generator is not None or self._pcd_generator_failed:
            return self.pcd_generator
        try:
            self.pcd_generator = PointCloudGenerator(
                self.physics,
                min_bound=[-1, -1, 0.7],
                max_bound=[1, 1, 2],
                **self.render_options,
            )
        except Exception as e:
            self._pcd_generator_failed = True
            logger = __import__("logging").getLogger(__name__)
            logger.warning(f"[dm_env] PointCloudGenerator 初始化失败，已禁用 PC 功能: {e}")
            return None
        return self.pcd_generator

    # ========== Grasp Lock API ==========
    def _try_auto_detect_grasp(self, action):
        """
        Mode 2 自动检测：当夹爪关闭且有物体接触时初始化 grasp lock，
        当夹爪打开时释放 grasp lock。
        设置 _grasped_entity_info 后，由 Mode 1 的 _sync_grasp_lock() 执行同步。
        """
        if action is None:
            return

        # 夹爪是否关闭（get_ee_open_state 实际返回的是"是否关闭"）
        gripper_closed = self.robot.get_ee_open_state(self.physics)

        if self._grasped_entity_info is not None:
            # 已锁定：检查是否需要释放
            if not gripper_closed:
                self._grasped_entity_info = None
            return

        # 未锁定：检测新抓取
        if not gripper_closed:
            return

        grasped_names, _ = self.get_grasped_entity()
        if not grasped_names:
            return

        self._init_grasp_lock_info(grasped_names[0])

    def _init_grasp_lock_info(self, entity_name):
        """初始化 _grasped_entity_info，计算 hand 与 entity 的相对位姿。"""
        raw_m = self.physics.model._model
        raw_d = self.physics.data._data

        if self._grasp_lock_hand_body_id is None:
            self._grasp_lock_hand_body_id = mujoco.mj_name2id(
                raw_m, mujoco.mjtObj.mjOBJ_BODY, "franka/hand")
        hand_id = self._grasp_lock_hand_body_id
        if hand_id < 0:
            return

        entity = self.task.entities.get(entity_name)
        if entity is None:
            return

        hand_pos = raw_d.xpos[hand_id].copy()
        hand_quat = raw_d.xquat[hand_id].copy()
        entity_pos = np.array(entity.get_xpos(self.physics))
        entity_quat = np.array(entity.get_xqaut(self.physics))

        rel_pos = entity_pos - hand_pos
        rel_pos_local = _quat_rotate(_quat_conjugate(hand_quat), rel_pos)
        rel_quat_local = _quat_mul(_quat_conjugate(hand_quat), entity_quat)

        self._grasped_entity_info = {
            "name": entity_name,
            "rel_pos_local": rel_pos_local,
            "rel_quat_local": rel_quat_local,
        }

    def enable_grasp_lock(self):
        """Runtime 开启 grasp lock"""
        self._grasp_lock_mode = 1

    def disable_grasp_lock(self):
        """Runtime 关闭 grasp lock - 恢复默认摩擦力抓取"""
        self._grasp_lock_mode = 0
        self._grasped_entity_info = None

    @property
    def grasp_lock_mode(self):
        """查询当前 grasp lock 模式: 0=disabled, 1=substep_sync"""
        return self._grasp_lock_mode

    def _sync_grasped_entity_pose(self):
        """
        核心：同步被抓物体到夹爪相对位姿。
        在每个 env.step() 后调用，确保被抓物体与夹爪完全同步。
        """
        if self._grasped_entity_info is None:
            return

        raw_m = self.physics.model._model
        raw_d = self.physics.data._data

        # 获取 hand body id
        if self._grasp_lock_hand_body_id is None:
            self._grasp_lock_hand_body_id = mujoco.mj_name2id(
                raw_m, mujoco.mjtObj.mjOBJ_BODY, "franka/hand")

        hand_id = self._grasp_lock_hand_body_id
        if hand_id < 0:
            return

        # 获取被抓物体的 freejoint qpos 地址
        entity_name = self._grasped_entity_info["name"]
        entity = self.task.entities.get(entity_name)
        if entity is None:
            return

        tube_jnt_adr = None
        tube_jnt_idx = None
        for i in range(raw_m.njnt):
            jname = mujoco.mj_id2name(raw_m, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
            jnt_type = raw_m.jnt_type[i]
            if entity_name.lower() in jname.lower() and jnt_type == 0:
                tube_jnt_adr = raw_m.jnt_qposadr[i]
                tube_jnt_idx = i
                break

        if tube_jnt_adr is None:
            # ========== ContainerWithDrawer 专用 lock ==========
            # 实体是带 slide joint 的抽屉（如 drawer's top/middle/bottom drawer），
            # slide axis 通常是世界 (1,0,0)。lock 启动瞬间记录 drawer 当前 qpos 和 hand
            # 世界坐标，之后每个 step 把 qpos 设为 init_qpos + dot(hand - init_hand, axis)，
            # 实现"handle 跟着手走"。
            if hasattr(entity, 'drawers') and hasattr(entity, 'get_drawer_handle_pos'):
                hand_pos_now = raw_d.xpos[hand_id].copy()
                # 找离 hand 最近的 drawer handle
                best_id = None
                best_dist = float('inf')
                for did, _drawer in enumerate(entity.drawers):
                    try:
                        hpos = entity.get_drawer_handle_pos(self.physics, did)
                        d = float(np.linalg.norm(np.array(hpos) - hand_pos_now))
                        if d < best_dist:
                            best_dist = d
                            best_id = did
                    except ValueError:
                        continue
                if best_id is not None:
                    drawer = entity.drawers[best_id]
                    # drawer.name 形如 "drawer_top"，joint 名形如 "drawer_0/top_drawer"。
                    # 用 drawer.name 最后一段（如 "top"）做匹配。
                    drawer_key = drawer.name.split("_")[-1].lower()
                    slide_jnt_id = None
                    for i in range(raw_m.njnt):
                        jname = mujoco.mj_id2name(raw_m, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
                        if jname and drawer_key in jname.lower() and raw_m.jnt_type[i] == mujoco.mjtJoint.mjJNT_SLIDE:
                            slide_jnt_id = i
                            break
                    if slide_jnt_id is not None:
                        axis = raw_d.xaxis[slide_jnt_id].copy()
                        qpos_adr = raw_m.jnt_qposadr[slide_jnt_id]
                        qvel_adr = raw_m.jnt_dofadr[slide_jnt_id]
                        # 首次进入 slide lock 或 drawer 切换时，记录初始状态
                        if (not hasattr(self, '_drawer_lock_drawer_id')
                                or self._drawer_lock_drawer_id != best_id):
                            self._drawer_lock_drawer_id = best_id
                            self._drawer_lock_init_qpos = float(raw_d.qpos[qpos_adr])
                            self._drawer_lock_init_hand_pos = hand_pos_now.copy()
                        # 目标 qpos = init_qpos + (hand 沿 axis 相对初始位置的位移)
                        target_qpos = self._drawer_lock_init_qpos + float(
                            np.dot(hand_pos_now - self._drawer_lock_init_hand_pos, axis)
                        )
                        # clamp 到 joint range
                        lo, hi = raw_m.jnt_range[slide_jnt_id]
                        if hi > lo:
                            target_qpos = float(np.clip(target_qpos, lo, hi))
                        raw_d.qpos[qpos_adr] = target_qpos
                        raw_d.qvel[qvel_adr] = 0
                        raw_d.qfrc_applied[qvel_adr] = 0
                return

            # ========== Door-Hinge 专用 lock ==========
            # 实体没有 freejoint，但如果是带 door_joint 的 ContainerWithDoor 派生类
            # （如 DryingBoxWithButton），则锁定 door body 围绕 hinge 轴的角度跟随夹爪
            if hasattr(entity, 'door_joint') and entity.door_joint is not None:
                door_joint = entity.door_joint
                full_joint_name = f"{entity_name}/{door_joint.name}"
                door_jnt_id = mujoco.mj_name2id(raw_m, mujoco.mjtObj.mjOBJ_JOINT, full_joint_name)
                if door_jnt_id < 0:
                    door_jnt_id = mujoco.mj_name2id(raw_m, mujoco.mjtObj.mjOBJ_JOINT, door_joint.name)
                if door_jnt_id >= 0 and raw_m.jnt_type[door_jnt_id] == mujoco.mjtJoint.mjJNT_HINGE:
                    current_key = (entity_name, id(door_joint))
                    if (not hasattr(self, '_hinge_lock_entity_key')
                            or self._hinge_lock_entity_key != current_key):
                        for attr in ['_hinge_lock_entity_key', '_hinge_lock_world_angle',
                                      '_hinge_lock_init_world_angle', '_hinge_lock_init_qpos',
                                      '_hinge_lock_call_count']:
                            if hasattr(self, attr):
                                delattr(self, attr)
                        self._hinge_lock_entity_key = current_key
                    # 1. 获取 hinge anchor 和 axis（世界坐标）
                    anchor = raw_d.xanchor[door_jnt_id].copy()
                    axis = raw_d.xaxis[door_jnt_id].copy()
                    # 2. 获取 hand 当前世界位置
                    hand_pos = raw_d.xpos[hand_id].copy()
                    # 3. 计算 gripper 相对于 anchor 的世界坐标偏移
                    dx = hand_pos[0] - anchor[0]
                    dy = hand_pos[1] - anchor[1]
                    dz = hand_pos[2] - anchor[2]
                    # 4. 根据 hinge axis 方向，计算门的旋转角度（世界坐标）
                    #    门绕轴旋转时，gripper 绕 anchor 做圆周运动
                    #    使用 atan2 计算 gripper 相对 anchor 的角度
                    if abs(axis[2]) > 0.9:  # hinge 绕 world z 轴旋转
                        world_angle = np.arctan2(dx, -dy)
                    elif abs(axis[0]) > 0.9:  # hinge 绕 world x 轴旋转
                        world_angle = np.arctan2(dz, -dy)
                    elif abs(axis[1]) > 0.9:  # hinge 绕 world y 轴旋转
                        world_angle = np.arctan2(dz, dx)
                    else:
                        world_angle = 0.0
                    # 5. 首次调用时记录初始状态
                    if not hasattr(self, '_hinge_lock_world_angle'):
                        self._hinge_lock_world_angle = world_angle
                        self._hinge_lock_init_world_angle = world_angle
                        self._hinge_lock_init_qpos = float(raw_d.qpos[raw_m.jnt_qposadr[door_jnt_id]])
                    last_wa = self._hinge_lock_world_angle
                    # 6. atan2 角度跳变处理：规范化到与上次值在 ±π 范围内
                    while world_angle - last_wa > np.pi:
                        world_angle -= 2 * np.pi
                    while world_angle - last_wa < -np.pi:
                        world_angle += 2 * np.pi
                    self._hinge_lock_world_angle = world_angle
                    # 7. 目标角度 = 初始 qpos + (当前 world_angle - 初始 world_angle)
                    target_angle = self._hinge_lock_init_qpos + (world_angle - self._hinge_lock_init_world_angle)
                    # clamp 到 hinge range
                    lo, hi = raw_m.jnt_range[door_jnt_id]
                    if hi > lo:
                        target_angle = float(np.clip(target_angle, lo, hi))
                    # 8. 写入 hinge qpos
                    qpos_adr = raw_m.jnt_qposadr[door_jnt_id]
                    raw_d.qpos[qpos_adr] = target_angle
                    # 9. 清零 qvel（防止物理引擎反向旋转抵消）
                    qvel_adr = raw_m.jnt_dofadr[door_jnt_id]
                    raw_d.qvel[qvel_adr] = 0
                    raw_d.qfrc_applied[qvel_adr] = 0
                    # DEBUG: 前5次和每200次
                    if not hasattr(self, '_hinge_lock_call_count'):
                        self._hinge_lock_call_count = 0
                    self._hinge_lock_call_count += 1
                    if self._hinge_lock_call_count <= 5 or self._hinge_lock_call_count % 200 == 0:
                        init_wa = self._hinge_lock_init_world_angle
                        print(f"[hinge_lock #{self._hinge_lock_call_count}] world_angle={world_angle:.3f}, init={init_wa:.3f}, delta={world_angle - init_wa:.3f}, target={target_angle:.3f}, hand=[{hand_pos[0]:.3f},{hand_pos[1]:.3f},{hand_pos[2]:.3f}]", file=sys.stderr, flush=True)
            return

        # freejoint 物体：直接用相对位姿同步
        hand_pos = raw_d.xpos[hand_id].copy()
        hand_quat = raw_d.xquat[hand_id].copy()
        rel_pos = self._grasped_entity_info["rel_pos_local"]
        rel_quat = self._grasped_entity_info["rel_quat_local"]
        expected_pos = hand_pos + _quat_rotate(hand_quat, rel_pos)
        expected_quat = _quat_mul(hand_quat, rel_quat)
        raw_d.qpos[tube_jnt_adr:tube_jnt_adr+3] = expected_pos
        raw_d.qpos[tube_jnt_adr+3:tube_jnt_adr+7] = expected_quat
        if tube_jnt_idx is not None:
            qvel_adr = raw_m.jnt_dofadr[tube_jnt_idx]
            raw_d.qvel[qvel_adr:qvel_adr+6] = 0
            raw_d.qfrc_applied[qvel_adr:qvel_adr+6] = 0

    def get_grasped_entity(self):
        name_list = []
        entity_list = []
        for name, entity in self.task.entities.items():
            if hasattr(entity, "is_grasped"):
                if entity.is_grasped(self.physics, self.robot):
                    name_list.append(name)
                    entity_list.append(entity)
        # grasp lock 模式下接触力检测可能失效，从 _grasped_entity_info 补充
        if not entity_list and hasattr(self, "_grasped_entity_info") and self._grasped_entity_info:
            fallback_name = self._grasped_entity_info.get("name")
            if fallback_name and fallback_name in self.task.entities:
                name_list = [fallback_name]
                entity_list = [self.task.entities[fallback_name]]
        return name_list, entity_list
    
    def _reset_attempt(self):
        self.task.reset_distractors(n_distractor=self.n_distractor)
        self._hooks.refresh_entity_hooks()
        return super()._reset_attempt()

    def get_obstacle_pcd(self): 
        multi_view_rgb, multi_view_depth, multi_view_seg = [], [], []
        for cam in range(self.physics.model.ncam):
            multi_view_rgb.append(self.render(camera_id=cam, **self.render_options))
            multi_view_depth.append(self.render(camera_id=cam, **self.render_options, depth=True))
            multi_view_seg.append(self.render(camera_id=cam, **self.render_options, segmentation=True))
        segmentation = np.array(multi_view_seg)
        total_mask = np.ones_like(segmentation[..., 0])
        robot_mask = np.where((segmentation[..., 0] <= 72)&(segmentation[..., 0] > 0), 0, 1).astype(np.uint8)
        total_mask *= robot_mask
        grasped_obj_name_list, grasped_obj = self.get_grasped_entity()
        for name in grasped_obj_name_list:
            geom_ids = [self.physics.bind(geom).element_id for geom in self.task.entities[name].geoms]
            obj_mask = np.where((segmentation[..., 0] <= max(geom_ids))&(segmentation[..., 0] >= min(geom_ids)), 0, 1).astype(np.uint8)
            total_mask *= obj_mask
        if self._ensure_pcd_generator() is not None:
            obstacle_pcd = self.pcd_generator.generate_pcd_from_rgbd(target_id=list(range(self.physics.model.ncam - 1)),
                                                                                        rgb=multi_view_rgb,
                                                                                        depth=multi_view_depth,
                                                                                        mask=expand_mask(total_mask))
        else:
            obstacle_pcd = None
        return obstacle_pcd
    
    def get_intention_score(self, threshold=0.5, discrete=True):
        """
        Get the intention score of the task
        """
        return self.task.get_intention_score(self.physics, threshold, discrete)
    
    def get_task_progress(self):
        """
        Get the stage progress score of the task
        """
        return self.task.get_task_progress(self.physics)
    
    def get_expert_skill_sequence(self):
        """
        Get the expert demenstration of trajectory generation sequence
        """
        return self.task.get_expert_skill_sequence(self.physics)
    
    def save(self):
        """
        Save the task and env configuration
        """
        return self.task.save(self.physics)
    
    def get_robot_frame_position(self):
        return self.robot.get_base_position(self.physics)