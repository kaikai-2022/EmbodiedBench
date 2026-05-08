import os
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
        #   1 = enabled (qpos sync - 位置修正同步)
        #   2 = enabled (weld constraint - 物理焊接约束，完全同步)
        # Enable via: os.environ["DM_ENV_GRASP_LOCK"] = "1" or "2" or call enable_grasp_lock(mode)
        self._grasp_lock_mode = int(os.environ.get("DM_ENV_GRASP_LOCK", "2"))
        self._grasped_entity_info = None  # {"name": str, "rel_pos_local": np, "rel_quat_local": np}
        self._grasp_lock_hand_body_id = None  # cached hand body id
        self._weld_constraint_id = None  # track active weld constraint

        # 预编译的 grasp lock callback 函数（避免重复创建）
        self._grasp_lock_callback = None
        # callback 会在第一个 step 时注册
        self._grasp_lock_callback_registered = False
        
    def reset(self):
        self.timestep = 0
        self._grasped_entity_info = None  # 清除 grasp lock 状态
        self._grasp_lock_hand_body_id = None  # 清除缓存
        self._weld_constraint_id = None  # 清除 weld 约束

        # 清除全局 MuJoCo callback（防止残留 callback 导致错误）
        try:
            mujoco.set_mjcb_control(None)
        except Exception:
            pass
        self._grasp_lock_callback_registered = False

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
    _GRASP_LOCK_SYNC_INTERVAL = 5

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
        self.pcd_generator.physics = self.physics
        if require_pcd:
            observation["masked_point_cloud"] = self.pcd_generator.generate_pcd_from_rgbd(target_id=list(range(self.physics.model.ncam - 1)), 
                                                                                            rgb=multi_view_rgb,
                                                                                            depth=multi_view_depth,
                                                                                            mask=expand_mask(observation["robot_mask"]))
            observation["point_cloud"] = self.pcd_generator.generate_pcd_from_rgbd(target_id=list(range(self.physics.model.ncam - 1)),
                                                                                    rgb=multi_view_rgb,
                                                                                    depth=multi_view_depth)
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
        self.pcd_generator = PointCloudGenerator(self.physics,
                                                 min_bound=[-1, -1, 0.7],
                                                 max_bound=[1, 1, 2],
                                                 **self.render_options)

    def _setup_grasp_lock_callback(self):
        """
        设置 MuJoCo 控制回调，实现每 substep 同步。
        这个回调会在每个 physics substep 前被调用。
        """
        def _grasp_lock_callback(model_ptr, data_ptr):
            """
            静态回调函数 - 使用闭包访问 self
            注意：MuJoCo callback 接收原生 mujoco model/data 指针
            """
            if self._grasp_lock_mode < 1 or self._grasped_entity_info is None:
                return

            # 获取原生 model/data 结构
            raw_m = self.physics.model._model
            raw_d = self.physics.data._data

            # 检查 physics 是否已初始化
            if raw_m is None or raw_d is None:
                return

            # 获取 hand body id（缓存）
            if self._grasp_lock_hand_body_id is None:
                try:
                    self._grasp_lock_hand_body_id = mujoco.mj_name2id(
                        raw_m, mujoco.mjtObj.mjOBJ_BODY, "franka/hand")
                except Exception:
                    return
            hand_id = self._grasp_lock_hand_body_id
            if hand_id < 0:
                return

            # 获取被抓物体的 freejoint 信息
            entity_name = self._grasped_entity_info["name"]
            entity = self.task.entities.get(entity_name)
            if entity is None:
                return

            # 查找 freejoint 的 qpos 和 qvel 地址
            tube_jnt_adr = None
            tube_jnt_idx = None
            for i in range(raw_m.njnt):
                jname = mujoco.mj_id2name(raw_m, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
                jnt_type = raw_m.jnt_type[i]
                if entity_name.lower() in jname.lower() and jnt_type == 0:  # freejoint
                    tube_jnt_adr = raw_m.jnt_qposadr[i]
                    tube_jnt_idx = i
                    break

            if tube_jnt_adr is None:
                return

            try:
                # 获取 hand 当前位姿
                hand_pos = raw_d.xpos[hand_id].copy()
                hand_quat = raw_d.xquat[hand_id].copy()

                # 计算期望的物体位姿
                rel_pos = self._grasped_entity_info["rel_pos_local"]
                rel_quat = self._grasped_entity_info["rel_quat_local"]

                expected_pos = hand_pos + _quat_rotate(hand_quat, rel_pos)
                expected_quat = _quat_mul(hand_quat, rel_quat)

                # 获取当前位置
                current_pos = raw_d.qpos[tube_jnt_adr:tube_jnt_adr+3].copy()
                pos_delta = np.linalg.norm(expected_pos - current_pos)

                # DEBUG: 打印同步信息
                print(f"[CALLBACK] entity={entity_name}, rel_pos={rel_pos}, hand={hand_pos}, expected={expected_pos}, current={current_pos}, delta={pos_delta:.4f}m")

                # 写入 qpos - 强制物体跟随夹爪
                raw_d.qpos[tube_jnt_adr:tube_jnt_adr+3] = expected_pos
                raw_d.qpos[tube_jnt_adr+3:tube_jnt_adr+7] = expected_quat

                # 清零 qvel 和外力
                if tube_jnt_idx is not None:
                    qvel_adr = raw_m.jnt_dofadr[tube_jnt_idx]
                    raw_d.qvel[qvel_adr:qvel_adr+6] = 0
                    raw_d.qfrc_applied[qvel_adr:qvel_adr+6] = 0
            except (IndexError, ValueError):
                # 数据可能无效，跳过这次同步
                pass

        # 注册 MuJoCo 回调
        # 注意：set_mjcb_control 设置的是全局回调，每个 physics step 都会调用
        try:
            mujoco.set_mjcb_control(_grasp_lock_callback)
            self._grasp_lock_callback = _grasp_lock_callback
        except AttributeError:
            # MuJoCo 版本不支持，降级到 step-level sync
            print("[GraspLock] WARNING: mujoco.set_mjcb_control not available, using step-level sync only")
            self._grasp_lock_callback = None

    # ========== Grasp Lock API ==========
    def enable_grasp_lock(self, mode=1):
        """
        Runtime 开启 grasp lock。

        Args:
            mode: 1 = qpos sync (位置修正), 2 = weld constraint (物理焊接，完全同步)
        """
        assert mode in (1, 2), f"Invalid mode {mode}, must be 1 or 2"
        self._grasp_lock_mode = mode
        if mode == 2:
            self._setup_weld_constraint()
        elif mode == 1:
            self._remove_weld_constraint()

    def disable_grasp_lock(self):
        """Runtime 关闭 grasp lock - 恢复默认摩擦力抓取"""
        self._grasp_lock_mode = 0
        self._grasped_entity_info = None
        self._remove_weld_constraint()

    @property
    def grasp_lock_mode(self):
        """查询当前 grasp lock 模式: 0=disabled, 1=qpos_sync, 2=weld"""
        return self._grasp_lock_mode

    def _is_gripper_closed(self):
        """检查夹爪是否关闭（通过 finger joint 位置判断）"""
        finger_pos = self.physics.data.qpos[7:9]  # finger_joint1, finger_joint2
        return np.max(finger_pos) < 0.02  # 闭合阈值 2cm

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
            return

        # 获取 hand 当前位姿
        hand_pos = raw_d.xpos[hand_id].copy()
        hand_quat = raw_d.xquat[hand_id].copy()

        # 计算期望的物体位姿
        rel_pos = self._grasped_entity_info["rel_pos_local"]
        rel_quat = self._grasped_entity_info["rel_quat_local"]

        expected_pos = hand_pos + _quat_rotate(hand_quat, rel_pos)
        expected_quat = _quat_mul(hand_quat, rel_quat)

        # 直接写入 qpos - 强制物体跟随夹爪
        raw_d.qpos[tube_jnt_adr:tube_jnt_adr+3] = expected_pos
        raw_d.qpos[tube_jnt_adr+3:tube_jnt_adr+7] = expected_quat

        # 清零 qvel 和外力 - 防止速度累积和碰撞力干扰
        if tube_jnt_idx is not None:
            qvel_adr = raw_m.jnt_dofadr[tube_jnt_idx]
            raw_d.qvel[qvel_adr:qvel_adr+6] = 0
            raw_d.qfrc_applied[qvel_adr:qvel_adr+6] = 0

    def _setup_weld_constraint(self):
        """
        为被抓物体创建 weld 约束，将其焊接到夹爪 hand body。
        Weld 约束在物理引擎层面保证两 body 完全同步。
        """
        if self._grasped_entity_info is None:
            print("[GraspLock] ⚠ Weld setup failed: _grasped_entity_info is None")
            return
        if self._weld_constraint_id is not None:
            print(f"[GraspLock] Weld already exists: eq_{self._weld_constraint_id}")
            return  # 已存在约束

        raw_m = self.physics.model._model
        raw_d = self.physics.data._data

        # 获取 hand body id
        if self._grasp_lock_hand_body_id is None:
            self._grasp_lock_hand_body_id = mujoco.mj_name2id(
                raw_m, mujoco.mjtObj.mjOBJ_BODY, "franka/hand")
        hand_id = self._grasp_lock_hand_body_id
        if hand_id < 0:
            print("[GraspLock] ⚠ Weld setup failed: hand body not found")
            return

        # 获取被抓物体的 body id
        entity_name = self._grasped_entity_info["name"]
        entity = self.task.entities.get(entity_name)
        if entity is None:
            print(f"[GraspLock] ⚠ Weld setup failed: entity '{entity_name}' not found")
            return

        # 获取 entity 在 MuJoCo 中的实际 body 名称
        # subentity 的 body 名称可能带命名空间前缀
        obj_body_name = entity.mjcf_model.model  # 使用 mjcf_model.model 获取实际名称
        obj_body_id = mujoco.mj_name2id(raw_m, mujoco.mjtObj.mjOBJ_BODY, obj_body_name)
        if obj_body_id < 0:
            # MuJoCo 中 subentity 的 body name 格式为 "parent/child" 或 "entity_name/"
            # 尝试添加斜杠后缀匹配
            obj_body_id = mujoco.mj_name2id(raw_m, mujoco.mjtObj.mjOBJ_BODY, entity_name + "/")
            if obj_body_id >= 0:
                obj_body_name = entity_name + "/"

            # 如果还找不到，尝试模糊匹配
            if obj_body_id < 0:
                for i in range(raw_m.nbody):
                    body_n = mujoco.mj_id2name(raw_m, mujoco.mjtObj.mjOBJ_BODY, i)
                    if body_n and entity_name.lower() in body_n.lower():
                        obj_body_id = i
                        obj_body_name = body_n
                        print(f"[GraspLock] DEBUG - 匹配到 body: '{body_n}' (id={i})")
                        break

            if obj_body_id < 0:
                print(f"[GraspLock] ⚠ Weld setup failed: obj body not found (tried '{obj_body_name}' and '{entity_name}')")
                return

        # 计算局部坐标系偏移（物体相对于 hand 的初始相对位姿）
        rel_pos = self._grasped_entity_info["rel_pos_local"]
        rel_quat = self._grasped_entity_info["rel_quat_local"]

        # 分配 weld 约束
        # 使用 mj_addEquality 动态添加约束（支持 neq=0 的情况）
        try:
            # mj_addEquality(model, data, type, obj1id, obj2id, data_vec)
            # type=2 表示 weld 约束
            eq_adr = mujoco._functions.mj_addEquality(
                raw_m, raw_d,
                2,  # mjEQ_WELD
                hand_id,
                obj_body_id,
                np.concatenate([rel_pos, rel_quat])
            )
            if eq_adr < 0:
                print(f"[GraspLock] ⚠ Weld setup failed: mj_addEquality returned {eq_adr}")
                return
        except AttributeError:
            # mj_addEquality 不可用（MuJoCo < 3.3），使用增强版 qpos sync
            # mode 2 降级为增强 sync：每次物理步进前后都强制同步位姿
            print("[GraspLock] mj_addEquality not available, using enhanced qpos sync (pre+post step)")
            self._sync_grasped_entity_pose()
            return

        self._weld_constraint_id = eq_adr
        print(f"[GraspLock] ✓ Weld constraint created: eq_{eq_adr}, hand={hand_id}, obj={obj_body_name}(id={obj_body_id})")

    def _remove_weld_constraint(self):
        """移除 weld 约束"""
        if self._weld_constraint_id is None:
            return

        raw_m = self.physics.model._model
        raw_m.eq_active[self._weld_constraint_id] = 0
        print(f"[GraspLock] Weld constraint removed: eq_{self._weld_constraint_id}")
        self._weld_constraint_id = None

    def _maintain_weld_constraint(self):
        """
        Weld 模式下定期检查约束有效性。
        如果被抓物体脱离了（不再被抓着），移除约束。
        """
        if self._weld_constraint_id is None:
            return

        # 检查物体是否仍然被抓着
        entity_name = self._grasped_entity_info.get("name")
        if entity_name:
            entity = self.task.entities.get(entity_name)
            if entity and hasattr(entity, "is_grasped"):
                if not entity.is_grasped(self.physics, self.robot):
                    # 不再被抓着，移除约束
                    self._remove_weld_constraint()
                    self._grasped_entity_info = None

    def get_grasped_entity(self):
        name_list = []
        entity_list = []
        for name, entity in self.task.entities.items():
            if hasattr(entity, "is_grasped"):
                if entity.is_grasped(self.physics, self.robot):
                    name_list.append(name)
                    entity_list.append(entity)
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
        obstacle_pcd= self.pcd_generator.generate_pcd_from_rgbd(target_id=list(range(self.physics.model.ncam - 1)), 
                                                                                        rgb=multi_view_rgb,
                                                                                        depth=multi_view_depth,
                                                                                        mask=expand_mask(total_mask))
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