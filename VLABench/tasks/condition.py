import numpy as np
from VLABench.utils.register import register
from VLABench.utils.utils import distance, quaternion_to_euler, matrix_to_quaternion
from VLABench.tasks.components.entity import Entity

class Condition:
    def __init__(self):
        self._initial_state_recorded = False
        self._met = False  # 条件已满足的标志

    def record_initial_state(self, physics=None):
        """在技能执行前调用，记录初始状态。子类可重写以支持前态-终态对比。"""
        self._initial_state_recorded = True
        self._met = False  # 重置条件满足标志

    def is_met(self, physics=None):
        raise NotImplementedError()

    def met_progress(self, physics=None):
        return self.is_met(physics)

@register.add_condition("order")
class OrderCondition(Condition):
    """
    check the position order of the given entities.
    params:
        entities: the given entities should in the expected order.
        axis: the  axis to check the order.
        offset: the acceptable offset between the entities in other axis.
    """
    def __init__(self, entities, axis=[0], offset=0.1):
        self.entities = entities
        self.axis = axis
        self.offset = offset
        
    def is_met(self, physics):
        if isinstance(self.entities[-1], Entity):
            self.entities_mjcf = [entity.mjcf_model.worldbody for entity in self.entities]
        entity_points = [physics.bind(entity_mjcf).xpos for entity_mjcf in self.entities_mjcf]
        for axis in [0, 1, 2]: # x y z
            if axis in self.axis:
                if not all([entity_points[i][axis] < entity_points[i+1][axis] for i in range(len(entity_points)-1)]):
                    return False
            else:
                if not all([(entity_points[i][axis] - entity_points[i+1][axis]) < self.offset for i in range(len(entity_points)-1)]):
                    return False
        return True

@register.add_condition("contain")
class ContainCondition(Condition):
    """
    Check if the container contains the target entities
    params:
        container: the container to contain the target eneities
        entities: the target entities to be contained
    """
    def __init__(self, container, entities, **kwargs):
        assert container is not None, "container must be provided"
        self.container = container
        self.entities = entities
        self.kwargs = kwargs
    
    def is_met(self, physics=None):
        if isinstance(self.entities[-1], Entity):
            self.entities_mjcf = [entity.mjcf_model.worldbody for entity in self.entities]
        entity_points = [physics.bind(entity_mjcf).xpos for entity_mjcf in self.entities_mjcf]
        for point in entity_points:
            if not self.container.contain(point, physics, **self.kwargs):
                return False
        return True

    def met_progress(self, physics=None):
        # TODO: return the progress of the condition
        return super().met_progress(physics)
    
@register.add_condition("not_contain")
class NotContainCondition(Condition):
    """
    Check if the container does not contain the target entities.
    params:
        container: the container to not contain the target entities
        entities: the target entities to be not contained
    """
    def __init__(self, container, entities):
        assert container is not None, "container must be provided"
        self.container = container
        self.entities = entities

    def is_met(self, physics=None):
        if isinstance(self.entities[-1], Entity):
            self.entities_mjcf = [entity.mjcf_model.worldbody for entity in self.entities]
        entity_points = [physics.bind(entity_mjcf).xpos for entity_mjcf in self.entities_mjcf]
        for point in entity_points:
            if self.container.contain(point, physics):
                return False
        return True
    
    def met_progress(self, physics=None):
        # TODO: return the progress of the condition
        return super().met_progress(physics)

@register.add_condition("is_grasped")
class IsGraspedCondition(Condition):
    """
    Check if the target entities are grasped
    """
    def __init__(self, entities, robot):
        self.entities = entities
        self.robot = robot
        
    def is_met(self, physics=None):
        for entity in self.entities:
            if not entity.is_grasped(physics, self.robot):
                return False
        return True

@register.add_condition("press_button")
class ButtonPressedCondition(Condition):
    """
    Check if the button is pressed
    """
    def __init__(self, target_button):
        self.button = target_button
        
    def is_met(self, physics=None):
        return self.button.is_pressed()
    
@register.add_condition("on")
class OnCondition(Condition):
    """
    Check if the target entities are on the target surface.
    In most of cases, only on entity in entities.
    """
    def __init__(self, entities, container):
        self.entities = entities
        self.container = container
        self._z_tolerance = 0.01  # 1cm 容差，处理浮点精度问题

    def is_met(self, physics=None):
        import mujoco as mj
        raw_m = physics.model._model
        raw_d = physics.data._data

        # 使用 ncon 正确遍历 contacts
        ncon = raw_d.ncon
        contacts = raw_d.contact

        container_geoms_id = [physics.bind(geom).element_id for geom in self.container.geoms]
        container_geoms_xpos = [physics.bind(geom).xpos for geom in self.container.geoms]
        max_xpos_z = max([xpos[-1] for xpos in container_geoms_xpos])

        print(f"DEBUG [OnCondition]: === 开始条件判断 ===")
        print(f"DEBUG [OnCondition]: 容器 geom 数量: {len(container_geoms_id)}")
        print(f"DEBUG [OnCondition]: 容器 geom IDs: {container_geoms_id}")
        print(f"DEBUG [OnCondition]: 容器 max_xpos_z: {max_xpos_z:.4f}")

        # 打印容器所有 geom 的名称
        for geom in self.container.geoms:
            gid = physics.bind(geom).element_id
            gname = mj.mj_id2name(raw_m, mj.mjtObj.mjOBJ_GEOM, gid) or '(unnamed)'
            gpos = physics.bind(geom).xpos
            print(f"DEBUG [OnCondition]:   容器 geom[{gid}] name={gname} pos=({gpos[0]:.4f},{gpos[1]:.4f},{gpos[2]:.4f})")

        # 打印所有 contact
        print(f"DEBUG [OnCondition]: === MuJoCo contacts 总数: {ncon} ===")
        for i in range(ncon):
            # MuJoCo contact 是 C 数组，直接用下标访问
            c = contacts[i]
            g1name = mj.mj_id2name(raw_m, mj.mjtObj.mjOBJ_GEOM, c.geom1) or '(unnamed)'
            g2name = mj.mj_id2name(raw_m, mj.mjtObj.mjOBJ_GEOM, c.geom2) or '(unnamed)'
            print(f"DEBUG [OnCondition]:   contact[{i}]: geom1={c.geom1}({g1name}) <-> geom2={c.geom2}({g2name})")

        for entity in self.entities:
            entity_geom_ids = [physics.bind(geom).element_id for geom in entity.geoms]
            entity_xpos = physics.bind(entity.mjcf_model.worldbody).xpos
            print(f"DEBUG [OnCondition]: 实体: {entity.name if hasattr(entity, 'name') else 'unknown'}")
            print(f"DEBUG [OnCondition]: 实体 worldbody xpos: ({entity_xpos[0]:.4f}, {entity_xpos[1]:.4f}, {entity_xpos[2]:.4f})")
            print(f"DEBUG [OnCondition]: 实体 geom IDs: {entity_geom_ids}")

            # 先检查接触 - 使用索引方式访问 contacts
            is_contacted = False
            contact_count = 0
            for i in range(ncon):
                c = contacts[i]
                g1, g2 = c.geom1, c.geom2
                if (g1 in container_geoms_id and g2 in entity_geom_ids) or \
                    (g2 in container_geoms_id and g1 in entity_geom_ids):
                    g1name = mj.mj_id2name(raw_m, mj.mjtObj.mjOBJ_GEOM, g1) or '(unnamed)'
                    g2name = mj.mj_id2name(raw_m, mj.mjtObj.mjOBJ_GEOM, g2) or '(unnamed)'
                    print(f"DEBUG [OnCondition]:   找到匹配接触: {g1}({g1name}) <-> {g2}({g2name})")
                    is_contacted = True
                    contact_count += 1
            print(f"DEBUG [OnCondition]: 接触点数: {contact_count}, is_contacted={is_contacted}")

            if not is_contacted:
                print(f"DEBUG [OnCondition]: ✗ 接触检查失败 (物体未接触到容器)")
                return False

            # 再检查 Z 轴位置
            z_check = entity_xpos[-1] >= max_xpos_z - self._z_tolerance
            print(f"DEBUG [OnCondition]: Z 轴检查: entity_z={entity_xpos[-1]:.4f} >= max_xpos_z-{self._z_tolerance}={max_xpos_z - self._z_tolerance:.4f} → {z_check}")

            if not z_check:
                print(f"DEBUG [OnCondition]: ✗ Z 轴检查失败 (物体高度低于容器)")
                return False

        print(f"DEBUG [OnCondition]: ✓ 所有检查通过")
        return True

@register.add_condition("above")
class AboveCondition(Condition):
    """
    Entity above the platform/container condition. Usually used for the entity is above the flat container. 
    params:
        target_entity: the target entity to be above the platform
        platform: the platform/flat container
    """
    def __init__(self, target_entity, platform):
        self.target_entity = target_entity
        self.platform = platform
    
    def is_met(self, physics):
        target_entity_xpos = physics.bind(self.target_entity.mjcf_model.worldbody).xpos
        z_platform = physics.bind(self.platform.mjcf_model.worldbody).xpos[-1]
        if target_entity_xpos[-1] < z_platform:
            return False
        else:
            point_to_check = target_entity_xpos
            point_to_check[-1] = z_platform + 0.01
            if self.platform.contain(point_to_check, physics): return True    
        return False

@register.add_condition("pour")
class PourCondition(Condition):
    """
    The cup/shaker/other_entity is poured.
    As mujoco does not support the liquid simulation, use this condition to simplify.
    The condition is the top site z pos is lower than the bottom site z pos.
    params:
        target_entity: the target entity to be poured
        threshold: the threshold of z_top - z_bottom, to confirm whether the entity is poured.

    Once the container is detected as tilted (bottom_site above top_site) at any frame
    during the step, self._met is latched to True, so the condition remains met even if
    the container is later returned to an upright posture within the same step
    (e.g. via a follow-up insert_to_entity skill). This mirrors the latch pattern used
    by ShakeCondition.
    """
    def __init__(self, target_entity, threshold=0):
        super().__init__()
        self.target_entity = target_entity
        self.threshold = threshold

    def is_met(self, physics):
        if self._met:
            return True
        top_site = self.target_entity.mjcf_model.worldbody.find("site", "top_site")
        bottom_site = self.target_entity.mjcf_model.worldbody.find("site", "bottom_site")
        top_site_xpos, bottom_site_xpos = physics.bind(top_site).xpos, physics.bind(bottom_site).xpos
        if (bottom_site_xpos[-1] - top_site_xpos[-1]) > self.threshold:
            self._met = True
            return True
        return False

@register.add_condition("pour_into")
class PourIntoCondition(Condition):
    """
    Strict pour-into-container check: the source container must be grasped by the robot,
    tilted (bottom_site above top_site), and its mouth (top_site) must be positioned
    within the receiver container's XY AABB AND above the receiver's top Z plane.

    As mujoco does not support liquid simulation, this condition uses geometric checks.
    Requires ALL of the following to be satisfied:
      1. target_entity is currently grasped by the robot
      2. target_entity is tilted (bottom_site above top_site) by more than tilt_threshold
      3. top_site (mouth) is within receiver_container's XY AABB (via contain())
         AND at least z_clearance meters above the receiver's top plane

    Latches true on first success — once satisfied, subsequent is_met() calls return True
    even if the container returns to an upright posture within the same step.

    params:
        target_entity: the container to be poured (e.g., test tube)
        receiver_container: the container receiving the liquid (e.g., beaker)
        robot: the robot (used for is_grasped check)
        tilt_threshold: minimum (z_bottom - z_top) to confirm tilt, default 0
        z_clearance: minimum height of mouth above receiver top Z, default 0.01m
    """
    def __init__(self, target_entity, receiver_container, robot,
                 tilt_threshold=0, z_clearance=0.01):
        super().__init__()
        self.target_entity = target_entity
        self.receiver_container = receiver_container
        self.robot = robot
        self.tilt_threshold = tilt_threshold
        self.z_clearance = z_clearance
        self._transfer_applied = False

    def record_initial_state(self, physics=None):
        super().record_initial_state(physics)
        self._transfer_applied = False

    def _transfer_solution(self, physics):
        """
        源容器清空 + 目标容器按源当前颜色灌入。
        任一不是 SolutionMixin 则静默 return。
        """
        target = self.receiver_container
        if not hasattr(target, "fill_solution"):
            print(f"[DEBUG _transfer] target {getattr(target, 'name', '?')} has no fill_solution, returning")
            return
        source = self.target_entity
        src_rgba = getattr(source, "_current_solution_rgba", None)
        print(f"[DEBUG _transfer] source={getattr(source, 'name', '?')} _current_solution_rgba={src_rgba}")
        print(f"[DEBUG _transfer] target={getattr(target, 'name', '?')}")
        target.fill_solution(physics, source_solution_rgba=src_rgba)
        print(f"[DEBUG _transfer] after fill: target._current_solution_rgba={getattr(target, '_current_solution_rgba', None)}")
        if hasattr(source, "clear_solution"):
            source.clear_solution(physics)
            print(f"[DEBUG _transfer] source cleared, now _current_solution_rgba={getattr(source, '_current_solution_rgba', None)}")

    def is_met(self, physics):
        if self._met:
            return True

        # Gate 1: object must be grasped by the robot
        if not self.target_entity.is_grasped(physics, self.robot):
            return False

        # Gate 2: object must be tilted (bottom higher than top)
        top_site = self.target_entity.mjcf_model.worldbody.find("site", "top_site")
        bottom_site = self.target_entity.mjcf_model.worldbody.find("site", "bottom_site")
        top_xpos = physics.bind(top_site).xpos
        bottom_xpos = physics.bind(bottom_site).xpos
        if (bottom_xpos[-1] - top_xpos[-1]) <= self.tilt_threshold:
            return False

        # Gate 3: mouth must be within receiver container's XY AABB and above it
        if not hasattr(self.receiver_container, 'contain'):
            # Fallback: simple XY Euclidean distance if receiver has no contain()
            receiver_xpos = physics.bind(
                self.receiver_container.mjcf_model.worldbody).xpos
            xy_dist = np.linalg.norm(top_xpos[:2] - receiver_xpos[:2])
            if xy_dist > 0.1:  # 10cm fallback tolerance
                return False
        else:
            # Use contain() for precise AABB check
            keysites = self.receiver_container.key_sites(physics)
            if not keysites:
                # Fallback: simple XY Euclidean distance if no key_sites
                receiver_xpos = physics.bind(
                    self.receiver_container.mjcf_model.worldbody).xpos
                xy_dist = np.linalg.norm(top_xpos[:2] - receiver_xpos[:2])
                if xy_dist > 0.1:
                    return False
            else:
                keypoints = np.array([physics.bind(kp).xpos for kp in keysites])
                max_z = keypoints[:, 2].max()

                # Step 3a: XY must be inside AABB (temporarily set Z to max_z)
                point_to_check = top_xpos.copy()
                point_to_check[2] = max_z
                if not self.receiver_container.contain(point_to_check, physics):
                    return False

                # Step 3b: Z must be high enough above the receiver top
                if top_xpos[2] < max_z + self.z_clearance:
                    return False

        # All gates passed — trigger solution transfer, then latch
        if not self._transfer_applied:
            self._transfer_solution(physics)
            self._transfer_applied = True
        self._met = True
        return True

@register.add_condition("on_position")
class OnPositionCondition(Condition):
    """
    Entities should close to the target position within a certain distance
    params:
        entities: list, the target entities to be close to the target positions
        positions: list, the target positions
        tolerance_distance: the acceptable distance between the entities and the target positions
        dimension: the dimension of the target positions, 2 or 3
    """
    def __init__(self, entities, positions, tolerance_distance=0.03, dimension=2):
        self.entities = entities
        self.target_positions = positions
        self.tolerance_distance = tolerance_distance
        self.dimension = dimension
    
    def is_met(self, physics=None):
        for entity, target_pos in zip(self.entities, self.target_positions):
            entity_xpos = physics.bind(entity.mjcf_model.worldbody).xpos
            if distance(entity_xpos[:self.dimension], target_pos[:self.dimension]) > self.tolerance_distance:
                return False
        return True

@register.add_condition("contact")
class ContactCondition(Condition):
    """
    Check if the target entities are in contact with each other
    params:
        entity1: the first entity, Entity class of list of Entity
        entity2: the second entity, Entity class of list of Entity
    """
    def __init__(self, entity1, entity2):
        self.entity1 = entity1
        self.entity2 = entity2
        self.entity_geoms_id = None
        
    def is_met(self, physics=None):
        contacts = physics.data.contact
        if self.entity_geoms_id is None:
            self.entity_geoms_id = dict(
                entity1=[],
                entity2=[]
            )
            if isinstance(self.entity1, Entity):                
                self.entity_geoms_id["entity1"].extend([physics.bind(geom).element_id for geom in self.entity1.geoms])
            elif isinstance(self.entity1, list):    
                self.entity_geoms_id["entity1"].extend([physics.bind(geom).element_id for geom in self.entity1])
                
            if isinstance(self.entity2, Entity):                
                self.entity_geoms_id["entity2"].extend([physics.bind(geom).element_id for geom in self.entity2.geoms])
            elif isinstance(self.entity2, list):    
                self.entity_geoms_id["entity2"].extend([physics.bind(geom).element_id for geom in self.entity2])
            
        for contact in contacts:
            if (contact.geom1 in self.entity_geoms_id["entity1"] and contact.geom2 in self.entity_geoms_id["entity2"]) or \
                (contact.geom2 in self.entity_geoms_id["entity2"] and contact.geom1 in self.entity_geoms_id["entity1"]):
                return True
        return False
    
@register.add_condition("joint_in_range")
class JointInRangeCondition(Condition):
    """
    The target joint should be in specific range.
    Params:
        entities: the target entities, list of Entity with a single joint (hinge or slide)
        target_pos_range: the target position range
    """
    def __init__(self, entities, target_pos_range):
        self.entities = entities
        self.target_pos_range = target_pos_range
    
    def is_met(self, physics=None):
        for entity in self.entities:
            joints = entity.joints
            assert len(joints) == 1, "The number of joints should be equal to the target position range"
            if physics.bind(joints[-1]).qpos < self.target_pos_range[0] or physics.bind(joints[-1]).qpos > self.target_pos_range[1]:
                return False
        return True

@register.add_condition("lift")
class LiftCondition(Condition):
    """
    The entity should be lifted above the target height.

    支持两种模式：
    1. 绝对高度: target_height 指定固定高度（如 0.9m）
    2. 相对高度: lift_height 指定相对于初始位置的高度增量（通过 record_initial_state 记录前态）

    params:
        entities: the target entities to be lifted
        target_height: the target height to achieve (absolute height in meters)
        lift_height: the height to lift relative to initial position (in meters)
    """
    def __init__(self, entities, target_height=None, lift_height=None):
        super().__init__()
        self.entities = entities
        self.target_height = target_height
        self.lift_height = lift_height
        self._initial_z = {}  # entity_name -> initial z height
        self._tolerance = 0.05  # 5cm tolerance

    def record_initial_state(self, physics=None):
        """技能执行前记录物体高度"""
        super().record_initial_state(physics)
        for entity in self.entities:
            name = entity.name if hasattr(entity, 'name') else str(id(entity))
            xpos = physics.bind(entity.mjcf_model.worldbody).xpos
            self._initial_z[name] = xpos[-1]

    def is_met(self, physics=None):
        for entity in self.entities:
            entity_xpos = physics.bind(entity.mjcf_model.worldbody).xpos
            name = entity.name if hasattr(entity, 'name') else str(id(entity))

            if self.lift_height is not None:
                if not self._initial_z:
                    return False
                initial_z = self._initial_z.get(name, entity_xpos[-1])
                target_z = initial_z + self.lift_height - self._tolerance
                if entity_xpos[-1] < target_z:
                    return False
            elif self.target_height is not None:
                if entity_xpos[-1] < self.target_height - self._tolerance:
                    return False
        return True

@register.add_condition("wait_for")
class WaitForCondition(Condition):
    """
    等待外部条件满足。检查机械臂抓夹未接触目标实体，表示实体在等待期间未被触碰。
    用于人机协同评测场景，如"等待人工添加液体"、"等待溶液变色"。

    成功条件：目标实体当前未被机械臂抓夹接触（即未被抓住）。

    params:
        entity: 要监控的实体
        robot: 机器人
        wait_duration: 等待时间（秒），默认 2.0（由 skill 使用）
        change_type: "add_solution" | "solution_change_color" | None
        solution: 溶液名称（如 "CuSO4"）
        color: RGBA 颜色列表（如 [1, 0, 0, 0.4]）
    """
    def __init__(self, entity, robot=None, wait_duration=2.0, change_type=None,
                 solution=None, color=None):
        super().__init__()
        self.entity = entity
        self.robot = robot
        self.wait_duration = wait_duration
        self.change_type = change_type
        self.solution = solution
        self.color = color
        self._change_applied = False

    def record_initial_state(self, physics=None):
        super().record_initial_state(physics)
        self._change_applied = False

    def is_met(self, physics=None):
        if self._change_applied:
            return True

        # 检查机械臂是否未接触目标实体
        if self.robot is not None and self.entity is not None:
            if hasattr(self.entity, 'is_grasped') and self.entity.is_grasped(physics, self.robot):
                return False

        # 实体未被抓住，条件满足，应用环境变化
        self._apply_change(physics)
        self._change_applied = True
        return True

    def _apply_change(self, physics):
        if self.change_type == "add_solution" and self.solution:
            if hasattr(self.entity, 'set_solution_rgba'):
                self.entity.set_solution_rgba(physics, self.solution)
        elif self.change_type == "solution_change_color" and self.color:
            if hasattr(self.entity, 'set_solution_rgba'):
                self.entity.set_solution_rgba(physics, target_rgba=self.color)

class ConditionSet:
    """
    A set of conditions, the condition set is met only when all the conditions are satisfied simutanously.
    """
    def __init__(self, conditions):
        self.conditions = conditions

    def __len__(self):
        return len(self.conditions)

    def is_met(self, physics=None):
        conditions_are_met = [condition.is_met(physics) for condition in self.conditions]
        return all(conditions_are_met)

    def add(self, condition):
        self.conditions.append(condition)

    def met_progress(self, physics=None):
        """
        compute the progress of the condition set.
        Return the ratio of the conditions that are met and those conditions are met.
        """
        conditions_are_met = [condition.is_met(physics) for condition in self.conditions]
        met_conditions = []
        for condition, met in zip(self.conditions, conditions_are_met):
            if met: met_conditions.append(condition)
        return sum(conditions_are_met) / len(conditions_are_met), met_conditions


class SequentialConditionSet:
    """
    顺序条件集合：只有上一个条件满足以后，下一个条件才参与判定。
    一旦条件被满足就被"锁住"，后续 is_met() 调用时跳过已锁住的条件；
    下一个尚未锁住的条件失败时，is_met() 返回 False，且不锁住该条件。

    is_met() 只有在所有条件都已被锁住时才返回 True。
    顺序由 conditions 列表的顺序决定（与 condition_plan 中 step_id 顺序一致）。
    """
    def __init__(self, conditions):
        self.conditions = conditions
        self._locked = [False] * len(conditions)

    def __len__(self):
        return len(self.conditions)

    def is_sequential(self):
        return True

    def is_met(self, physics=None):
        for i, cond in enumerate(self.conditions):
            if self._locked[i]:
                continue
            if not cond.is_met(physics):
                return False
            self._locked[i] = True
        return True

    def add(self, condition):
        self.conditions.append(condition)
        self._locked.append(False)

    def reset_locks(self):
        self._locked = [False] * len(self.conditions)

    def per_condition_met(self, physics=None):
        """逐个强制重新检查每个条件，返回每个条件当前的 met 状态（不修改 lock）。"""
        return [cond.is_met(physics) for cond in self.conditions]

    def met_progress(self, physics=None):
        n_locked = sum(self._locked)
        for i, cond in enumerate(self.conditions):
            if self._locked[i]:
                continue
            if not cond.is_met(physics):
                return n_locked / len(self.conditions), []
            self._locked[i] = True
            n_locked += 1
        return 1.0, list(self.conditions)


@register.add_condition("asyn_sequence")
class AsynSequenceCondition(Condition):
    """
    A time sequence of conditions, the condition is met only when all the sub-conditions are satisfied asynchronously.
    Different with single condition set, the sub-conditions' met are maintained in a member value.
    params:
        condition_sets: a list of ConditionSet, each ConditionSet contains a list of conditions.   
    
    """
    def __init__(self, condition_sets):
        assert isinstance(condition_sets, list) and isinstance(condition_sets[0], ConditionSet), "condition_set should be a list of condition_set"
        self.condition_has_been_met = [False for _ in range(len(condition_sets))]
        self.condition_sets = condition_sets
        
    def is_met(self, physics=None):
        for i, condition in enumerate(self.condition_sets):
            if not self.condition_has_been_met[i]:
                if condition.is_met(physics):
                    self.condition_has_been_met[i] = True
        if all(self.condition_has_been_met): return True
        else: return False
    
    def met_progress(self, physics=None):
        met_scores = []
        # subcontion_num = 0
        for condition_set in self.condition_sets:
            progress_score, _ = condition_set.met_progress(physics)
            met_scores.append(progress_score)
        return np.mean(met_scores), None

@register.add_condition("or")
class OrCondition(Condition):
    """
    Any one of the conditions in the condition set is met.
    params:
        condition_sets: a list of ConditionSet, each ConditionSet contains a list of conditions.
    """
    def __init__(self, condition_sets):
        assert isinstance(condition_sets, list) and isinstance(condition_sets[0], ConditionSet), "condition_sets should be a list of condition sets"
        self.condition_sets = condition_sets

    def is_met(self, physics=None):
        return any([condition_set.is_met(physics) for condition_set in self.condition_sets])

@register.add_condition("heated")
class HeatedCondition(Condition):
    """
    通用加热条件：判断目标物体是否被加热源充分加热。

    设计思路：
    - 空间判定：目标物体必须在加热源正上方（Z 高于加热源 且 XY 在加热源范围内）
    - 时间判定：采用"累积计时"而非"连续计时"，即目标物体移开后不清零已累积的时间，
      只是停止计时。这样做是为了对机械臂控制抖动更鲁棒——轻微晃动不会导致进度丢失。
    - 通用性：heat_source 参数不绑定特定加热工具，本生灯、酒精灯、电炉等均可。

    判定流程（每个 control step 调用一次 is_met）：
    1. 计算当前 step 距上次调用的时间差 dt（通过 physics.data.time）
    2. 检查目标物体是否在加热源正上方：
       a. Z 轴：目标物体 Z > 加热源 Z
       b. XY 轴：优先使用 heat_source.contain() 判断；若不支持则 fallback 到 XY 欧氏距离 < xy_tolerance
    3. 若在上方，accumulated_time += dt
    4. accumulated_time >= duration 时条件达成

    参数：
        target_entity: 被加热的物体（如试管）
        heat_source:   加热工具（如本生灯），任何 Entity 均可
        duration:      需要累积的加热秒数，默认 5.0 秒
        xy_tolerance:  XY 距离阈值（仅在 heat_source 无 contain 方法时使用），默认 0.08m
    """
    def __init__(self, target_entity, heat_source, duration=5.0, xy_tolerance=0.08):
        self.target_entity = target_entity
        self.heat_source = heat_source
        self.duration = duration
        self.xy_tolerance = xy_tolerance
        self.accumulated_time = 0.0
        self.last_time = None

    def _is_above_heat_source(self, physics):
        """
        判断目标物体是否在加热源正上方。
        两步检查：(1) Z 高于加热源顶部  (2) XY 在加热源范围内

        注意：对于 subentity（如试管架上的试管），worldbody.xpos 在被抓起后可能不更新。
        因此优先使用 geom 位置的平均值作为实际位置。
        """
        # 获取目标物体的实际位置（优先用 geom 均值，对 subentity 更可靠）
        target_geoms = self.target_entity.mjcf_model.find_all('geom')
        if target_geoms:
            target_xpos = np.mean([physics.bind(g).xpos for g in target_geoms], axis=0)
        else:
            target_xpos = physics.bind(self.target_entity.mjcf_model.worldbody).xpos

        # 获取加热源顶部 z（用所有 geom 的最高点）
        source_xpos = physics.bind(self.heat_source.mjcf_model.worldbody).xpos
        source_geoms = self.heat_source.mjcf_model.find_all('geom')
        if source_geoms:
            source_top_z = max(physics.bind(g).xpos[2] for g in source_geoms)
        else:
            source_top_z = source_xpos[2]

        # 步骤1：Z 轴检查 — 目标必须高于加热源顶部
        if target_xpos[2] <= source_top_z:
            return False

        # 步骤2：XY 范围检查
        if hasattr(self.heat_source, 'contain') and callable(self.heat_source.contain):
            point_to_check = target_xpos.copy()
            point_to_check[2] = source_xpos[2] + 0.01
            return self.heat_source.contain(point_to_check, physics)
        else:
            # fallback：简单的 XY 欧氏距离判定
            xy_dist = np.sqrt((target_xpos[0] - source_xpos[0])**2 +
                              (target_xpos[1] - source_xpos[1])**2)
            return xy_dist < self.xy_tolerance

    def is_met(self, physics):
        # 通过仿真时间计算 dt（每次 is_met 调用间隔）
        current_time = physics.data.time
        dt = (current_time - self.last_time) if self.last_time is not None else 0.0
        self.last_time = current_time

        # 空间判定 + 时间累积
        if self._is_above_heat_source(physics):
            self.accumulated_time += dt

        return self.accumulated_time >= self.duration

    def met_progress(self, physics=None):
        """返回加热进度 (0.0 ~ 1.0)，方便评估和 debug"""
        progress = min(self.accumulated_time / self.duration, 1.0) if self.duration > 0 else 1.0
        return progress, []

@register.add_condition("on_orientation")
class OnOrientationCondition(Condition):
    """
    Check if the entity's orientation matches the target orientation within tolerance.
    params:
        entities: list of entity names
        orientations: list of target orientations in Euler angles (roll, pitch, yaw) in radians
        tolerance_angle: maximum allowed angle difference in radians (default: pi/6)
        check_axes: list of axes to check (default: [0, 1, 2] meaning roll, pitch, yaw)
    """
    def __init__(self, entities, orientations, tolerance_angle=np.pi/6, check_axes=None):
        self.entities = entities
        self.orientations = np.array(orientations)
        self.tolerance_angle = tolerance_angle
        self.check_axes = check_axes if check_axes is not None else [0, 1, 2]  # Default: check all axes

    def is_met(self, physics=None):
        for i, entity in enumerate(self.entities):
            entity_quat = physics.bind(entity.mjcf_model.worldbody).xquat
            # Convert quaternion to Euler angles
            from VLABench.utils.utils import quaternion_to_euler
            current_euler = quaternion_to_euler(entity_quat)

            # Get possible target orientations (support multiple targets)
            if len(self.orientations.shape) == 1:
                # Single target orientation
                target_orientations = [self.orientations]
            else:
                # Multiple target orientations (any one can match)
                target_orientations = self.orientations

            # Check if current orientation matches any target
            orientation_matched = False
            for target_euler in target_orientations:
                # Calculate angular difference for each axis
                angle_diff = np.abs(current_euler - target_euler)
                # Handle angle wrapping (e.g., 359° vs 1° should be close)
                angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)

                # Only check specified axes
                angle_diff_to_check = angle_diff[self.check_axes]

                # Check if all specified angles are within tolerance
                if np.all(angle_diff_to_check < self.tolerance_angle):
                    orientation_matched = True
                    break

            if not orientation_matched:
                return False
        return True

@register.add_condition("shake")
class ShakeCondition(Condition):
    """
    摇晃成功判定：物体被握住后，偏角方向反复切换达到一定次数。

    判定流程（每个 simulation step 调用一次 is_met）：
    1. 检查物体是否被握住，未被握住则直接失败
    2. 获取当前 Euler 角，减去初始 Euler 角得到相对偏角
    3. 判断偏角方向（正/负），如果方向翻转且偏角超过阈值则计数 +1
    4. 方向变化次数 >= min_direction_changes 时条件达成

    params:
        entities: 要检查的物体列表
        robot: 机器人对象（用于检查抓取状态）
        min_direction_changes: 最小方向变化次数，默认 3
        min_angle_threshold: 有效偏角阈值（弧度），默认 0.1
        check_axis: 检查的旋转轴索引 (0=X, 1=Y, 2=Z)，默认 1 (Y轴)
    """
    def __init__(self, entities, robot, min_direction_changes=3,
                 min_angle_threshold=0.1, check_axis=1):
        super().__init__()
        self.entities = entities
        self.robot = robot
        self.min_direction_changes = min_direction_changes
        self.min_angle_threshold = min_angle_threshold
        self.check_axis = check_axis

        self._initial_euler = {}
        self._last_direction = {}
        self._direction_changes = {}

    def record_initial_state(self, physics):
        super().record_initial_state(physics)
        self._initial_euler = {}
        self._last_direction = {}
        self._direction_changes = {}

        for entity in self.entities:
            name = entity.name if hasattr(entity, 'name') else str(id(entity))
            quat = self._get_entity_quat(entity, physics)
            euler = quaternion_to_euler(quat)
            self._initial_euler[name] = euler.copy()
            self._last_direction[name] = 0
            self._direction_changes[name] = 0

    @staticmethod
    def _get_entity_quat(entity, physics):
        """获取实体的世界坐标系四元数。subentity 的 worldbody 姿态在抓取后可能不更新，优先用 geom。"""
        geoms = entity.mjcf_model.find_all('geom')
        if geoms:
            return matrix_to_quaternion(physics.bind(geoms[0]).xmat)
        return physics.bind(entity.mjcf_model.worldbody).xquat.copy()

    def is_met(self, physics):
        for entity in self.entities:
            name = entity.name if hasattr(entity, 'name') else str(id(entity))

            if not entity.is_grasped(physics, self.robot):
                return False

            quat = self._get_entity_quat(entity, physics)
            current_euler = quaternion_to_euler(quat)
            initial_euler = self._initial_euler.get(name, current_euler)
            relative_angle = current_euler[self.check_axis] - initial_euler[self.check_axis]

            # 归一化到 [-pi, pi]
            relative_angle = (relative_angle + np.pi) % (2 * np.pi) - np.pi

            if abs(relative_angle) < self.min_angle_threshold:
                continue

            current_direction = 1 if relative_angle > 0 else -1
            last_dir = self._last_direction.get(name, 0)

            if last_dir != 0 and current_direction != last_dir:
                self._direction_changes[name] = self._direction_changes.get(name, 0) + 1

            self._last_direction[name] = current_direction

        total_changes = sum(self._direction_changes.values())
        met = total_changes >= self.min_direction_changes
        if met:
            self._met = True
        return met

    def met_progress(self, physics):
        total_changes = sum(self._direction_changes.values())
        progress = min(total_changes / self.min_direction_changes, 1.0) if self.min_direction_changes > 0 else 1.0
        return progress, []

@register.add_condition("stir")
class StirCondition(Condition):
    """
    搅拌成功判定：搅拌工具插入容器后，累积 XY 方向运动距离达标且未与容器壁发生硬碰撞。

    判定流程（每个 simulation step 调用一次 is_met）：
    1. 检查工具是否被握住，未被握住则直接失败
    2. 获取工具 tip 位置（优先 bottom_site，退化为 geom 均值或 worldbody xpos）
    3. 插入验证：tip 必须在容器 AABB 内（通过 container.contain()），否则失败
    4. 累积 XY 方向路径长度（只统计 X、Y 平面位移，忽略 Z 轴垂直下降/上升）
    5. 扫描 contacts，工具与容器壁接触时记录软警告（_collision_warnings[name] = True）
       但不阻止成功判定，最终评分时可查阅
    6. 累积距离 >= min_distance 时锁存成功

    params:
        entities: 要检查的搅拌工具实体列表（通常为玻璃棒）
        container: 目标容器（如烧杯），用于插入验证和碰撞检测
        robot: 机器人对象（用于检查抓取状态）
        min_distance: 累积 XY 运动距离阈值（m），默认 0.15
        tool_tip_site: 工具上用于读取位置的 site 名称，默认 "bottom_site"
    """
    def __init__(self, entities, container, robot,
                 min_distance=0.15, tool_tip_site="bottom_site"):
        super().__init__()
        self.entities = entities
        self.container = container
        self.robot = robot
        self.min_distance = min_distance
        self.tool_tip_site = tool_tip_site

        # 每个工具实体的累加器
        self._last_tip_pos = {}
        self._cumulative_distance = {}
        self._collision_warnings = {}  # 软警告：记录发生过碰撞

        # 碰撞 geom id 缓存（首次 is_met 时构建）
        self._tool_geom_ids = None
        self._container_geom_ids = None

    def record_initial_state(self, physics):
        print(f"[DEBUG StirCondition.record_initial_state] Called, setting _initial_state_recorded=True")
        super().record_initial_state(physics)
        self._last_tip_pos = {}
        self._cumulative_distance = {}
        self._collision_warnings = {}
        # 重置碰撞缓存，reset 后 geom 可能重新绑定
        self._tool_geom_ids = None
        self._container_geom_ids = None
        for entity in self.entities:
            name = entity.name if hasattr(entity, 'name') else str(id(entity))
            tip = self._get_tool_tip_pos(entity, physics)
            self._last_tip_pos[name] = tip.copy()
            self._cumulative_distance[name] = 0.0
            self._collision_warnings[name] = False

    def _get_tool_tip_pos(self, entity, physics):
        """获取工具 tip 的世界坐标。优先读取 named site，退化为 geom 均值或 worldbody。"""
        site = entity.mjcf_model.find("site", self.tool_tip_site) if self.tool_tip_site else None
        if site is not None:
            return np.array(physics.bind(site).xpos).copy()
        geoms = entity.mjcf_model.find_all('geom')
        if geoms:
            return np.mean([physics.bind(g).xpos for g in geoms], axis=0).copy()
        return np.array(physics.bind(entity.mjcf_model.worldbody).xpos).copy()

    def is_met(self, physics):
        if not self._initial_state_recorded:
            print(f"[DEBUG StirCondition.is_met] _initial_state_recorded=False, returning False")
            return False
        try:
            result = self._is_met_impl(physics)
            print(f"[DEBUG StirCondition.is_met] _is_met_impl returned {result}")
            return result
        except RecursionError:
            # geom id 缓存可能指向旧模型，清理后下个 step 重建
            print(f"[DEBUG StirCondition.is_met] RecursionError caught, resetting caches")
            self._tool_geom_ids = None
            self._container_geom_ids = None
            return False

    def _is_met_impl(self, physics):
        # 1. 懒构建 geom id 集合（ContactCondition 模式）
        if self._tool_geom_ids is None:
            self._tool_geom_ids = set()
            for e in self.entities:
                self._tool_geom_ids.update(
                    physics.bind(g).element_id for g in e.geoms)
            self._container_geom_ids = set(
                physics.bind(g).element_id for g in self.container.geoms)

        # 2. 扫描 contacts，记录工具与容器壁的软警告
        for c in physics.data.contact:
            if (c.geom1 in self._tool_geom_ids and c.geom2 in self._container_geom_ids) or \
               (c.geom2 in self._tool_geom_ids and c.geom1 in self._container_geom_ids):
                for entity in self.entities:
                    name = entity.name if hasattr(entity, 'name') else str(id(entity))
                    self._collision_warnings[name] = True

        # 3. 逐个工具更新距离并判断
        all_passed = True
        for entity in self.entities:
            name = entity.name if hasattr(entity, 'name') else str(id(entity))
            if not entity.is_grasped(physics, self.robot):
                return False

            tip = self._get_tool_tip_pos(entity, physics)

            # 插入验证：tip 必须在容器 AABB 内
            if not self.container.contain(tip, physics):
                return False

            # 累积 XY 方向路径长度（忽略 Z 轴垂直位移）
            last = self._last_tip_pos.get(name, tip)
            xy_distance = np.linalg.norm(tip[:2] - last[:2])
            self._cumulative_distance[name] = self._cumulative_distance.get(name, 0.0) \
                                              + float(xy_distance)
            self._last_tip_pos[name] = tip

            if self._cumulative_distance[name] < self.min_distance:
                all_passed = False

        if all_passed:
            self._met = True
        return self._met

    def met_progress(self, physics):
        total = sum(self._cumulative_distance.values())
        progress = min(total / self.min_distance, 1.0) if self.min_distance > 0 else 1.0
        return progress, []

@register.add_condition("cap_open")
class CapOpenCondition(Condition):
    """
    Check if the cap of a ContainerWithCap is unscrewed open.
    Uses slide joint RELATIVE displacement from initial state (recorded via
    record_initial_state), not absolute qpos, to avoid false positives from
    static-balance physics deformation of the slide joint at MJCF compile time.

    Rationale: the cap is held down by physical contact with the body — even after
    the user unscrews and lifts it, MuJoCo's contact forces pull the slide joint
    back to its static-balance qpos (~0.011m) and keep the collision meshes in
    penetration. So a contact-based "no overlap" check never fires reliably in
    the current physics setup. Using slide joint relative displacement captures
    the *intent* of the user (cap is being lifted) without depending on the
    physically-impossible complete separation.

    params:
        entities: list of ContainerWithCap entities
        open_threshold: minimum hinge rotation DELTA from initial (radians) to consider cap open
        lift_threshold: minimum slide displacement DELTA from initial (meters) to consider cap open
    """
    def __init__(self, entities, open_threshold=0.5*np.pi, lift_threshold=0.002):
        super().__init__()
        self.entities = entities
        self.open_threshold = open_threshold
        self.lift_threshold = lift_threshold
        self._initial_joint_pos = {}
        self._initial_slide_pos = {}

    def record_initial_state(self, physics):
        super().record_initial_state(physics)
        for entity in self.entities:
            name = entity.mjcf_model.model
            if entity.cap_joint is not None:
                qpos = physics.bind(entity.cap_joint).qpos
                self._initial_joint_pos[name] = float(qpos.item() if hasattr(qpos, 'item') else qpos)
            if entity.slide_joint is not None:
                qpos = physics.bind(entity.slide_joint).qpos
                self._initial_slide_pos[name] = float(qpos.item() if hasattr(qpos, 'item') else qpos)

    def is_met(self, physics=None):
        if not self._initial_state_recorded:
            return False
        for entity in self.entities:
            name = entity.mjcf_model.model
            # 条件 1: door (hinge) 累计旋转量 >= open_threshold (默认 0.5π)
            if entity.cap_joint is not None and name in self._initial_joint_pos:
                qpos = physics.bind(entity.cap_joint).qpos
                current_door = float(qpos.item() if hasattr(qpos, 'item') else qpos)
                if abs(current_door - self._initial_joint_pos[name]) > self.open_threshold:
                    return True
            # 条件 2: slide 相对位移 > lift_threshold (向后兼容)
            if entity.slide_joint is not None and name in self._initial_slide_pos:
                qpos = physics.bind(entity.slide_joint).qpos
                current_slide = float(qpos.item() if hasattr(qpos, 'item') else qpos)
                if current_slide - self._initial_slide_pos[name] > self.lift_threshold:
                    return True
        return False


@register.add_condition("drawer_open")
class DrawerOpenCondition(Condition):
    """
    抽屉打开判定：针对 ContainerWithDrawer 类型的实体（如 cabinet）。

    记录初始状态时遍历所有抽屉关节（slide joint）的 qpos，
    若任意抽屉关节相对初始位置移动超过 open_threshold（默认 0.05m），
    视为该抽屉已被拉开。

    params:
        entities: Cabinet 实体列表（每个 entity 应有 self.joints 与 self.drawers）
        open_threshold: 抽屉相对初始位置的最大位移（m），默认 0.05
    """
    def __init__(self, entities, open_threshold=0.05):
        super().__init__()
        self.entities = entities
        self.open_threshold = open_threshold
        self._initial_drawer_qpos = {}

    def record_initial_state(self, physics):
        super().record_initial_state(physics)
        for entity in self.entities:
            name = entity.mjcf_model.model
            self._initial_drawer_qpos[name] = [
                float(physics.bind(joint).qpos[0]) for joint in entity.joints
            ]

    def is_met(self, physics=None):
        if not self._initial_state_recorded:
            return False
        for entity in self.entities:
            name = entity.mjcf_model.model
            if name not in self._initial_drawer_qpos:
                continue
            initial_qpos = self._initial_drawer_qpos[name]
            for joint, init_q in zip(entity.joints, initial_qpos):
                current = float(physics.bind(joint).qpos[0])
                if abs(current - init_q) > self.open_threshold:
                    return True
        return False

@register.add_condition("door_open")
class DoorOpenCondition(Condition):
    """
    门打开判定：针对 ContainerWithDoor 类型的实体（如 drying_box）。

    记录初始状态时门关节（hinge joint）的 qpos，
    若门关节相对初始位置移动超过 open_threshold（默认 0.1 弧度），
    视为门已被打开。

    params:
        entities: ContainerWithDoor 实体列表
        open_threshold: 门相对初始位置的最大旋转角度（弧度），默认 0.1
    """
    def __init__(self, entities, open_threshold=0.1):
        super().__init__()
        self.entities = entities
        self.open_threshold = open_threshold
        self._initial_door_qpos = {}

    def record_initial_state(self, physics):
        super().record_initial_state(physics)
        for entity in self.entities:
            if hasattr(entity, 'door_joint') and entity.door_joint is not None:
                name = entity.mjcf_model.model
                self._initial_door_qpos[name] = float(physics.bind(entity.door_joint).qpos[0])

    def is_met(self, physics=None):
        if not self._initial_state_recorded:
            return False
        for entity in self.entities:
            if not (hasattr(entity, 'door_joint') and entity.door_joint is not None):
                continue
            name = entity.mjcf_model.model
            if name not in self._initial_door_qpos:
                continue
            initial_qpos = self._initial_door_qpos[name]
            current_qpos = float(physics.bind(entity.door_joint).qpos[0])
            if abs(current_qpos - initial_qpos) > self.open_threshold:
                return True
        return False

@register.add_condition("always_false")
class AlwaysFalseCondition(Condition):
    """
    永远返回false的条件，用于测试模型而不触发任务成功。
    """
    def __init__(self):
        super().__init__()

    def is_met(self, physics=None):
        return False
