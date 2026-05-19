import numpy as np
from VLABench.utils.register import register
from VLABench.utils.utils import distance
from VLABench.tasks.components.entity import Entity

class Condition:
    def __init__(self):
        pass

    def record_initial_state(self, physics=None):
        """在技能执行前调用，记录初始状态。子类可重写以支持前态-终态对比。"""
        pass

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
        
    def is_met(self, physics=None):
        contacts = physics.data.contact
        
        container_geoms_id = [physics.bind(geom).element_id for geom in self.container.geoms]
        container_geoms_xpos = [physics.bind(geom).xpos for geom in self.container.geoms]
        max_xpos_z = max([xpos[-1] for xpos in container_geoms_xpos])
        for entity in self.entities:
            is_contacted = False
            entity_geom_ids = [physics.bind(geom).element_id for geom in entity.geoms]
            entity_xpos = physics.bind(entity.mjcf_model.worldbody).xpos
            # z position detection
            if entity_xpos[-1] <= max_xpos_z:
                return False
            # on contact detection
            for contact in contacts:
                if (contact.geom1 in container_geoms_id and contact.geom2 in entity_geom_ids) or \
                    (contact.geom2 in entity_geom_ids and contact.geom1 in container_geoms_id):
                    is_contacted = True
                    break
            if is_contacted is False:
                return False
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
    """
    def __init__(self, target_entity, threshold=0):
        self.target_entity = target_entity
        self.threshold = threshold
        
    def is_met(self, physics):
        top_site = self.target_entity.mjcf_model.worldbody.find("site", "top_site")
        bottom_site = self.target_entity.mjcf_model.worldbody.find("site", "bottom_site")
        top_site_xpos, bottom_site_xpos = physics.bind(top_site).xpos, physics.bind(bottom_site).xpos
        if (bottom_site_xpos[-1] - top_site_xpos[-1]) > self.threshold:
            return True
        else:
            return False

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
        self.entities = entities
        self.target_height = target_height
        self.lift_height = lift_height
        self._initial_z = {}  # entity_name -> initial z height
        self._tolerance = 0.05  # 5cm tolerance

    def record_initial_state(self, physics=None):
        """技能执行前记录物体高度"""
        for entity in self.entities:
            name = entity.name if hasattr(entity, 'name') else str(id(entity))
            xpos = physics.bind(entity.mjcf_model.worldbody).xpos
            self._initial_z[name] = xpos[-1]

    def is_met(self, physics=None):
        for entity in self.entities:
            entity_xpos = physics.bind(entity.mjcf_model.worldbody).xpos
            name = entity.name if hasattr(entity, 'name') else str(id(entity))

            if self.lift_height is not None and self._initial_z:
                initial_z = self._initial_z.get(name, entity_xpos[-1])
                target_z = initial_z + self.lift_height - self._tolerance
                if entity_xpos[-1] < target_z:
                    return False
            elif self.target_height is not None:
                if entity_xpos[-1] < self.target_height - self._tolerance:
                    return False
        return True

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
