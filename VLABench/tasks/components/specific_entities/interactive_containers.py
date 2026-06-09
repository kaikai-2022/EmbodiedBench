"""
Register the interactive containers/recaptacles in daily life, such as electrical device
"""
import numpy as np
from VLABench.utils.register import register
from VLABench.utils.utils import rotate_point_around_axis
from VLABench.tasks.components.container import CommonContainer, ContainerWithDoor

@register.add_entity("CoffeeMachine")
class CoffeeMachine(CommonContainer):
    """
    Coffee manchine that can be interactived by the user, press the button and the coffee fluid will be shown
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._is_pressed = False
    
    @property
    def fluid_sites(self):
        fluid_sites = []
        sites = self.mjcf_model.worldbody.find_all("site")
        for site in sites:
            if hasattr(site, "name") and ("fluid" in site.name or "liquid" in site.name):
                fluid_sites.append(site)
        return fluid_sites
    
    @property
    def start_button(self):
        return self.mjcf_model.worldbody.find("geom", "start_button")
        
    def get_start_button_pos(self, physics):
        return physics.bind(self.start_button).xpos
    
    def show_fluid(self, physics):
        for fluid_site in self.fluid_sites:
            physics.bind(fluid_site).rgba = np.concatenate([physics.bind(fluid_site).rgba[:3], [1]])
    
    def hidden_fluid(self, physics):
        for fluid_site in self.fluid_sites:
            physics.bind(fluid_site).rgba = np.concatenate([physics.bind(fluid_site).rgba[:3], [0]])   
    
    def is_activate(self, physics):
        contacts = physics.data.contact
        contact_goems = [contact.geom1 for contact in contacts] + [contact.geom2 for contact in contacts]
   
        if physics.bind(self.start_button).element_id in contact_goems:
            self._is_pressed = True
            return True
        else:
            self._is_pressed = False
            return False
    
    def after_substep(self, physics, random_state):
        if self.is_activate(physics): self.show_fluid(physics)
        else: self.hidden_fluid(physics)
    
    def initialize_episode(self, physics, random_state):
        if self.is_activate(physics): self.show_fluid(physics)
        else: self.hidden_fluid(physics)
        return super().initialize_episode(physics, random_state)
    
    def is_pressed(self):
        return self._is_pressed

@register.add_entity("Juicer")
class Juicer(CommonContainer):
    """
    Juicer that can be interactived by the user, press the button and the juicer will be activated
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._is_pressed = False
    
    @property
    def start_button(self):
        return self.mjcf_model.worldbody.find("geom", "start_button")
    
    def is_activate(self, physics):
        contacts = physics.data.contact
        contact_goems = [contact.geom1 for contact in contacts] + [contact.geom2 for contact in contacts]
   
        if physics.bind(self.start_button).element_id in contact_goems:
            self._is_pressed = True
            return True
        else:
            self._is_pressed = False
            return False
    
    def initialize_episode(self, physics, random_state):
        return super().initialize_episode(physics, random_state)
    
    def is_pressed(self):
        return self._is_pressed

@register.add_entity("Microwave")
class Microwave(ContainerWithDoor):
    def _build(self, 
               name:str="microwave",
               target_force_range=(1, 5),
               **kwargs):
        self._is_activated = False
        self._min_force, self._max_force = target_force_range
        super()._build(name=name, **kwargs)
    
    @property
    def start_button(self):
        return self.mjcf_model.worldbody.find("geom", "start_button")
        
    def get_start_button_pos(self, physics):
        return physics.bind(self.start_button).xpos
    
    def is_activate(self, physics):
        contacts = physics.data.contact
        contact_goems = [contact.geom1 for contact in contacts] + [contact.geom2 for contact in contacts]
   
        if physics.bind(self.start_button).element_id in contact_goems:
            self._is_pressed = True
            return True
        else:
            self._is_pressed = False
            return False

    def is_pressed(self):
        return self._is_pressed

@register.add_entity("ContainerWithCap")
class ContainerWithCap(CommonContainer):
    """
    Container with a cap connected by hinge + slide joints.
    The cap rotates around the hinge axis while the slide joint allows
    the cap to lift (passively, driven by external contact during unscrew).

    与 ContainerWithDoor 并列：直接继承 CommonContainer。
    避免 ContainerWithDoor 中 FIXME 的 is_grasped() 占位实现污染 pick 判定。
    """
    def __init__(self,
                 open_threshold=3*np.pi/2,
                 close_threshold=np.pi/6,
                 unlock_rotation=0.5*np.pi,
                 unlock_slide_range=0.02,
                 *args,
                 **kwargs):
        self.open_threshold = open_threshold
        self.close_threshold = close_threshold
        self.unlock_rotation = unlock_rotation
        self.unlock_slide_range = unlock_slide_range
        self._slide_unlocked = False
        self._initial_door_qpos = 0.0
        super().__init__(*args, **kwargs)

    @property
    def cap_joint(self):
        for joint in self.joints:
            if "door" in joint.name:
                return joint
        return None

    @property
    def slide_joint(self):
        for joint in self.joints:
            if "slide" in joint.name:
                return joint
        return None

    def get_cap_body(self):
        if self.cap_joint is None:
            return None
        return self.cap_joint.parent

    def is_cap_open(self, physics, initial_joint_qpos=None, initial_slide_qpos=None):
        if self.cap_joint is None:
            return False
        initial_j = initial_joint_qpos if initial_joint_qpos is not None else 0.0
        # 阈值改用 unlock_rotation（默认 4π），与"旋转 2 圈才解锁"语义一致
        if abs(physics.bind(self.cap_joint).qpos - initial_j) > self.unlock_rotation:
            return True
        if self.slide_joint is not None and initial_slide_qpos is not None:
            if abs(physics.bind(self.slide_joint).qpos - initial_slide_qpos) > 0.01:
                return True
        return False

    def is_cap_closed(self, physics, initial_joint_qpos=None):
        if self.cap_joint is None:
            return True
        initial_j = initial_joint_qpos if initial_joint_qpos is not None else 0.0
        return abs(physics.bind(self.cap_joint).qpos - initial_j) < self.close_threshold

    def is_slide_unlocked(self):
        """Whether cap_slide joint has been unlocked (range > 0)."""
        return getattr(self, '_slide_unlocked', False)

    def record_initial_door_qpos(self, physics):
        """Record the initial door qpos. Call this in initialize_episode / after reset."""
        if self.cap_joint is None:
            return
        qpos = physics.bind(self.cap_joint).qpos
        self._initial_door_qpos = float(qpos.item() if hasattr(qpos, 'item') else qpos)
        self._slide_unlocked = False

    def check_and_unlock_slide(self, physics):
        """
        If door has rotated >= unlock_rotation from initial, unlock the cap_slide
        joint by setting its range to [0, unlock_slide_range] AND lowering damping
        to 2 (so external force can lift the cap after unlock). Idempotent.

        Should be called every step (e.g. in the task's after_step hook).
        Safe to call when slide_joint is None.
        """
        if getattr(self, '_slide_unlocked', False):
            return
        if self.cap_joint is None or self.slide_joint is None:
            return
        qpos = physics.bind(self.cap_joint).qpos
        current_door = float(qpos.item() if hasattr(qpos, 'item') else qpos)
        rotation_delta = abs(current_door - self._initial_door_qpos)
        print(f"[check_and_unlock_slide] door qpos: {current_door:.4f}, "
              f"initial: {self._initial_door_qpos:.4f}, "
              f"delta: {rotation_delta:.4f} / {self.unlock_rotation:.4f} "
              f"({rotation_delta/np.pi:.2f}*pi / {self.unlock_rotation/np.pi:.2f}*pi)")
        if rotation_delta >= self.unlock_rotation:
            slide_id = physics.bind(self.slide_joint).element_id
            slide_dof = int(physics.model.jnt_dofadr[slide_id])
            physics.model.jnt_range[slide_id] = np.array(
                [0.0, float(self.unlock_slide_range)]
            )
            # 降低 damping 使 cap 在解锁后能被外力提起
            # XML 中初始 damping=100000（锁住），解锁后改为 2（自由）
            physics.model.dof_damping[slide_dof] = 2.0
            self._slide_unlocked = True
            print(f"[check_and_unlock_slide] ★ SLIDE UNLOCKED! "
                  f"range -> [0, {self.unlock_slide_range}], damping -> 2.0")

    def is_cap_separated(self, physics, penetration_threshold=0.0):
        """Check if cap body geoms and bottle body geoms have no real penetration.

        For unscrewed-state detection. A penetration threshold of 0 means we only
        consider real overlap (dist < 0) as contact. Note: at rest, MuJoCo's static
        balance may place the cap in tangential contact (dist == 0) without penetration,
        which we treat as "not yet separated". The caller is responsible for
        ensuring record_initial_state has been called to avoid reset-loop false
        positives — at reset time the slide joint is at its static-balance qpos
        (~0.011m) and there are 0 penetrating contacts, but the calling Condition
        gates is_met() with _initial_state_recorded.
        """
        cap_body = self.get_cap_body()
        if cap_body is None:
            return False
        cap_geom_ids = {physics.bind(g).element_id for g in cap_body.find_all('geom')}
        all_geom_ids = {physics.bind(g).element_id for g in self.geoms}
        body_geom_ids = all_geom_ids - cap_geom_ids
        if not cap_geom_ids or not body_geom_ids:
            return False
        for c in physics.data.contact:
            if c.dist >= -penetration_threshold:
                continue
            pair = {c.geom1, c.geom2}
            if pair & cap_geom_ids and pair & body_geom_ids:
                return False
        return True

    def get_cap_slide_pos(self, physics):
        if self.slide_joint is None:
            return 0.0
        return physics.bind(self.slide_joint).qpos

    def get_grasped_keypoints(self, physics):
        """
        Only return grasp sites on the cap body (not the bottle body),
        so pick() targets the cap rather than the body.
        """
        cap_body = self.get_cap_body()
        if cap_body is None:
            return super().get_grasped_keypoints(physics)
        cap_geoms = cap_body.find_all('geom')
        cap_geom_ids = {physics.bind(g).element_id for g in cap_geoms}

        keypoints = []
        for site in self.grasp_sites(physics):
            site_pos = physics.bind(site).xpos
            # 通过 site 找到它所属的 body
            site_body = site.parent
            site_body_geom_ids = {physics.bind(g).element_id for g in site_body.find_all('geom')}
            if site_body_geom_ids & cap_geom_ids:
                keypoints.append(site_pos)
        return keypoints

    def is_grasped(self, physics, robot):
        """
        Contact-based grasp detection: gripper geoms touching any entity geoms
        (body + cap). Checking only cap geoms is too restrictive — the cap
        collision cylinder is tiny, and contacts often register on the body.
        """
        entity_geom_ids = {physics.bind(g).element_id for g in self.geoms}
        gripper_geom_ids = {physics.bind(g).element_id for g in robot.gripper_geoms}
        for c in physics.data.contact:
            if (c.geom1 in gripper_geom_ids and c.geom2 in entity_geom_ids) or \
               (c.geom2 in gripper_geom_ids and c.geom1 in entity_geom_ids):
                return True
        return False