"""
Mixin class for entities with interactive button(s).
Provides reusable button position detection, press state tracking, and visual feedback.
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Default colors for the button visual feedback
BUTTON_COLOR_IDLE   = np.array([0.6, 0.05, 0.05, 1.0])  # red
BUTTON_COLOR_ACTIVE = np.array([0.1, 0.6,  0.1,  1.0])  # green


class EntityWithButton:
    """
    Mixin class that adds button interaction capability to any entity.

    Requirements on the entity's MuJoCo XML:
        - A geom named "start_button" (used for contact detection)
        - One material that the base class can resolve as the button's visual.
          Resolution priority:
            1. Class attribute `_button_material_name` (explicit string)
            2. `start_button` geom's own `material` reference
            3. Heuristic: first material in the entity whose rgba looks red-ish
               (R > 0.4, G < 0.2, B < 0.2) OR name contains "button"

    Behavior:
        - On press:   button material changes from red  -> green
        - On release: button material changes from green -> red
        - Color transitions occur only on state change, not every step.

    Subclass usage:

        @register.add_entity("MyDevice")
        class MyDevice(CommonContainer, EntityWithButton):
            _button_material_name = "my_button_mat"   # optional explicit name
    """

    # Subclasses can override this to specify which material changes color.
    _button_material_name = None

    @property
    def start_button(self):
        """Return the start_button geom from the MuJoCo model."""
        return self.mjcf_model.worldbody.find("geom", "start_button")

    def get_start_button_pos(self, physics):
        """
        Get the world-space **pressable surface** position of the start button.

        Uses physics.bind(mjcf_element).xpos which returns the live world-space
        position of the geom without needing the named-data scope prefix.

        For cylinder-type buttons, the surface is offset by the half-height
        in the geom's local +Z direction (the press-from-above direction).
        For box-type buttons, the surface is offset by half-height in local +Z.
        """
        btn = self.start_button
        model_name = getattr(self.mjcf_model, 'model', '') or ''
        if btn is None:
            return np.zeros(3)
        try:
            body_xpos = np.array(physics.bind(btn).xpos)
            # Get the rotation matrix of the geom's body frame
            body_xmat = physics.bind(btn).xmat.reshape(3, 3)
            # Surface offset in local +Z direction (press from above)
            # Half-height is the second element of geom size
            size = np.array(btn.size)
            half_height = size[1] if len(size) > 1 else size[0] * 0.5
            surface_offset_local = np.array([0, 0, half_height])
            surface_offset_world = body_xmat @ surface_offset_local
            surface_pos = body_xpos + surface_offset_world
            return surface_pos
        except Exception:
            return np.zeros(3)

    def _resolve_button_material(self):
        """
        Return the mjcf material element whose RGBA should be changed when
        the button is pressed. Returns None if not resolvable.
        """
        # 1. Explicit class-level override
        if self._button_material_name is not None:
            mat = self.mjcf_model.asset.find("material", self._button_material_name)
            if mat is not None:
                return mat

        # 2. Material referenced by the start_button geom itself
        btn = self.start_button
        if btn is not None and getattr(btn, "material", None) is not None:
            mat = btn.material
            if getattr(mat, "name", None):
                return mat

        # 3. Heuristic: any material whose name contains "button"
        #    or whose rgba looks red (R dominant, G/B near zero)
        for mat in self.mjcf_model.asset.find_all("material"):
            name = (getattr(mat, "name", "") or "").lower()
            if "button" in name:
                return mat
            try:
                r, g, b = float(mat.rgba[0]), float(mat.rgba[1]), float(mat.rgba[2])
            except (TypeError, ValueError, IndexError):
                continue
            if r > 0.4 and g < 0.2 and b < 0.2:
                return mat
        return None

    def is_activate(self, physics):
        """
        Check whether the button is being pressed and update visual feedback.

        - Performs contact detection on the start_button geom.
        - Sets self._is_pressed to reflect the current state.
        - When the press state changes (False->True or True->False), rewrites
          the resolved button material's RGBA in the live mj_model.
        - Returns True iff the button is currently in contact.

        Subclasses that override this method should call
        `super().is_activate(physics)` to keep color feedback working.
        """
        btn = self.start_button
        if btn is None:
            print("[is_activate] start_button geom is None!")
            return False
        try:
            # Use full path (body_name/geom_name) since MuJoCo compiles with namespace
            body_name = self.mjcf_model.model
            btn_id = physics.model.name2id(f"{body_name}/start_button", "geom")
        except Exception as e:
            print(f"[is_activate] name2id FAILED: {e}")
            return False
        contacts = physics.data.contact
        contact_geoms = [c.geom1 for c in contacts] + [c.geom2 for c in contacts]
        currently_pressed = btn_id in contact_geoms

        prev_pressed = getattr(self, "_is_pressed", False)
        if currently_pressed and not prev_pressed:
            print(f"[is_activate] STATE CHANGE False->True! btn_id={btn_id}, calling _set_button_color(active=True)")
            self._set_button_color(physics, active=True)
        elif not currently_pressed and prev_pressed:
            print(f"[is_activate] STATE CHANGE True->False! calling _set_button_color(active=False)")
            self._set_button_color(physics, active=False)

        self._is_pressed = currently_pressed
        if currently_pressed:
            print(f"[is_activate] PRESSED! (call returned True)")
        return currently_pressed

    def _set_button_color(self, physics, active: bool):
        """Rewrite the button material rgba in the live mj_model."""
        mat = self._resolve_button_material()
        if mat is None:
            print(f"[_set_button_color] FAILED: material not resolved! active={active}")
            return
        try:
            target = BUTTON_COLOR_ACTIVE if active else BUTTON_COLOR_IDLE
            body_name = self.mjcf_model.model
            print(f"[_set_button_color] mat.name={mat.name}, target={target}")

            # Approach: find the material ID in the raw MjModel by name, then write directly.
            # dm_control's physics.named.model.mat_rgba and physics.bind(mat).rgba both
            # write to views that may not propagate to the renderer. Writing directly to
            # the underlying MjModel.mat_rgba[mat_id] is the most reliable approach.
            import mujoco
            raw_m = physics.model._model
            target_full_name = f"{body_name}/{mat.name}"
            mat_id = None
            for i in range(raw_m.nmat):
                addr = raw_m.name_matadr[i]
                name_bytes = raw_m.names[addr:].split(b'\x00')[0]
                name = name_bytes.decode('utf-8', errors='replace')
                if name == target_full_name or name == mat.name:
                    mat_id = i
                    break

            if mat_id is None:
                print(f"[_set_button_color] FAILED: material '{target_full_name}' not found in raw MjModel")
                return

            print(f"[_set_button_color] before: mat_id={mat_id}, rgba={raw_m.mat_rgba[mat_id]}")
            raw_m.mat_rgba[mat_id] = target
            print(f"[_set_button_color] after:  mat_id={mat_id}, rgba={raw_m.mat_rgba[mat_id]}")
            print(f"[_set_button_color] SUCCESS via raw MjModel write")
        except (KeyError, AttributeError) as e:
            print(f"[_set_button_color] FAILED: {type(e).__name__}: {e}")
            pass

    def is_pressed(self):
        """Return the current press state."""
        return getattr(self, '_is_pressed', False)
