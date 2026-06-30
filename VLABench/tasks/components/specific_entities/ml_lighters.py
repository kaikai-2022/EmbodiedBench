"""
Alcohol lamp entity with flame lighting/extinguishing support.
Provides a unified `set_flame_state` method to control flame visibility
by modifying the rgba alpha channel of the flame geoms.
"""
from VLABench.tasks.components.entity import CommonGraspedEntity
from VLABench.utils.register import register


@register.add_entity("AlcoholLamp")
class AlcoholLamp(CommonGraspedEntity):
    """
    Alcohol lamp entity with flame visibility control.

    Default state (XML): flame geoms have alpha=0, so flame is invisible.
    When `set_flame_state(lit=True)` is called, the flame geoms' rgba alpha
    is restored to their original values, making the flame visible.

    Inherits from CommonGraspedEntity to support pick/lift/grasp operations.
    """

    # Hardcoded original rgba values (from alcohol_lamp.xml)
    # These are used to restore flame color when lit=True
    LIT_RGBA_OUTER = [1.0, 0.45, 0.05, 0.48]
    LIT_RGBA_INNER = [1.0, 0.95, 0.8, 0.85]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flame_lit = False

    def set_flame_state(self, physics, lit=True):
        """
        Control flame visibility by modifying geom rgba alpha.

        Args:
            physics: MuJoCo physics instance
            lit: True to show flame (restore original rgba),
                 False to hide flame (alpha=0)
        """
        self._flame_lit = lit

        flame_configs = [
            ("flame_outer", self.LIT_RGBA_OUTER),
            ("flame_inner", self.LIT_RGBA_INNER),
        ]

        for geom_name, lit_rgba in flame_configs:
            geom = self.mjcf_model.worldbody.find("geom", geom_name)
            if geom is None:
                continue

            if lit:
                # Restore original rgba when lighting
                physics.bind(geom).rgba = list(lit_rgba)
            else:
                # Keep RGB, set alpha to 0 when extinguishing
                rgba = list(lit_rgba)
                rgba[3] = 0.0
                physics.bind(geom).rgba = rgba

    def initialize_episode(self, physics, random_state):
        """Ensure flame is off at episode start."""
        self.set_flame_state(physics, lit=False)
        return super().initialize_episode(physics, random_state)