"""
Mixin class for entities with interactive button(s).
Provides reusable button position detection and press state tracking.
"""


class EntityWithButton:
    """
    Mixin class that adds button interaction capability to any entity.

    Requirements:
        - The entity's MuJoCo XML must contain a geom named "start_button"
        - The button geom should be a collision geom for contact detection

    Usage:
        @register.add_entity("MyDevice")
        class MyDevice(CommonContainer, EntityWithButton):
            pass
    """

    @property
    def start_button(self):
        """Return the start_button geom from the MuJoCo model."""
        return self.mjcf_model.worldbody.find("geom", "start_button")

    def get_start_button_pos(self, physics):
        """Get the world-space position of the start button."""
        return physics.bind(self.start_button).xpos

    def is_activate(self, physics):
        """
        Check if the button is being pressed (contact detection).

        Returns:
            bool: True if button geom is in contact with any other geom
        """
        contacts = physics.data.contact
        contact_geoms = [c.geom1 for c in contacts] + [c.geom2 for c in contacts]

        if physics.bind(self.start_button).element_id in contact_geoms:
            self._is_pressed = True
            return True
        else:
            self._is_pressed = False
            return False

    def is_pressed(self):
        """Return the current press state."""
        return getattr(self, '_is_pressed', False)
