"""
Pipette and dropper entities that can hold and dispense chemistry solutions.
Used with SkillLib.aspirate / SkillLib.dispense for fine-grained liquid handling.
"""
import numpy as np
from VLABench.tasks.components.entity import CommonGraspedEntity
from VLABench.utils.register import register


@register.add_entity("Pipette")
class Pipette(CommonGraspedEntity):
    """
    Mechanical pipette entity with aspirate site at the tip.

    Can store a held solution's name and rgba after successful aspiration,
    and pass it to the target container during dispense.

    Expected XML layout:
      - <site name="aspirate_site" .../> at the tip (group=3, used for contact detection)
      - <site class="grasppoint" .../> at the bulb end (where gripper picks it up)
      - <site class="keypoint" name="top_site" .../> at the tip (liquid surface reference)
      - <site class="keypoint" name="bottom_site" .../> at the bulb base
    """
    def __init__(self, solution=None, solution_rgba=None, **kwargs):
        self.solution = solution        # e.g. "CuSO4"
        self.solution_rgba = solution_rgba  # [r, g, b, a]
        super().__init__(**kwargs)

    @property
    def has_solution(self):
        return self.solution is not None

    def get_aspirate_site(self):
        """Get the aspirate site element from the MJCF model."""
        return self.mjcf_model.worldbody.find("site", "aspirate_site")

    def get_aspirate_pos(self, physics):
        """Get the aspirate site world position, or None if not defined."""
        site = self.get_aspirate_site()
        if site is None:
            return None
        return physics.bind(site).xpos

    def store_solution(self, solution, solution_rgba=None):
        """
        Store solution info after successful aspiration.

        Args:
            solution: substance name string, e.g. "CuSO4"
            solution_rgba: [r, g, b, a] color list, or None to infer from solute2rgba
        """
        self.solution = solution
        self.solution_rgba = solution_rgba

    def clear_solution(self):
        """Clear stored solution after dispensing."""
        self.solution = None
        self.solution_rgba = None

    def initialize_episode(self, physics, random_state):
        # Ensure clean state at episode start
        self.clear_solution()
        return super().initialize_episode(physics, random_state)


@register.add_entity("Dropper")
class Dropper(CommonGraspedEntity):
    """
    Rubber bulb dropper (滴管) with aspirate site at the tip.
    Shares the same aspirate/dispense skills and solution storage as Pipette.
    """
    def __init__(self, solution=None, solution_rgba=None, **kwargs):
        self.solution = solution
        self.solution_rgba = solution_rgba
        super().__init__(**kwargs)

    @property
    def has_solution(self):
        return self.solution is not None

    def get_aspirate_site(self):
        return self.mjcf_model.worldbody.find("site", "aspirate_site")

    def get_aspirate_pos(self, physics):
        site = self.get_aspirate_site()
        if site is None:
            return None
        return physics.bind(site).xpos

    def store_solution(self, solution, solution_rgba=None):
        self.solution = solution
        self.solution_rgba = solution_rgba

    def clear_solution(self):
        self.solution = None
        self.solution_rgba = None

    def initialize_episode(self, physics, random_state):
        self.clear_solution()
        return super().initialize_episode(physics, random_state)
