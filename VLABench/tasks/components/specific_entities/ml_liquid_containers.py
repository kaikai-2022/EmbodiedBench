"""
Liquid container entities with solution rendering support.
Provides a unified `solution` parameter interface for chemistry containers
(e.g., beaker, flask, petri dish) to display colored liquid in MuJoCo rendering.
"""
import numpy as np
from VLABench.tasks.components.entity import CommonGraspedEntity
from VLABench.tasks.components.container import ContainerMiXin
from VLABench.utils.register import register


class SolutionMixin:
    """
    Solution rendering Mixin — shared capability for all containers that hold liquid.
    All subclasses share a global solvent color mapping table (solution2rgba).
    """

    solution2rgba = {
        "CuCl2": [0.141, 1.0, 0.174, 0.4],
        "CuSO4": [0, 0.45, 1, 0.4],
        "FeCl3": [0.6475, 0.5686, 0.023, 0.4],
        "KMnO4": [0.5, 0, 0.5, 0.4],
        "I2": [0.3, 0.13, 0.0, 0.4],
        "K2CrO4": [0.57, 0.12, 0.013, 0.4],
        "NaCl": [1, 1, 1, 0.3],
        "AgNO3": [1, 1, 1, 0.3],
        "BaCl2": [1, 1, 1, 0.3],
        "H2SO4": [1, 1, 1, 0.3],
        "NaOH": [1, 1, 1, 0.3],
        "Ba(NO3)2": [1, 1, 1, 0.3],
        "Pb(NO3)2": [1, 1, 1, 0.3],
        "Na2CO3": [1, 1, 1, 0.3],
        "CaCl2": [1, 1, 1, 0.3],
        "HCl": [1, 1, 1, 0.3],
        "CaSO4": [1, 1, 1, 0.7],
    }

    _solution_geom_name = "solution"

    def __init__(self, solution=None, solution_rgba=None, **kwargs):
        self.solution = solution
        self.solution_rgba = solution_rgba
        super().__init__(**kwargs)

    def set_solution_rgba(self, physics, solution_name=None, target_rgba=None):
        """
        Set the rgba color of the solution geom.

        Priority: target_rgba > solution_name > self.solution_rgba > self.solution
        """
        geom = self.mjcf_model.worldbody.find("geom", self._solution_geom_name)
        if geom is None:
            return
        if target_rgba is not None:
            physics.bind(geom).rgba = target_rgba
        elif solution_name is not None:
            rgba = self.solution2rgba.get(solution_name, [1, 1, 1, 0.3])
            physics.bind(geom).rgba = rgba
        elif self.solution_rgba is not None:
            physics.bind(geom).rgba = self.solution_rgba
        elif self.solution is not None:
            rgba = self.solution2rgba.get(self.solution, [1, 1, 1, 0.3])
            physics.bind(geom).rgba = rgba
        else:
            physics.bind(geom).rgba = [1, 1, 1, 0]

    def get_solution(self):
        """Return the current solvent name."""
        return self.solution

    def initialize_episode(self, physics, random_state):
        self.set_solution_rgba(physics)
        return super().initialize_episode(physics, random_state)

    def save(self, physics):
        data = super().save(physics)
        data["solution"] = self.solution
        return data


@register.add_entity("ChemistryBeaker")
class ChemistryBeaker(SolutionMixin, ContainerMiXin, CommonGraspedEntity):
    """
    Beaker with solution rendering capability.
    Pass `solution` parameter (e.g., solution="CuSO4") to display colored liquid.
    Without `solution`, the beaker renders empty.

    Inherits from ContainerMiXin to support place operations (has get_place_point).
    """
    def get_place_point(self, physics):
        """
        Get place points for placing objects into/on the beaker.
        For beaker, returns the grasp sites positions (opening area).
        """
        grasp_sites = self.grasp_sites(physics)
        if grasp_sites:
            place_points = [physics.bind(site).xpos for site in grasp_sites]
            return place_points
        # Fallback: use worldbody position with small offset
        return [self.get_xpos(physics) + [0, 0, 0.05]]

    def contain(self, point, physics):
        """
        Judge whether the target point is inside the beaker.
        Uses key_sites to determine the bounding box of the beaker interior.
        """
        try:
            keysites = self.key_sites(physics)
            if not keysites:
                raise AttributeError("No key_sites found")
            keypoints = np.array([physics.bind(kp).xpos for kp in keysites])
            if keypoints.ndim != 2 or keypoints.shape[1] != 3:
                raise ValueError(f"Unexpected keypoints shape: {keypoints.shape}")
            minX, maxX, minY, maxY, minZ, maxZ = (
                keypoints[:, 0].min(), keypoints[:, 0].max(),
                keypoints[:, 1].min(), keypoints[:, 1].max(),
                keypoints[:, 2].min(), keypoints[:, 2].max()
            )
            return (minX <= point[0] <= maxX and
                    minY <= point[1] <= maxY and
                    minZ <= point[2] <= maxZ)
        except (AttributeError, TypeError, ValueError):
            # Fallback: use beaker center and approximate dimensions
            center = self.get_xpos(physics)
            radius = 0.05  # approximate beaker radius
            in_cylinder = ((point[0] - center[0])**2 + (point[1] - center[1])**2) < radius**2
            in_height = center[2] <= point[2] <= center[2] + 0.15
            return in_cylinder and in_height
