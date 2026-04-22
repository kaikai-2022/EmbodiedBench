"""
Liquid container entities with solution rendering support.
Provides a unified `solution` parameter interface for chemistry containers
(e.g., beaker, flask, petri dish) to display colored liquid in MuJoCo rendering.
"""
from VLABench.tasks.components.entity import CommonGraspedEntity
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

    def __init__(self, solution=None, **kwargs):
        self.solution = solution
        super().__init__(**kwargs)

    def set_solution_rgba(self, physics, solution_name=None):
        """
        Set the rgba color of the solution geom based on solvent name.
        Subclasses can override `_solution_geom_name` to target a different geom.

        Args:
            physics: MuJoCo physics instance
            solution_name: solvent name. If None, uses self.solution.
        """
        target = solution_name if solution_name is not None else self.solution
        geom = self.mjcf_model.worldbody.find("geom", self._solution_geom_name)
        if geom is None:
            return
        if target is None:
            # No solution: make the liquid geom fully transparent
            physics.bind(geom).rgba = [1, 1, 1, 0]
            return
        rgba = self.solution2rgba.get(target, [1, 1, 1, 0.3])
        physics.bind(geom).rgba = rgba

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
class ChemistryBeaker(SolutionMixin, CommonGraspedEntity):
    """
    Beaker with solution rendering capability.
    Pass `solution` parameter (e.g., solution="CuSO4") to display colored liquid.
    Without `solution`, the beaker renders empty.
    """
    pass
