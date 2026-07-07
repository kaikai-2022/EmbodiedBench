"""
Liquid container entities with solution rendering support.
Provides a unified `solution` parameter interface for chemistry containers
(e.g., beaker, flask, petri dish) to display colored liquid in MuJoCo rendering.
"""
import numpy as np
from VLABench.tasks.components.entity import CommonGraspedEntity
from VLABench.tasks.components.container import ContainerMiXin
from VLABench.utils.register import register
from VLABench.tasks.components.specific_entities.solute_reaction import SOLUTE2RGBA, resolve_color_from_solutes, resolve_substance_name


class SolutionMixin:
    """
    Solution rendering Mixin — shared capability for all containers that hold liquid.
    All subclasses share a global solute color mapping table (solute2rgba).
    """

    # 类属性别名，保持向后兼容（旧的 solution2rgba 引用继续有效）
    solute2rgba = SOLUTE2RGBA
    solution2rgba = SOLUTE2RGBA  # deprecated alias

    _solution_geom_name = "solution"

    def __init__(self, solution=None, solution_rgba=None, solutes=None, **kwargs):
        self.solution = solution          # 初始物质名（向后兼容，不随倾倒更新）
        self.solution_rgba = solution_rgba  # LLM 给的 fallback RGBA
        self._current_solution_rgba = None  # 最近生效的 rgba
        # 当前瓶内所有溶质（含反应产物）；向后兼容：从 solution 推断单元素列表
        if solutes is not None:
            self.solutes = [resolve_substance_name(s) for s in solutes]
        elif solution is not None:
            self.solutes = [resolve_substance_name(solution)]
        else:
            self.solutes = []
        super().__init__(**kwargs)

    def set_solution_rgba(self, physics, solution_name=None, target_rgba=None):
        """
        Set the rgba color of the solution geom.

        Priority: target_rgba > solution_name > solutes 查表 > self.solution_rgba (LLM fallback)
        """
        geom = self.mjcf_model.worldbody.find("geom", self._solution_geom_name)
        if geom is None:
            return
        if target_rgba is not None:
            physics.bind(geom).rgba = target_rgba
        elif solution_name is not None:
            # solution_name 传入时：追加到 solutes（人工"添加溶液"语义）
            if solution_name not in self.solutes:
                self.solutes.append(solution_name)
            rgba = self.solute2rgba.get(solution_name, [1, 1, 1, 0.3])
            physics.bind(geom).rgba = rgba
        else:
            # 无显式参数：按 solutes 列表求颜色，fallback 到 LLM 给的 solution_rgba
            rgba = resolve_color_from_solutes(self.solutes, self.solution_rgba)
            physics.bind(geom).rgba = rgba
        self._current_solution_rgba = list(physics.bind(geom).rgba)

    def get_solution(self):
        """Return the current solute list."""
        return self.solutes

    def clear_solution(self, physics):
        """
        清空容器中的溶液（设置为透明）。
        """
        geom = self.mjcf_model.worldbody.find("geom", self._solution_geom_name)
        if geom is None:
            return
        physics.bind(geom).rgba = [1, 1, 1, 0]
        self._current_solution_rgba = [1, 1, 1, 0]
        self.solutes = []
        self.solution = None
        self.solution_rgba = None

    def fill_solution(self, physics, source_solutes=None, source_solution_rgba=None):
        """
        向容器中灌入溶液（含溶质列表和颜色）。

        接受两种调用方式（向后兼容）：
          - 新路径（推荐）：fill_solution(physics, source_solutes=[...], source_solution_rgba=[...])
          - 旧路径：fill_solution(physics, source_solution_rgba=[...])  — 仅 rgba，不带物质列表
        """
        geom = self.mjcf_model.worldbody.find("geom", self._solution_geom_name)
        if geom is None:
            return

        # 优先使用物质列表
        if source_solutes is not None:
            self.solutes = list(source_solutes)
            rgba = resolve_color_from_solutes(self.solutes, source_solution_rgba)
        elif source_solution_rgba is not None:
            rgba = list(source_solution_rgba)
        else:
            rgba = [1, 1, 1, 0.3]

        physics.bind(geom).rgba = rgba
        self._current_solution_rgba = list(rgba)
        self.solution_rgba = list(rgba)

    def initialize_episode(self, physics, random_state):
        self.set_solution_rgba(physics)
        return super().initialize_episode(physics, random_state)

    def save(self, physics):
        data = super().save(physics)
        data["solution"] = self.solution       # 初始物质名（向后兼容）
        data["solutes"] = list(self.solutes)   # 当前溶质列表（含反应产物）
        data["solution_rgba"] = list(self.solution_rgba) if self.solution_rgba is not None else None
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
