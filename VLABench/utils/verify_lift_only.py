"""
测试: 只提拉 5cm，看 slide 关节是否会自动上升。
模拟 unscrew_cap 流程但 rotation_angle=0（不旋转），只走 lift。
"""
import sys, os
sys.path.insert(0, "/ssd/mkqin/workspace/VLABench")
os.environ.setdefault("VLABENCH_ROOT", "/ssd/mkqin/workspace/VLABench/VLABench")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("DM_ENV_GRASP_LOCK", "0")

import importlib, importlib.util
importlib.import_module("VLABench.robots")
importlib.import_module("VLABench.tasks.autogen_tasks")
importlib.import_module("VLABench.tasks.hierarchical_tasks.composite")
import VLABench.tasks.components

import glob
series_path = sorted(glob.glob("/ssd/mkqin/workspace/VLABench/VLABench/tasks/autogen_tasks/unscrew_*_series.py"))[-1]
mod_name = os.path.basename(series_path).replace(".py", "")
spec = importlib.util.spec_from_file_location(mod_name, series_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

from VLABench.configs import name2config as configs_name2config
import VLABench.envs as envs_module
configs_name2config[mod_name] = [mod_name.replace("_series", "")]
envs_module.name2config[mod_name] = [mod_name.replace("_series", "")]
envs_module.TASK_CONFIG[mod_name] = {}

import numpy as np
from VLABench.envs import load_env
from VLABench.utils.skill_lib import SkillLib
from VLABench.algorithms.utils import interpolate_path

env = load_env(mod_name.replace("_series", ""))
bottle = env.task.entities["pill_bottle_0"]
physics = env.physics

# 找 CapOpenCondition
cap_open_cond = None
cond_set = env.task.conditions
for c in cond_set.conditions if hasattr(cond_set, 'conditions') else []:
    if type(c).__name__ == 'CapOpenCondition':
        cap_open_cond = c
        break

# record initial state (重要)
cap_open_cond.record_initial_state(physics)
initial_slide = cap_open_cond._initial_slide_pos[list(cap_open_cond._initial_slide_pos.keys())[0]]
print(f"Initial slide: {initial_slide:.6f}")

# pick (抓 cap)
print("\n=== 抓取 cap ===")
obs, wp, pick_success, _ = SkillLib.pick(env, "pill_bottle_0", prior_eulers=[[np.pi, 0, 0]])
print(f"pick_success: {pick_success}")

# 跳过旋转阶段，直接做 lift
print("\n=== 跳过旋转，直接 lift 5cm ===")
gripper_state = SkillLib._get_gripper_state(env)
start_pos = env.robot.get_end_effector_pos(env.physics)
start_quat = env.robot.get_end_effector_quat(env.physics)
target_pos = np.array(start_pos) + np.array([0, 0, 0.05])

interplate_path, interplate_quat = interpolate_path(
    [start_pos, target_pos],
    [np.array(start_quat), np.array(start_quat)])
obs, wp, _, _ = SkillLib.step_trajectory(env, interplate_path, interplate_quat, gripper_state)

# 释放
print("=== 释放 ===")
SkillLib.open_gripper(env)

# 检查结果
final_slide = float(physics.bind(bottle.slide_joint).qpos)
delta = final_slide - initial_slide
print(f"\nFinal slide: {final_slide:.6f}")
print(f"Delta: {delta:.6f}  (threshold: 0.003)")
print(f"cap_open.is_met: {cap_open_cond.is_met(physics)}  (expected: False if delta<0.003)")
