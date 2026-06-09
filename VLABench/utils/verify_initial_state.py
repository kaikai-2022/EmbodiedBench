"""
验证 episode 一启动时，cap_open 条件是 True 还是 False。
期望是 False（瓶盖初始应该是关闭的）。
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

# 加载最新生成的 series（动态发现）
import glob
series_files = sorted(glob.glob("/ssd/mkqin/workspace/VLABench/VLABench/tasks/autogen_tasks/unscrew_*_series.py"))
print(f"Found series files: {series_files}")
series_path = series_files[-1] if series_files else None
if not series_path:
    print("No series file found!")
    sys.exit(1)

mod_name = os.path.basename(series_path).replace(".py", "")
spec = importlib.util.spec_from_file_location(mod_name, series_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# 找到 task_name（从装饰器拿不到，直接用文件名）
task_name = mod_name.replace("_series", "")
print(f"task_name: {task_name}")

from VLABench.configs import name2config as configs_name2config
import VLABench.envs as envs_module
configs_name2config[mod_name] = [task_name]
envs_module.name2config[mod_name] = [task_name]
envs_module.TASK_CONFIG[mod_name] = {}

import numpy as np
from VLABench.envs import load_env
env = load_env(task_name)
bottle = env.task.entities["pill_bottle_0"]
physics = env.physics

# 找 CapOpenCondition
cap_open_cond = None
cond_set = env.task.conditions
for c in cond_set.conditions if hasattr(cond_set, 'conditions') else []:
    if type(c).__name__ == 'CapOpenCondition':
        cap_open_cond = c
        break
print(f"\nFound CapOpenCondition: {cap_open_cond is not None}")

# 检查初始状态
slide_joint = bottle.slide_joint
hinge_joint = bottle.cap_joint
print(f"\n=== Episode 初始状态（reset 完成后）===")
print(f"  slide qpos: {float(physics.bind(slide_joint).qpos):.6f}")
print(f"  hinge qpos: {float(physics.bind(hinge_joint).qpos):.6f}")
print(f"  cap_open.is_met (before record): {cap_open_cond.is_met(physics)}  (expected: False due to _initial_state_recorded=False)")

# 模拟 simulation_node 行为：record_initial_state
cap_open_cond.record_initial_state(physics)
print(f"\n=== record_initial_state 之后 ===")
print(f"  recorded: {cap_open_cond._initial_state_recorded}")
print(f"  initial slide pos: {cap_open_cond._initial_slide_pos}")
print(f"  cap_open.is_met: {cap_open_cond.is_met(physics)}  (expected: False if slide not moved)")

# 检查阈值
print(f"  lift_threshold: {cap_open_cond.lift_threshold}")
print(f"  delta: {float(physics.bind(slide_joint).qpos) - cap_open_cond._initial_slide_pos[list(cap_open_cond._initial_slide_pos.keys())[0]]:.6f}")
