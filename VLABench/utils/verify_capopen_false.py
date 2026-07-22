"""
验证 cap_open 条件不会误返回 True。
让 unscrew_cap 只转很小角度（0.1*pi ≈ 18°），看:
1. slide 关节位移是否 > 0.003m （如果 >0.003m 就不应该）
2. cap_open 条件是否被触发
3. stage_success (skill 层) 是否为 False

用法: python VLABench/utils/verify_capopen_false.py
"""
import sys, os
sys.path.insert(0, "/ssd/liuzirui/VLAbench_LZR")
os.environ.setdefault("VLABENCH_ROOT", "/ssd/liuzirui/VLAbench_LZR/VLABench")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("DM_ENV_GRASP_LOCK", "0")

import importlib, importlib.util
importlib.import_module("VLABench.robots")
importlib.import_module("VLABench.tasks.autogen_tasks")
importlib.import_module("VLABench.tasks.hierarchical_tasks.composite")
import VLABench.tasks.components

# 加载流水线生成的 unscrew_bottle_series
series_path = "/ssd/liuzirui/VLAbench_LZR/VLABench/tasks/autogen_tasks/unscrew_bottle_series.py"
spec = importlib.util.spec_from_file_location("unscrew_bottle_series", series_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

from VLABench.configs import name2config as configs_name2config
import VLABench.envs as envs_module
configs_name2config["unscrew_bottle_series"] = ["unscrew_bottle"]
envs_module.name2config["unscrew_bottle_series"] = ["unscrew_bottle"]
envs_module.TASK_CONFIG["unscrew_bottle_series"] = {}

import numpy as np
from VLABench.envs import load_env
from VLABench.utils.skill_lib import SkillLib

env = load_env("unscrew_bottle")
bottle = env.task.entities["pill_bottle_0"]
physics = env.physics

# 找 CapOpenCondition
cap_open_cond = None
cond_set = env.task.conditions
for c in cond_set.conditions if hasattr(cond_set, 'conditions') else []:
    if type(c).__name__ == 'CapOpenCondition':
        cap_open_cond = c
        break
print(f"Found CapOpenCondition: {cap_open_cond is not None}")

# 记录初始 state (必须)
cap_open_cond.record_initial_state(physics)
print(f"recorded: {cap_open_cond._initial_state_recorded}")
print(f"initial slide pos: {cap_open_cond._initial_slide_pos}")
print(f"is_met (before any action): {cap_open_cond.is_met(physics)}  (expected: False)")

# 调用 unscrew_cap 但只转 0.1*pi
print("\n=== Running unscrew_cap with rotation_angle=0.1*pi (18°) ===")
observations, waypoints, stage_success, task_success = SkillLib.unscrew_cap(
    env, target_entity_name="pill_bottle_0",
    rotation_angle=0.1 * np.pi,  # 18° - 远不够拧开
    lift_height=0.02,
)
print(f"\n=== Result ===")
print(f"stage_success (skill): {stage_success}  (expected: False)")
print(f"task_success: {task_success}  (expected: False)")

# 检查 slide 关节最终值
final_slide = float(physics.bind(bottle.slide_joint).qpos)
initial_slide = cap_open_cond._initial_slide_pos[list(cap_open_cond._initial_slide_pos.keys())[0]]
delta = final_slide - initial_slide
print(f"final slide qpos: {final_slide:.6f}")
print(f"initial slide qpos: {initial_slide:.6f}")
print(f"delta: {delta:.6f}  (threshold: 0.003)")

# 检查 cap_open condition
print(f"\ncap_open.is_met (after skill): {cap_open_cond.is_met(physics)}  (expected: False)")

# 总结
print("\n=== Conclusion ===")
if not cap_open_cond.is_met(physics) and not stage_success:
    print("✅ cap_open 条件正确返回 False，没有误判")
else:
    print("❌ cap_open 条件可能误判！")
