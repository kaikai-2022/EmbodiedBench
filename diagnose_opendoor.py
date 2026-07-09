"""
诊断脚本：检查 open_door 技能中所有关键数值
使用方法：
    VLABENCH_ROOT=/ssd/mkqin/workspace/VLABench/VLABench python diagnose_opendoor.py
"""
import os, sys, numpy as np
sys.path.insert(0, '/ssd/mkqin/workspace/VLABench')
os.environ['VLABENCH_ROOT'] = '/ssd/mkqin/workspace/VLABench/VLABench'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['DM_ENV_GRASP_LOCK'] = '0'
import warnings
warnings.filterwarnings('ignore')

import VLABench.tasks.components
import importlib
importlib.import_module('VLABench.robots')

series_name = 'open_dryingbox_series'
series_path = '/ssd/mkqin/workspace/VLABench/VLABench/tasks/autogen_tasks/open_dryingbox_series.py'
spec = importlib.util.spec_from_file_location(series_name, series_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

from VLABench.configs import name2config as configs_name2config
import VLABench.envs as envs_module
configs_name2config[series_name] = ['open_dryingbox']
envs_module.name2config[series_name] = ['open_dryingbox']

from VLABench.envs import load_env
env = load_env('open_dryingbox', robot='franka')
env.reset()
physics = env.physics

entity = env.task.entities['drying_box_0']

print("=" * 60)
print("1. DryingBox 实体世界位置")
print("=" * 60)
drying_box_pos = entity.get_xpos(physics)
drying_box_quat = entity.get_xqaut(physics)
print(f"   实体 root 世界位置: [{drying_box_pos[0]:.4f}, {drying_box_pos[1]:.4f}, {drying_box_pos[2]:.4f}]")
print(f"   实体 root 世界四元数: [{drying_box_quat[0]:.4f}, {drying_box_quat[1]:.4f}, {drying_box_quat[2]:.4f}, {drying_box_quat[3]:.4f}]")

print()
print("=" * 60)
print("2. 门轴关节 (door_joint) — 关键检查")
print("=" * 60)
door_joint = entity.door_joint
xanchor = physics.bind(door_joint).xanchor
xaxis = physics.bind(door_joint).xaxis
qpos = float(physics.bind(door_joint).qpos[0])
joint_range = physics.bind(door_joint).range
print(f"   joint.name: {door_joint.name}")
print(f"   xanchor (reported): [{xanchor[0]:.4f}, {xanchor[1]:.4f}, {xanchor[2]:.4f}]")
print(f"   xaxis  (reported): [{xaxis[0]:.4f}, {xaxis[1]:.4f}, {xaxis[2]:.4f}]")
print(f"   qpos: {qpos:.6f}")
print(f"   joint range: [{joint_range[0]:.2f}, {joint_range[1]:.2f}]")

print()
print("   关键验证 — xanchor 应该接近门轴世界坐标:")
print(f"   如果 drying_box 绕 z 转 90°，门轴世界位置应该是:")
print(f"   x ≈ 0.3 + 0.03 = 0.33 或 0.3 - 0.37 = -0.07 (取决于朝向)")
print(f"   y ≈ 0.45 + 0.03 = 0.48 或 0.45 - 0.37 = 0.08")
print(f"   如果 xanchor 值接近桌面高度 z≈0.8 或 0.03，说明可能用了局部坐标!")

print()
print("=" * 60)
print("3. 所有 grasp sites (group=4)")
print("=" * 60)
gs = entity.grasp_sites(physics)
for s in gs:
    pos = physics.bind(s).xpos
    g = physics.bind(s).group
    # Site 所在 body
    print(f"   {s.name}: group={g}, world_pos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

print()
print("=" * 60)
print("4. get_handle_pos() 输出 — 这是 open_door 轨迹的起点")
print("=" * 60)
try:
    handle_pos = entity.get_handle_pos(physics)
    print(f"   ✓ get_handle_pos: [{handle_pos[0]:.4f}, {handle_pos[1]:.4f}, {handle_pos[2]:.4f}]")
except ValueError as e:
    print(f"   ✗ ValueError: {e}")
    gs2 = entity.grasp_sites(physics)
    print(f"   grasp_sites 返回了 {len(gs2)} 个 (期望 > 0)")
    print(f"   如果是 0，检查 site 的 group 是否为 4")
    for s in entity.sites:
        g = physics.bind(s).group
        print(f"     site={s.name}, group={g}")

print()
print("=" * 60)
print("5. get_open_trajectory() 输出")
print("=" * 60)
try:
    traj = entity.get_open_trajectory(physics)
    print(f"   轨迹点数: {len(traj)}")
    print(f"   第一个点 (轨迹起点): [{traj[0][0]:.4f}, {traj[0][1]:.4f}, {traj[0][2]:.4f}]")
    print(f"   最后一个点:         [{traj[-1][0]:.4f}, {traj[-1][1]:.4f}, {traj[-1][2]:.4f}]")
    print(f"   桌面高度参考: z≈0.78")
    print(f"   夹爪移动方向: Δz = {traj[0][2] - handle_pos[2]:.4f}")
except Exception as e:
    print(f"   ✗ 异常: {e}")
    import traceback; traceback.print_exc()

print()
print("=" * 60)
print("6. 机器人末端执行器初始位置")
print("=" * 60)
ee_pos = env.robot.get_end_effector_pos(physics)
print(f"   夹爪世界位置: [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]")
if 'handle_pos' in dir():
    dist = np.linalg.norm(handle_pos - ee_pos)
    print(f"   夹爪到门把手距离: {dist:.4f} m")
    print(f"   夹爪是否高于把手: {'是' if ee_pos[2] > handle_pos[2] else '否'}")

print()
print("=" * 60)
print("7. XML 中的原始值 (参考)")
print("=" * 60)
print(f"   door_handle_grasp site 局部坐标: [-0.2265, -0.0014, 0.0569]")
print(f"   drying_box_door_joint pos(局部): [-0.2882, -0.3704, 0.0331]")
print(f"   drying_box_door_joint axis:      [0, 0, 1]")
print(f"   drying_box 初始朝向 (YAML):      [0, 0, 1.5708] (90°)")
print(f"   drying_box 初始位置 (YAML):      [0.3, 0.45, 0.8]")

env.close()
