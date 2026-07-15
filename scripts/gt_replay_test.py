"""
GT Replay Test: 直接从 HDF5 数据集读取真实轨迹执行，验证数据转换一致性

用法:
    python gt_replay_test.py --hdf5-path /path/to/data_0.hdf5 --vis
"""
import os
import sys
import argparse
import h5py
import json
import numpy as np
import mediapy

# 设置环境
os.environ["MUJOCO_GL"] = "egl"
os.environ["VLABENCH_ROOT"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from VLABench.robots import *
from VLABench.tasks import *
from VLABench.envs import load_env
from VLABench.utils.utils import euler_to_quaternion, get_logger
from VLABench.configs import name2config
from scipy.spatial.transform import Rotation as R


def load_trajectory_from_hdf5(hdf5_path):
    """从 HDF5 文件加载 trajectory 和 episode config"""
    with h5py.File(hdf5_path, "r") as f:
        data_group = f["data"]
        # 获取第一个 timestamp
        timestamp = list(data_group.keys())[0]
        episode = data_group[timestamp]

        # 加载观测数据
        images = episode["observation"]["rgb"][()]
        ee_state = episode["observation"]["ee_state"][()]
        q_state = episode["observation"]["q_state"][()]

        # 加载轨迹（actions）
        trajectory = episode["trajectory"][()]

        # 加载 episode config
        episode_config_bytes = np.asarray(episode["meta_info"]["episode_config"]).astype('S')
        episode_config = episode_config_bytes.item().decode('utf-8')
        episode_config = json.loads(episode_config)

        # 加载 instruction
        instruction = np.array(episode["instruction"])[0].decode("utf-8")

        # 加载 task name
        task_str = None
        if "task" in episode["meta_info"]:
            task_str = np.array(episode["meta_info"]["task"])[0].decode("utf-8")

    return {
        "images": images,
        "ee_state": ee_state,
        "q_state": q_state,
        "trajectory": trajectory,
        "episode_config": episode_config,
        "instruction": instruction,
        "task": task_str,
    }


def replay_trajectory(hdf5_path, save_video=False, save_dir=None):
    """
    GT Replay: 从 HDF5 读取 trajectory，直接作为 action 执行
    """
    print(f"\n{'='*60}")
    print(f"GT Replay Test: {hdf5_path}")
    print(f"{'='*60}")

    # 加载数据
    print("[1/4] Loading HDF5 data...")
    data = load_trajectory_from_hdf5(hdf5_path)
    trajectory = data["trajectory"]
    episode_config = data["episode_config"]
    instruction = data["instruction"]

    print(f"  - Trajectory length: {len(trajectory)}")
    print(f"  - Instruction: {instruction}")
    print(f"  - Episode config robot: {episode_config.get('robot', 'default')}")

    # 解析任务名
    task_name = data["task"]
    if task_name is None:
        # 从文件名推断
        task_name = os.path.basename(os.path.dirname(hdf5_path))
    print(f"  - Task name: {task_name}")

    # 创建环境（使用相同的 episode config）
    print(f"\n[2/4] Creating environment...")

    # 从 episode_config 提取关键信息
    robot_config = episode_config.get("robot", {})
    print(f"  - Robot position: {robot_config.get('position', [0, -0.4, 0.78])}")

    # 加载环境（使用相同的 episode config）
    print(f"\n[2/4] Creating environment...")

    # 从 episode_config 提取关键信息
    robot_config = episode_config.get("robot", {})
    print(f"  - Robot position: {robot_config.get('position', [0, -0.4, 0.78])}")

    # 加载环境（传入 episode_config 确保确定性布局）
    env = load_env(task_name, robot="franka", eval=False, episode_config=episode_config)

    # 启用 grasp_lock（与轨迹生成一致）
    env.enable_grasp_lock()
    print("  - Grasp lock enabled")

    print("  - Environment loaded with episode config")

    # 获取条件检查
    task_success = False
    condition_results = {}

    # 顺序条件评测准备
    if hasattr(env.task, 'conditions') and env.task.conditions is not None:
        if hasattr(env.task.conditions, 'reset_locks'):
            env.task.conditions.reset_locks()
        for condition in env.task.conditions.conditions:
            if hasattr(condition, 'record_initial_state'):
                condition.record_initial_state(env.physics)

    # 执行 trajectory
    print(f"\n[3/4] Executing trajectory...")
    frames = []
    success = False
    prev_gripper = None
    gripper_change_points = []

    for i, action in enumerate(trajectory):
        gripper_val = action[6] if len(action) >= 7 else action[-1]

        # 检测 gripper 值变化
        if prev_gripper is not None:
            if abs(gripper_val - prev_gripper) > 0.001:
                gripper_change_points.append((i, prev_gripper, gripper_val))
                print(f"  *** Gripper change at step {i}: {prev_gripper:.4f} -> {gripper_val:.4f} ***")

        # 打印前几个、每30个、以及 gripper 变化点附近
        should_print = (
            i < 5 or  # 前5个
            i % 30 == 0 or  # 每30个
            any(abs(i - cp[0]) <= 2 for cp in gripper_change_points[-3:])  # 变化点附近
        )

        if should_print:
            print(f"  Step {i:3d}/{len(trajectory)}: pos=[{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}], "
                  f"euler=[{action[3]:.3f}, {action[4]:.3f}, {action[5]:.3f}], "
                  f"gripper={gripper_val:.4f}")

        prev_gripper = gripper_val

        # 从 action 提取 gripper
        # action format: [pos3, euler3, gripper1, gripper2] 或 [pos3, euler3, gripper1]
        if len(action) == 8:
            gripper_state = action[6:8]  # 使用两个 gripper 值
        else:
            # 7D format
            gripper_width = action[6]
            gripper_state = np.array([gripper_width, gripper_width])

        # 将 euler 转换为 quat 用于逆运动学
        pos = action[:3]
        euler = action[3:6]
        quat = euler_to_quaternion(*euler)

        # 获取 qpos 从 ee pose
        _, action_qpos = env.robot.get_qpos_from_ee_pos(
            physics=env.physics,
            pos=pos + np.array(episode_config.get("robot", {}).get("position", [0, -0.4, 0.78])),
            quat=quat
        )

        # 构造完整 action: [arm_qpos(7), gripper(2)]
        full_action = np.concatenate([action_qpos, gripper_state])

        # 执行 step
        timestep = env.step(full_action)

        # 录制视频
        if save_video:
            obs = env.get_observation(require_pcd=False)
            rgb = obs["rgb"]
            # 组合四个视角
            frame = np.vstack([np.hstack(rgb[:2]), np.hstack(rgb[2:4])])
            frames.append(frame)

        # 检查任务成功
        if timestep.last():
            task_success = True
            print(f"  -> Task success at step {i}!")
            break

        # 顺序条件检查
        if hasattr(env.task, 'conditions') and env.task.conditions is not None:
            env.task.conditions.is_met(env.physics)

    # 收集条件结果
    print(f"\n[4/4] Evaluating conditions...")
    if hasattr(env.task, 'conditions') and env.task.conditions is not None:
        for cond in env.task.conditions.conditions:
            cond_name = type(cond).__name__
            condition_results[cond_name] = cond.is_met(env.physics)
            print(f"  {cond_name}: {'PASS' if condition_results[cond_name] else 'FAIL'}")

    env.close()

    # 保存视频
    if save_video and frames and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        video_path = os.path.join(save_dir, f"gt_replay_{os.path.basename(hdf5_path).replace('.hdf5', '.mp4')}")
        mediapy.write_video(video_path, frames, fps=10)
        print(f"\n  Video saved: {video_path}")

    # 打印总结
    print(f"\n{'='*60}")
    print(f"GT Replay Summary:")
    print(f"  - Trajectory executed: {min(i+1, len(trajectory))}/{len(trajectory)} steps")
    print(f"  - Task success: {task_success}")
    print(f"  - Conditions passed: {sum(condition_results.values())}/{len(condition_results)}")
    print(f"{'='*60}")

    return task_success, condition_results


def main():
    parser = argparse.ArgumentParser(description="GT Replay Test - Verify trajectory data integrity")
    parser.add_argument("--hdf5-path", type=str, required=True, help="Path to HDF5 trajectory file")
    parser.add_argument("--vis", action="store_true", help="Save visualization video")
    parser.add_argument("--save-dir", type=str, default="/ssd/qinmaokai/workspace/SciVLABench/logs/gt_replay", help="Directory to save videos")
    args = parser.parse_args()

    if not os.path.exists(args.hdf5_path):
        print(f"Error: HDF5 file not found: {args.hdf5_path}")
        sys.exit(1)

    success, conditions = replay_trajectory(args.hdf5_path, save_video=args.vis, save_dir=args.save_dir)

    # 返回码：成功为 0，失败为 1
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
