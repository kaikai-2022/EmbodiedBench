"""
LeRobot GT Replay Test: 从 LeRobot parquet 读取 action，验证转换一致性

用法:
    python lerobot_gt_replay.py --dataset-path /path/to/lerobot/dataset --episode 0 --vis
"""
import os
import sys
import argparse
import numpy as np
import json

os.environ["MUJOCO_GL"] = "egl"
os.environ["VLABENCH_ROOT"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 设置 PYTHONPATH
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/ssd/qinmaokai/workspace/SciVLABench")

from VLABench.robots import *
from VLABench.tasks import *
from VLABench.envs import load_env
from VLABench.utils.utils import euler_to_quaternion, quaternion_to_euler, get_logger
from VLABench.configs import name2config


def find_hdf5_with_episode(hdf5_dir, episode_idx):
    """
    在目录中查找包含指定 episode 的 HDF5 文件
    返回 HDF5 路径和 episode_config
    """
    from pathlib import Path
    hdf5_dir = Path(hdf5_dir)

    # 查找所有 HDF5 文件
    hdf5_files = sorted(hdf5_dir.glob("*/data_*.hdf5"))
    if not hdf5_files:
        hdf5_files = sorted(hdf5_dir.glob("data_*.hdf5"))

    # 返回第一个文件（假设 episode_idx 对应第一个文件）
    if hdf5_files:
        return str(hdf5_files[0])
    return None


def load_lerobot_actions(dataset_path, episode_idx=0, hdf5_path=None):
    """
    从 LeRobot 数据集加载 actions
    支持直接从 parquet 文件读取
    """
    import glob
    from pathlib import Path

    data_dir = Path(dataset_path)

    # 找到 episode 对应的 parquet 文件
    parquet_files = sorted(data_dir.glob("data/chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}/data/chunk-*/")

    # 找到对应 episode
    target_file = None
    for pf in parquet_files:
        # 从文件名解析 episode index
        ep_str = pf.stem.split('_')[1]  # episode_000000 -> 000000
        if int(ep_str) == episode_idx:
            target_file = pf
            break

    if target_file is None:
        raise FileNotFoundError(f"Episode {episode_idx} not found in {parquet_files}")

    print(f"  Loading: {target_file}")

    # 使用 pyarrow 读取 parquet
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(target_file)
        df = table.to_pandas()
    except ImportError:
        # 备选：使用 pandas
        import pandas as pd
        df = pd.read_parquet(target_file)

    # 提取关键列
    actions = np.array(df['actions'].tolist()) if 'actions' in df.columns else None
    states = np.array(df['observation.state'].tolist()) if 'observation.state' in df.columns else None

    # 从 info.json 获取任务信息
    info_file = data_dir / "meta" / "info.json"
    task_name = "unknown"
    if info_file.exists():
        with open(info_file) as f:
            info = json.load(f)
            task_name = info.get("repo_id", "unknown")

    # 尝试加载 episode_config（用于确定性初始化）
    episode_config = None
    if hdf5_path and os.path.exists(hdf5_path):
        try:
            import h5py
            with h5py.File(hdf5_path, 'r') as f:
                ts = list(f['data'].keys())[0]
                config_bytes = np.asarray(f['data'][ts]['meta_info']['episode_config']).astype('S')
                episode_config = json.loads(config_bytes.item().decode('utf-8'))
                print(f"  Loaded episode_config from: {hdf5_path}")
        except Exception as e:
            print(f"  Warning: Could not load episode_config: {e}")
    else:
        # 尝试自动查找
        auto_hdf5 = find_hdf5_with_episode(data_dir.parent, episode_idx)
        if auto_hdf5 and os.path.exists(auto_hdf5):
            try:
                import h5py
                with h5py.File(auto_hdf5, 'r') as f:
                    ts = list(f['data'].keys())[0]
                    config_bytes = np.asarray(f['data'][ts]['meta_info']['episode_config']).astype('S')
                    episode_config = json.loads(config_bytes.item().decode('utf-8'))
                    print(f"  Loaded episode_config from: {auto_hdf5}")
            except Exception as e:
                print(f"  Warning: Could not load episode_config: {e}")

    return {
        "actions": actions,  # (N, 7): [pos3, euler3, gripper_binary]
        "states": states,    # (N, 7)
        "num_frames": len(actions) if actions is not None else 0,
        "task_name": task_name,
        "episode_config": episode_config,
    }


def lerobot_gt_replay(dataset_path, episode_idx=0, task=None, hdf5_path=None, save_video=False, save_dir=None):
    """
    从 LeRobot 数据集读取 action，直接执行 GT Replay
    """
    print(f"\n{'='*60}")
    print(f"LeRobot GT Replay Test")
    print(f"Dataset: {dataset_path}")
    print(f"Episode: {episode_idx}")
    print(f"{'='*60}")

    # 1. 加载 LeRobot 数据集
    print("\n[1/4] Loading LeRobot dataset...")
    try:
        data = load_lerobot_actions(dataset_path, episode_idx, hdf5_path=hdf5_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nTrying alternative loading method...")
        # 备选：使用 lerobot 库
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
            dataset = LeRobotDataset(dataset_path)
            # 获取 episode 数据
            from lerobot.common.datasets.utils import load_episodes
            episodes = load_episodes(dataset.root / dataset.meta_dir)
            # 这里需要进一步处理...
            raise NotImplementedError("Please implement lerobot dataset loading")
        except Exception as e2:
            print(f"Alternative loading also failed: {e2}")
            sys.exit(1)

    actions = data["actions"]
    num_frames = data["num_frames"]
    task_name = data["task_name"]

    print(f"  - Dataset frames: {num_frames}")
    print(f"  - Action shape: {actions.shape}")  # (N, 7)
    print(f"  - Task: {task_name}")

    # 2. 创建环境
    print(f"\n[2/4] Creating environment...")

    # 确定任务名
    if task:
        inferred_task = task
    else:
        # 从 meta/tasks.jsonl 尝试推断
        import json
        meta_tasks_file = os.path.join(dataset_path, "meta", "tasks.jsonl")
        if os.path.exists(meta_tasks_file):
            with open(meta_tasks_file) as f:
                for line in f:
                    task_info = json.loads(line)
                    if task_info.get("task_index", 0) == 0:
                        task_instruction = task_info.get("task", "")
                        # 尝试从 instruction 匹配已知任务
                        if "place" in task_instruction.lower() and "heat" in task_instruction.lower():
                            inferred_task = "place_beaker_pres_heat_device"
                        elif "lift" in task_instruction.lower():
                            inferred_task = "lift_beaker"
                        else:
                            inferred_task = "lift_beaker"
                        break
        else:
            inferred_task = "lift_beaker"

    # 使用 episode_config 创建确定性环境
    episode_config = data.get("episode_config")
    if episode_config:
        print(f"  Using episode_config for deterministic initialization")
        env = load_env(inferred_task, robot="franka", eval=False, episode_config=episode_config)
    else:
        print(f"  WARNING: No episode_config, using random initialization")
        env = load_env(inferred_task, robot="franka", eval=False)
    env.enable_grasp_lock()
    print(f"  - Environment: {inferred_task}")
    print(f"  - Robot frame: {env.get_robot_frame_position()}")

    # 顺序条件评测准备
    if hasattr(env.task, 'conditions') and env.task.conditions is not None:
        if hasattr(env.task.conditions, 'reset_locks'):
            env.task.conditions.reset_locks()
        for condition in env.task.conditions.conditions:
            if hasattr(condition, 'record_initial_state'):
                condition.record_initial_state(env.physics)

    # 3. 执行 LeRobot actions
    print(f"\n[3/4] Executing LeRobot actions...")
    frames = []
    task_success = False
    robot_frame_pos = env.get_robot_frame_position()

    prev_gripper = None

    for i in range(num_frames):
        # 读 LeRobot action: [pos3_local, euler3_local, gripper_binary]
        action = actions[i]

        pos_local = action[:3]
        euler_local = action[3:6]
        gripper_binary = action[6]

        # DEBUG: 检测 gripper 变化
        if prev_gripper is not None and abs(gripper_binary - prev_gripper) > 0.5:
            print(f"  *** Gripper change at step {i}: {prev_gripper:.1f} -> {gripper_binary:.1f} ***")

        # 打印关键帧
        should_print = (
            i < 5 or
            i % 20 == 0 or
            (prev_gripper is not None and abs(gripper_binary - prev_gripper) > 0.5)
        )

        if should_print:
            print(f"  Step {i:3d}/{num_frames}: local_pos=[{pos_local[0]:.3f}, {pos_local[1]:.3f}, {pos_local[2]:.3f}], "
                  f"euler=[{euler_local[0]:.3f}, {euler_local[1]:.3f}, {euler_local[2]:.3f}], "
                  f"gripper_binary={gripper_binary:.1f}")

        prev_gripper = gripper_binary

        # 2. local → world frame
        target_pos = pos_local + robot_frame_pos

        # 3. gripper binary → 物理值
        # LeRobot 中: 0=闭合, 1=张开
        if gripper_binary >= 0.5:
            gripper_state = np.ones(2) * 0.04  # 张开
        else:
            gripper_state = np.zeros(2)        # 闭合

        # 4. euler → quat
        quat = euler_to_quaternion(*euler_local)

        # 5. IK: world EE pose → qpos
        _, action_qpos = env.robot.get_qpos_from_ee_pos(
            physics=env.physics,
            pos=target_pos,
            quat=quat
        )

        # 6. 构造完整 action
        full_action = np.concatenate([action_qpos, gripper_state])

        # 7. 执行
        timestep = env.step(full_action)

        # 录制
        if save_video:
            try:
                obs = env.get_observation(require_pcd=False)
                rgb = obs["rgb"]
                frame = np.vstack([np.hstack(rgb[:2]), np.hstack(rgb[2:4])])
                frames.append(frame)
            except Exception as e:
                print(f"Warning: Failed to capture frame: {e}")

        # 检查成功
        if timestep.last():
            task_success = True
            print(f"  -> Task success at step {i}!")
            break

        # 条件检查
        if hasattr(env.task, 'conditions') and env.task.conditions is not None:
            env.task.conditions.is_met(env.physics)

    # 4. 收集结果
    print(f"\n[4/4] Evaluating conditions...")
    condition_results = {}
    if hasattr(env.task, 'conditions') and env.task.conditions is not None:
        for cond in env.task.conditions.conditions:
            cond_name = type(cond).__name__
            condition_results[cond_name] = cond.is_met(env.physics)
            print(f"  {cond_name}: {'PASS' if condition_results[cond_name] else 'FAIL'}")

    env.close()

    # 保存视频
    if save_video and frames and save_dir:
        try:
            import cv2
            os.makedirs(save_dir, exist_ok=True)
            video_path = os.path.join(save_dir, f"lerobot_gt_replay_ep{episode_idx}.mp4")

            # 获取帧尺寸
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 10, (w, h))

            for frame in frames:
                # cv2 需要 BGR 格式
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            out.release()
            print(f"\n  Video saved: {video_path}")
        except Exception as e:
            print(f"Warning: Failed to save video: {e}")

    # 总结
    print(f"\n{'='*60}")
    print(f"LeRobot GT Replay Summary:")
    print(f"  - Frames executed: {min(i+1, num_frames)}/{num_frames}")
    print(f"  - Task success: {task_success}")
    print(f"  - Conditions passed: {sum(condition_results.values())}/{len(condition_results)}")
    print(f"{'='*60}")

    return task_success, condition_results


def main():
    parser = argparse.ArgumentParser(description="LeRobot GT Replay Test")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to LeRobot dataset directory (containing data/ and meta/)")
    parser.add_argument("--task", type=str, default=None,
                        help="Task name to load (e.g., lift_beaker, place_beaker_pres_heat_device). "
                             "If not specified, tries to infer from dataset meta or use a default.")
    parser.add_argument("--hdf5-path", type=str, default=None,
                        help="Path to original HDF5 file for episode_config. "
                             "If not specified, will try to find it automatically.")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to replay")
    parser.add_argument("--vis", action="store_true", help="Save visualization video")
    parser.add_argument("--save-dir", type=str,
                        default="/ssd/qinmaokai/workspace/SciVLABench/logs/lerobot_gt_replay",
                        help="Directory to save videos")
    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path not found: {args.dataset_path}")
        sys.exit(1)

    success, conditions = lerobot_gt_replay(
        args.dataset_path,
        episode_idx=args.episode,
        task=args.task,
        hdf5_path=args.hdf5_path,
        save_video=args.vis,
        save_dir=args.save_dir
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
