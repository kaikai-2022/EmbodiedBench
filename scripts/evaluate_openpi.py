import collections
import dataclasses
import json
import logging
import os
import sys
import traceback

import numpy as np
import random
import mediapy
from openpi_client import websocket_client_policy as _websocket_client_policy
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_VLABENCH_ROOT = os.path.dirname(_SCRIPT_DIR)

from VLABench.robots import *
from VLABench.tasks import *
from VLABench.envs import load_env
from VLABench.configs import name2config
from VLABench.utils.utils import euler_to_quaternion, quaternion_to_euler, find_key_by_value

# 触发 autogen_tasks 下所有已生成任务的 @register 装饰器注册
import glob, importlib.util
_autogen_dir = os.path.join(os.environ.get("VLABENCH_ROOT", _VLABENCH_ROOT), "tasks", "autogen_tasks")
for _path in glob.glob(os.path.join(_autogen_dir, "*_series.py")):
    _name = os.path.splitext(os.path.basename(_path))[0]
    _spec = importlib.util.spec_from_file_location(_name, _path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
print("[DEBUG] autogen_tasks 模块导入成功", flush=True)
print("[DEBUG] 所有模块导入成功", flush=True)

logger = logging.getLogger(__name__)

VLABENCH_DUMMY_ACTION = [0.0] * 6 + [0.04, 0.04]


class Pi0:
    """OpenPI policy client via WebSocket."""

    def __init__(self, client, replan_steps=5):
        self.model = client
        self.replan_steps = replan_steps
        self.action_plan = collections.deque(maxlen=replan_steps)

    def reset(self):
        self.action_plan.clear()

    def predict(self, obs, **kwargs):
        if len(self.action_plan) == 0:
            second_image, _, image, image_wrist = obs["rgb"]
            state = obs["ee_state"]
            last_action = obs["last_action"].copy()
            pos, quat, gripper_state = state[:3], state[3:7], state[-1]
            ee_euler = quaternion_to_euler(quat)
            pos -= np.array([0, -0.4, 0.78])
            state = np.concatenate([pos, ee_euler, np.array(gripper_state).reshape(-1)])
            instruction = obs["instruction"]
            print(f"[DEBUG] Prompt: {instruction}")
            policy_input = {
                "observation/image": image,
                "observation/second_image": second_image,
                "observation/wrist_image": image_wrist,
                "observation/state": state,
                "prompt": instruction,
            }
            action_chunk = self.model.infer(policy_input)["actions"]
            assert (
                len(action_chunk) >= self.replan_steps
            ), f"We want to replan every {self.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
            self.action_plan.extend(action_chunk[: self.replan_steps])
        action = self.action_plan.popleft()
        target_pos, target_euler, gripper = action[:3], action[3:6], action[-1]
        if gripper >= 0.1:
            gripper_state = np.ones(2) * 0.04
        else:
            gripper_state = np.zeros(2)
        target_pos = target_pos.copy()
        target_pos += np.array([0, -0.4, 0.78])
        return target_pos, target_euler, gripper_state

    @property
    def name(self):
        return "pi0"

    @property
    def control_mode(self):
        return "ee"


def run_episode(env, agent, max_episode_length=200, save_video=False):
    """运行单个 episode，返回 condition 结果和基本信息。"""
    print("[DEBUG] run_episode 开始", flush=True)
    env.reset()
    print("[DEBUG] env.reset() 完成", flush=True)
    success = False
    last_action = None
    robot_frame = env.get_robot_frame_position()
    frames_to_save = []

    # 关掉 env 层的 episode 自动 reset（与 simulation 节点行为对齐）
    env._skill_execution_mode = True

    # 重置顺序条件锁：每个 episode 都从第一个条件开始顺序检查
    if hasattr(env.task, 'conditions') and env.task.conditions is not None:
        if hasattr(env.task.conditions, 'reset_locks'):
            env.task.conditions.reset_locks()

    # 记录条件初始状态
    if hasattr(env.task, 'conditions') and env.task.conditions is not None:
        for condition in env.task.conditions.conditions:
            if hasattr(condition, 'record_initial_state'):
                condition.record_initial_state(env.physics)

    for i in range(max_episode_length):
        observation = env.get_observation(require_pcd=False)
        observation["instruction"] = env.task.get_instruction()
        ee_state = observation["ee_state"]
        observation['robot_frame'] = robot_frame
        if last_action is None:
            last_action = np.concatenate([ee_state[:3], quaternion_to_euler(ee_state[3:7])])
        observation["last_action"] = last_action

        if save_video:
            frames_to_save.append(observation["rgb"])

        pos, euler, gripper_state = agent.predict(observation)
        last_action = np.concatenate([pos, euler])
        quat = euler_to_quaternion(*euler)
        _, action = env.robot.get_qpos_from_ee_pos(physics=env.physics, pos=pos, quat=quat)
        action = np.concatenate([action, gripper_state])

        timestep = env.step(action)
        if timestep.last():
            success = True
            break

        # 顺序条件：调用 is_met() 推进 SequentialConditionSet 内部锁状态
        # 不基于 is_met 提前 break，跑满 max_episode_length 后在末尾统一统计每个 condition
        if hasattr(env.task, 'conditions') and env.task.conditions is not None:
            env.task.conditions.is_met(env.physics)

    # 收集各条件的最终结果
    condition_results = {}
    if hasattr(env.task, 'conditions') and env.task.conditions is not None:
        for cond in env.task.conditions.conditions:
            cond_name = type(cond).__name__
            condition_results[cond_name] = cond.is_met(env.physics)

    env.close()
    return success, i + 1, condition_results, frames_to_save


def main():
    import argparse

    print("[DEBUG] main() 函数开始", flush=True)
    print(f"[DEBUG] VLABENCH_ROOT={os.environ.get('VLABENCH_ROOT')}", flush=True)

    parser = argparse.ArgumentParser(description="Evaluate OpenPI pi0 model on VLABench tasks")
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--replan_steps", default=5, type=int)
    parser.add_argument("--tasks", nargs='+', required=True, help="Task names to evaluate")
    parser.add_argument("--n_episode", default=1, type=int, help="Number of episodes per task")
    parser.add_argument("--save_dir", default=None, type=str, help="Directory to save results and videos")
    parser.add_argument("--visualization", action="store_true", default=False)
    parser.add_argument("--max_episode_length", default=200, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    # 读取 task_config.json 获取 max_episode_length
    vlabench_root = os.environ.get("VLABENCH_ROOT")
    task_configs = {}
    if vlabench_root:
        config_path = os.path.join(vlabench_root, "configs/task_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                task_configs = json.load(f)

    # 连接 OpenPI server
    print("[DEBUG] 正在连接 OpenPI server...", args.host, args.port, flush=True)
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print("[DEBUG] OpenPI server 连接成功", flush=True)
    agent = Pi0(client=client, replan_steps=args.replan_steps)
    print("[DEBUG] Pi0 agent 创建成功", flush=True)

    all_results = {}

    for task in args.tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task} ({args.n_episode} episodes)")
        print(f"{'='*60}")

        # 获取该任务的 max_episode_length
        max_ep_len = args.max_episode_length
        if task_configs:
            config_key = find_key_by_value(name2config, task)
            if config_key and task_configs.get(config_key, {}).get("evaluation", {}).get("max_episode_length"):
                max_ep_len = task_configs[config_key]["evaluation"]["max_episode_length"]

        episode_data = []
        print(f"[DEBUG] 开始评测 {task}, n_episode={args.n_episode}", flush=True)
        for ep_i in tqdm(range(args.n_episode), desc=f"Evaluating {task}"):
            print(f"[DEBUG] Episode {ep_i}: agent.reset() 开始", flush=True)
            agent.reset()
            print(f"[DEBUG] Episode {ep_i}: agent.reset() 完成", flush=True)
            try:
                print(f"[DEBUG] Episode {ep_i}: load_env 开始", flush=True)
                # 检查机器人注册状态
                from VLABench.utils.register import register
                print(f"[DEBUG] Episode {ep_i}: 已注册的机器人: {list(register._robots.keys())}", flush=True)
                print(f"[DEBUG] Episode {ep_i}: 已注册的任务数: {len(register._tasks)}", flush=True)
                print(f"[DEBUG] Episode {ep_i}: 'lift_beaker' in tasks: {'lift_beaker' in register._tasks}", flush=True)
                np.random.seed(args.seed + ep_i)
                random.seed(args.seed + ep_i)
                env = load_env(task, random_init=True, eval=False, run_mode="eval")
                print(f"[DEBUG] Episode {ep_i}: load_env 完成", flush=True)
                success, steps, cond_results, frames = run_episode(
                    env, agent, max_episode_length=max_ep_len, save_video=args.visualization
                )
                # 保存视频
                if args.visualization and args.save_dir and frames:
                    video_dir = os.path.join(args.save_dir, task, "videos")
                    os.makedirs(video_dir, exist_ok=True)
                    frames_grid = [np.vstack([np.hstack(f[:2]), np.hstack(f[2:4])]) for f in frames]
                    video_path = os.path.join(video_dir, f"ep{ep_i}_success_{success}_steps_{steps}.mp4")
                    mediapy.write_video(video_path, frames_grid, fps=10)
                    print(f"  Video saved to {video_path}")
            except Exception as e:
                print(f"  Episode {ep_i} error: {e}")
                traceback.print_exc()
                success, steps, cond_results = False, 0, {}
            episode_data.append({
                "episode": ep_i,
                "success": success,
                "steps": steps,
                "conditions": cond_results,
            })
            # 每个 episode 完成后打印简短结果
            cond_summary = ", ".join(
                f"{name}={'OK' if met else 'FAIL'}" for name, met in cond_results.items()
            ) if cond_results else "no conditions"
            print(f"  Episode {ep_i}: success={success}, steps={steps}, conditions=[{cond_summary}]")

        # 汇总该任务的结果
        print(f"\n--- {task} Summary ---")

        # 收集所有 condition 名称
        all_cond_names = set()
        for ep in episode_data:
            all_cond_names.update(ep["conditions"].keys())
        all_cond_names = sorted(all_cond_names)

        if not all_cond_names:
            print("  No conditions defined for this task.")
            task_score = 100.0 if all(ep["success"] for ep in episode_data) else 0.0
        else:
            # 打印每个 condition 的通过率
            for cond_name in all_cond_names:
                met_count = sum(1 for ep in episode_data if ep["conditions"].get(cond_name, False))
                rate = met_count / len(episode_data) * 100
                print(f"  {cond_name}: {met_count}/{len(episode_data)} ({rate:.1f}%)")

            # 最终打分：每个 episode 通过的 condition 均值 / 总 condition 数 * 100
            n_total = len(all_cond_names)
            avg_met = np.mean([
                sum(1 for v in ep["conditions"].values() if v)
                for ep in episode_data
            ])
            task_score = avg_met / n_total * 100

        success_rate = np.mean([ep["success"] for ep in episode_data]) * 100
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Condition score: {task_score:.1f}%")

        all_results[task] = {
            "success_rate": success_rate,
            "condition_score": task_score,
            "per_condition": {
                cond: sum(1 for ep in episode_data if ep["conditions"].get(cond, False)) / len(episode_data)
                for cond in all_cond_names
            },
            "episodes": episode_data,
        }

    # 保存结果
    if args.save_dir:
        result_path = os.path.join(args.save_dir, "evaluation_result.json")
        with open(result_path, "w") as f:
            json.dump(all_results, f, indent=4, default=str)
        print(f"\nResults saved to {result_path}")

    # 打印总结
    print(f"\n{'='*60}")
    print("Overall Results")
    print(f"{'='*60}")
    for task, result in all_results.items():
        print(f"  {task}: success_rate={result['success_rate']:.1f}%, condition_score={result['condition_score']:.1f}%")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
