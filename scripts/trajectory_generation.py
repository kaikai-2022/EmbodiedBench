"""
The scripts to launch auto scene load and key-point based trajectory generation.
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import json
import numpy as np
import os
import types
import open3d as o3d
import mediapy
import argparse
import traceback
import time
from dm_control import viewer
from tqdm import tqdm
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from VLABench.robots import Franka
from VLABench.tasks import *
from VLABench.utils.data_utils import save_single_data, process_observations
from VLABench.utils.utils import find_key_by_value, get_logger
from VLABench.envs import load_env
from VLABench.utils.skill_lib import SkillLib
from VLABench.configs import name2config

def get_args():
    parser = argparse.ArgumentParser(description='Generate trajectory for a task')
    parser.add_argument('--task-name', default="select_poker", type=str, help='task name')
    parser.add_argument('--record-video', action='store_true', default=True, help='record video')
    parser.add_argument('--save-dir', default="/media/shiduo/LENOVO_USB_HDD/dataset/VLABench")
    parser.add_argument('--n-sample', default=1, type=int, help='number of samples to generate')
    parser.add_argument('--start-id', default=0, type=int, help='start index for data storage')
    parser.add_argument('--robot', default="franka", type=str, help='robot name')
    parser.add_argument('--debug', action="store_true", default=False, help='debug mode')
    parser.add_argument('--early-stop', action="store_true", default=False, help='whether use early stop when skill failed to carry out')
    parser.add_argument('--max-episode', default=100, type=int, help='max episode number in the directory')
    parser.add_argument('--eval-unseen', default=False, action="store_true", help='evaluate unseen object categories')
    args = parser.parse_args()
    return args

def get_all_hdf5_files(directory):
    hdf5_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.hdf5'):
                hdf5_files.append(os.path.join(root, file))
    return hdf5_files

def generate_trajectory(args, index, logger):
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"[TIMING] generate_trajectory 开始 (index={index})")
    print(f"{'='*60}")

    print(f"[TIMING] 加载环境...")
    t1 = time.time()
    env = load_env(args.task_name, robot=args.robot, eval=args.eval_unseen)
    print(f"[TIMING] load_env 完成, 耗时 {time.time()-t1:.1f}s")

    # Disable grasp lock for normal physics simulation
    env.disable_grasp_lock()
    print(f"[DEBUG] 已禁用 grasp_lock 模式")

    t1 = time.time()
    env.reset()
    print(f"[TIMING] env.reset() 完成, 耗时 {time.time()-t1:.1f}s")

    # 技能执行模式：避免 step 中 should_terminate_episode 触发 reset 导致重入死循环
    env._skill_execution_mode = True

    # 方案 A Patch：录制期间完全屏蔽终止判定
    # 让所有 skill 完整跑完，最后再用 conditions.is_met(physics) 决定 task_success
    # 避免三条件 AND 在 insert_to_entity RRT 中途碰巧成立时截断 trajectory
    _orig_should_terminate = env.task.should_terminate_episode
    def _never_terminate(physics):
        return False
    env.task.should_terminate_episode = _never_terminate
    print(f"[DEBUG] 已 patch should_terminate_episode → 录制期强制 False")

    # 初始化顺序条件评测（与 test_simulation_only.py 一致）：
    # 1) 重置条件锁 2) 记录初始状态 3) 挂载逐帧回调
    if hasattr(env.task, 'conditions') and env.task.conditions is not None:
        if hasattr(env.task.conditions, 'reset_locks'):
            env.task.conditions.reset_locks()
        for condition in env.task.conditions.conditions:
            if hasattr(condition, 'record_initial_state'):
                condition.record_initial_state(env.physics)
        if not hasattr(env.task, '_per_step_condition_callbacks'):
            env.task._per_step_condition_callbacks = []
        if env.task.conditions.is_met not in env.task._per_step_condition_callbacks:
            env.task._per_step_condition_callbacks.append(env.task.conditions.is_met)

    episode_config = env.save()

    # load key prior information and task specific variables
    target_entity = env.task.config_manager.target_entity
    instruction = env.task.get_instruction()
    meta_info = dict(
        target_entity=[target_entity],
        entities=list(env.task.entities.keys()),
        instruction=[instruction],
    )
    print(f"[DEBUG] target_entity={target_entity}, entities={list(env.task.entities.keys())}")

    # register the expert sequence
    t1 = time.time()
    skill_seq = env.get_expert_skill_sequence()
    print(f"[TIMING] get_expert_skill_sequence 完成, 耗时 {time.time()-t1:.1f}s")
    print(f"[DEBUG] 技能序列共 {len(skill_seq)} 个技能")

    # start auto trajectory generation
    observations, waypoints= [], []
    task_success = False
    if skill_seq is not None: # normal case
        for skill_idx, skill in enumerate(skill_seq):
            skill_name = skill.func.__name__
            skill_params = {k: v for k, v in skill.keywords.items() if k != 'env'}
            print(f"\n[TIMING] ===== 技能 {skill_idx+1}/{len(skill_seq)}: {skill_name} =====")
            print(f"[DEBUG] 参数: {skill_params}")
            t_skill = time.time()
            obs, waypoint, stage_success, skill_task_success = skill(env)
            elapsed = time.time() - t_skill
            print(f"[TIMING] 技能 {skill_name} 完成, 耗时 {elapsed:.1f}s")
            print(f"[DEBUG] stage_success={stage_success}, skill_task_success={skill_task_success}")
            print(f"[DEBUG] 生成 {len(obs)} 个观测, {len(waypoint)} 个路径点")

            # 打印试管当前位置
            try:
                target = env.task.entities[env.task.target_entity]
                tube_pos = env.physics.bind(target.mjcf_model.worldbody).xpos
                print(f"[DEBUG] 试管当前位置: {tube_pos}")
                # 打印试管架位置
                for key, entity in env.task.entities.items():
                    if "tube_stand" in key:
                        stand_pos = entity.get_xpos(env.physics)
                        print(f"[DEBUG] 试管架位置: {stand_pos}")
                        break
                # 打印 _tube_has_left_stand
                if hasattr(env.task, '_tube_has_left_stand'):
                    print(f"[DEBUG] _tube_has_left_stand={env.task._tube_has_left_stand}")
            except Exception as e:
                print(f"[DEBUG] 获取位置信息失败: {e}")

            if args.debug:
                for o in obs: observations.append(dict(rgb=o["rgb"]))
            else:
                observations.extend(obs)
                waypoints.extend(waypoint)
            if args.early_stop and not stage_success:
                logger.warning(f"{skill} failed, early quit...")
                break
            if skill_task_success:
                task_success = True
                print(f"[DEBUG] 技能返回 task_success=True, 提前退出")
                break

        # 使用条件（conditions）来判断任务是否成功
        print(f"\n[TIMING] 检查任务条件...")
        t1 = time.time()
        if hasattr(env.task, 'conditions') and env.task.conditions is not None:
            try:
                for ci, cond in enumerate(env.task.conditions.conditions):
                    cond_result = cond.is_met(env.physics)
                    print(f"[DEBUG]   Condition {ci} ({type(cond).__name__}): {cond_result}")
                result = env.task.conditions.is_met(env.physics)
                print(f"[DEBUG] Overall Conditions is_met: {result}")
                if result:
                    task_success = True
            except Exception as e:
                print(f"[DEBUG] Error evaluating conditions: {e}")
                import traceback as tb
                print(tb.format_exc())
        else:
            print("[DEBUG] No conditions found")
        print(f"[TIMING] 条件检查完成, 耗时 {time.time()-t1:.1f}s")
    else: # TODO: some special tasks should be handled based on the feedback
        raise NotImplementedError("No expert skill sequence found")

    task_dir = os.path.join(args.save_dir, args.task_name)
    print(f"\n[TIMING] 所有技能执行完成, 总观测数={len(observations)}, task_success={task_success}")
    print(f"[TIMING] 技能执行总耗时: {time.time()-t0:.1f}s")

    try:
        if args.record_video:
            print(f"[TIMING] 开始生成视频 ({len(observations)} 帧)...")
            t1 = time.time()
            frames = []
            for o in observations:
                frames.append(np.vstack([np.hstack(o["rgb"][:2]), np.hstack(o["rgb"][2:4])]))
            if not os.path.exists(task_dir):
                os.makedirs(task_dir)
            video_path = os.path.join(task_dir, f"demo_{index}_success_{task_success}.mp4")
            mediapy.write_video(video_path, frames, fps=10)
            print(f"[TIMING] 视频生成完成, 耗时 {time.time()-t1:.1f}s, 路径: {video_path}")
        if not task_success:
            logger.warning("Task failed, skip saving data")
            print(f"[TIMING] generate_trajectory 结束 (失败), 总耗时: {time.time()-t0:.1f}s")
            return
        else:
            logger.info("Task success, saving data")

        # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data_to_save = process_observations(observations)

        robot_position = env.robot.robot_config["position"]
        robot_frame_waypoints = [np.array(waypoint) - np.concatenate([robot_position, np.zeros(5)]) for waypoint in waypoints]
        data_to_save["trajectory"] = robot_frame_waypoints
        data_to_save["entities"] = meta_info["entities"]
        data_to_save["target_entity"] = meta_info["target_entity"]
        data_to_save["episode_config"] = json.dumps(episode_config)
        data_to_save["instruction"] =meta_info["instruction"]
        save_single_data(data_to_save,
                         save_dir=task_dir,
                         filename=f"data_{index}.hdf5",
                         )
    finally:
        # 无论任务成功/失败都恢复 patch，避免影响后续 episode
        try:
            env.task.should_terminate_episode = _orig_should_terminate
        except Exception:
            pass
        try:
            env._skill_execution_mode = False
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass

        
if __name__ == "__main__":
    args = get_args()

    # 动态注册 _series 后缀任务：触发 ConfigManager/Task 的 @register 装饰器
    # 与 test_simulation_only.py 的处理逻辑一致：
    # autogen_tasks 默认 __init__ 不会递归加载 primitive/ 子目录下的 series 文件，
    # 因此需要在脚本入口显式 importlib 把对应文件加载进来，再注入 name2config。
    if args.task_name.endswith("_series"):
        import importlib.util as _ilu
        _vlabench_root = os.environ.get("VLABENCH_ROOT") or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "VLABench"
        )
        _series_path = os.path.join(
            _vlabench_root, "tasks", "autogen_tasks", "primitive", f"{args.task_name}.py"
        )
        if not os.path.exists(_series_path):
            _series_path = os.path.join(
                _vlabench_root, "tasks", "autogen_tasks", f"{args.task_name}.py"
            )
        if os.path.exists(_series_path):
            _spec = _ilu.spec_from_file_location(args.task_name, _series_path)
            _mod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            _base_name = args.task_name[: -len("_series")]
            from VLABench.configs import name2config as _n2c
            from VLABench.utils.register import register as _register
            import VLABench.envs as _envs_mod
            _n2c[args.task_name] = [_base_name]
            _envs_mod.name2config[args.task_name] = [_base_name]
            # series 名也指向同一个 Task 类，让 load_env(series_name) 可解析
            if _base_name in _register._tasks and args.task_name not in _register._tasks:
                _register._tasks[args.task_name] = _register._tasks[_base_name]
            if _base_name in _register._config_managers and args.task_name not in _register._config_managers:
                _register._config_managers[args.task_name] = _register._config_managers[_base_name]
            print(f"✓ 动态注册 series: {args.task_name} -> {_base_name}")
        else:
            print(f"[WARN] 找不到 series 文件: {args.task_name}")

    logger = get_logger()
    for i in tqdm(range(args.n_sample)):
        i += args.start_id
        try:
            h5_files = get_all_hdf5_files(os.path.join(args.save_dir, args.task_name))
            if len(h5_files) >= args.max_episode:
                logger.info(f"Task {args.task_name} has reached the maximum episode number, skip")
                break
            # Skip if this index already exists
            existing_files = [f for f in h5_files if f"data_{i}." in f]
            if existing_files:
                logger.info(f"Index {i} already exists, skipping")
                continue
            generate_trajectory(args, i, logger)
        except Exception as e:
            err = traceback.TracebackException.from_exception(e)
            print("".join(err.format()))
            continue