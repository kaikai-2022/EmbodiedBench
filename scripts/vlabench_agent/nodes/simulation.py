"""
Simulation Node - 在 MuJoCo 中执行物理仿真

加载注册好的任务，执行专家技能序列，收集观测数据和轨迹，
验证任务是否成功完成。
"""

import json
import logging
import os
import signal
import traceback
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# 设置 MuJoCo 渲染后端
os.environ.setdefault("MUJOCO_GL", "egl")

# 单个技能执行超时（秒）
SKILL_TIMEOUT = 300  # 5 分钟
# 整个仿真超时（秒）
SIMULATION_TIMEOUT = 600  # 10 分钟


class SkillTimeoutError(Exception):
    """技能执行超时"""
    pass


def _timeout_handler(signum, frame):
    raise SkillTimeoutError("技能执行超时")


def extract_skill_sequence(skill_seq, entity_names: List[str]) -> List[Dict]:
    """
    从 partial 对象列表中提取 operation_sequence 格式的技能序列。

    Args:
        skill_seq: list of functools.partial objects
        entity_names: 场景中的实体名称列表（用于映射到索引）

    Returns:
        [{"name": "pick", "params": {"target_entity_name": 1}}, ...]
    """
    # 构建名称到索引的映射
    name_to_index = {}
    for i, name in enumerate(entity_names):
        name_to_index[name] = i

    sequence = []
    for skill in skill_seq:
        name = skill.func.__name__
        params = {}

        keywords = skill.keywords if hasattr(skill, "keywords") else {}

        if "target_entity_name" in keywords:
            entity_name = keywords["target_entity_name"]
            params["target_entity_name"] = name_to_index.get(entity_name, entity_name)

        if "target_container_name" in keywords:
            container_name = keywords["target_container_name"]
            params["target_container_name"] = name_to_index.get(container_name, container_name)

        sequence.append({"name": name, "params": params})

    return sequence


def simulation_node(state: Dict) -> Dict:
    """
    仿真节点 - 在 MuJoCo 中执行技能序列

    复用 trajectory_generation.py 的核心逻辑：
    1. 加载环境
    2. 获取专家技能序列
    3. 逐个执行技能
    4. 保存数据和视频

    Args:
        state: VLABenchAgentState

    Returns:
        更新后的状态字典
    """
    logger.info("=" * 60)
    logger.info("[Simulation] 开始 MuJoCo 物理仿真...")

    task_analysis = state.get("task_analysis", {})
    task_name = task_analysis.get("task_name", "custom_task")

    try:
        # 1. 加载环境（需要先导入 robots 和 tasks 以触发 @register 装饰器注册）
        import importlib
        importlib.import_module("VLABench.robots")  # 触发 @register.add_robot
        importlib.import_module("VLABench.tasks.hierarchical_tasks.primitive")  # 触发 @register.add_task
        importlib.import_module("VLABench.tasks.hierarchical_tasks.composite")
        from VLABench.envs import load_env

        logger.info(f"[Simulation] 加载环境: {task_name}")
        env = load_env(task_name, robot="franka")

        # 2. 保存环境配置
        episode_config = env.save()
        logger.info("[Simulation] ✓ 环境加载完成")

        # 3. 获取任务元信息
        target_entity = env.task.config_manager.target_entity
        instruction = env.task.get_instruction()
        entities = list(env.task.entities.keys())

        logger.info(f"[Simulation]   目标实体: {target_entity}")
        logger.info(f"[Simulation]   指令: {instruction}")
        logger.info(f"[Simulation]   场景实体: {entities}")

        # 4. 获取专家技能序列
        skill_seq = env.get_expert_skill_sequence()
        if skill_seq is None:
            raise RuntimeError("get_expert_skill_sequence() 返回 None，任务类未正确实现该方法")

        logger.info(f"[Simulation]   技能序列: {[s.func.__name__ for s in skill_seq]}")

        # 提取 operation_sequence 格式
        # 构建完整的实体名称列表（包含 components 中的固定实体）
        component_names = []
        if hasattr(env.task, "config_manager") and hasattr(env.task.config_manager, "config"):
            for comp in env.task.config_manager.config.get("task", {}).get("components", []):
                component_names.append(comp.get("name", ""))

        executed_skill_sequence = extract_skill_sequence(skill_seq, component_names or entities)

        # 5. 执行技能序列（带超时保护）
        observations, waypoints = [], []
        task_success = False

        # 设置整体仿真超时
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(SIMULATION_TIMEOUT)

        try:
            for skill_idx, skill in enumerate(skill_seq):
                skill_name = skill.func.__name__
                logger.info(f"[Simulation]   执行技能 {skill_idx + 1}/{len(skill_seq)}: {skill_name}")

                # 每个技能重置超时
                signal.alarm(SKILL_TIMEOUT)

                obs, waypoint, stage_success, skill_task_success = skill(env)
                observations.extend(obs)
                waypoints.extend(waypoint)

                if not stage_success:
                    logger.warning(f"[Simulation]   ⚠ 技能 {skill_name} 执行失败")
                    break

                if skill_task_success:
                    task_success = True
                    break
        except SkillTimeoutError:
            logger.error(f"[Simulation]   ✗ 技能执行超时 (>{SKILL_TIMEOUT}s)")
        finally:
            signal.alarm(0)  # 取消超时
            signal.signal(signal.SIGALRM, old_handler)  # 恢复原处理器

        # 6. 检查任务条件
        if not task_success and hasattr(env.task, "conditions") and env.task.conditions is not None:
            try:
                result = env.task.conditions.is_met(env.physics)
                if result:
                    task_success = True
                    logger.info("[Simulation]   ✓ 任务条件满足")
            except Exception as e:
                logger.warning(f"[Simulation]   ⚠ 条件检查异常: {e}")

        logger.info(f"[Simulation]   任务结果: {'成功' if task_success else '失败'}")

        # 7. 保存数据
        vlabench_root = os.environ.get("VLABENCH_ROOT")
        project_root = Path(vlabench_root).parent if vlabench_root else Path(".")
        save_dir = project_root / "dataset" / "training_data"
        task_dir = save_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        # 保存视频
        video_path = None
        if observations:
            try:
                import mediapy
                frames = []
                for o in observations:
                    if "rgb" in o and len(o["rgb"]) >= 4:
                        frames.append(np.vstack([
                            np.hstack(o["rgb"][:2]),
                            np.hstack(o["rgb"][2:4])
                        ]))
                if frames:
                    video_path = str(task_dir / f"demo_0_success_{task_success}.mp4")
                    mediapy.write_video(video_path, frames, fps=10)
                    logger.info(f"[Simulation]   ✓ 视频保存: {video_path}")
            except Exception as e:
                logger.warning(f"[Simulation]   ⚠ 视频保存失败: {e}")

        # 保存 HDF5 数据（仅在成功时）
        hdf5_path = None
        if task_success and observations:
            try:
                from VLABench.utils.data_utils import save_single_data, process_observations

                data_to_save = process_observations(observations)
                robot_position = env.robot.robot_config["position"]
                robot_frame_waypoints = [
                    np.array(wp) - np.concatenate([robot_position, np.zeros(5)])
                    for wp in waypoints
                ]

                data_to_save["trajectory"] = robot_frame_waypoints
                data_to_save["entities"] = entities
                data_to_save["target_entity"] = [target_entity]
                data_to_save["episode_config"] = json.dumps(episode_config)
                data_to_save["instruction"] = [instruction]

                hdf5_path = str(task_dir / "data_0.hdf5")
                save_single_data(data_to_save, save_dir=str(task_dir), filename="data_0.hdf5")
                logger.info(f"[Simulation]   ✓ HDF5 数据保存: {hdf5_path}")
            except Exception as e:
                logger.warning(f"[Simulation]   ⚠ HDF5 保存失败: {e}")

        env.close()

        if task_success:
            logger.info("[Simulation] ✓ 仿真成功完成")
            logger.info("=" * 60 + "\n")

            return {
                "simulation_success": True,
                "simulation_video_path": video_path,
                "simulation_hdf5_path": hdf5_path,
                "episode_config": episode_config,
                "executed_skill_sequence": executed_skill_sequence,
                "error_feedback": None,
                "current_stage": "vlm_data",
            }
        else:
            logger.warning("[Simulation] ✗ 仿真失败 - 任务未完成")
            logger.info("=" * 60 + "\n")

            return {
                "simulation_success": False,
                "simulation_video_path": video_path,
                "episode_config": episode_config,
                "executed_skill_sequence": executed_skill_sequence,
                "error_feedback": (
                    f"仿真执行完毕但任务未成功完成。\n"
                    f"执行的技能: {[s.func.__name__ for s in skill_seq]}\n"
                    f"请检查 get_expert_skill_sequence 中的技能参数是否正确，"
                    f"特别是 target_entity_name 和 target_container_name 是否与实际实体名称匹配。"
                ),
                "current_stage": "simulation",
            }

    except Exception as e:
        error_tb = traceback.format_exc()
        logger.error(f"[Simulation] ✗ 仿真异常: {e}")
        logger.error(error_tb)

        return {
            "simulation_success": False,
            "error_feedback": f"仿真异常:\n{error_tb}",
            "current_stage": "simulation",
        }
