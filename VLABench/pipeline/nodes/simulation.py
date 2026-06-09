"""
Simulation Node - 在 MuJoCo 中执行物理仿真

加载注册好的任务，执行专家技能序列，收集观测数据和轨迹，
验证任务是否成功完成。
"""

import json
import logging
import os
import signal
import time
import traceback
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from .node_logger import log_node_output_file

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


def _save_video(observations, task_dir, task_success):
    """保存视频，无论仿真成功或失败都调用"""
    if not observations:
        return None
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
            task_dir.mkdir(parents=True, exist_ok=True)
            video_path = str(task_dir / f"demo_0_success_{task_success}.mp4")
            mediapy.write_video(video_path, frames, fps=10)
            logger.info(f"[Simulation]   ✓ 视频保存: {video_path}")
            return video_path
    except Exception as e:
        logger.warning(f"[Simulation]   ⚠ 视频保存失败: {e}")
    return None


def extract_skill_sequence(skill_seq, entity_names: List[str]) -> List[Dict]:
    """
    从 partial 对象列表中提取 operation_sequence 格式的技能序列。

    Args:
        skill_seq: list of functools.partial objects
        entity_names: 场景中的实体名称列表（用于映射到索引）

    Returns:
        [{"name": "pick", "params": {"target_entity_name": "small_beaker_0"}}, ...]
        注意：直接返回实体名称字符串，不做索引转换，避免映射错误
    """
    sequence = []
    for skill in skill_seq:
        name = skill.func.__name__
        params = {}

        keywords = skill.keywords if hasattr(skill, "keywords") else {}

        # 直接使用实体名称字符串，不做索引映射
        if "target_entity_name" in keywords:
            params["target_entity_name"] = keywords["target_entity_name"]

        if "target_container_name" in keywords:
            params["target_container_name"] = keywords["target_container_name"]

        sequence.append({"name": name, "params": params})

    return sequence


# ========== Per-Step Condition Checking ==========
# 数值参数键集合（这些参数不做 entity name 解析）
NUMERIC_PARAM_KEYS = {
    "positions", "target_pos_range", "orientations",
    "duration", "xy_tolerance", "target_height", "tolerance_distance",
    "tolerance_angle", "dimension", "offset", "threshold", "check_axes",
    "layer", "tilt_angle", "wait_time", "insert_depth", "lift_height",
    "push_distance", "rotation_angle", "gripper_state",
}


def _resolve_condition_params(params: Dict, entities_dict: Dict, robot) -> Dict:
    """
    将 condition params 中的字符串 entity name 解析为实际的 Entity 对象。

    Args:
        params: condition 配置中的参数字典，包含 entity name 字符串
        entities_dict: env.task.entities 字典，key 为 entity name
        robot: robot Entity 对象

    Returns:
        解析后的参数字典，entity name 字符串已被替换为 Entity 对象
    """
    from VLABench.utils.register import register
    resolved = {}
    for k, v in params.items():
        if k == "robot":
            resolved[k] = robot
        elif k in NUMERIC_PARAM_KEYS:
            resolved[k] = v
        elif isinstance(v, str):
            # 字符串可能是 entity name，尝试解析
            resolved[k] = entities_dict.get(v, v)
        elif isinstance(v, list):
            resolved[k] = [
                entities_dict.get(item, item) if isinstance(item, str) else item
                for item in v
            ]
        else:
            resolved[k] = v
    return resolved


def _create_condition(step_id: int, condition_entry: Dict, entities_dict: Dict, robot, physics):
    """
    创建 condition 实例并记录初始状态（技能执行前调用）。

    Returns:
        (condition_instance, condition_type) or (None, "pass")
    """
    from VLABench.utils.register import register

    cond_type = condition_entry.get("condition_type", "pass")
    if cond_type == "pass":
        return None, "pass"

    params = condition_entry.get("params", {})
    try:
        condition_cls = register.load_condition(cond_type)
        resolved_params = _resolve_condition_params(params, entities_dict, robot)
        condition = condition_cls(**resolved_params)
        condition.record_initial_state(physics)
        return condition, cond_type
    except Exception as e:
        logger.warning(f"[Simulation]   ⚠ Step {step_id} condition creation error: {e}")
        return None, cond_type


def _evaluate_condition(step_id: int, condition, cond_type: str, physics) -> Dict:
    """
    检查 condition 是否满足（技能执行后调用）。

    Returns:
        {"step_id": int, "condition_type": str, "met": bool, "error": str or None}
    """
    if condition is None:
        return {"step_id": step_id, "condition_type": cond_type, "met": True, "error": None}

    # 如果 callback 在执行期间已经设置了 _met 标志，说明条件已满足
    if hasattr(condition, '_met') and condition._met:
        return {"step_id": step_id, "condition_type": cond_type, "met": True, "error": None}

    try:
        met = condition.is_met(physics)
        return {"step_id": step_id, "condition_type": cond_type, "met": met, "error": None}
    except Exception as e:
        logger.warning(f"[Simulation]   ⚠ Step {step_id} condition check error: {e}")
        return {"step_id": step_id, "condition_type": cond_type, "met": False, "error": str(e)}


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

    # 日志输出 condition_plan（方便调试）
    condition_plan = state.get("condition_plan")
    if condition_plan:
        logger.info(f"[Simulation] 接收 condition_plan: {len(condition_plan)} 个 conditions")
        for cp in condition_plan:
            logger.info(f"  Step {cp.get('step_id')}: {cp.get('condition_type')} - params={cp.get('params', {})}")
    else:
        logger.info("[Simulation] 未收到 condition_plan（或为 None），将跳过 per-step condition 检查")

    # 确保 VLABENCH_ROOT 环境变量已设置（assets 在 VLABench 子目录下）
    if not os.environ.get("VLABENCH_ROOT"):
        os.environ["VLABENCH_ROOT"] = "/ssd/mkqin/workspace/VLABench/VLABench"

    task_analysis = state.get("task_analysis", {})
    task_name = task_analysis.get("task_name", "custom_task").replace(" ", "_")
    observations = []  # 提前初始化，确保异常路径也能保存视频

    try:
        # 1. 加载环境（需要先导入 robots 和 tasks 以触发 @register 装饰器注册）
        import importlib
        importlib.import_module("VLABench.robots")  # 触发 @register.add_robot
        importlib.import_module("VLABench.tasks.autogen_tasks")  # 触发 @register.add_task
        importlib.import_module("VLABench.tasks.hierarchical_tasks.composite")
        from VLABench.envs import load_env

        # 动态加载 series 文件（由 code_generator 生成，注册 task_name task）
        vlabench_root = os.environ.get("VLABENCH_ROOT", "/ssd/mkqin/workspace/VLABench/VLABench")
        series_path = os.path.join(
            os.path.dirname(vlabench_root),
            "VLABench", "tasks", "autogen_tasks",
            f"{task_name}_series.py"
        )
        if os.path.exists(series_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location(f"{task_name}_series", series_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            logger.info(f"[Simulation] ✓ 已加载 series 文件: {series_path}")
        else:
            logger.warning(f"[Simulation] ⚠ series 文件不存在: {series_path}")

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
        waypoints = []
        task_success = False

        # 获取 condition_plan 和 skill_plan
        condition_plan = state.get("condition_plan", [])
        skill_plan = state.get("skill_plan", {})
        global_skill_plan = skill_plan.get("global_skill_plan", [])

        # 初始化 per-step condition 检查结果
        step_condition_results = []

        # ========== 时间戳记录初始化 ==========
        video_start_time = time.time()  # 视频开始录制的绝对时间
        step_timestamps = []  # 记录每个大 step 和原子操作的时间戳

        # 构建 step_id -> skills 数量的映射
        # atomic_sequence 中的 skill 数量决定何时检查 condition
        step_skill_counts = []
        step_skill_ends = []  # 累计索引，用于判断某 step 的 skills 何时完成
        step_skill_starts = []  # 每个 step 的起始 skill 索引
        for step in global_skill_plan:
            num_skills = len(step.get("atomic_sequence", []))
            step_skill_counts.append(num_skills)
            if step_skill_ends:
                step_skill_ends.append(step_skill_ends[-1] + num_skills)
            else:
                step_skill_ends.append(num_skills)

        # 计算每个 step 的起始索引
        for i, count in enumerate(step_skill_counts):
            if i == 0:
                step_skill_starts.append(0)
            else:
                step_skill_starts.append(step_skill_starts[-1] + step_skill_counts[i - 1])

        # 存储 step_idx -> (condition_instance, condition_type) 的映射
        step_conditions = {}

        logger.info(f"[Simulation]   共有 {len(global_skill_plan)} 个 steps, {len(skill_seq)} 个 skills")
        logger.info(f"[Simulation]   Step skill counts: {step_skill_counts}, starts: {step_skill_starts}, ends: {step_skill_ends}")

        # 设置整体仿真超���
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(SIMULATION_TIMEOUT)

        # 技能执行期间禁用自动 reset，让技能能完整执行完所有动作
        env._skill_execution_mode = True

        try:
            current_atomic_timestamps = []
            # 当前 step 挂载的 condition callback（用于逐帧调用 is_met）
            active_condition_callback = None

            for skill_idx, skill in enumerate(skill_seq):
                skill_name = skill.func.__name__

                # step 开始前：创建 condition 实例并记录初始状态，挂载逐帧 callback
                for step_idx, start_idx in enumerate(step_skill_starts):
                    if skill_idx == start_idx:
                        step_condition_entry = None
                        if condition_plan:
                            step_condition_entry = next(
                                (c for c in condition_plan if c.get("step_id") == step_idx),
                                None
                            )
                        if step_condition_entry:
                            cond_instance, cond_type = _create_condition(
                                step_idx, step_condition_entry,
                                env.task.entities, env.robot, env.physics
                            )
                            step_conditions[step_idx] = (cond_instance, cond_type)
                            # 挂载逐帧 callback，让 condition.is_met 在每个 physics step 被调用
                            if cond_instance is not None:
                                if not hasattr(env.task, '_per_step_condition_callbacks'):
                                    env.task._per_step_condition_callbacks = []
                                active_condition_callback = cond_instance.is_met
                                env.task._per_step_condition_callbacks.append(active_condition_callback)
                        break

                logger.info(f"[Simulation]   执行技能 {skill_idx + 1}/{len(skill_seq)}: {skill_name}")

                # 记录原子操作开始时间
                atomic_start = time.time() - video_start_time

                # 每个技能重置超时
                signal.alarm(SKILL_TIMEOUT)

                obs, waypoint, stage_success, skill_task_success = skill(env)
                observations.extend(obs)
                waypoints.extend(waypoint)

                # 记录原子操作结束时间
                atomic_end = time.time() - video_start_time
                current_atomic_timestamps.append({
                    "atomic_idx": skill_idx,
                    "start": atomic_start,
                    "end": atomic_end
                })

                if not stage_success:
                    logger.warning(f"[Simulation]   ⚠ 技能 {skill_name} 执行失败")
                    break

                if skill_task_success:
                    task_success = True
                    break

                # step 结束后：卸载逐帧 callback，评估 condition
                for step_idx, end_idx in enumerate(step_skill_ends):
                    if skill_idx == end_idx - 1:
                        # 卸载逐帧 callback
                        if active_condition_callback is not None:
                            env.task._per_step_condition_callbacks.remove(active_condition_callback)
                            active_condition_callback = None

                        step_end = time.time() - video_start_time
                        step_timestamps.append({
                            "step_id": step_idx,
                            "atomic_timestamps": current_atomic_timestamps,
                            "step_end": step_end
                        })
                        current_atomic_timestamps = []

                        if step_idx in step_conditions:
                            cond_instance, cond_type = step_conditions[step_idx]
                            result = _evaluate_condition(
                                step_idx, cond_instance, cond_type, env.physics
                            )
                            step_condition_results.append(result)
                            status_icon = "✓" if result["met"] else "✗"
                            logger.info(
                                f"[Simulation]   Step {step_idx} condition '{result['condition_type']}': "
                                f"{status_icon} (skill_idx={skill_idx})"
                            )
                        break
        except SkillTimeoutError:
            logger.error(f"[Simulation]   ✗ 技能执行超时 (>{SKILL_TIMEOUT}s)")
        finally:
            signal.alarm(0)  # 取消超时
            signal.signal(signal.SIGALRM, old_handler)  # 恢复原处理器
            env._skill_execution_mode = False  # 恢复自动 reset
            # 确保 condition callback 被清理
            if active_condition_callback is not None and hasattr(env.task, '_per_step_condition_callbacks'):
                try:
                    env.task._per_step_condition_callbacks.remove(active_condition_callback)
                except ValueError:
                    pass

        # 6. Per-step condition 检查结果汇总
        # 如果有 condition_plan，检查是否所有 conditions 都满足
        all_conditions_met = True
        if condition_plan and not step_condition_results:
            # condition_plan 存在但 step_condition_results 为空：
            # 通常是 skill 中途失败 break 跳出，跳过了 condition 评估。
            # 此时不能算"成功"——条件压根没被判断。
            all_conditions_met = False
            logger.warning("[Simulation]   ✗ step conditions 未被评估（skill 中途失败跳出），不能算成功")
        elif condition_plan and step_condition_results:
            failed_conditions = [r for r in step_condition_results if not r["met"]]
            if failed_conditions:
                all_conditions_met = False
                logger.warning(f"[Simulation]   ✗ {len(failed_conditions)} 个 step conditions 未满足:")
                for fc in failed_conditions:
                    error_info = f", error: {fc['error']}" if fc.get("error") else ""
                    logger.warning(f"[Simulation]     - Step {fc['step_id']}: {fc['condition_type']}{error_info}")
            else:
                logger.info(f"[Simulation]   ✓ 所有 {len(step_condition_results)} 个 step conditions 都满足")

        # 综合判定：skill 执行完成 + 所有 conditions 满足
        if not task_success:
            if not condition_plan:
                # 没有 condition_plan 时，回退到原有的条件检查机制
                if hasattr(env.task, "conditions") and env.task.conditions is not None:
                    try:
                        result = env.task.conditions.is_met(env.physics)
                        if result:
                            task_success = True
                            logger.info("[Simulation]   ✓ 任务条件满足（fallback）")
                    except Exception as e:
                        logger.warning(f"[Simulation]   ⚠ 条件检查异常: {e}")
            elif all_conditions_met:
                task_success = True
                logger.info("[Simulation]   ✓ 所有 step conditions 满足，任务成功")

        logger.info(f"[Simulation]   任务结果: {'成功' if task_success else '失败'}")

        # 8. 保存数据
        vlabench_root = os.environ.get("VLABENCH_ROOT")
        project_root = Path(vlabench_root).parent if vlabench_root else Path(".")
        task_dir = project_root / "dataset" / "autogen_tasks" / task_name

        # 保存视频（无论成功或失败都保存）
        video_path = _save_video(observations, task_dir, task_success)

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

            output = {
                "simulation_success": True,
                "simulation_video_path": video_path,
                "simulation_hdf5_path": hdf5_path,
                "episode_config": episode_config,
                "executed_skill_sequence": executed_skill_sequence,
                "step_condition_results": step_condition_results,
                "step_timestamps": step_timestamps,  # 时间戳记录，供 Reviewer 使用
                "error_feedback": None,
                "current_stage": "reviewer",  # 修改为进入 Reviewer 节点
            }
            log_node_output_file("simulation", state, output)
            return output
        else:
            logger.warning("[Simulation] ✗ 仿真失败 - 任务未完成")
            logger.info("=" * 60 + "\n")

            # 构建详细的错误反馈，包含 condition 检查结果
            error_msg = f"仿真执行完毕但任务未成功完成。\n执行的技能: {[s.func.__name__ for s in skill_seq]}\n"
            if condition_plan and step_condition_results:
                failed_conditions = [r for r in step_condition_results if not r["met"]]
                if failed_conditions:
                    error_msg += f"\n未满足的 conditions:\n"
                    for fc in failed_conditions:
                        error_info = f" (error: {fc['error']})" if fc.get("error") else ""
                        error_msg += f"  - Step {fc['step_id']}: {fc['condition_type']}{error_info}\n"
            error_msg += "请检查 get_expert_skill_sequence 中的技能参数是否正确，特别是 target_entity_name 和 target_container_name 是否与实际实体名称匹配。"

            output = {
                "simulation_success": False,
                "simulation_video_path": video_path,
                "episode_config": episode_config,
                "executed_skill_sequence": executed_skill_sequence,
                "step_condition_results": step_condition_results,
                "step_timestamps": step_timestamps,  # 时间戳记录，供 Reviewer 使用
                "error_feedback": error_msg,
                "current_stage": "reviewer",  # 修改为进入 Reviewer 节点
            }
            log_node_output_file("simulation", state, output)
            return output

    except Exception as e:
        error_tb = traceback.format_exc()
        logger.error(f"[Simulation] ✗ 仿真异常: {e}")
        logger.error(error_tb)

        # 即使崩溃也尝试保存已有观测的视频
        video_path = None
        try:
            if observations:
                vlabench_root = os.environ.get("VLABENCH_ROOT")
                project_root = Path(vlabench_root).parent if vlabench_root else Path(".")
                task_dir = project_root / "dataset" / "autogen_tasks" / task_name
                video_path = _save_video(observations, task_dir, False)
        except Exception:
            pass

        try:
            env.close()
        except Exception:
            pass

        output = {
            "simulation_success": False,
            "simulation_video_path": video_path,
            "step_timestamps": step_timestamps if 'step_timestamps' in dir() else [],  # 时间戳记录，供 Reviewer 使用
            "error_feedback": f"仿真异常:\n{error_tb}",
            "current_stage": "reviewer",  # 修改为进入 Reviewer 节点
        }
        log_node_output_file("simulation", state, output)
        return output
