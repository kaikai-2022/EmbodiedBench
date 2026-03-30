"""
Registration Node - 动态注册 LLM 生成的任务类

将生成的 Python 任务文件动态导入，注册到 VLABench 的任务注册表中，
并更新 name2config 和 TASK_CONFIG 使 load_env() 能够找到该任务。
"""

import importlib
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


def registration_node(state: Dict) -> Dict:
    """
    注册节点 - 动态注册生成的任务类

    Args:
        state: VLABenchAgentState

    Returns:
        更新后的状态字典
    """
    logger.info("=" * 60)
    logger.info("[Registration] 开始动态注册任务...")

    task_analysis = state.get("task_analysis", {})
    asset_status = state.get("asset_status", {})
    task_module_path = state.get("task_module_path")
    task_name = task_analysis.get("task_name", "custom_task")
    series_name = f"{task_name}_series"
    module_name = f"VLABench.tasks.hierarchical_tasks.primitive.{task_name}_series"

    if not task_module_path:
        logger.error("[Registration] ✗ 缺少 task_module_path")
        return {
            "registration_success": False,
            "error_feedback": "task_module_path 为空，代码生成步骤可能失败",
            "current_stage": "registration",
        }

    try:
        # 1. 清理旧注册（重试时需要）
        from VLABench.utils.register import register

        if task_name in register._tasks:
            del register._tasks[task_name]
            logger.info(f"[Registration] 清理旧 task 注册: {task_name}")
        if task_name in register._config_managers:
            del register._config_managers[task_name]
            logger.info(f"[Registration] 清理旧 config_manager 注册: {task_name}")

        # 清理旧模块缓存
        if module_name in sys.modules:
            del sys.modules[module_name]
            logger.info(f"[Registration] 清理旧模块缓存: {module_name}")

        # 2. 更新 name2config（内存）
        from VLABench.configs import name2config

        name2config[series_name] = [task_name]
        logger.info(f"[Registration] ✓ 更新 name2config: {series_name} -> [{task_name}]")

        # 3. 更新 TASK_CONFIG（内存 + JSON 文件）
        import VLABench.envs as envs_module

        vlabench_root = os.environ.get("VLABENCH_ROOT")
        scene_name = task_analysis.get("scene", "laboratory") + "_0"

        # 构建 series 配置
        # 使用 canonical_name（name2class_xml 中的注册名）而非 LLM 输出的名称
        objects_canonical = []
        containers_canonical = []

        for obj_name, info in asset_status.items():
            canonical = info.get("canonical_name", obj_name)
            obj_class = info.get("class", "")
            if "Container" in obj_class or "Stand" in obj_class:
                containers_canonical.append(canonical)
            else:
                objects_canonical.append(canonical)

        series_config = {
            "task": {
                "asset": {
                    "seen_object": objects_canonical if objects_canonical else [list(asset_status.values())[0].get("canonical_name", list(asset_status.keys())[0])],
                    "unseen_object": objects_canonical if objects_canonical else [list(asset_status.values())[0].get("canonical_name", list(asset_status.keys())[0])],
                },
                "scene": {"name": scene_name},
                "components": [
                    {
                        "name": "table",
                        "xml_path": "obj/meshes/table/table.xml",
                        "class": "Table",
                        "randomness": {"texture": False},
                    }
                ],
            }
        }

        if containers_canonical:
            series_config["task"]["asset"]["seen_container"] = containers_canonical
            series_config["task"]["asset"]["unseen_container"] = containers_canonical

        # 更新内存中的 TASK_CONFIG
        envs_module.TASK_CONFIG[series_name] = series_config
        logger.info(f"[Registration] ✓ 更新 envs.TASK_CONFIG[{series_name}]")

        # 同时写入 task_config.json（持久化）
        if vlabench_root:
            config_json_path = Path(vlabench_root) / "configs" / "task_config.json"
            if config_json_path.exists():
                with open(config_json_path, "r") as f:
                    all_config = json.load(f)
                all_config[series_name] = series_config
                with open(config_json_path, "w") as f:
                    json.dump(all_config, f, indent=2)
                logger.info(f"[Registration] ✓ 持久化写入 task_config.json")

        # 4. 动态导入模块（触发 @register 装饰器）
        logger.info(f"[Registration] 导入模块: {module_name}")
        importlib.import_module(module_name)

        # 5. 验证注册成功
        if task_name not in register._tasks:
            raise RuntimeError(f"任务 '{task_name}' 未在 register._tasks 中注册")
        if task_name not in register._config_managers:
            raise RuntimeError(f"配置管理器 '{task_name}' 未在 register._config_managers 中注册")

        logger.info(f"[Registration] ✓ 任务注册成功: {task_name}")
        logger.info(f"[Registration]   Task class: {register._tasks[task_name].__name__}")
        logger.info(f"[Registration]   ConfigManager class: {register._config_managers[task_name].__name__}")
        logger.info("=" * 60 + "\n")

        return {
            "registration_success": True,
            "error_feedback": None,
            "current_stage": "simulation",
        }

    except Exception as e:
        error_tb = traceback.format_exc()
        logger.error(f"[Registration] ✗ 注册失败: {e}")
        logger.error(error_tb)

        return {
            "registration_success": False,
            "error_feedback": f"注册失败:\n{error_tb}\n\n生成的代码路径: {task_module_path}",
            "current_stage": "registration",
        }
