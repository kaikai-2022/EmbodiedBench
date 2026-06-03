"""
Registration Node - 动态注册 LLM 生成的任务类

将生成的 Python 任务文件动态导入，注册到 VLABench 的任务注册表中，
并更新 name2config 和 TASK_CONFIG 使 load_env() 能够找到该任务。

适配新 pipeline 数据源：
  - normalized_context（替代旧 task_graph）
  - asset_status 中的 class_name（替代 spec 字段）
"""

import importlib
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Dict

from .node_logger import log_node_output_file

logger = logging.getLogger(__name__)


def registration_node(state: Dict) -> Dict:
    """
    注册节点 - 动态注册生成的任务类

    适配新 pipeline：从 normalized_context 读取 instances，
    从 asset_status 读取 class_name 推断容器/物体分类。
    """
    logger.info("=" * 60)
    logger.info("[Registration] 开始动态注册任务...")

    task_analysis = state.get("task_analysis", {})
    normalized_context = state.get("normalized_context", {})
    asset_status = state.get("asset_status", {})
    task_module_path = state.get("task_module_path")
    task_name = task_analysis.get("task_name", "custom_task").replace(" ", "_")
    series_name = f"{task_name}_series"
    module_name = f"VLABench.tasks.autogen_tasks.{task_name}_series"

    if not task_module_path:
        logger.error("[Registration] ✗ 缺少 task_module_path")
        output = {
            "registration_success": False,
            "error_feedback": "task_module_path 为空，代码生成步骤可能失败",
            "current_stage": "registration",
        }
        log_node_output_file("registration", state, output)
        return output

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
        # 同时更新 VLABench.configs 中的和 envs 模块中的 name2config，
        # 避免因 import 顺序导致 load_env() 看不到动态注册的任务
        from VLABench.configs import name2config as configs_name2config
        import VLABench.envs as envs_module

        configs_name2config[series_name] = [task_name]
        envs_module.name2config[series_name] = [task_name]
        logger.info(f"[Registration] ✓ 更新 name2config: {series_name} -> [{task_name}]")

        # 3. 更新 TASK_CONFIG（内存 + JSON 文件）


        vlabench_root = os.environ.get("VLABENCH_ROOT")
        scene_name = task_analysis.get("scene", "laboratory") + "_0"

        # 从 normalized_context["instances"] 收集所有实体 spec
        # 简化：不再区分 container 和 object，所有实体统一放入 seen_object
        instances = normalized_context.get("instances", [])
        all_specs = []

        if instances:
            for inst in instances:
                if not inst.get("is_physical", True):
                    continue
                spec = inst.get("spec", inst.get("uid", ""))
                if spec not in all_specs:
                    all_specs.append(spec)
        elif asset_status:
            logger.info("[Registration] 无 normalized_context.instances，使用 asset_status 兼容模式")
            for uid, info in asset_status.items():
                spec = info.get("spec", uid)
                if spec not in all_specs:
                    all_specs.append(spec)

        series_config = {
            "task": {
                "asset": {
                    "seen_object": all_specs,
                    "unseen_object": all_specs,
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

        # 更新内存中的 TASK_CONFIG
        envs_module.TASK_CONFIG[series_name] = series_config
        logger.info(f"[Registration] ✓ 更新 envs.TASK_CONFIG[{series_name}]")

        # 写入 task_config.json（持久化）
        # 重要：每次都强制覆盖，确保使用最新的动态生成的配置
        # 这样可以避免旧配置（如之前测试 tube 时生成的）残留导致问题
        if vlabench_root:
            config_json_path = Path(vlabench_root) / "configs" / "task_config.json"
            if config_json_path.exists():
                with open(config_json_path, "r") as f:
                    all_config = json.load(f)
                all_config[series_name] = series_config
                with open(config_json_path, "w") as f:
                    json.dump(all_config, f, indent=2)
                logger.info(f"[Registration] ✓ 写入 task_config.json: {series_name}")

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

        output = {
            "registration_success": True,
            "error_feedback": None,
            "current_stage": "simulation",
        }
        log_node_output_file("registration", state, output)
        return output

    except Exception as e:
        error_tb = traceback.format_exc()
        logger.error(f"[Registration] ✗ 注册失败: {e}")
        logger.error(error_tb)

        output = {
            "registration_success": False,
            "error_feedback": f"注册失败:\n{error_tb}\n\n生成的代码路径: {task_module_path}",
            "current_stage": "registration",
        }
        log_node_output_file("registration", state, output)
        return output
