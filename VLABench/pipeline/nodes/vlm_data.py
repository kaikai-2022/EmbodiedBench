"""
VLM Data Node - 从仿真结果生成 VLM 评测数据集

将仿真的 episode_config、instruction、operation_sequence 输出为
VLM 评测数据格式，并调用渲染脚本生成多视角图像。
"""

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


def vlm_data_node(state: Dict) -> Dict:
    """
    VLM 数据节点 - 输出 VLM 评测数据集

    Args:
        state: VLABenchAgentState

    Returns:
        更新后的状态字典
    """
    logger.info("=" * 60)
    logger.info("[VLM Data] 开始生成 VLM 评测数据...")

    task_analysis = state.get("task_analysis", {})
    task_name = task_analysis.get("task_name", "custom_task")
    episode_config = state.get("episode_config")
    executed_skill_sequence = state.get("executed_skill_sequence", [])

    vlabench_root = os.environ.get("VLABENCH_ROOT")
    if not vlabench_root:
        return {
                    "errors": state.get("errors", []) + ["VLABENCH_ROOT 未设置"],
        }

    project_root = Path(vlabench_root).parent
    dataset_root = project_root / "dataset" / "vlm_evaluation_v1.0"
    dimension = "M&T"
    task_dir = dataset_root / dimension / task_name / "example0"

    try:
        # 1. 创建目录结构
        (task_dir / "env_config").mkdir(parents=True, exist_ok=True)
        (task_dir / "input").mkdir(parents=True, exist_ok=True)
        (task_dir / "output").mkdir(parents=True, exist_ok=True)

        # 2. 保存 env_config.json
        if episode_config:
            config_path = task_dir / "env_config" / "env_config.json"
            with open(config_path, "w") as f:
                json.dump(episode_config, f, indent=4)
            logger.info(f"[VLM Data] ✓ 保存 env_config.json")
        else:
            logger.warning("[VLM Data] ⚠ episode_config 为空，跳过保存")

        # 3. 保存 instruction.txt
        instruction = state.get("user_instruction", "")
        (task_dir / "input" / "instruction.txt").write_text(instruction)
        logger.info(f"[VLM Data] ✓ 保存 instruction.txt: {instruction}")

        # 4. 保存 operation_sequence.json
        operation_seq = {"skill_sequence": executed_skill_sequence}
        with open(task_dir / "output" / "operation_sequence.json", "w") as f:
            json.dump(operation_seq, f, indent=4)
        logger.info(f"[VLM Data] ✓ 保存 operation_sequence.json ({len(executed_skill_sequence)} 步)")

        # 5. 调用渲染脚本生成图像
        rendered_images = []
        render_script = project_root / "scripts" / "render_vlm_dataset.py"

        if render_script.exists():
            logger.info("[VLM Data] 调用渲染脚本...")

            conda_activate = "source /ssd/mkqin/miniconda3/etc/profile.d/conda.sh && conda activate vlabench_2"
            render_cmd = (
                f"python {render_script} --task {task_name} "
                f"--dimension '{dimension}' --num-examples 1"
            )
            full_cmd = f"{conda_activate} && {render_cmd}"

            try:
                result = subprocess.run(
                    full_cmd,
                    shell=True,
                    executable="/bin/bash",
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=project_root,
                )

                if result.returncode == 0:
                    images = list((task_dir / "input").glob("*.png"))
                    rendered_images = [str(img) for img in images]
                    logger.info(f"[VLM Data] ✓ 渲染成功: {len(rendered_images)} 张图像")
                else:
                    logger.warning(f"[VLM Data] ⚠ 渲染失败: {result.stderr[:500]}")
            except subprocess.TimeoutExpired:
                logger.warning("[VLM Data] ⚠ 渲染超时")
            except Exception as e:
                logger.warning(f"[VLM Data] ⚠ 渲染异常: {e}")
        else:
            logger.warning(f"[VLM Data] ⚠ 渲染脚本不存在: {render_script}")

        # 6. 读取验证报告（如果有）
        validation_report = {}
        validation_path = task_dir / "env_config" / "validation_report.json"
        if validation_path.exists():
            with open(validation_path) as f:
                validation_report = json.load(f)

        logger.info(f"[VLM Data] ✓ VLM 评测数据生成完成")
        logger.info(f"[VLM Data]   路径: {task_dir}")
        logger.info("=" * 60 + "\n")

        return {
            "env_config": episode_config,
            "task_save_path": str(task_dir),
            "rendered_images": rendered_images,
            "validation_report": validation_report,
                }

    except Exception as e:
        logger.error(f"[VLM Data] ✗ VLM 数据生成失败: {e}")
        return {
                    "errors": state.get("errors", []) + [f"VLM 数据生成失败: {e}"],
        }
