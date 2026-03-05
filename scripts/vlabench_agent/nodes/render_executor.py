"""
Render Executor Node - 渲染执行智能体

调用 render_vlm_dataset.py 生成图像并验证场景
"""

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


def render_vlm_task(task_name: str, dimension: str = "M&T") -> Dict:
    """
    调用 render_vlm_dataset.py 渲染任务场景

    Args:
        task_name: 任务名称
        dimension: 评测维度

    Returns:
        {
            "success": bool,
            "rendered_images": List[str],
            "validation_report": Dict,
            "error": str (optional)
        }
    """
    logger.info(f"  开始渲染任务: {task_name}")

    vlabench_root = os.environ.get('VLABENCH_ROOT')
    if not vlabench_root:
        return {
            "success": False,
            "rendered_images": [],
            "validation_report": {},
            "error": "VLABENCH_ROOT 未设置"
        }

    # render_vlm_dataset.py 在项目根目录的 scripts/ 下
    project_root = Path(vlabench_root).parent
    script_path = project_root / "scripts" / "render_vlm_dataset.py"

    if not script_path.exists():
        return {
            "success": False,
            "rendered_images": [],
            "validation_report": {},
            "error": f"render_vlm_dataset.py 不存在: {script_path}"
        }

    try:
        # 需要激活 conda 环境并运行渲染脚本
        conda_activate = "source /ssd/mkqin/miniconda3/etc/profile.d/conda.sh && conda activate vlabench_2"
        # dimension 包含 & 符号，需要用引号包裹
        render_cmd = f"python {script_path} --task {task_name} --dimension '{dimension}' --num-examples 1"

        full_cmd = f"{conda_activate} && {render_cmd}"

        result = subprocess.run(
            full_cmd,
            shell=True,
            executable='/bin/bash',
            capture_output=True,
            text=True,
            timeout=300,  # 5 分钟超时
            cwd=project_root  # 在项目根目录运行
        )

        if result.returncode == 0:
            # 查找渲染的图像
            dataset_root = project_root / "dataset" / "vlm_evaluation_v1.0"
            task_dir = dataset_root / dimension / task_name / "example0" / "input"
            images = list(task_dir.glob("*.png"))

            # 读取验证报告
            validation_path = task_dir.parent / "env_config" / "validation_report.json"
            validation_report = {}
            if validation_path.exists():
                with open(validation_path) as f:
                    validation_report = json.load(f)

            logger.info(f"  ✓ 渲染成功: {len(images)} 张图像")

            return {
                "success": True,
                "rendered_images": [str(img) for img in images],
                "validation_report": validation_report
            }
        else:
            logger.error(f"  ✗ 渲染失败: {result.stderr}")
            return {
                "success": False,
                "rendered_images": [],
                "validation_report": {},
                "error": result.stderr
            }

    except subprocess.TimeoutExpired:
        logger.error(f"  ✗ 渲染超时 (>5分钟)")
        return {
            "success": False,
            "rendered_images": [],
            "validation_report": {},
            "error": "渲染超时"
        }
    except Exception as e:
        logger.error(f"  ✗ 渲染异常: {e}")
        return {
            "success": False,
            "rendered_images": [],
            "validation_report": {},
            "error": str(e)
        }


def render_executor_node(state: Dict) -> Dict:
    """
    渲染执行节点 - 调用渲染脚本并验证

    Args:
        state: VLABenchAgentState

    Returns:
        更新后的状态字典
    """
    logger.info("=" * 60)
    logger.info("[Render Executor] 开始渲染场景...")

    task_analysis = state.get('task_analysis', {})
    task_name = task_analysis.get('task_name')

    if not task_name:
        logger.error("[Render Executor] ✗ 任务名称未找到")
        return {
            "current_stage": "error",
            "errors": state.get("errors", []) + ["任务名称未找到"]
        }

    # 1. 渲染场景
    render_result = render_vlm_task(task_name)

    if not render_result['success']:
        error_msg = f"渲染失败: {render_result.get('error', 'Unknown')}"
        logger.error(f"[Render Executor] ✗ {error_msg}")
        return {
            "current_stage": "error",
            "errors": state.get("errors", []) + [error_msg]
        }

    # 2. 检查验证报告
    validation = render_result['validation_report']
    warnings = []

    if validation:
        overall_status = validation.get('overall_status', 'UNKNOWN')
        logger.info(f"[Render Executor] 场景验证: {overall_status}")

        if overall_status != 'PASS':
            warnings.append("场景验证未通过,请检查验证报告")
            for error in validation.get('errors', []):
                warnings.append(f"  - {error}")
                logger.warning(f"[Render Executor]   ⚠ {error}")

    # 3. 输出结果
    logger.info(f"[Render Executor] ✓ 渲染完成")
    logger.info(f"[Render Executor] 图像数量: {len(render_result['rendered_images'])}")
    for img in render_result['rendered_images']:
        logger.info(f"  - {img}")

    logger.info("=" * 60 + "\n")

    return {
        "rendered_images": render_result['rendered_images'],
        "validation_report": validation,
        "warnings": state.get("warnings", []) + warnings,
        "current_stage": "done"
    }
