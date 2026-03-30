#!/usr/bin/env python3
"""
VLM 评测任务批量生成脚本（v2 - 基于仿真环境）

针对已有 series 文件的任务，通过 MuJoCo 仿真环境批量生成 VLM 评测数据集。
每个 example 通过 env.reset() 获得不同的随机化场景配置。

工作流程:
  1. load_env(task_name) 加载任务环境
  2. 循环 N 次:
     a. env.reset() -> 随机化场景
     b. env.save() -> 获取 episode_config (即 env_config.json)
     c. env.get_expert_skill_sequence() -> 获取技能序列
     d. extract_skill_sequence() -> 转换为 operation_sequence.json 格式
     e. env.task.get_instruction() -> 获取 instruction
     f. 写入 dataset/vlm_evaluation_v1.0/{dimension}/{task_name}/example{i}/
  3. 可选: 调用 render_vlm_dataset.py 渲染图像

前置条件:
  - 任务的 series 文件已创建 (VLABench/tasks/hierarchical_tasks/primitive/{task}_series.py)
  - 任务已注册 (configs/__init__.py 和 primitive/__init__.py)
  - 任务已通过 trajectory_generation.py 调试验证

使用示例:
    # 生成 10 个样本
    python scripts/create_vlm_task.py --task shake_tube --num-examples 10

    # 生成并渲染
    python scripts/create_vlm_task.py --task shake_tube --num-examples 100 --dimension "M&T" --render

    # 干运行模式（不创建文件，只预览）
    python scripts/create_vlm_task.py --task shake_tube --num-examples 5 --dry-run

    # 列出所有可用的已注册任务
    python scripts/create_vlm_task.py --list-tasks

作者: Claude + QMK
日期: 2026-03-24 (v2 改造)
"""

import os
import sys
import json
import argparse
import subprocess
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

# 添加项目根目录到路径
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 设置环境变量（必须在导入 VLABench 之前）
os.environ.setdefault("VLABENCH_ROOT", str(PROJECT_ROOT / "VLABench"))
os.environ.setdefault("MUJOCO_GL", "egl")

DATASET_ROOT = PROJECT_ROOT / "dataset" / "vlm_evaluation_v1.0"


# ==================== 技能序列提取 ====================

def extract_skill_sequence(skill_seq, entity_names: List[str]) -> List[Dict]:
    """
    从 partial 对象列表中提取 operation_sequence 格式的技能序列。
    复用自 simulation.py 的逻辑。

    Args:
        skill_seq: list of functools.partial objects
        entity_names: 场景中的实体名称列表（用于映射到索引）

    Returns:
        [{"name": "pick", "params": {"target_entity_name": 1}}, ...]
    """
    name_to_index = {name: i for i, name in enumerate(entity_names)}

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


# ==================== 核心功能 ====================

def create_example_from_env(
    env,
    task_name: str,
    example_idx: int,
    dimension: str,
    dry_run: bool = False,
) -> Tuple[bool, str]:
    """
    从仿真环境创建单个 VLM 评测样本。

    在调用前，env 应该已经 reset() 过，场景已初始化。

    Args:
        env: 已加载的 MuJoCo 环境
        task_name: 任务名称
        example_idx: 样本索引
        dimension: 评测维度
        dry_run: 是否为干运行模式

    Returns:
        (是否成功, 消息)
    """
    try:
        # 1. 获取 episode_config (env_config.json 的内容)
        episode_config = env.save()

        # 2. 获取 instruction
        instruction = env.task.get_instruction()

        # 3. 获取专家技能序列并转换格式
        skill_seq = env.get_expert_skill_sequence()
        if skill_seq is None:
            return False, f"example{example_idx}: get_expert_skill_sequence() 返回 None"

        # 从 episode_config 的 components 中提取名称列表（与 env_config.json 顺序一致）
        # 这样 extract_skill_sequence 才能将实体名正确映射为数字索引
        entity_names = []
        for comp in episode_config.get("task", {}).get("components", []):
            entity_names.append(comp.get("name", ""))
        if not entity_names:
            entity_names = list(env.task.entities.keys())

        operation_seq = {
            "skill_sequence": extract_skill_sequence(skill_seq, entity_names)
        }

        skill_names = [s.func.__name__ for s in skill_seq]

        if dry_run:
            print(f"  [DRY RUN] example{example_idx}:")
            print(f"    Instruction: {instruction}")
            print(f"    Entities: {', '.join(entity_names)}")
            print(f"    Skills: {' -> '.join(skill_names)}")
            return True, "dry-run"

        # 4. 创建目录并写入文件
        example_path = DATASET_ROOT / dimension / task_name / f"example{example_idx}"
        (example_path / "input").mkdir(parents=True, exist_ok=True)
        (example_path / "output").mkdir(parents=True, exist_ok=True)
        (example_path / "env_config").mkdir(parents=True, exist_ok=True)

        # instruction.txt
        with open(example_path / "input" / "instruction.txt", "w") as f:
            f.write(instruction)

        # env_config.json
        with open(example_path / "env_config" / "env_config.json", "w") as f:
            json.dump(episode_config, f, indent=4)

        # operation_sequence.json
        with open(example_path / "output" / "operation_sequence.json", "w") as f:
            json.dump(operation_seq, f, indent=4)

        return True, f"example{example_idx}: {instruction} [{' -> '.join(skill_names)}]"

    except Exception as e:
        return False, f"example{example_idx}: {e}"


def render_task_images(task_name: str, dimension: str, overwrite: bool = False) -> bool:
    """调用 render_vlm_dataset.py 渲染任务图像"""
    render_script = SCRIPT_DIR / "render_vlm_dataset.py"

    cmd = [
        sys.executable,
        str(render_script),
        "--task", task_name,
        "--dimension", dimension,
    ]
    if overwrite:
        cmd.append("--overwrite")

    print(f"\n{'='*60}")
    print("渲染任务图像...")
    print(f"{'='*60}")

    env_vars = os.environ.copy()
    env_vars["MUJOCO_GL"] = "osmesa"

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, env=env_vars)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"渲染失败: {e}")
        return False


def list_registered_tasks():
    """列出所有已注册的任务"""
    from VLABench.utils.register import register
    import VLABench.robots
    import VLABench.tasks.hierarchical_tasks.primitive
    import VLABench.tasks.hierarchical_tasks.composite

    print("\n" + "=" * 60)
    print("已注册的任务列表")
    print("=" * 60)

    for task_name in sorted(register._tasks.keys()):
        print(f"  {task_name}")

    print(f"\n共 {len(register._tasks)} 个任务")
    print("=" * 60)


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(
        description="VLM 评测任务批量生成脚本（基于仿真环境）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成 10 个样本
  %(prog)s --task shake_tube --num-examples 10

  # 生成并渲染
  %(prog)s --task shake_tube --num-examples 100 --dimension "M&T" --render

  # 干运行模式（预览）
  %(prog)s --task shake_tube --num-examples 5 --dry-run

  # 列出所有可用任务
  %(prog)s --list-tasks

前置条件:
  任务的 series 文件必须已创建并通过调试验证。
        """,
    )

    parser.add_argument("--task", type=str, help="任务名称（已注册的任务）")
    parser.add_argument("--dimension", type=str, default="M&T", help="评测维度 (默认: M&T)")
    parser.add_argument("--num-examples", type=int, default=1, help="生成样本数量 (默认: 1)")
    parser.add_argument("--render", action="store_true", help="创建后自动渲染图像")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的数据")
    parser.add_argument("--dry-run", action="store_true", help="干运行模式（不创建文件，仅预览）")
    parser.add_argument("--list-tasks", action="store_true", help="列出所有已注册的任务")

    args = parser.parse_args()

    if args.list_tasks:
        list_registered_tasks()
        return 0

    if not args.task:
        parser.error("必须指定 --task 参数")

    task_name = args.task
    dimension = args.dimension
    num_examples = args.num_examples

    print("\n" + "=" * 60)
    print("VLM 评测数据批量生成")
    print("=" * 60)
    print(f"任务: {task_name}")
    print(f"维度: {dimension}")
    print(f"样本数: {num_examples}")
    if args.dry_run:
        print("模式: DRY RUN")
    print()

    # 导入并加载环境
    try:
        import VLABench.robots
        import VLABench.tasks.hierarchical_tasks.primitive
        import VLABench.tasks.hierarchical_tasks.composite
        from VLABench.envs import load_env

        print(f"加载任务环境: {task_name} ...")
        env = load_env(task_name, robot="franka")
        print("环境加载成功\n")
    except Exception as e:
        print(f"环境加载失败: {e}")
        traceback.print_exc()
        return 1

    # 批量生成
    success_count = 0
    failed_count = 0

    for i in range(num_examples):
        # 第一次 load_env 已经 reset 过了，后续需要手动 reset
        if i > 0:
            env.reset()

        success, message = create_example_from_env(
            env=env,
            task_name=task_name,
            example_idx=i,
            dimension=dimension,
            dry_run=args.dry_run,
        )

        if success:
            success_count += 1
            print(f"  [OK] {message}")
        else:
            failed_count += 1
            print(f"  [FAIL] {message}")

    env.close()

    # 总结
    print("\n" + "=" * 60)
    if args.dry_run:
        print("DRY RUN 完成")
    else:
        print("生成完成")
    print("=" * 60)
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    print(f"输出: {DATASET_ROOT / dimension / task_name}/")
    print()

    if args.dry_run:
        print("提示: 移除 --dry-run 以实际创建文件")
        return 0

    # 渲染
    if args.render and success_count > 0:
        if render_task_images(task_name, dimension, args.overwrite):
            print("\n渲染完成")
        else:
            print("\n渲染失败，请手动运行:")
            print(f"  python scripts/render_vlm_dataset.py --task {task_name} --dimension \"{dimension}\"")

    elif success_count > 0 and not args.render:
        print("下一步:")
        print(f"  1. 渲染: python scripts/render_vlm_dataset.py --task {task_name} --dimension \"{dimension}\"")
        print(f"  2. 评测: python scripts/evaluate_vlm.py --vlm_name Qwen2_VL --eval-dimension \"{dimension}\" --tasks {task_name}")

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
