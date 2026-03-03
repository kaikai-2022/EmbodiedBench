#!/usr/bin/env python3
"""
VLM 评测任务批量生成 Pipeline

功能强大的自动化工具，用于批量创建 VLM 评测任务数据集

特性:
- 支持命令行参数配置
- 支持自定义任务模板 (YAML/JSON)
- 自动验证物体资产路径
- 自动调用渲染脚本生成图像
- 支持多种操作类型 (pick, place, pour, lift, etc.)
- 批量生成多个样本
- 干运行模式 (dry-run) 预览

使用示例:
    # 基础用法
    python scripts/create_vlm_task.py --task pour_tube --dimension "M&T" --num-examples 10

    # 使用自定义模板
    python scripts/create_vlm_task.py --config templates/my_task.yaml --render

    # 干运行模式（不创建文件，只预览）
    python scripts/create_vlm_task.py --task lift_flask --dry-run

作者: Claude + QMK
日期: 2026-02-06
"""

import os
import sys
import json
import yaml
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

# 添加项目根目录到路径
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASET_ROOT = PROJECT_ROOT / "dataset" / "vlm_evaluation_v1.0"

# ==================== 任务模板定义 ====================

# 预定义的任务模板库
BUILTIN_TEMPLATES = {
    "pour_tube": {
        "description": "倾倒试管任务 - 抓取并倾倒试管",
        "dimension": "M&T",
        "scene": "laboratory_0",
        "floor_texture": "lab_floor",
        "components": [
            {
                "name": "table",
                "xml_path": "obj/meshes/table/table.xml",
                "position": [0, 0, 0],
                "orientation": [1, 0, 0, 0],
                "class": "Table",
                "materials": ["wood3"]
            }
        ],
        "task_objects": [
            {
                "type": "test_tube",
                "xml_path": "obj/meshes/tube/tube/tube.xml",
                "class": "ChemistryTube",
                "position_base": [0.0, 0.0, 0.8],
                "orientation": [1, 0, 0, 0],
                "extra_params": {
                    "solution": "CuSO4"
                },
                "variations": [
                    {"solution": "CuSO4"},
                    {"solution": "CuCl2"},
                    {"solution": "FeCl3"},
                    {"solution": "KMnO4"}
                ]
            }
        ],
        "instruction_template": "Pour the {object_name}",
        "operation_sequence": [
            {"name": "pick", "params": {"target_entity_name": 1}},
            {"name": "pour", "params": {"target_container_name": 0}}
        ],
        "conditions": {
            "pour": {
                "target_entity": "{object_name}",
                "threshold": 0
            }
        }
    },

    "lift_object": {
        "description": "举起物体任务 - 抓取并举起物体到指定高度",
        "dimension": "M&T",
        "scene": "laboratory_0",
        "floor_texture": "lab_floor",
        "components": [
            {
                "name": "table",
                "xml_path": "obj/meshes/table/table.xml",
                "position": [0, 0, 0],
                "orientation": [1, 0, 0, 0],
                "class": "Table",
                "materials": ["wood3"]
            }
        ],
        "task_objects": [
            {
                "type": "target_object",
                "xml_path": "obj/meshes/containers/boxes/giftbox/giftbox_0/giftbox.xml",
                "class": "CommonContainer",
                "position_base": [0.0, 0.0, 0.8],
                "orientation": [1, 0, 0, 0]
            }
        ],
        "instruction_template": "Lift the {object_name}",
        "operation_sequence": [
            {"name": "pick", "params": {"target_entity_name": 1}},
            {"name": "lift", "params": {"target_height": 0.9}}
        ],
        "conditions": {
            "lift": {
                "entities": ["{object_name}"],
                "target_height": 0.9
            }
        }
    },

    "place_in_container": {
        "description": "放置任务 - 将物体放入容器",
        "dimension": "M&T",
        "scene": "laboratory_0",
        "floor_texture": "lab_floor",
        "components": [
            {
                "name": "table",
                "xml_path": "obj/meshes/table/table.xml",
                "position": [0, 0, 0],
                "orientation": [1, 0, 0, 0],
                "class": "Table",
                "materials": ["wood3"]
            }
        ],
        "task_objects": [
            {
                "type": "target",
                "xml_path": "obj/meshes/poker/poker_asset.xml",
                "class": "Poker",
                "position_base": [-0.2, 0.0, 0.8],
                "orientation": [1, 0, 0, 0]
            },
            {
                "type": "container",
                "xml_path": "obj/meshes/containers/boxes/giftbox/giftbox_0/giftbox.xml",
                "class": "CommonContainer",
                "position_base": [0.3, 0.2, 0.8],
                "orientation": [1, 0, 0, 0]
            }
        ],
        "instruction_template": "Put the {target} into the {container}",
        "operation_sequence": [
            {"name": "pick", "params": {"target_entity_name": 1}},
            {"name": "place", "params": {"target_container_name": 2}}
        ],
        "conditions": {
            "contain": {
                "container": "{container}",
                "entities": ["{target}"]
            }
        }
    },

    "pick_tube": {
        "description": "试管取出任务 - 从试管架上取下试管并放置在桌面上",
        "dimension": "M&T",
        "scene": "laboratory_0",
        "floor_texture": "lab_floor",
        "components": [
            {
                "name": "table",
                "xml_path": "obj/meshes/table/table.xml",
                "position": [0, 0, 0],
                "orientation": [1, 0, 0, 0],
                "class": "Table",
                "materials": ["wood3"]
            },
            {
                "name": "tube_stand",
                "xml_path": "obj/meshes/tube/tube_container/tube_stand.xml",
                "position": [0.0, 0.0, 0.76],
                "orientation": [1, 0, 0, 0],
                "class": "TubeStand"
            }
        ],
        "task_objects": [
            {
                "type": "test_tube",
                "xml_path": "obj/meshes/tube/tube/tube.xml",
                "class": "ChemistryTube",
                "position_base": [0.0, 0.0, 0.759],
                "orientation": [1, 0, 0, 0],
                "extra_params": {
                    "solution": "CuSO4"
                },
                "variations": [
                    {"solution": "CuSO4"},
                    {"solution": "CuCl2"},
                    {"solution": "FeCl3"},
                    {"solution": "KMnO4"}
                ]
            }
        ],
        "instruction_template": "Take out the {object_name} from the tube stand and place it on the table",
        "operation_sequence": [
            {"name": "pick", "params": {"target_entity_name": 2}},
            {"name": "place", "params": {"target_container_name": 0}}
        ],
        "conditions": {
            "not_contain": {
                "container": "tube_stand",
                "entities": ["{object_name}"]
            }
        }
    },

    "flip_petri_dish": {
        "description": "翻转培养皿任务 - 抓取并翻转培养皿（灰色肉培养皿）",
        "dimension": "M&T",
        "scene": "laboratory_0",
        "floor_texture": "lab_floor",
        "components": [
            {
                "name": "table",
                "xml_path": "obj/meshes/table/table.xml",
                "position": [0, 0, 0],
                "orientation": [1, 0, 0, 0],
                "class": "Table",
                "materials": ["wood3"]
            }
        ],
        "task_objects": [
            {
                "type": "petri_dish",
                "xml_path": "obj/meshes/lab_equipment/petri_dish/petri_dish.xml",
                "class": "Plate",
                "position_base": [0.0, 0.0, 0.8],
                "orientation": [1, 0, 0, 0]
            }
        ],
        "instruction_template": "Flip the {object_name}",
        "operation_sequence": [
            {"name": "pick", "params": {"target_entity_name": 1}},
            {"name": "flip", "params": {}}
        ],
        "conditions": {
            "is_grasped": {
                "entities": ["{object_name}"],
                "robot": "franka"
            }
        }
    },

    "insert_tube_centrifuge": {
        "description": "离心机试管插入任务 - 将试管插入离心机",
        "dimension": "M&T",
        "scene": "laboratory_0",
        "floor_texture": "lab_floor",
        "components": [
            {
                "name": "table",
                "xml_path": "obj/meshes/table/table.xml",
                "position": [0, 0, 0],
                "orientation": [1, 0, 0, 0],
                "class": "Table",
                "materials": ["wood3"]
            },
            {
                "name": "tube_stand",
                "xml_path": "obj/meshes/tube/tube_container/tube_stand.xml",
                "position": [-0.25, 0.0, 0.76],
                "orientation": [1, 0, 0, 0],
                "class": "TubeStand"
            }
        ],
        "task_objects": [
            {
                "type": "test_tube",
                "xml_path": "obj/meshes/tube/tube/tube.xml",
                "class": "ChemistryTube",
                "position_base": [-0.25, 0.0, 0.805],
                "orientation": [1, 0, 0, 0],
                "extra_params": {
                    "solution": "CuSO4"
                },
                "variations": [
                    {"solution": "CuSO4"},
                    {"solution": "CuCl2"},
                    {"solution": "FeCl3"},
                    {"solution": "KMnO4"},
                    {"solution": "I2"}
                ]
            },
            {
                "type": "centrifuge",
                "xml_path": "obj/meshes/lab_equipment/centrifuge/centrifuge.xml",
                "class": "CommonContainer",
                "position_base": [0.25, 0.15, 0.82],
                "orientation": [1, 0, 0, 0]
            }
        ],
        "instruction_template": "Insert the {target} into the {container}",
        "operation_sequence": [
            {"name": "pick", "params": {"target_entity_name": 2}},
            {"name": "place", "params": {"target_container_name": 3}}
        ],
        "conditions": {
            "contain": {
                "container": "{container}",
                "entities": ["{target}"]
            }
        }
    }
}

# ==================== 核心功能函数 ====================

def validate_xml_path(xml_path: str) -> bool:
    """验证 XML 资产路径是否存在"""
    # 尝试多个可能的路径
    possible_paths = [
        PROJECT_ROOT / "VLABench" / "assets" / xml_path,
        PROJECT_ROOT / "VLABench" / xml_path,
        PROJECT_ROOT / xml_path
    ]

    for path in possible_paths:
        if path.exists():
            return True

    return False

def add_position_variation(base_pos: List[float], idx: int, variation_range: float = 0.15) -> List[float]:
    """为物体位置添加变化（避免重叠）"""
    random.seed(idx)  # 使用索引作为种子确保可重现
    x_offset = random.uniform(-variation_range, variation_range)
    y_offset = random.uniform(-variation_range, variation_range)
    return [base_pos[0] + x_offset, base_pos[1] + y_offset, base_pos[2]]

def select_tube_slot(idx: int) -> tuple:
    """
    为试管选择试管架的孔洞位置

    试管架有5列2行共10个孔洞:
    - col_pos: [-0.16, -0.08, 0, 0.08, 0.16]
    - row_pos: [-0.05, 0.05]

    Args:
        idx: 样本索引

    Returns:
        (x_offset, y_offset): 相对于试管架中心的偏移
    """
    col_pos = [-0.16, -0.08, 0, 0.08, 0.16]
    row_pos = [-0.05, 0.05]

    # 循环使用孔洞位置
    slot_idx = idx % (len(col_pos) * len(row_pos))
    col_idx = slot_idx % len(col_pos)
    row_idx = slot_idx // len(col_pos)

    return (col_pos[col_idx], row_pos[row_idx])

def generate_env_config(template: Dict, example_idx: int) -> Dict:
    """
    根据模板生成环境配置

    Args:
        template: 任务模板
        example_idx: 样本索引

    Returns:
        环境配置字典
    """
    components = []

    # 添加固定组件（如桌子）
    for comp in template["components"]:
        components.append(comp.copy())

    # 检测是否有试管架（用于特殊处理试管位置）
    tube_stand_pos = None
    for comp in components:
        if comp.get("class") == "TubeStand":
            tube_stand_pos = comp["position"]
            break

    # 添加任务物体
    object_names = []
    for obj_idx, task_obj in enumerate(template["task_objects"]):
        obj_name = f"{task_obj['type']}_{obj_idx}" if len(template["task_objects"]) > 1 else task_obj["type"]
        object_names.append(obj_name)

        # 位置计算
        if tube_stand_pos is not None and task_obj["class"] == "ChemistryTube":
            # 试管+试管架：使用固定孔洞位置
            x_offset, y_offset = select_tube_slot(example_idx)
            position = [
                tube_stand_pos[0] + x_offset,
                tube_stand_pos[1] + y_offset,
                task_obj["position_base"][2]  # z 保持不变
            ]
        else:
            # 其他物体：使用随机位置变化
            position = add_position_variation(task_obj["position_base"], example_idx)

        # 基础配置
        obj_config = {
            "name": obj_name,
            "xml_path": task_obj["xml_path"],
            "position": position,
            "orientation": task_obj["orientation"],
            "class": task_obj["class"]
        }

        # 处理变体（如不同化学溶液）
        if "variations" in task_obj and example_idx < len(task_obj["variations"]):
            variation = task_obj["variations"][example_idx % len(task_obj["variations"])]
            obj_config.update(variation)
        elif "extra_params" in task_obj:
            obj_config.update(task_obj["extra_params"])

        components.append(obj_config)

    # 生成指令（替换占位符）
    instruction = template["instruction_template"]
    for i, name in enumerate(object_names):
        placeholder = f"{{object_name}}" if i == 0 else f"{{object_{i}}}"
        instruction = instruction.replace(placeholder, name)

    # 替换所有可能的占位符
    instruction = instruction.replace("{target}", object_names[0] if object_names else "target")
    if len(object_names) > 1:
        instruction = instruction.replace("{container}", object_names[1])

    # 生成条件（替换占位符）
    conditions = json.loads(json.dumps(template["conditions"]))  # 深拷贝
    conditions_str = json.dumps(conditions)
    for i, name in enumerate(object_names):
        conditions_str = conditions_str.replace("{object_name}", name)
        conditions_str = conditions_str.replace("{target}", object_names[0] if object_names else "target")
        if len(object_names) > 1:
            conditions_str = conditions_str.replace("{container}", object_names[1])
    conditions = json.loads(conditions_str)

    # 组装完整配置
    env_config = {
        "task": {
            "components": components,
            "scene": {
                "name": template["scene"],
                "position": [0, 0, 0],
                "orientation": [1, 0, 0, 0],
                "floor_textures": [template["floor_texture"]]
            },
            "instructions": [instruction],
            "conditions": conditions
        }
    }

    return env_config, object_names

def generate_operation_sequence(template: Dict, object_names: List[str]) -> Dict:
    """
    生成操作序列

    Args:
        template: 任务模板
        object_names: 物体名称列表

    Returns:
        操作序列字典
    """
    return {
        "skill_sequence": template["operation_sequence"]
    }

def create_task_example(
    task_name: str,
    template: Dict,
    example_idx: int,
    dimension: str,
    dry_run: bool = False
) -> Tuple[bool, str]:
    """
    创建单个任务样本

    Args:
        task_name: 任务名称
        template: 任务模板
        example_idx: 样本索引
        dimension: 评测维度
        dry_run: 是否为干运行模式

    Returns:
        (是否成功, 消息)
    """
    example_path = DATASET_ROOT / dimension / task_name / f"example{example_idx}"

    if not dry_run:
        # 创建目录结构
        (example_path / "input").mkdir(parents=True, exist_ok=True)
        (example_path / "output").mkdir(parents=True, exist_ok=True)
        (example_path / "env_config").mkdir(parents=True, exist_ok=True)

    # 生成配置
    env_config, object_names = generate_env_config(template, example_idx)
    operation_seq = generate_operation_sequence(template, object_names)
    instruction = env_config["task"]["instructions"][0]

    # 验证资产路径
    invalid_paths = []
    for comp in env_config["task"]["components"]:
        xml_path = comp.get("xml_path")
        if xml_path and not validate_xml_path(xml_path):
            invalid_paths.append(xml_path)

    if invalid_paths:
        return False, f"无效的资产路径: {', '.join(invalid_paths)}"

    if dry_run:
        # 干运行：仅打印信息
        print(f"  [DRY RUN] example{example_idx}:")
        print(f"    Instruction: {instruction}")
        print(f"    Objects: {', '.join(object_names)}")
        print(f"    Operations: {' -> '.join([op['name'] for op in operation_seq['skill_sequence']])}")
        return True, "dry-run"

    # 写入文件
    try:
        # instruction.txt
        with open(example_path / "input" / "instruction.txt", "w") as f:
            f.write(instruction)

        # env_config.json
        with open(example_path / "env_config" / "env_config.json", "w") as f:
            json.dump(env_config, f, indent=4)

        # operation_sequence.json
        with open(example_path / "output" / "operation_sequence.json", "w") as f:
            json.dump(operation_seq, f, indent=4)

        return True, f"✓ example{example_idx}: {instruction}"

    except Exception as e:
        return False, f"✗ example{example_idx}: {str(e)}"

def render_task_images(task_name: str, dimension: str, overwrite: bool = False) -> bool:
    """
    调用渲染脚本生成任务图像

    Args:
        task_name: 任务名称
        dimension: 评测维度
        overwrite: 是否覆盖已存在的图像

    Returns:
        是否成功
    """
    render_script = SCRIPT_DIR / "render_vlm_dataset.py"

    cmd = [
        sys.executable,
        str(render_script),
        "--task", task_name,
        "--dimension", dimension
    ]

    if overwrite:
        cmd.append("--overwrite")

    print(f"\n{'='*60}")
    print("🎨 渲染任务图像...")
    print(f"{'='*60}")

    # 设置环境变量以使用 OSMesa 离线渲染
    env = os.environ.copy()
    env['MUJOCO_GL'] = 'osmesa'

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, env=env)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"渲染失败: {e}")
        return False

def load_template_from_file(template_path: str) -> Dict:
    """从 YAML/JSON 文件加载自定义模板"""
    path = Path(template_path)

    if not path.exists():
        raise FileNotFoundError(f"模板文件不存在: {template_path}")

    with open(path, 'r') as f:
        if path.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif path.suffix == '.json':
            return json.load(f)
        else:
            raise ValueError(f"不支持的文件格式: {path.suffix}")

def list_available_templates():
    """列出所有可用的内置模板"""
    print("\n" + "="*60)
    print("📋 可用的内置任务模板")
    print("="*60)

    for name, template in BUILTIN_TEMPLATES.items():
        print(f"\n  {name}")
        print(f"    描述: {template['description']}")
        print(f"    维度: {template['dimension']}")
        print(f"    物体数: {len(template['task_objects'])}")
        print(f"    操作: {' -> '.join([op['name'] for op in template['operation_sequence']])}")

    print("\n" + "="*60)

# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(
        description="VLM 评测任务批量生成 Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用内置模板创建任务
  %(prog)s --task pour_tube --num-examples 10 --render

  # 使用自定义模板
  %(prog)s --config my_template.yaml --dimension "M&T" --render

  # 干运行模式（预览）
  %(prog)s --task lift_object --num-examples 5 --dry-run

  # 列出所有可用模板
  %(prog)s --list-templates
        """
    )

    parser.add_argument("--task", type=str, help="任务名称（使用内置模板）")
    parser.add_argument("--config", type=str, help="自定义模板配置文件路径 (YAML/JSON)")
    parser.add_argument("--dimension", type=str, default="M&T", help="评测维度 (默认: M&T)")
    parser.add_argument("--num-examples", type=int, default=1, help="生成样本数量 (默认: 1)")
    parser.add_argument("--render", action="store_true", help="创建后自动渲染图像")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的图像")
    parser.add_argument("--dry-run", action="store_true", help="干运行模式（不创建文件，仅预览）")
    parser.add_argument("--list-templates", action="store_true", help="列出所有可用的内置模板")

    args = parser.parse_args()

    # 列出模板
    if args.list_templates:
        list_available_templates()
        return 0

    # 验证参数
    if not args.task and not args.config:
        parser.error("必须指定 --task 或 --config")

    # 加载模板
    if args.config:
        print(f"📖 加载自定义模板: {args.config}")
        template = load_template_from_file(args.config)
        task_name = Path(args.config).stem
        dimension = template.get("dimension", args.dimension)
    else:
        if args.task not in BUILTIN_TEMPLATES:
            print(f"❌ 错误: 未知的任务模板 '{args.task}'")
            print(f"\n可用模板: {', '.join(BUILTIN_TEMPLATES.keys())}")
            print(f"或使用 --list-templates 查看详细信息")
            return 1

        template = BUILTIN_TEMPLATES[args.task]
        task_name = args.task
        dimension = template.get("dimension", args.dimension)

    # 开始创建
    print("\n" + "="*60)
    print(f"🚀 VLM 任务批量生成 Pipeline")
    print("="*60)
    print(f"任务名称: {task_name}")
    print(f"评测维度: {dimension}")
    print(f"样本数量: {args.num_examples}")
    print(f"描述: {template.get('description', 'N/A')}")
    if args.dry_run:
        print(f"模式: 🔍 干运行 (DRY RUN)")
    print("")

    # 批量创建样本
    success_count = 0
    failed_count = 0

    for i in range(args.num_examples):
        success, message = create_task_example(
            task_name=task_name,
            template=template,
            example_idx=i,
            dimension=dimension,
            dry_run=args.dry_run
        )

        if success:
            success_count += 1
            if not args.dry_run:
                print(f"  {message}")
        else:
            failed_count += 1
            print(f"  {message}")

    # 总结
    print("\n" + "="*60)
    if args.dry_run:
        print(f"🔍 干运行完成")
    else:
        print(f"✅ 任务创建完成")
    print("="*60)
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    print("")

    if args.dry_run:
        print("💡 提示: 移除 --dry-run 参数以实际创建文件")
        return 0

    # 渲染图像
    if args.render and success_count > 0:
        render_success = render_task_images(task_name, dimension, args.overwrite)

        if render_success:
            print("\n✅ 图像渲染完成！")
        else:
            print("\n⚠️  图像渲染失败，请手动运行渲染脚本")

    # 后续步骤提示
    if success_count > 0 and not args.render:
        print("📝 下一步:")
        print(f"  1. 渲染图像:")
        print(f"     python scripts/render_vlm_dataset.py --task {task_name} --dimension \"{dimension}\"")
        print(f"  2. 测试评估:")
        print(f"     python scripts/evaluate_vlm.py --vlm_name Qwen2_VL --eval-dimension \"{dimension}\" --tasks {task_name}")
        print("")

    return 0 if failed_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
