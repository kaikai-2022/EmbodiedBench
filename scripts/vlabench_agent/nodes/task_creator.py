"""
Task Creator Node - 任务生成智能体

基于分析结果和资产状态,生成 env_config.json 和任务目录
"""

import json
import logging
import random
import os
from pathlib import Path
from typing import Dict, List

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

from ..config import AgentConfig

logger = logging.getLogger(__name__)


def generate_env_config(
    task_name: str,
    objects: List[Dict],
    scene: str,
    instruction: str,
    conditions: Dict
) -> Dict:
    """
    生成 VLABench env_config.json

    Args:
        task_name: 任务名称
        objects: 物体列表,每个包含 {name, xml_path, class, position, orientation}
        scene: 场景名称
        instruction: 任务指令
        conditions: 评测条件

    Returns:
        完整的 env_config dict
    """
    # 基础配置
    env_config = {
        "task": {
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
            "scene": {
                "name": scene,
                "position": [0, 0, 0],
                "orientation": [1, 0, 0, 0],
                "floor_textures": ["lab_floor"]
            },
            "instructions": [instruction],
            "conditions": conditions
        }
    }

    # 添加任务物体
    for obj in objects:
        component = {
            "name": obj["name"],
            "xml_path": obj["xml_path"],
            "position": obj.get("position", [0.0, 0.0, 0.8]),
            "orientation": obj.get("orientation", [1, 0, 0, 0]),
            "class": obj["class"]
        }

        # 添加额外参数
        if "extra_params" in obj:
            component.update(obj["extra_params"])

        env_config["task"]["components"].append(component)

    return env_config


def task_creator_node(state: Dict) -> Dict:
    """
    任务创建节点 - 生成 env_config 和任务目录结构

    Args:
        state: VLABenchAgentState

    Returns:
        更新后的状态字典
    """
    if ChatAnthropic is None:
        logger.error("[Task Creator] langchain_anthropic 未安装")
        return {
            "current_stage": "error",
            "errors": state.get("errors", []) + ["langchain_anthropic 未安装"]
        }

    logger.info("=" * 60)
    logger.info("[Task Creator] 开始生成任务配置...")

    task_analysis = state.get('task_analysis', {})
    asset_status = state.get('asset_status', {})

    # 1. 构建物体列表
    objects = []
    for i, obj_name in enumerate(task_analysis.get('objects', [])):
        asset_info = asset_status.get(obj_name, {})

        if not asset_info.get('found'):
            logger.error(f"[Task Creator] ✗ 物体 {obj_name} 资产未找到")
            return {
                "current_stage": "error",
                "errors": state.get("errors", []) + [f"物体 {obj_name} 资产未找到"]
            }

        # 生成随机位置(避免碰撞)
        angle = (i * 360 / len(task_analysis['objects'])) * 3.14159 / 180
        radius = 0.15
        position = [
            radius * random.uniform(0.8, 1.2) * (1 if i % 2 == 0 else -1),
            radius * random.uniform(0.8, 1.2) * (1 if (i // 2) % 2 == 0 else -1),
            0.8  # 桌面高度
        ]

        objects.append({
            "name": obj_name,
            "xml_path": asset_info['xml_path'],
            "class": asset_info['class'],
            "position": position,
            "orientation": [1, 0, 0, 0]
        })

    logger.info(f"[Task Creator] 生成了 {len(objects)} 个物体配置")

    # 2. 推断操作序列 (使用 LLM - 统一配置)
    llm = ChatAnthropic(**AgentConfig.get_llm_config())

    operation_prompt = f"""你是机器人操纵任务专家。请根据任务生成操作序列。

任务信息:
- 任务指令: {task_analysis.get('instruction_en')}
- 物体列表: {[o['name'] for o in objects]}
- 操作类型: {task_analysis.get('operation_type')}

请生成 VLABench 操作序列,返回 JSON 格式:
[
    {{"name": "pick", "params": {{"target_entity_name": <物体索引>}}}},
    {{"name": "操作名称", "params": {{...}}}}
]

重要规则:
1. 物体索引从 1 开始 (0 是桌子)
2. 操作名称: pick, place, pour, lift
3. pick 操作: {{"name": "pick", "params": {{"target_entity_name": 索引}}}}
4. place 操作: {{"name": "place", "params": {{"target_container_name": 索引}}}}
5. pour 操作: {{"name": "pour", "params": {{"target_container_name": 索引}}}}
6. lift 操作: {{"name": "lift", "params": {{"target_height": 高度}}}}

只返回 JSON 数组,不要其他文字。"""

    try:
        op_response = llm.invoke(operation_prompt)

        # 解析响应 (处理 markdown 包裹的 JSON)
        content = op_response.content.strip()

        # 如果响应被 markdown 代码块包裹,去除包裹
        if content.startswith("```"):
            # 移除开头的 ```json 或 ```
            lines = content.split('\n')
            if lines[0].startswith("```"):
                lines = lines[1:]
            # 移除结尾的 ```
            if lines[-1].strip() == "```":
                lines = lines[:-1]
            content = '\n'.join(lines)

        operation_sequence = json.loads(content)
        logger.info(f"[Task Creator] ✓ 生成操作序列: {operation_sequence}")

    except Exception as e:
        logger.error(f"[Task Creator] ✗ 操作序列生成失败: {e}")
        # 使用默认序列
        operation_sequence = [
            {"name": "pick", "params": {"target_entity_name": 1}}
        ]
        logger.info(f"[Task Creator] 使用默认操作序列: {operation_sequence}")

    # 3. 生成评测条件
    conditions = {}
    operation_type = task_analysis.get('operation_type', 'pick')

    if operation_type == 'pour':
        conditions = {
            "pour": {
                "target_entity": objects[0]["name"],
                "threshold": 0
            }
        }
    elif operation_type == 'place' and len(objects) >= 2:
        conditions = {
            "contain": {
                "container": objects[1]["name"],
                "entities": [objects[0]["name"]]
            }
        }
    elif operation_type == 'lift':
        conditions = {
            "lift": {
                "entities": [objects[0]["name"]],
                "target_height": 0.9
            }
        }

    logger.info(f"[Task Creator] ✓ 生成评测条件: {list(conditions.keys())}")

    # 4. 生成 env_config
    env_config = generate_env_config(
        task_name=task_analysis.get('task_name', 'custom_task'),
        objects=objects,
        scene=task_analysis.get('scene', 'laboratory') + "_0",
        instruction=task_analysis.get('instruction_en', 'Complete the task'),
        conditions=conditions
    )

    # 5. 创建任务目录结构
    vlabench_root = os.environ.get('VLABENCH_ROOT')
    if not vlabench_root:
        logger.error("[Task Creator] ✗ VLABENCH_ROOT 未设置")
        return {
            "current_stage": "error",
            "errors": state.get("errors", []) + ["VLABENCH_ROOT 未设置"]
        }

    # dataset 在项目根目录，不在 VLABench 包内
    project_root = Path(vlabench_root).parent
    dataset_root = project_root / "dataset" / "vlm_evaluation_v1.0"
    task_dir = dataset_root / "M&T" / task_analysis['task_name'] / "example0"
    task_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[Task Creator] 创建任务目录: {task_dir}")

    # 保存 env_config.json
    config_dir = task_dir / "env_config"
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "env_config.json"

    with open(config_path, 'w') as f:
        json.dump(env_config, f, indent=4)
    logger.info(f"[Task Creator] ✓ 保存 env_config.json")

    # 保存 instruction.txt
    input_dir = task_dir / "input"
    input_dir.mkdir(exist_ok=True)
    (input_dir / "instruction.txt").write_text(task_analysis['instruction_en'])
    logger.info(f"[Task Creator] ✓ 保存 instruction.txt")

    # 保存 operation_sequence.json
    output_dir = task_dir / "output"
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "operation_sequence.json", 'w') as f:
        json.dump({"skill_sequence": operation_sequence}, f, indent=4)
    logger.info(f"[Task Creator] ✓ 保存 operation_sequence.json")

    logger.info(f"[Task Creator] ✓ 任务配置生成完成")
    logger.info("=" * 60 + "\n")

    return {
        "env_config": env_config,
        "task_save_path": str(task_dir),
        "current_stage": "rendering"
    }
