"""
Analyzer Node - 任务理解智能体

使用 LLM 理解用户自然语言指令,提取任务关键信息
"""

import json
import logging
from typing import Dict

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

from ..config import AgentConfig

logger = logging.getLogger(__name__)


def analyzer_node(state: Dict) -> Dict:
    """
    任务分析节点 - 使用 LLM 理解用户意图

    Args:
        state: VLABenchAgentState

    Returns:
        更新后的状态字典
    """
    if ChatAnthropic is None:
        logger.error("langchain_anthropic 未安装,请运行: pip install langchain-anthropic")
        return {
            "current_stage": "error",
            "errors": state.get("errors", []) + ["langchain_anthropic 未安装"]
        }

    logger.info("=" * 60)
    logger.info("[Analyzer] 开始分析任务...")
    logger.info(f"[Analyzer] 用户指令: {state['user_instruction']}")

    # 初始化 LLM (使用统一配置)
    llm = ChatAnthropic(**AgentConfig.get_llm_config())

    # 构建分析提示词
    analysis_prompt = f"""你是一个机器人操纵任务设计专家。用户描述了一个科研场景任务,请提取关键信息。

用户指令: "{state['user_instruction']}"

请分析并输出 JSON 格式:
{{
    "objects": ["物体1", "物体2", ...],
    "scene": "场景类型",
    "operation_type": "操作类型",
    "instruction_en": "规范化英文指令",
    "spatial_relations": ["关系描述"],
    "task_name": "任务名称(英文下划线)"
}}

注意事项:
1. objects: 识别科研器材的专业英文名称
   - 盖玻片 -> coverslip
   - 显微镜 -> microscope
   - 试管 -> test_tube
   - 烧杯 -> beaker
   - 载玻片 -> microscope_slide

2. scene: 场景类型选择
   - 实验室场景 -> laboratory
   - 厨房场景 -> kitchen
   - 客厅场景 -> living_room

3. operation_type: 操作类型
   - 抓取 -> pick
   - 放置 -> place
   - 倾倒 -> pour
   - 举起 -> lift
   - 滑动 -> slide

4. instruction_en: 生成简洁的英文指令 (动词 + the + 物体名称)

5. spatial_relations: 物体间的空间关系描述

6. task_name: 使用英文下划线格式,如 place_coverslip_on_microscope

只返回 JSON,不要其他文字。"""

    try:
        # 调用 LLM
        response = llm.invoke(analysis_prompt)

        # 解析响应 (处理 markdown 包裹的 JSON)
        content = response.content.strip()

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

        task_analysis = json.loads(content)

        logger.info(f"[Analyzer] ✓ 识别到物体: {task_analysis['objects']}")
        logger.info(f"[Analyzer] ✓ 场景类型: {task_analysis['scene']}")
        logger.info(f"[Analyzer] ✓ 操作类型: {task_analysis['operation_type']}")
        logger.info(f"[Analyzer] ✓ 任务名称: {task_analysis['task_name']}")
        logger.info(f"[Analyzer] ✓ 英文指令: {task_analysis['instruction_en']}")
        logger.info("=" * 60 + "\n")

        return {
            "task_analysis": task_analysis,
            "current_stage": "asset_check",
            "messages": state.get("messages", []) + [response]
        }

    except json.JSONDecodeError as e:
        logger.error(f"[Analyzer] ✗ JSON 解析失败: {e}")
        logger.error(f"[Analyzer] 响应内容: {response.content}")
        return {
            "current_stage": "error",
            "errors": state.get("errors", []) + [f"任务分析失败: JSON 解析错误 - {e}"]
        }

    except Exception as e:
        logger.error(f"[Analyzer] ✗ 分析失败: {e}")
        return {
            "current_stage": "error",
            "errors": state.get("errors", []) + [f"任务分析失败: {e}"]
        }
