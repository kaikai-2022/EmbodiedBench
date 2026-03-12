"""
场景优化节点 - Phase 2: 智能优化

基于渲染结果和用户反馈，自动调整物体的位置、朝向、尺寸等参数
"""

import json
import logging
from typing import Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)


def analyze_visual_issues(rendered_image_path: str, validation_report: Dict) -> List[Dict]:
    """
    分析渲染图像中的视觉问题

    Args:
        rendered_image_path: 渲染图像路径
        validation_report: 验证报告

    Returns:
        问题列表，每个问题包含 {type, description, severity, suggested_fix}
    """
    issues = []

    # 1. 检查validation report中的问题
    if validation_report.get('overall_status') == 'FAILED':
        for error in validation_report.get('errors', []):
            issues.append({
                "type": "validation_error",
                "description": error,
                "severity": "high",
                "suggested_fix": "根据validation建议调整配置"
            })

    # 2. 检查warnings
    for warning in validation_report.get('warnings', []):
        issues.append({
            "type": "validation_warning",
            "description": warning,
            "severity": "medium",
            "suggested_fix": "检查物体可见性和位置"
        })

    # 3. 启发式检查：从review目录下载的模型可能存在的问题
    # 这些模型通常是从3D资产库下载的，可能存在：
    # - 尺寸不合理（太大或太小）
    # - 朝向不正确（躺倒、倒置等）
    # - 位置不合理（嵌入桌面、悬浮等）

    return issues


def generate_optimization_prompt(
    env_config: Dict,
    validation_report: Dict,
    user_feedback: str = None
) -> str:
    """
    生成优化提示词，用于LLM分析和修复配置

    Args:
        env_config: 当前环境配置
        validation_report: 验证报告
        user_feedback: 用户反馈（可选）

    Returns:
        优化提示词
    """
    # 提取任务物体（排除table等固定组件）
    task_objects = []
    for comp in env_config['task']['components']:
        if comp.get('class') not in ['Table', 'TubeStand']:
            task_objects.append(comp)

    prompt = f"""你是一个机器人场景配置专家。当前场景存在以下问题需要优化：

## 当前配置
```json
{json.dumps(task_objects, indent=2, ensure_ascii=False)}
```

## 验证报告
- 整体状态: {validation_report.get('overall_status', 'UNKNOWN')}
- 错误: {validation_report.get('errors', [])}
- 警告: {validation_report.get('warnings', [])}

## 已知问题模式
对于从review目录下载的3D模型（路径包含"review/"），常见问题包括：

1. **尺寸问题**: 模型原始尺寸可能与实验室器材的真实尺寸不符
   - **显微镜**: 通常需要缩放到 30% (scale=0.3) 以匹配真实尺寸（高度约0.3-0.5米）
   - 烧杯通常高度约0.1-0.2米，直径约0.05-0.1米
   - 试管通常长度约0.1-0.15米，直径约0.01-0.02米

2. **朝向问题**: 模型可能以非标准朝向建模
   - **显微镜**: 通常需要绕X轴旋转90度 (euler=[1.57, 0, 0]) 才能直立放置（目镜在上，底座在下）
   - 烧杯应该开口向上
   - 试管应该开口向上或水平放置
   - **重要**: 检查验证报告中的朝向信息，如果 angle_with_vertical 接近 90° 但有 "特殊旋转物体" 注释，说明朝向已正确

3. **位置问题**: 物体应该合理放置在桌面上
   - 桌面高度通常为0.8米
   - 物体底部应该接触桌面（z坐标 = 0.8 + 物体高度/2）
   - 物体之间应该有合理间距，避免重叠

## 优化要求
请分析当前配置，并提供优化建议。对于每个物体，考虑：

1. **是否需要调整尺寸** (通过添加scale参数)
   - 如果模型过大或过小，建议合理的缩放比例
   - scale范围通常在0.1-2.0之间

2. **是否需要调整朝向** (通过修改orientation或添加euler参数)
   - 如果模型躺倒或倒置，建议正确的旋转角度
   - euler格式: [rx, ry, rz] (弧度制)
   - 常用旋转: 90度 = 1.57, 180度 = 3.14

3. **是否需要调整位置** (通过修改position参数)
   - 确保物体正确放置在桌面上
   - 考虑物体的实际尺寸和重心位置

## 输出格式
请以JSON格式输出优化后的配置，只包含需要修改的物体：

```json
{{
  "optimized_components": [
    {{
      "name": "microscope",
      "position": [x, y, z],
      "orientation": [w, x, y, z],
      "scale": 0.5,  // 可选，如果需要缩放
      "euler": [rx, ry, rz],  // 可选，如果需要旋转
      "explanation": "优化说明"
    }}
  ],
  "optimization_summary": "总体优化说明"
}}
```

注意：
- 只输出JSON，不要包含其他文字
- scale和euler参数是可选的，只在需要时添加
- position和orientation是必需的
- 提供清晰的explanation说明每个优化的原因
"""

    if user_feedback:
        prompt += f"\n\n## 用户反馈\n{user_feedback}\n"

    return prompt


def apply_optimization(env_config: Dict, optimization_result: Dict) -> Dict:
    """
    应用优化结果到环境配置

    Args:
        env_config: 原始环境配置
        optimization_result: LLM生成的优化结果

    Returns:
        优化后的环境配置
    """
    optimized_config = json.loads(json.dumps(env_config))  # 深拷贝

    optimized_components = optimization_result.get('optimized_components', [])

    for opt_comp in optimized_components:
        # 查找对应的组件
        for i, comp in enumerate(optimized_config['task']['components']):
            if comp['name'] == opt_comp['name']:
                # 更新position
                if 'position' in opt_comp:
                    comp['position'] = opt_comp['position']

                # 更新orientation
                if 'orientation' in opt_comp:
                    comp['orientation'] = opt_comp['orientation']

                # 添加scale参数（如果有）
                if 'scale' in opt_comp:
                    if 'randomness' not in comp:
                        comp['randomness'] = {}
                    comp['randomness']['scale'] = opt_comp['scale']

                # 添加euler参数（如果有）
                if 'euler' in opt_comp:
                    # 将euler转换为quaternion并应用到orientation
                    from VLABench.utils.utils import euler_to_quaternion, quaternion_multiply
                    import numpy as np

                    euler = opt_comp['euler']
                    # 将euler角度转换为quaternion
                    euler_quat = np.array(euler_to_quaternion(euler[0], euler[1], euler[2]))

                    # 与当前orientation相乘（如果有的话）
                    current_quat = np.array(comp.get('orientation', [1, 0, 0, 0]))
                    new_quat = quaternion_multiply(current_quat, euler_quat)

                    # 更新orientation
                    comp['orientation'] = new_quat.tolist()

                    # 同时保留euler信息在metadata中，方便调试
                    if 'metadata' not in comp:
                        comp['metadata'] = {}
                    comp['metadata']['euler'] = euler

                logger.info(f"  ✓ 优化 {comp['name']}: {opt_comp.get('explanation', '无说明')}")
                break

    return optimized_config


def optimization_node(state: Dict) -> Dict:
    """
    场景优化节点 - 基于验证报告和渲染结果自动优化配置

    Args:
        state: Agent状态

    Returns:
        更新后的状态
    """
    logger.info("============================================================")
    logger.info("[Optimization] 开始场景优化...")

    validation_report = state.get('validation_report', {})
    env_config = state.get('env_config')
    rendered_images = state.get('rendered_images', [])

    if not env_config:
        logger.error("[Optimization] ✗ 缺少env_config，无法优化")
        return {
            "current_stage": "error",
            "errors": state.get('errors', []) + ["优化失败：缺少env_config"]
        }

    # 1. 分析问题
    issues = analyze_visual_issues(
        rendered_images[0] if rendered_images else None,
        validation_report
    )

    # 如果有用户反馈，强制执行优化
    user_feedback = state.get('user_feedback')
    if not user_feedback and not issues and validation_report.get('overall_status') == 'PASS':
        logger.info("[Optimization] ✓ 场景配置良好，无需优化")
        return {
            "current_stage": "done"
        }

    if user_feedback:
        logger.info(f"[Optimization] 用户反馈: {user_feedback}")
    if issues:
        logger.info(f"[Optimization] 发现 {len(issues)} 个问题")

    # 2. 生成优化提示词
    optimization_prompt = generate_optimization_prompt(
        env_config,
        validation_report,
        user_feedback=state.get('user_feedback')
    )

    # 3. 调用LLM生成优化方案
    try:
        from anthropic import Anthropic
        from scripts.vlabench_agent.config import AgentConfig

        client = Anthropic(
            api_key=AgentConfig.ANTHROPIC_API_KEY,
            base_url=AgentConfig.BASE_URL
        )

        logger.info("[Optimization] 调用LLM生成优化方案...")

        response = client.messages.create(
            model=AgentConfig.MODEL_NAME,
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": optimization_prompt
            }]
        )

        # 解析LLM响应
        content = response.content[0].text

        # 提取JSON（可能被```json包裹）
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        optimization_result = json.loads(content)

        logger.info(f"[Optimization] ✓ 优化方案: {optimization_result.get('optimization_summary', '无说明')}")

        # 4. 应用优化
        optimized_config = apply_optimization(env_config, optimization_result)

        # 5. 保存优化后的配置
        task_save_path = state.get('task_save_path')
        if task_save_path:
            config_path = Path(task_save_path) / "env_config" / "env_config.json"
            with open(config_path, 'w') as f:
                json.dump(optimized_config, f, indent=4)
            logger.info(f"[Optimization] ✓ 保存优化后的配置: {config_path}")

        logger.info("[Optimization] ✓ 优化完成")
        logger.info("============================================================\n")

        return {
            "env_config": optimized_config,
            "current_stage": "re_rendering",  # 需要重新渲染
            "optimization_applied": True,
            "optimization_summary": optimization_result.get('optimization_summary', '')
        }

    except Exception as e:
        logger.error(f"[Optimization] ✗ 优化失败: {e}")
        import traceback
        traceback.print_exc()

        return {
            "current_stage": "error",
            "errors": state.get('errors', []) + [f"优化失败: {str(e)}"]
        }
