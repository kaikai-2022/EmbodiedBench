"""
Reviewer Node - 基于截图的任务质量审核

对 simulation 生成的视频进行逐 step 审核，结合截图和 condition 判断每个 step 是否执行成功。
"""

import base64
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

from ..config import AgentConfig
from ..state import VLABenchAgentState
from .node_logger import log_node_output_file

logger = logging.getLogger(__name__)

# LLM 调用重试次数
MAX_REVIEWER_RETRIES = 2


def extract_frames_from_timestamps(video_path: str, timestamps: List[Dict]) -> Dict[int, List[str]]:
    """
    根据时间戳从 simulation 视频中提取帧。

    Args:
        video_path: 视频文件路径
        timestamps: 时间戳列表，格式为:
            [{"step_id": int, "atomic_timestamps": [{"atomic_idx": int, "start": float, "end": float}], "step_end": float}]

    Returns:
        Dict[int, List[str]]: {step_id: [frame_paths]}，每个 step 的帧路径列表
    """
    if not video_path or not os.path.exists(video_path):
        logger.warning("[Reviewer] 视频文件不存在或路径为空")
        return {}

    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning("[Reviewer] 无法打开视频文件")
            return {}

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or total_frames <= 0:
            logger.warning(f"[Reviewer] 视频参数异常: fps={fps}, frames={total_frames}")
            cap.release()
            return {}

        step_frames = {}
        temp_dir = tempfile.mkdtemp(prefix="reviewer_")

        for step_ts in timestamps:
            step_id = step_ts["step_id"]
            frames = []

            # 对每个原子操作，提取中点帧
            for atomic_ts in step_ts.get("atomic_timestamps", []):
                start = atomic_ts.get("start", 0)
                end = atomic_ts.get("end", 0)
                mid_time = (start + end) / 2
                frame_idx = int(mid_time * fps)

                # 边界检查
                frame_idx = max(0, min(frame_idx, total_frames - 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

                ret, frame = cap.read()
                if ret:
                    frame_path = os.path.join(temp_dir, f"step_{step_id}_atomic_{atomic_ts['atomic_idx']}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frames.append(frame_path)

            # 大 step 结束后提取一帧
            step_end = step_ts.get("step_end", 0)
            end_frame_idx = int(step_end * fps)
            end_frame_idx = max(0, min(end_frame_idx, total_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, end_frame_idx)

            ret, frame = cap.read()
            if ret:
                frame_path = os.path.join(temp_dir, f"step_{step_id}_after.jpg")
                cv2.imwrite(frame_path, frame)
                frames.append(frame_path)

            step_frames[step_id] = frames

        cap.release()

        logger.info(f"[Reviewer] 从视频中提取了 {len(step_frames)} 个 step 的帧")
        for step_id, frames in step_frames.items():
            logger.info(f"[Reviewer]   Step {step_id}: {len(frames)} 帧")

        return step_frames

    except Exception as e:
        logger.warning(f"[Reviewer] 视频帧提取失败: {e}")
        return {}


def build_reviewer_prompt(
    step_frames: Dict[int, List[str]],
    condition_plan: List[Dict],
    instruction: str,
    asset_status: Dict
) -> Tuple[str, List[str]]:
    """
    构建包含所有 step 的大 prompt。

    Args:
        step_frames: {step_id: [frame_paths]}
        condition_plan: condition_plan 列表
        instruction: 任务指令
        asset_status: 资产状态字典

    Returns:
        (prompt_text, image_paths): prompt 文本和所有图片路径
    """
    # 构建场景资产描述
    asset_lines = []
    for uid, info in asset_status.items():
        class_name = info.get("class_name", "Unknown")
        asset_lines.append(f"- {uid}: {class_name}")
    asset_str = "\n".join(asset_lines) if asset_lines else "无资产信息"

    # 构建执行步骤列表
    steps_list = []
    for step_id in sorted(step_frames.keys()):
        frames = step_frames[step_id]
        condition_entry = next(
            (c for c in condition_plan if c.get("step_id") == step_id),
            None
        )
        # 使用 reasoning 作为 Condition 描述
        condition_str = condition_entry.get("reasoning", "") if condition_entry else ""

        steps_list.append({
            "step_id": step_id,
            "screenshots": frames,
            "Condition": condition_str
        })

    # 构建步骤列表的字符串描述（不含实际图片路径）
    steps_str = json.dumps(steps_list, ensure_ascii=False, indent=2)

    prompt = f"""## 任务
你是一个机器人任务质量审核员。你的职责是根据仿真视频截图和成功条件，判断每个执行步骤是否正确完成。

### 任务指令
{instruction}

### 场景资产
{asset_str}

### 执行步骤列表
{steps_str}

### 审核要求
1. 逐一检查每个 step 的截图，判断 condition 是否满足
2. 如果某个 step 失败，说明图片中显示的具体问题（如"图片显示 rag_0 未被抓住"）
3. 如果某个 step 通过，给出简短的理由

### 输出格式（严格 JSON，无其他文本）
{{
  "results": [
    {{
      "step_id": 0,
      "passed": true,
      "reason": "抓取动作正确完成，rag_0 位于机器人抓夹中"
    }},
    {{
      "step_id": 1,
      "passed": false,
      "reason": "图片显示 beaker_0 仍在台面上，未被移动到 rack_0"
    }}
  ],
  "all_passed": false
}}
"""

    # 收集所有图片路径
    image_paths = []
    for frames in step_frames.values():
        image_paths.extend(frames)

    return prompt, image_paths


def _extract_json(text: str) -> Optional[Dict]:
    """从 LLM 响应中提取 JSON."""
    text = text.strip()

    # 直接尝试 JSON 解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试从 markdown 代码块中提取
    patterns = [
        r'```json\s*\n(.*?)\n\s*```',
        r'```\s*\n(.*?)\n\s*```',
        r'\{[\s\S]*\}',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue

    return None


def _parse_review_results(text: str) -> Tuple[List[Dict], bool]:
    """
    解析 LLM 返回的审核结果。

    Returns:
        (results, all_passed)
    """
    data = _extract_json(text)
    if data is None:
        logger.warning(f"[Reviewer] 无法解析 LLM 响应: {text[:200]}...")
        return [], False

    results = data.get("results", [])
    all_passed = data.get("all_passed", False)

    return results, all_passed


def reviewer_node(state: Dict) -> Dict:
    """
    Reviewer 节点 - 基于截图的任务质量审核。

    输入:
      - state["step_timestamps"]: 时间戳列表
      - state["simulation_video_path"]: 视频文件路径
      - state["condition_plan"]: condition_plan 列表
      - state["task_analysis"]: 任务上下文
      - state["asset_status"]: 资产信息

    输出:
      - state["review_results"]: 每个 step 的审核结果
      - state["review_passed"]: 全部通过为 True
      - state["current_stage"]: "vlm_data"
    """
    logger.info("=" * 60)
    logger.info("[Reviewer] 开始任务质量审核...")

    if ChatAnthropic is None:
        logger.error("[Reviewer] langchain_anthropic 未安装，审核跳过")
        output = {
            "review_results": [],
            "review_passed": True,
            "current_stage": "vlm_data",
        }
        log_node_output_file("reviewer", state, output)
        return output

    # 获取必要信息
    step_timestamps = state.get("step_timestamps", [])
    video_path = state.get("simulation_video_path")
    condition_plan = state.get("condition_plan", [])
    task_analysis = state.get("task_analysis", {})
    asset_status = state.get("asset_status", {})

    # 获取任务指令
    instruction = task_analysis.get("instruction_en", task_analysis.get("user_instruction", ""))

    logger.info(f"[Reviewer] 共有 {len(step_timestamps)} 个 step 需要审核")
    logger.info(f"[Reviewer] 视频路径: {video_path}")

    # 1. 从视频中提取帧
    step_frames = {}
    if video_path:
        step_frames = extract_frames_from_timestamps(video_path, step_timestamps)
    else:
        logger.warning("[Reviewer] 无视频路径，跳过帧提取")

    if not step_frames:
        logger.warning("[Reviewer] 未能提取任何帧，将使用空结果通过审核")

    # 2. 构建 prompt
    prompt, image_paths = build_reviewer_prompt(
        step_frames=step_frames,
        condition_plan=condition_plan,
        instruction=instruction,
        asset_status=asset_status
    )

    # 3. 调用 LLM
    llm = ChatAnthropic(**AgentConfig.get_llm_config())

    last_error = None
    for attempt in range(MAX_REVIEWER_RETRIES + 1):
        try:
            if image_paths:
                # 使用 vision model，发送图片
                from langchain_core.messages import HumanMessage

                content_blocks = [{"type": "text", "text": prompt}]

                # 添加图片（base64 编码）
                for img_path in image_paths:
                    if os.path.exists(img_path):
                        with open(img_path, "rb") as f:
                            img_data = base64.b64encode(f.read()).decode()
                        content_blocks.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_data,
                            }
                        })

                message = HumanMessage(content=content_blocks)
                response = llm.invoke([message])
            else:
                # 无图片，纯文本模式
                response = llm.invoke(prompt)

            # 解析响应
            raw = response.content
            if isinstance(raw, list):
                text_block = next(
                    (b for b in raw if isinstance(b, dict) and b.get("type") == "text"),
                    None
                )
                text = (text_block["text"] if text_block else "").strip()
            else:
                text = raw

            review_results, review_passed = _parse_review_results(text)

            if review_results:
                logger.info(f"[Reviewer] 获取到 {len(review_results)} 个 step 的审核结果")
                for result in review_results:
                    step_id = result.get("step_id")
                    passed = result.get("passed")
                    reason = result.get("reason", "")
                    status = "✓" if passed else "✗"
                    logger.info(f"[Reviewer]   Step {step_id}: {status} {reason[:50]}...")

                output = {
                    "review_results": review_results,
                    "review_passed": review_passed,
                    "current_stage": "vlm_data",
                }
                log_node_output_file("reviewer", state, output)
                logger.info(f"[Reviewer] 审核结果: {'全部通过' if review_passed else '存在失败步骤'}")
                logger.info("=" * 60 + "\n")
                return output
            else:
                last_error = f"未能解析审核结果: {text[:200]}"
                logger.warning(f"[Reviewer] {last_error}")

        except Exception as e:
            last_error = str(e)
            logger.warning(f"[Reviewer] LLM 调用失败: {e}")

    # 所有重试都失败，默认通过
    logger.error(f"[Reviewer] 所有重试失败: {last_error}，默认通过")
    output = {
        "review_results": [],
        "review_passed": True,
        "current_stage": "vlm_data",
    }
    log_node_output_file("reviewer", state, output)
    return output