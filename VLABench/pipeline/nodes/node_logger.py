"""
Node Logger - 节点输出日志工具

将每个节点的输出 state 序列化到日志文件，供核对调试。
每次运行生成独立文件，文件名含时间戳。
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# 脱敏字段
SENSITIVE_KEYS = {"messages", "error_traceback"}


def _sanitize_for_json(obj):
    """将对象转为 JSON 可序列化格式"""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        return repr(obj)


def _mask_sensitive(obj, sensitive_keys: set):
    """对敏感字段进行脱敏"""
    if not isinstance(obj, dict):
        return obj
    result = {}
    for k, v in obj.items():
        if k in sensitive_keys:
            result[k] = f"<{len(str(v))} chars>"
        elif isinstance(v, dict):
            result[k] = _mask_sensitive(v, sensitive_keys)
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            result[k] = [_mask_sensitive(item, sensitive_keys) for item in v]
        else:
            result[k] = v
    return result


def init_run_log(instruction: str) -> str:
    """
    初始化一次 pipeline 运行的日志文件。
    在 pipeline 入口调用一次，返回日志文件路径字符串。

    Args:
        instruction: 用户指令文本（用于文件名）

    Returns:
        日志文件路径字符串
    """
    log_root = Path(os.environ.get("VLABENCH_LOG_ROOT", "/ssd/mkqin/workspace/VLABench/logs"))
    log_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in instruction[:20])
    filename = f"pipeline_{timestamp}_{safe_name}.log"
    log_filepath = log_root / filename

    header = f"""{'=' * 80}
Pipeline Log
指令: {instruction}
时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{'=' * 80}
"""
    log_filepath.write_text(header, encoding="utf-8")
    logger.info(f"[NodeLogger] 日志文件: {log_filepath}")
    return str(log_filepath)


def log_node_output_file(node_name: str, state: Dict, output: Dict) -> None:
    """
    将节点输出追加到当前运行的日志文件。
    不同节点间用长横线分隔。

    Args:
        node_name: 节点名称
        state: 节点输入（含 _log_filepath）
        output: 节点输出
    """
    log_filepath_str = state.get("_log_filepath")
    if not log_filepath_str:
        logger.warning(f"[NodeLogger] 无 _log_filepath，跳过日志")
        return

    log_filepath = Path(log_filepath_str)
    if not log_filepath.exists():
        log_filepath.write_text("", encoding="utf-8")

    entry = {
        "node": node_name,
        "timestamp": datetime.now().isoformat(),
        "output_keys": list(output.keys()),
        "output": output,
    }
    entry = _mask_sensitive(entry, SENSITIVE_KEYS)

    try:
        entry_json = json.dumps(entry, indent=2, ensure_ascii=False, default=_sanitize_for_json)
        separator = f"\n{'—' * 80}\n"
        block = f"{separator}[{node_name}] {datetime.now().strftime('%H:%M:%S')}\n{separator}\n{entry_json}\n"

        with open(log_filepath, "a", encoding="utf-8") as f:
            f.write(block)

        logger.info(f"[NodeLogger] ✓ {node_name} → {log_filepath.name}")
    except Exception as e:
        logger.error(f"[NodeLogger] ✗ {node_name} 日志写入失败: {e}")
