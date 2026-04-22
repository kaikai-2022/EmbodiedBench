"""Apply targeted edits to code_generator.py using Python file manipulation."""
import re

path = "/ssd/mkqin/workspace/VLABench/scripts/vlabench_agent/nodes/code_generator.py"
with open(path) as f:
    content = f.read()

# Fix 1: Rename target_uid -> target_entity_name in _format_skill_line_v2
old = '''def _format_skill_line_v2(entry: Dict, uid_to_spec: Dict) -> str:
    """新格式：将单个 atomic_sequence 条目转为 partial(...) 调用字符串

    新格式中 params 的 uid 直接是 entity name（spec），不需要占位符替换。
    """
    skill = entry.get("skill", "")
    params = entry.get("params", {})

    # uid → spec 映射
    resolved_params = {}
    for k, v in params.items():
        if isinstance(v, str):
            resolved_params[k] = uid_to_spec.get(v, v)
        elif isinstance(v, list):
            resolved_params[k] = [
                uid_to_spec.get(item, item) if isinstance(item, str) else item
                for item in v
            ]
        else:
            resolved_params[k] = v'''

new = '''def _format_skill_line_v2(entry: Dict, uid_to_spec: Dict) -> str:
    """新格式：将单个 atomic_sequence 条目转为 partial(...) 调用字符串"""
    skill = entry.get("skill", "")
    params = entry.get("params", {})

    # uid → spec 映射，同时重命名 SkillLib 参数名
    resolved_params = {}
    param_rename = {
        "target_uid": "target_entity_name",
        "target_container_name": "target_container",
    }
    for k, v in params.items():
        # 重命名参数名以匹配 SkillLib 签名
        k = param_rename.get(k, k)
        if isinstance(v, str):
            resolved_params[k] = uid_to_spec.get(v, v)
        elif isinstance(v, list):
            resolved_params[k] = [
                uid_to_spec.get(item, item) if isinstance(item, str) else item
                for item in v
            ]
        else:
            resolved_params[k] = v'''

if old in content:
    content = content.replace(old, new)
    print("Fixed _format_skill_line_v2: added param_rename")
else:
    print("WARNING: _format_skill_line_v2 not found exactly")
    idx = content.find("def _format_skill_line_v2")
    if idx >= 0:
        print(f"Found at {idx}: {content[idx:idx+500]}")

with open(path, 'w') as f:
    f.write(content)

# Verify syntax
import subprocess
result = subprocess.run(['python3', '-m', 'py_compile', path], capture_output=True, text=True)
if result.returncode == 0:
    print(f"OK: {path}")
else:
    print(f"ERROR: {result.stderr.decode()}")
