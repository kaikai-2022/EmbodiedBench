#!/usr/bin/env python3
"""
任务重命名脚本
用法: python rename_task.py <旧任务名> <新任务名>
示例: python rename_task.py pick_small_beaker_place_small_beaker place_beaker

该脚本会同步修改以下内容：
1. tasks/autogen_tasks/{old_name}_series.py 文件重命名
2. Python 文件内的类名和装饰器注册名
3. tasks/autogen_tasks/__init__.py 中的 import 语句
4. configs/__init__.py 中的 name2config
5. configs/task_config.json 中的 JSON key
"""

import os
import sys
import json
import re
import shutil
import argparse
from pathlib import Path

VLABENCH_ROOT = Path("/ssd/mkqin/workspace/VLABench/VLABench")


def parse_args():
    parser = argparse.ArgumentParser(description="重命名 VLABench 任务")
    parser.add_argument("old_name", help="旧任务名，如 pick_small_beaker_place_small_beaker")
    parser.add_argument("new_name", help="新任务名，如 place_beaker")
    return parser.parse_args()


def to_class_name(name):
    """将任务名转换为类名 (snake_case -> PascalCase)"""
    return "".join(word.capitalize() for word in name.split("_"))


def check_files_exist(old_name):
    """检查相关文件是否存在"""
    autogen_dir = VLABENCH_ROOT / "tasks/autogen_tasks"
    for sub in (autogen_dir, autogen_dir / "primitive"):
        if (sub / f"{old_name}_series.py").exists():
            return True
    return False


def check_name_conflict(new_name):
    """检查新名字是否已存在"""
    autogen_dir = VLABENCH_ROOT / "tasks/autogen_tasks"
    for sub in (autogen_dir, autogen_dir / "primitive"):
        if (sub / f"{new_name}_series.py").exists():
            return f"新文件名已存在: {sub / (new_name + '_series.py')}"

    for init_path in (autogen_dir / "__init__.py", autogen_dir / "primitive" / "__init__.py"):
        if not init_path.exists():
            continue
        content = init_path.read_text()
        if f"from VLABench.tasks.autogen_tasks.{new_name}_series import" in content:
            return f"{init_path.name} 中已存在 {new_name}_series 的 import"
        if f"from VLABench.tasks.autogen_tasks.primitive.{new_name}_series import" in content:
            return f"{init_path.name} 中已存在 primitive.{new_name}_series 的 import"

    task_json = VLABENCH_ROOT / "configs" / "task_config.json"
    with open(task_json) as f:
        if f'"{new_name}_series"' in f.read():
            return f"task_config.json 中已存在 {new_name}_series 的配置"

    return None


def locate_series_file(name, autogen_dir):
    """在 autogen 目录或其 primitive 子目录中定位 series 文件"""
    for sub in (autogen_dir, autogen_dir / "primitive"):
        candidate = sub / f"{name}_series.py"
        if candidate.exists():
            return candidate, sub
    return None, None


def rename_task(old_name, new_name):
    """执行重命名操作"""
    autogen_dir = VLABENCH_ROOT / "tasks/autogen_tasks"
    series_file, source_subdir = locate_series_file(old_name, autogen_dir)
    if series_file is None:
        print(f"错误: 找不到任务文件: {old_name}_series.py")
        sys.exit(1)

    new_series_file = source_subdir / f"{new_name}_series.py"
    init_file = source_subdir / "__init__.py"
    configs_init = VLABENCH_ROOT / "configs" / "__init__.py"
    task_json = VLABENCH_ROOT / "configs" / "task_config.json"

    # 1. 重命名文件
    shutil.move(series_file, new_series_file)
    print(f"✓ 重命名文件: {series_file.name} -> {new_series_file.name}")

    # 2. 修改 Python 文件内容
    old_class_prefix = to_class_name(old_name)
    new_class_prefix = to_class_name(new_name)

    content = new_series_file.read_text()

    # 替换装饰器注册名 (两次: ConfigManager 和 Task)
    content = content.replace(f'@register.add_config_manager("{old_name}")', f'@register.add_config_manager("{new_name}")')
    content = content.replace(f'@register.add_task("{old_name}")', f'@register.add_task("{new_name}")')

    # 替换类名
    content = content.replace(old_class_prefix, new_class_prefix)

    new_series_file.write_text(content)
    print(f"✓ 修改 Python 文件内容: {old_class_prefix}* -> {new_class_prefix}*")

    # 3. 修改子目录对应的 __init__.py (顶层或 primitive/__init__.py)
    init_content = init_file.read_text()
    init_content = init_content.replace(f"{old_name}_series", f"{new_name}_series")
    init_file.write_text(init_content)
    print(f"✓ 修改 {init_file.relative_to(VLABENCH_ROOT)}")

    # 4. 修改 configs/__init__.py
    configs_content = configs_init.read_text()
    configs_content = configs_content.replace(f'"{old_name}_series"', f'"{new_name}_series"')
    configs_content = configs_content.replace(f'["{old_name}"]', f'["{new_name}"]')
    configs_init.write_text(configs_content)
    print(f"✓ 修改 configs/__init__.py")

    # 5. 修改 task_config.json
    with open(task_json, "r") as f:
        json_content = json.load(f)

    if f"{old_name}_series" in json_content:
        json_content[f"{new_name}_series"] = json_content.pop(f"{old_name}_series")

    with open(task_json, "w") as f:
        json.dump(json_content, f, indent=2)
    print(f"✓ 修改 configs/task_config.json")

    print(f"\n重命名完成: {old_name} -> {new_name}")


def main():
    args = parse_args()
    old_name = args.old_name
    new_name = args.new_name

    print(f"开始重命名: {old_name} -> {new_name}\n")

    # 检查旧文件是否存在
    if not check_files_exist(old_name):
        print(f"错误: 找不到任务文件: {old_name}_series.py")
        sys.exit(1)

    # 检查新名字是否冲突
    conflict = check_name_conflict(new_name)
    if conflict:
        print(f"错误: {conflict}")
        sys.exit(1)

    try:
        rename_task(old_name, new_name)
    except Exception as e:
        print(f"\n错误: 重命名失败 - {e}")
        print("所有文件保持不变")
        sys.exit(1)


if __name__ == "__main__":
    main()
