#!/usr/bin/env python3
"""
VLABench Agent 安装验证脚本

检查所有依赖和模块是否正确安装
"""

import sys
import os
from pathlib import Path

# 设置环境
PROJECT_ROOT = Path(__file__).parent
os.environ['VLABENCH_ROOT'] = str(PROJECT_ROOT)

print("=" * 70)
print("🔍 VLABench Agent 安装验证")
print("=" * 70)
print()

# 1. 检查 Python 版本
print("✓ Python 版本")
print("-" * 70)
print(f"  Python: {sys.version.split()[0]}")
print()

# 2. 检查依赖包
print("✓ 依赖包检查")
print("-" * 70)

required_packages = {
    'langgraph': '>=0.2.0',
    'langchain': '>=0.3.0',
    'langchain_core': '>=0.3.0',
    'langchain_anthropic': '>=0.2.0',
    'anthropic': '>=0.40.0',
    'pydantic': '>=2.0.0'
}

all_installed = True
for package, requirement in required_packages.items():
    try:
        module = __import__(package)
        version = getattr(module, '__version__', 'installed')
        print(f"  ✓ {package:25s} {version:15s} {requirement}")
    except ImportError:
        print(f"  ✗ {package:25s} 未安装       {requirement}")
        all_installed = False

print()

# 3. 检查 VLABench Agent 模块
print("✓ Agent 模块检查")
print("-" * 70)

modules_ok = True
try:
    from VLABench.pipeline import build_vlabench_agent, VLABenchAgentState
    print("  ✓ 核心模块 (build_vlabench_agent, VLABenchAgentState)")

    from VLABench.pipeline.nodes import (
        analyzer_node, asset_manager_node, skill_planner_node, condition_planner_node,
        code_generator_node, registration_node, simulation_node, vlm_data_node
    )
    print("  ✓ 节点模块 (8 个节点)")

    from VLABench.pipeline.tools import check_asset_exists, download_asset
    print("  ✓ 工具模块 (check_asset_exists, download_asset)")

    # 尝试构建 Agent
    agent = build_vlabench_agent()
    print(f"  ✓ Agent 构建成功 (类型: {type(agent).__name__})")

except Exception as e:
    print(f"  ✗ 模块导入/构建失败: {e}")
    modules_ok = False

print()

# 4. 检查环境变量
print("✓ 环境变量检查")
print("-" * 70)

vlabench_root = os.environ.get('VLABENCH_ROOT', '未设置')
anthropic_key = os.environ.get('ANTHROPIC_API_KEY', '未设置')

print(f"  VLABENCH_ROOT: {vlabench_root}")

if anthropic_key == '未设置':
    print(f"  ⚠ ANTHROPIC_API_KEY: 未设置")
    print(f"    (需要设置才能运行 Agent)")
else:
    key_preview = anthropic_key[:15] + '...' if len(anthropic_key) > 15 else anthropic_key
    print(f"  ✓ ANTHROPIC_API_KEY: {key_preview}")

print()

# 5. 检查文件结构
print("✓ 文件结构检查")
print("-" * 70)

required_files = [
    'VLABench/pipeline/__init__.py',
    'VLABench/pipeline/agent.py',
    'VLABench/pipeline/state.py',
    'VLABench/pipeline/config.py',
    'VLABench/pipeline/nodes/analyzer.py',
    'VLABench/pipeline/nodes/asset_manager.py',
    'VLABench/pipeline/nodes/skill_planner.py',
    'VLABench/pipeline/nodes/condition_planner.py',
    'VLABench/pipeline/nodes/code_generator.py',
    'VLABench/pipeline/tools/asset_tools.py',
    'vlabench_agent_cli.py',
]

files_ok = True
for file_path in required_files:
    full_path = PROJECT_ROOT / file_path
    if full_path.exists():
        print(f"  ✓ {file_path}")
    else:
        print(f"  ✗ {file_path} (缺失)")
        files_ok = False

print()

# 总结
print("=" * 70)
print("📊 验证总结")
print("=" * 70)

has_api_key = anthropic_key != '未设置'

if all_installed and modules_ok and files_ok:
    print("✅ 所有依赖和模块都已正确安装!")
    print()

    if has_api_key:
        print("🎉 系统完全就绪! 可以立即使用 Agent!")
        print()
        print("运行示例:")
        print("  python3 vlabench_agent_cli.py --instruction '创建一个倾倒试管的任务'")
    else:
        print("⚠️  还需要一步: 设置 ANTHROPIC_API_KEY")
        print()
        print("设置方法:")
        print("  export ANTHROPIC_API_KEY='your-anthropic-api-key'")
        print()
        print("然后运行:")
        print("  python3 vlabench_agent_cli.py --instruction '创建一个倾倒试管的任务'")
else:
    print("❌ 安装未完成")
    print()

    if not all_installed:
        print("问题: 部分依赖包未安装")
        print("解决: pip install -r requirements_agent.txt")
        print()

    if not modules_ok:
        print("问题: Agent 模块导入失败")
        print("解决: 检查是否在正确的目录下,并确保依赖已安装")
        print()

    if not files_ok:
        print("问题: 部分项目文件缺失")
        print("解决: 重新运行 Phase 1 安装脚本")
        print()

print("=" * 70)
