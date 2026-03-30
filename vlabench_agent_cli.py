#!/usr/bin/env python3
"""
VLABench Agent CLI - 命令行接口

使用自然语言指令自动生成 VLABench 任务
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR  # CLI is in project root
sys.path.insert(0, str(PROJECT_ROOT))

# 设置 VLABENCH_ROOT 环境变量 (指向 VLABench 包目录，包含 assets/ 和 configs/)
if "VLABENCH_ROOT" not in os.environ:
    os.environ["VLABENCH_ROOT"] = str(PROJECT_ROOT / "VLABench")

from scripts.vlabench_agent.agent import build_vlabench_agent, create_initial_state
from scripts.vlabench_agent.config import AgentConfig


def setup_logging(verbose: bool = False):
    """配置日志"""
    from datetime import datetime
    from pathlib import Path

    level = logging.DEBUG if verbose else logging.INFO
    format_str = '%(asctime)s - %(levelname)s - %(message)s'

    # 创建日志目录
    log_dir = Path("/ssd/mkqin/workspace/VLABench/logs/任务生成日志")
    log_dir.mkdir(parents=True, exist_ok=True)

    # 生成时间戳文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"agent_{timestamp}.log"

    # 配置日志：同时输出到控制台和文件
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding='utf-8')
    ]

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=handlers
    )

    print(f"📝 日志保存至: {log_file}")


def print_banner():
    """打印启动横幅"""
    print("=" * 70)
    print("🤖 VLABench Agent - 自然语言任务生成智能体")
    print("=" * 70)
    print()


def print_result(result: dict):
    """打印执行结果"""
    print("\n" + "=" * 70)

    if result.get('current_stage') == 'done':
        print("任务完成!")
        print("=" * 70)

        # 打印任务信息
        task_analysis = result.get('task_analysis', {})
        if task_analysis:
            print(f"\n任务信息:")
            print(f"  - 任务名称: {task_analysis.get('task_name')}")
            print(f"  - 物体列表: {', '.join(task_analysis.get('objects', []))}")
            print(f"  - 操作类型: {task_analysis.get('operation_type')}")
            print(f"  - 英文指令: {task_analysis.get('instruction_en')}")

        # 打印生成的代码路径
        task_module_path = result.get('task_module_path')
        if task_module_path:
            print(f"\n生成的任务类: {task_module_path}")

        # 打印仿真结果
        if result.get('simulation_success'):
            print(f"\n仿真结果: 成功")
            if result.get('simulation_video_path'):
                print(f"  - 视频: {result['simulation_video_path']}")
            if result.get('simulation_hdf5_path'):
                print(f"  - HDF5: {result['simulation_hdf5_path']}")

        # 打印技能序列
        skill_seq = result.get('executed_skill_sequence', [])
        if skill_seq:
            print(f"\n执行的技能序列:")
            for i, skill in enumerate(skill_seq):
                print(f"  {i+1}. {skill.get('name')} {skill.get('params', {})}")

        # 打印 VLM 评测数据路径
        print(f"\nVLM 评测数据: {result.get('task_save_path')}")
        rendered = result.get('rendered_images', [])
        if rendered:
            print(f"渲染图像:")
            for img in rendered:
                print(f"  - {img}")

        # 打印警告
        warnings = result.get('warnings', [])
        if warnings:
            print(f"\n警告 ({len(warnings)}):")
            for warn in warnings:
                print(f"  - {warn}")

    else:
        print("❌ 任务失败")
        print("=" * 70)
        errors = result.get('errors', [])
        if errors:
            print(f"\n错误 ({len(errors)}):")
            for err in errors:
                print(f"  - {err}")

    print("\n" + "=" * 70)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="VLABench 自然语言任务生成 Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础用法
  python vlabench_agent_cli.py --instruction "创建一个倾倒试管的任务"

  # 复杂任务
  python vlabench_agent_cli.py --instruction "创建一个将盖玻片放在显微镜台上的任务"

  # 调试模式
  python vlabench_agent_cli.py --instruction "创建一个举起烧杯的任务" --verbose

环境变量:
  VLABENCH_ROOT: VLABench 项目根目录 (默认: 当前目录的父目录)
  ANTHROPIC_API_KEY: Claude API 密钥 (必需)
        """
    )

    parser.add_argument(
        '--instruction',
        type=str,
        required=True,
        help='任务描述 (自然语言,中文或英文)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细输出模式'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='可选: 提供 YAML 配置文件路径 (暂未实现)'
    )

    args = parser.parse_args()

    # 配置日志
    setup_logging(args.verbose)

    # 打印横幅
    print_banner()

    # 检查 API 配置 (支持环境变量或 config.py)
    api_key = os.environ.get("ANTHROPIC_API_KEY") or AgentConfig.ANTHROPIC_API_KEY
    if not api_key:
        print("❌ 错误: 未设置 ANTHROPIC_API_KEY")
        print("\n方式 1: 设置环境变量")
        print("  export ANTHROPIC_API_KEY='your-api-key'")
        print("\n方式 2: 修改配置文件")
        print("  编辑 scripts/vlabench_agent/config.py")
        print("  设置 ANTHROPIC_API_KEY = 'your-api-key'")
        sys.exit(1)

    print(f"📝 用户指令: {args.instruction}")
    print(f"📂 项目根目录: {os.environ.get('VLABENCH_ROOT', PROJECT_ROOT)}")
    print(f"🔑 API 配置: {'环境变量' if os.environ.get('ANTHROPIC_API_KEY') else 'config.py'}")
    print(f"🌐 API 端点: {AgentConfig.BASE_URL or '默认官方端点'}")
    print()

    # 初始化 Agent
    try:
        agent = build_vlabench_agent()
    except Exception as e:
        print(f"❌ Agent 初始化失败: {e}")
        sys.exit(1)

    # 创建初始状态
    initial_state = create_initial_state(args.instruction)

    # 运行 Agent
    try:
        print("🚀 开始执行任务...\n")
        result = agent.invoke(
            initial_state,
            config={"configurable": {"thread_id": "default"}}
        )

        # 输出结果
        print_result(result)

        # 返回退出码
        if result.get('current_stage') == 'done':
            sys.exit(0)
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(130)

    except Exception as e:
        print(f"\n\n❌ 执行异常: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
