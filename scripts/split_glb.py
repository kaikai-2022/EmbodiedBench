#!/usr/bin/env python3
"""
GLB 多几何体拆分脚本

将包含多个独立几何体的 GLB 文件拆分为单独的 GLB 文件，
然后可选地调用 process_local_glb.py 进行完整处理。

Usage:
    # 仅拆分
    python split_glb.py model.glb --output_dir /tmp/split

    # 拆分 + 自动处理
    python split_glb.py model.glb --process --keyword glassware

    # 拆分 + 处理 + 旋转
    python split_glb.py model.glb --process --keyword pipette --rotate_axis x --rotate_degrees -90
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh


def _guess_object_type(name: str, size: np.ndarray) -> str:
    """根据几何体名称和尺寸推测物体类型。"""
    name_lower = name.lower()
    height = max(size)
    diameter = min(size[0], size[1])

    if "erlenmeyer" in name_lower or "flask" in name_lower:
        return "锥形瓶 (Erlenmeyer Flask)"
    elif "cylinder" in name_lower:
        if height < 12:
            return "小量筒 (Small Graduated Cylinder)"
        elif height < 18:
            return "中量筒 (Medium Graduated Cylinder)"
        else:
            return "大量筒 (Large Graduated Cylinder)"
    elif "beaker" in name_lower:
        if diameter < 8:
            return "小烧杯 (Small Beaker)"
        else:
            return "大烧杯 (Large Beaker)"
    elif "tube" in name_lower or "test_tube" in name_lower:
        return "试管 (Test Tube)"
    elif "bottle" in name_lower:
        return "瓶子 (Bottle)"
    elif "bowl" in name_lower:
        return "碗 (Bowl)"
    elif "cup" in name_lower or "mug" in name_lower:
        return "杯子 (Cup/Mug)"
    else:
        return "未知物体"


def load_and_inspect(glb_path: str):
    """加载 GLB 并返回几何体列表。"""
    scene = trimesh.load(glb_path)

    if not hasattr(scene, 'geometry') or len(scene.geometry) == 0:
        print("该 GLB 文件不包含可拆分的几何体。")
        return None

    geometries = []
    for name, geom in scene.geometry.items():
        bounds = geom.bounds
        size = bounds[1] - bounds[0]
        vertices = len(geom.vertices) if hasattr(geom, 'vertices') else 0
        guess = _guess_object_type(name, size)

        geometries.append({
            "name": name,
            "mesh": geom,
            "bounds": bounds,
            "size": size,
            "vertices": vertices,
            "guess": guess,
        })

    return geometries


def interactive_naming(geometries: list) -> dict:
    """交互式命名每个几何体。返回 {文件名: 几何体索引}。"""
    print(f"\n检测到 {len(geometries)} 个几何体：\n")
    print("-" * 60)

    results = {}
    for i, g in enumerate(geometries):
        print(f"  [{i+1}/{len(geometries)}] {g['name']}")
        print(f"    顶点数: {g['vertices']}")
        print(f"    尺寸: {g['size'][0]:.1f} × {g['size'][1]:.1f} × {g['size'][2]:.1f}")
        print(f"    推测: {g['guess']}")

        while True:
            answer = input(f"    输出文件名 (回车跳过, 'done' 结束): ").strip()
            if answer.lower() == "done":
                print()
                return results
            elif answer == "":
                print(f"    → 跳过\n")
                break
            else:
                # 清理文件名
                clean_name = answer.replace(" ", "_").lower()
                if not clean_name.endswith(".glb"):
                    clean_name += ".glb"
                # 检查重名
                if clean_name in results:
                    print(f"    ⚠️  '{clean_name}' 已被使用，请换一个名字")
                    continue
                results[clean_name] = i
                print(f"    → {clean_name}\n")
                break

    return results


def export_geometries(geometries: list, naming: dict, output_dir: Path):
    """将选中的几何体导出为独立 GLB 文件。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = []

    for filename, geom_idx in naming.items():
        g = geometries[geom_idx]
        output_path = output_dir / filename

        # 创建只包含该几何体的新 Scene
        single_scene = trimesh.Scene()
        single_scene.add_geometry(g["mesh"], node_name=g["name"])

        # 导出 GLB
        glb_data = single_scene.export(file_type="glb")
        output_path.write_bytes(glb_data)

        print(f"  ✓ {output_path.name} ({g['vertices']} 顶点, {g['size'][0]:.1f}×{g['size'][1]:.1f}×{g['size'][2]:.1f})")
        exported.append(output_path)

    return exported


def process_exported(exported: list, keyword: str, rotate_axis: str = None, rotate_degrees: float = 0):
    """调用 process_local_glb.py 处理导出的 GLB 文件。"""
    if not exported:
        return

    scripts_dir = Path(__file__).resolve().parent
    process_script = scripts_dir / "process_local_glb.py"

    for glb_path in exported:
        cmd = [
            sys.executable, str(process_script),
            "--input_dir", str(glb_path.parent),
            "--keyword", keyword,
        ]
        if rotate_axis and rotate_degrees != 0:
            cmd.extend(["--rotate_axis", rotate_axis, "--rotate_degrees", str(rotate_degrees)])

        print(f"\n处理: {glb_path.name}")
        print(f"  命令: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(scripts_dir.parent))


def main():
    parser = argparse.ArgumentParser(description="将多几何体 GLB 拆分为单独文件")
    parser.add_argument("glb_file", help="输入 GLB 文件路径")
    parser.add_argument("--output_dir", help="输出目录 (默认: 输入文件同目录下 split/)")
    parser.add_argument("--process", action="store_true", help="拆分后自动调用 process_local_glb.py")
    parser.add_argument("--keyword", help="process 时的 keyword (默认: 从文件名推断)")
    parser.add_argument("--rotate_axis", help="传递给 process_local_glb.py 的旋转轴 (x/y/z)")
    parser.add_argument("--rotate_degrees", type=float, default=0, help="传递给 process_local_glb.py 的旋转角度")

    args = parser.parse_args()

    glb_path = Path(args.glb_file).resolve()
    if not glb_path.exists():
        print(f"文件不存在: {glb_path}")
        sys.exit(1)

    # 输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = glb_path.parent / "split"

    # keyword
    keyword = args.keyword or glb_path.stem

    print(f"输入: {glb_path}")
    print(f"输出: {output_dir}")
    print("=" * 60)

    # 加载并检测几何体
    geometries = load_and_inspect(str(glb_path))
    if geometries is None:
        sys.exit(0)

    if len(geometries) == 1:
        print(f"\n该 GLB 只包含 1 个几何体 ({geometries[0]['name']})，无需拆分。")
        sys.exit(0)

    # 交互式命名
    naming = interactive_naming(geometries)
    if not naming:
        print("没有选择任何几何体，退出。")
        sys.exit(0)

    # 导出
    print(f"导出 {len(naming)} 个几何体到 {output_dir}/")
    print("-" * 60)
    exported = export_geometries(geometries, naming, output_dir)

    print("-" * 60)
    print(f"\n拆分完成! 导出了 {len(exported)} 个文件到 {output_dir}/")

    # 可选处理
    if args.process:
        print("\n" + "=" * 60)
        print("开始自动处理...")
        print("=" * 60)
        process_exported(exported, keyword, args.rotate_axis, args.rotate_degrees)
    else:
        # 提示后续命令
        print(f"\n后续处理命令:")
        print(f"  python scripts/process_local_glb.py \\")
        print(f"    --input_dir {output_dir} \\")
        print(f"    --keyword {keyword}")
        if args.rotate_axis:
            print(f"    --rotate_axis {args.rotate_axis} --rotate_degrees {args.rotate_degrees}")


if __name__ == "__main__":
    main()
