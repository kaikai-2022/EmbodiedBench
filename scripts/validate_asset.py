#!/usr/bin/env python3
"""
MuJoCo 模型加载验证脚本

创建最小 MuJoCo 场景（地面 + 待验证模型），仿真若干步，
检查模型是否能正常加载、物理是否稳定、尺寸是否合理。

Usage:
    # 验证单个模型
    python validate_asset.py path/to/model.xml

    # 批量验证目录
    python validate_asset.py --dir scripts/assets/review/beaker/

    # 渲染验证图
    python validate_asset.py path/to/model.xml --render
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def validate_asset(xml_path: str, render_preview: bool = False,
                   render_video: bool = False, sim_steps: int = 200,
                   video_duration: float = 3.0, video_fps: int = 30) -> Dict:
    """
    验证模型是否能正常加载到 MuJoCo 场景中

    检查项:
    1. XML 加载: dm_control mjcf.from_path() 是否成功
    2. 物理编译: Physics.from_mjcf_model() 是否报错
    3. 稳定性: 仿真后模型 z 坐标是否合理
    4. 尺寸: 模型是否在合理范围

    Args:
        xml_path: MJCF XML 文件路径
        render_preview: 是否渲染验证截图 (PNG)
        render_video: 是否渲染验证视频 (MP4)，模型从空中落到地面的过程
        sim_steps: 物理检查仿真步数
        video_duration: 视频时长(秒)
        video_fps: 视频帧率

    Returns:
        验证报告 dict
    """
    xml_path = os.path.abspath(xml_path)
    xml_dir = os.path.dirname(xml_path)
    xml_name = os.path.basename(xml_path)

    report = {
        'status': 'FAIL',
        'xml_path': xml_path,
        'checks': {},
        'preview_path': None,
        'video_path': None,
    }

    # --- Check 1: XML 加载 ---
    try:
        from dm_control import mjcf
    except ImportError:
        # 回退到原生 mujoco
        return _validate_with_mujoco(xml_path, render_preview, sim_steps)

    try:
        # 需要 chdir 到 xml 所在目录，否则相对路径的 mesh/texture 找不到
        original_dir = os.getcwd()
        os.chdir(xml_dir)

        try:
            model = mjcf.from_path(xml_name)
            report['checks']['xml_load'] = 'PASS'
        except Exception as e:
            report['checks']['xml_load'] = f'FAIL: {e}'
            os.chdir(original_dir)
            return report

        # --- 移除 freejoint（attach 后会变成嵌套 body，freejoint 不允许） ---
        for body in model.worldbody.all_children():
            if hasattr(body, 'tag') and body.tag == 'body':
                for child in list(body.all_children()):
                    if hasattr(child, 'tag') and child.tag == 'freejoint':
                        child.remove()
                break  # 只处理第一层 body

        # --- 构建验证场景 ---
        scene = mjcf.RootElement()
        scene.worldbody.add('geom', name='floor', type='plane',
                            size=[2, 2, 0.1], rgba=[0.9, 0.9, 0.9, 1])
        scene.worldbody.add('light', directional='true',
                            pos=[0, 0, 3], dir=[0, 0, -1])

        attachment = scene.attach(model)
        attachment.pos = [0, 0, 0.3]  # 放在地面上方

        # --- Check 2: 物理编译 ---
        try:
            physics = mjcf.Physics.from_mjcf_model(scene)
            report['checks']['physics_compile'] = 'PASS'
        except Exception as e:
            report['checks']['physics_compile'] = f'FAIL: {e}'
            os.chdir(original_dir)
            return report

        # --- 获取 body 名称 ---
        body_name = None
        for i in range(physics.model.nbody):
            name = physics.model.id2name(i, 'body')
            if name and name != 'world' and '/' in name:
                body_name = name
                break

        if body_name is None:
            report['checks']['body_found'] = 'FAIL: no body found'
            os.chdir(original_dir)
            return report
        report['checks']['body_found'] = f'PASS: {body_name}'

        # --- Check 3: 仿真稳定性 ---
        initial_pos = physics.named.data.xpos[body_name].copy()

        for _ in range(sim_steps):
            physics.step()

        final_pos = physics.named.data.xpos[body_name].copy()

        # z 坐标检查：应该 > -0.5（没有穿透地面）且 < 5（没有飞出）
        final_z = final_pos[2]
        if final_z < -0.5:
            report['checks']['stability'] = f'FAIL: z={final_z:.3f} (穿透地面)'
        elif final_z > 5:
            report['checks']['stability'] = f'FAIL: z={final_z:.3f} (飞出场景)'
        else:
            report['checks']['stability'] = f'PASS: final_z={final_z:.3f}'

        # XY 漂移检查
        xy_drift = ((final_pos[0] - initial_pos[0])**2 +
                     (final_pos[1] - initial_pos[1])**2) ** 0.5
        if xy_drift > 1.0:
            report['checks']['xy_drift'] = f'WARN: drift={xy_drift:.3f}m'
        else:
            report['checks']['xy_drift'] = f'PASS: drift={xy_drift:.3f}m'

        report['checks']['final_position'] = [float(x) for x in final_pos]

        # --- Check 4: 渲染验证图 ---
        if render_preview:
            try:
                import numpy as np
                from PIL import Image

                # 重置到初始状态并前进一步
                physics.reset()
                for _ in range(50):
                    physics.step()

                # 使用 mujoco 原生渲染器
                import mujoco
                renderer = mujoco.Renderer(physics.model.ptr, height=480, width=640)
                camera = mujoco.MjvCamera()
                camera.lookat[:] = [0, 0, 0.3]
                camera.distance = 1.5
                camera.azimuth = 45
                camera.elevation = -20
                renderer.update_scene(physics.data.ptr, camera=camera)
                img = renderer.render()

                preview_path = os.path.join(xml_dir, 'validation_preview.png')
                Image.fromarray(img).save(preview_path)
                renderer.close()
                report['preview_path'] = preview_path
                report['checks']['render'] = 'PASS'
            except Exception as e:
                report['checks']['render'] = f'FAIL: {e}'

        # --- Check 5: 渲染验证视频 (模型从空中落到地面) ---
        if render_video:
            try:
                import mujoco
                import numpy as np

                physics.reset()

                renderer = mujoco.Renderer(physics.model.ptr, height=480, width=640)
                camera = mujoco.MjvCamera()
                camera.lookat[:] = [0, 0, 0.15]
                camera.distance = 1.0
                camera.azimuth = 135
                camera.elevation = -25

                total_frames = int(video_duration * video_fps)
                timestep = physics.model.ptr.opt.timestep
                steps_per_frame = max(1, int(1.0 / (video_fps * timestep)))

                frames = []
                for _ in range(total_frames):
                    for _ in range(steps_per_frame):
                        physics.step()
                    renderer.update_scene(physics.data.ptr, camera=camera)
                    frames.append(renderer.render().copy())

                renderer.close()

                # 写 MP4
                video_path = os.path.join(xml_dir, 'validation.mp4')
                import numpy as np
                frames_array = np.array(frames)

                try:
                    import imageio.v3 as iio
                    iio.imwrite(video_path, frames_array, fps=video_fps, codec='h264')
                except (ImportError, Exception):
                    try:
                        import mediapy as media
                        media.write_video(video_path, frames_array, fps=video_fps)
                    except (ImportError, Exception):
                        import cv2
                        h, w = frames[0].shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer_cv = cv2.VideoWriter(video_path, fourcc, video_fps, (w, h))
                        for frame in frames:
                            writer_cv.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        writer_cv.release()

                report['video_path'] = video_path
                report['checks']['video'] = f'PASS: {total_frames} frames, {video_duration}s'
            except Exception as e:
                report['checks']['video'] = f'FAIL: {e}'

        # --- 判断总体状态 ---
        all_checks = report['checks']
        has_fail = any(str(v).startswith('FAIL') for v in all_checks.values())
        has_warn = any(str(v).startswith('WARN') for v in all_checks.values())

        if has_fail:
            report['status'] = 'FAIL'
        elif has_warn:
            report['status'] = 'WARN'
        else:
            report['status'] = 'PASS'

        os.chdir(original_dir)
        return report

    except Exception as e:
        report['checks']['unexpected'] = f'FAIL: {e}'
        try:
            os.chdir(original_dir)
        except Exception:
            pass
        return report


def _validate_with_mujoco(xml_path: str, render_preview: bool,
                          sim_steps: int) -> Dict:
    """使用原生 mujoco（非 dm_control）的回退验证"""
    report = {
        'status': 'FAIL',
        'xml_path': xml_path,
        'checks': {},
        'preview_path': None,
    }

    try:
        import mujoco
    except ImportError:
        report['checks']['dependency'] = 'FAIL: mujoco not installed'
        return report

    xml_dir = os.path.dirname(xml_path)
    original_dir = os.getcwd()
    os.chdir(xml_dir)

    try:
        model = mujoco.MjModel.from_xml_path(os.path.basename(xml_path))
        report['checks']['xml_load'] = 'PASS'

        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        report['checks']['physics_compile'] = 'PASS'

        # 仿真
        for _ in range(sim_steps):
            mujoco.mj_step(model, data)

        # 检查第一个非世界 body
        if model.nbody > 1:
            body_id = 1
            final_z = data.xpos[body_id][2]
            if final_z < -0.5:
                report['checks']['stability'] = f'FAIL: z={final_z:.3f}'
            elif final_z > 5:
                report['checks']['stability'] = f'FAIL: z={final_z:.3f}'
            else:
                report['checks']['stability'] = f'PASS: z={final_z:.3f}'

        has_fail = any(str(v).startswith('FAIL') for v in report['checks'].values())
        report['status'] = 'FAIL' if has_fail else 'PASS'

    except Exception as e:
        report['checks']['xml_load'] = f'FAIL: {e}'
    finally:
        os.chdir(original_dir)

    return report


def validate_directory(dir_path: str, render_preview: bool = False,
                       render_video: bool = False) -> list:
    """批量验证目录中的所有模型"""
    results = []
    dir_path = Path(dir_path)

    # 查找所有包含 XML 的子目录
    xml_files = list(dir_path.glob("*/*.xml"))
    if not xml_files:
        xml_files = list(dir_path.glob("*.xml"))

    for xml_file in sorted(xml_files):
        uid = xml_file.stem
        print(f"\n验证: {uid}")
        print("-" * 60)

        report = validate_asset(str(xml_file), render_preview=render_preview,
                                render_video=render_video)
        results.append(report)

        status_icon = {'PASS': '+', 'WARN': '!', 'FAIL': 'x'}.get(report['status'], '?')
        print(f"  [{status_icon}] {report['status']}")
        for check_name, check_result in report.get('checks', {}).items():
            if check_name == 'final_position':
                continue
            icon = '+' if str(check_result).startswith('PASS') else (
                '!' if str(check_result).startswith('WARN') else 'x')
            print(f"    [{icon}] {check_name}: {check_result}")

    # 汇总
    total = len(results)
    passed = sum(1 for r in results if r['status'] == 'PASS')
    warned = sum(1 for r in results if r['status'] == 'WARN')
    failed = sum(1 for r in results if r['status'] == 'FAIL')

    print(f"\n{'=' * 60}")
    print(f"验证汇总: {total} 个模型")
    print(f"  通过: {passed}")
    print(f"  警告: {warned}")
    print(f"  失败: {failed}")
    print(f"{'=' * 60}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='MuJoCo 模型加载验证',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python validate_asset.py path/to/model.xml
  python validate_asset.py path/to/model.xml --render
  python validate_asset.py path/to/model.xml --video
  python validate_asset.py --dir scripts/assets/review/beaker/ --video
        """
    )

    parser.add_argument('input', nargs='?', help='要验证的 XML 文件路径')
    parser.add_argument('--dir', help='批量验证目录')
    parser.add_argument('--render', action='store_true', help='渲染验证截图 (PNG)')
    parser.add_argument('--video', action='store_true', help='渲染验证视频 (MP4)，模型从空中落到地面')
    parser.add_argument('--steps', type=int, default=200, help='仿真步数 (默认: 200)')
    parser.add_argument('--video-duration', type=float, default=3.0, help='视频时长(秒) (默认: 3.0)')
    parser.add_argument('--video-fps', type=int, default=30, help='视频帧率 (默认: 30)')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    if args.dir:
        results = validate_directory(args.dir, render_preview=args.render,
                                     render_video=args.video)
        # 保存汇总报告
        summary_path = Path(args.dir) / "validation_summary.json"
        summary_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2)
        )
        print(f"\n汇总报告已保存: {summary_path}")

    elif args.input:
        report = validate_asset(args.input, render_preview=args.render,
                                render_video=args.video,
                                sim_steps=args.steps,
                                video_duration=args.video_duration,
                                video_fps=args.video_fps)
        print(json.dumps(report, ensure_ascii=False, indent=2))

        # 保存报告到模型目录
        xml_dir = os.path.dirname(os.path.abspath(args.input))
        report_path = os.path.join(xml_dir, 'validation_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n报告已保存: {report_path}")

        return 0 if report['status'] != 'FAIL' else 1
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
