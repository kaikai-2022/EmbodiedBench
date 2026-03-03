#!/usr/bin/env python3
"""
VLM 数据集渲染脚本 - 带实例分割掩码版本

为自定义 VLM 评测任务生成渲染图像:
- input.png: 四视角堆叠的 RGB 图像
- input_mask.png: 彩色实例分割掩码 + 数字标签(与官方数据集一致)

支持命令行参数,可渲染任何任务
"""

import os
import sys
import json
import numpy as np
import cv2
import argparse
from collections import defaultdict

# 设置无头模式(在导入 VLABench 之前)
os.environ['MUJOCO_PY_NO_XR'] = '1'
os.environ['DISPLAY'] = ''
os.environ['MUJOCO_GL'] = 'osmesa'  # 使用 OSMesa 离线渲染，无需 X11 显示

# 设置环境变量(必须在导入 VLABench 之前)
if "VLABENCH_ROOT" not in os.environ:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.environ["VLABENCH_ROOT"] = project_root

# 导入 VLABench 组件
from VLABench import robots
from VLABench import tasks
from VLABench.envs import load_env

# 鲜艳半透明调色板（带 alpha 通道，用于半透明叠加）
COLOR_PALETTE_WITH_ALPHA = [
    [255, 0, 0, 100],      # 0: 半透明红
    [0, 255, 0, 100],      # 1: 半透明绿
    [0, 0, 255, 100],      # 2: 半透明蓝
    [255, 255, 0, 100],    # 3: 半透明黄
    [255, 0, 255, 100],    # 4: 半透明品红
    [0, 255, 255, 100],    # 5: 半透明青
    [255, 128, 0, 100],    # 6: 半透明橙
    [128, 0, 255, 100],    # 7: 半透明紫
    [0, 128, 255, 100],    # 8: 半透明浅蓝
    [255, 0, 128, 100],    # 9: 半透明玫瑰红
    [128, 255, 0, 100],    # 10: 半透明青柠色
    [0, 255, 128, 100],    # 11: 半透明蓝绿色
]

def create_prompt_overlay(rgb_image, segmentation_map, target_geom_to_seq, global_target_geom_ids):
    """
    在原 RGB 图像上创建半透明彩色提示 overlay，只高亮任务物体

    Args:
        rgb_image: (H, W, 3) 原始 RGB 图像
        segmentation_map: (H, W) 或 (H, W, 2) 的分割图
        target_geom_to_seq: 目标 geom ID 到顺序 ID 的映射
        global_target_geom_ids: 目标 geom ID 的集合

    Returns:
        (H, W, 3) 混合后的图像（保留纹理 + 彩色高亮）
    """
    # 提取 instance IDs
    if len(segmentation_map.shape) == 3:
        instance_ids = segmentation_map[:, :, 0].astype(np.int32)
    else:
        instance_ids = segmentation_map.astype(np.int32)

    # overlay 初始化为全黑（将在任务物体位置填充彩色）
    overlay = np.zeros_like(rgb_image)

    # 只为任务物体创建彩色 overlay
    for obj_id in np.unique(instance_ids):
        if obj_id <= 0 or obj_id not in global_target_geom_ids:
            continue  # 只处理任务物体

        seq_id = target_geom_to_seq[obj_id]
        color_with_alpha = COLOR_PALETTE_WITH_ALPHA[seq_id % len(COLOR_PALETTE_WITH_ALPHA)]
        color = np.array(color_with_alpha[:3], dtype=np.uint8)

        mask = (instance_ids == obj_id)
        overlay[mask] = color

    # 混合：原 RGB (70%) + 彩色 overlay (30%)，保留纹理同时添加彩色高亮
    blended = cv2.addWeighted(rgb_image, 0.70, overlay, 0.30, 0)

    return blended

def add_object_labels(image, instance_ids, target_geom_to_seq, font_scale=0.5):
    """
    在图像上添加数字标签（仅为任务物体标注，标签放在物体上方，避免遮挡）

    Args:
        image: (H, W, 3) 图像
        instance_ids: (H, W) instance ID 图
        target_geom_to_seq: 目标 geom ID 到顺序 ID 的映射
        font_scale: 字体大小（调小避免遮挡）

    Returns:
        带标签的图像
    """
    labeled = image.copy()

    for geom_id, seq_id in target_geom_to_seq.items():
        obj_mask = (instance_ids == geom_id)
        y_coords, x_coords = np.where(obj_mask)

        if len(y_coords) < 50:  # 面积阈值，避免小噪声
            continue

        # 计算 bounding box
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        min_x, max_x = np.min(x_coords), np.max(x_coords)

        # x 位置：bbox 中心
        center_x = int((min_x + max_x) / 2)

        # y 位置：标签放在物体顶部上方（避免遮挡主体）
        text = str(seq_id)
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        padding = 4  # 框紧凑点

        # 标签底部 y = 物体顶部 y - padding
        label_bottom_y = min_y - padding
        # 标签顶部 y = label_bottom_y - text_height - baseline
        label_top_y = label_bottom_y - text_height - baseline

        # 如果标签太靠近图像顶部，移到物体内部（备用，避免裁切）
        if label_top_y < 10:
            label_top_y = min_y + padding
            label_bottom_y = label_top_y + text_height + baseline

        top_left = (
            center_x - text_width // 2 - padding,
            label_top_y
        )
        bottom_right = (
            center_x + text_width // 2 + padding,
            label_bottom_y + padding  # 稍多一点底部 padding
        )

        # 绘制黑框
        cv2.rectangle(labeled, top_left, bottom_right, (0, 0, 0), -1)

        # 绘制白字（垂直居中）
        text_pos = (
            center_x - text_width // 2,
            label_bottom_y - padding // 2  # 略向上调整对齐
        )
        cv2.putText(labeled, text, text_pos, font, font_scale, (255, 255, 255), thickness)

    return labeled

def render_vlm_example(env_config_path, output_image_path, output_mask_path, camera_ids=[0, 1, 2, 3]):
    """渲染单个 VLM 样本"""
    print(f"渲染: {env_config_path}")

    with open(env_config_path, 'r') as f:
        env_config = json.load(f)

    try:
        # 创建环境
        env = load_env(
            task="simple_pickplace",
            episode_config=env_config,
            time_limit=float('inf'),
            reset_wait_step=0,
            random_init=False
        )

        # ==================== ID 映射逻辑 ====================
        # 1. 获取任务物体数量
        num_objects = len(env_config['task']['components'])
        print(f"  任务物体数量: {num_objects}")

        # 2. 从第一个成功的相机获取所有 geom ID
        target_geom_to_seq = {}
        global_target_geom_ids = set()

        for cam_id in camera_ids:
            try:
                seg = env.render(camera_id=cam_id, width=256, height=256, segmentation=True)
                if seg is not None:
                    # 提取 instance IDs
                    if len(seg.shape) == 3:
                        instance_ids = seg[:, :, 0].astype(int)
                    else:
                        instance_ids = seg.astype(int)

                    # 获取所有唯一 ID
                    unique_ids = np.unique(instance_ids)
                    unique_ids = unique_ids[unique_ids > 0]  # 排除背景 (ID=0)

                    if len(unique_ids) >= num_objects:
                        # 取最大的 num_objects 个 ID (这些是任务物体)
                        target_geom_ids = sorted(unique_ids)[-num_objects:]
                        global_target_geom_ids = set(target_geom_ids)

                        # 创建映射: geom_id -> sequential_id (0, 1, 2, ...)
                        for seq_id, geom_id in enumerate(target_geom_ids):
                            target_geom_to_seq[geom_id] = seq_id

                        print(f"  目标 geom IDs: {target_geom_ids}")
                        print(f"  ID 映射: {target_geom_to_seq}")
                        break
            except Exception as e:
                print(f"  相机 {cam_id} ID 映射失败: {e}")
                continue

        if not target_geom_to_seq:
            print(f"  警告: 无法创建 ID 映射，将使用原始 ID")

        # ==================== 渲染多视角 ====================
        rgb_images = []
        seg_images = []

        for cam_id in camera_ids:
            try:
                # 1. RGB 图像（作为 base）
                rgb = env.render(camera_id=cam_id, width=256, height=256)
                if rgb is None:
                    print(f"  相机 {cam_id} RGB 渲染失败")
                    continue
                rgb_images.append(rgb)  # 保持原 RGB 用于 input.png

                # 2. 生成 prompt 图像（RGB + 半透明 overlay + 标签）
                try:
                    seg = env.render(camera_id=cam_id, width=256, height=256, segmentation=True)
                    if seg is not None:
                        # 调试信息：打印所有 unique IDs
                        if len(seg.shape) == 3:
                            instance_ids = seg[:, :, 0].astype(int)
                        else:
                            instance_ids = seg.astype(int)

                        print(f"  相机 {cam_id} 所有 unique IDs: {sorted(np.unique(instance_ids))}")

                        # 创建半透明 overlay + blend（保留纹理）
                        blended = create_prompt_overlay(rgb, seg, target_geom_to_seq, global_target_geom_ids)

                        # 添加标签（在 blended 上画）
                        labeled = add_object_labels(blended, instance_ids, target_geom_to_seq)
                        seg_images.append(labeled)  # seg_images 现在是 prompt 图
                    else:
                        # segmentation 不可用，使用原 RGB
                        seg_images.append(rgb.copy())
                except Exception as e:
                    print(f"  相机 {cam_id} prompt 生成失败: {e}")
                    seg_images.append(rgb.copy())

            except Exception as e:
                print(f"  相机 {cam_id} 渲染失败: {e}")

        if len(rgb_images) == 0:
            print("  ✗ 未能渲染任何图像")
            return False

        # 堆叠为 2x2 网格
        if len(rgb_images) == 4:
            # RGB 图像堆叠
            row1 = np.hstack([rgb_images[0], rgb_images[1]])
            row2 = np.hstack([rgb_images[2], rgb_images[3]])
            stacked_rgb = np.vstack([row1, row2])

            # 分割图像堆叠
            if len(seg_images) == 4:
                row1_seg = np.hstack([seg_images[0], seg_images[1]])
                row2_seg = np.hstack([seg_images[2], seg_images[3]])
                stacked_seg = np.vstack([row1_seg, row2_seg])
            else:
                stacked_seg = np.zeros_like(stacked_rgb)
        else:
            stacked_rgb = np.vstack(rgb_images)
            if len(seg_images) == len(rgb_images):
                stacked_seg = np.vstack(seg_images)
            else:
                stacked_seg = np.zeros_like(stacked_rgb)

        # 转换 RGB -> BGR
        if len(stacked_rgb.shape) == 3 and stacked_rgb.shape[2] == 3:
            stacked_bgr = cv2.cvtColor(stacked_rgb, cv2.COLOR_RGB2BGR)
            stacked_seg_bgr = cv2.cvtColor(stacked_seg, cv2.COLOR_RGB2BGR)
        else:
            stacked_bgr = stacked_rgb
            stacked_seg_bgr = stacked_seg

        # 保存
        cv2.imwrite(output_image_path, stacked_bgr)
        cv2.imwrite(output_mask_path, stacked_seg_bgr)

        print(f"  ✓ 保存: {output_image_path} ({stacked_bgr.shape})")
        print(f"  ✓ 保存: {output_mask_path} ({stacked_seg_bgr.shape}) - 彩色分割掩码")

        env.close()
        return True

    except Exception as e:
        print(f"  ✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def render_vlm_task(task_name, num_examples, dimension="M&T", overwrite=False):
    """渲染整个任务"""
    if os.environ['VLABENCH_ROOT'].endswith('/VLABench'):
        base_root = os.path.dirname(os.environ['VLABENCH_ROOT'])
    else:
        base_root = os.environ['VLABENCH_ROOT']
    dataset_root = f"{base_root}/dataset/vlm_evaluation_v1.0/{dimension}/{task_name}"

    print("="*60)
    print("VLM 数据集渲染工具 - 分割掩码版")
    print("="*60)
    print(f"任务: {task_name}")
    print(f"样本数: {num_examples}")
    print(f"维度: {dimension}")
    print(f"路径: {dataset_root}")
    print(f"输出: input.png (RGB) + input_mask.png (彩色分割掩码+标签)")
    print("")

    success = fail = skip = 0

    for i in range(num_examples):
        example_dir = f"{dataset_root}/example{i}"
        env_config = f"{example_dir}/env_config/env_config.json"
        out_img = f"{example_dir}/input/input.png"
        out_mask = f"{example_dir}/input/input_mask.png"

        if not overwrite and os.path.exists(out_img) and os.path.exists(out_mask):
            print(f"example{i}: 已存在,跳过")
            skip += 1
            continue

        if not os.path.exists(env_config):
            print(f"example{i}: 配置不存在,跳过")
            fail += 1
            continue

        print(f"\n渲染 example{i}...")
        if render_vlm_example(env_config, out_img, out_mask):
            success += 1
        else:
            fail += 1

    print("\n" + "="*60)
    print(f"完成! 成功:{success} 跳过:{skip} 失败:{fail}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='VLM 数据集渲染工具 - 分割掩码版')
    parser.add_argument('--task', type=str, required=True, help='任务名称')
    parser.add_argument('--num-examples', type=int, default=None, help='样本数量(默认自动检测)')
    parser.add_argument('--dimension', type=str, default='M&T', help='评测维度')
    parser.add_argument('--overwrite', action='store_true', help='覆盖已存在的图像')
    parser.add_argument('--cameras', type=int, nargs='+', default=[0,1,2,3], help='相机ID')

    args = parser.parse_args()

    # 自动检测样本数
    if args.num_examples is None:
        if os.environ['VLABENCH_ROOT'].endswith('/VLABench'):
            base_root = os.path.dirname(os.environ['VLABENCH_ROOT'])
        else:
            base_root = os.environ['VLABENCH_ROOT']
        dataset_root = f"{base_root}/dataset/vlm_evaluation_v1.0/{args.dimension}/{args.task}"

        if os.path.exists(dataset_root):
            examples = [d for d in os.listdir(dataset_root) if d.startswith('example')]
            args.num_examples = len(examples)
        else:
            print(f"错误: 路径不存在 {dataset_root}")
            sys.exit(1)

    render_vlm_task(args.task, args.num_examples, args.dimension, args.overwrite)

if __name__ == "__main__":
    main()
