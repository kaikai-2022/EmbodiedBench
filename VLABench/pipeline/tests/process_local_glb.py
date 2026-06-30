#!/usr/bin/env python3
"""
本地 GLB 模型批处理脚本

对本地已有的 GLB 文件运行完整处理流水线：
GLB → OBJ → MJCF → 后处理 → 验证 → 朝向修正 → 尺寸修正

与 get_assets.py 的区别：跳过 Objaverse 搜索和下载步骤，
直接处理用户指定目录下的 GLB 文件。

关键设计:
  - 旋转在 GLB→OBJ 之后、postprocess 之前执行
  - postprocess 后自动补偿 Z 轴偏移（bottom-align），确保 mesh 底部在 body 原点
  - 生成的 XML body pos=[0,0,0]，由 Entity.init_pos 控制世界位置
  - 输出格式与 get_assets.py 完全一致

Usage:
    # 基本用法（不旋转）
    python process_local_glb.py --input_dir /path/to/glbs --keyword microscope

    # 指定旋转（绕X轴旋转+90度）
    python process_local_glb.py --input_dir /path/to/glbs --keyword pipettes_stand --rotate_axis x --rotate_degrees 90

    # 绕Y轴旋转-90度
    python process_local_glb.py --input_dir /path/to/glbs --keyword model --rotate_axis y --rotate_degrees -90

    # 递归扫描子目录
    python process_local_glb.py --input_dir /path/to/glbs --keyword lab_equip --recursive --rotate_axis x --rotate_degrees 90

    # 只处理目录中匹配 --file 的单个模型（不会处理其他 GLB）
    python process_local_glb.py --input_dir /path/to/glbs --keyword thermometer --file thermometer --rotate_axis x --rotate_degrees 90
"""

import argparse
import json
import logging
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./process_local_glb.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 将 scripts 目录加入 path，以便复用 get_assets.py 的方法
_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent.parent  # tests -> pipeline -> VLABench
_TESTS_DIR = _SCRIPTS_DIR
_TOOLS_DIR = _PROJECT_ROOT / "pipeline" / "tools"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

# Import from project root (must be after sys.path setup)
from fix_obj2mjcf_xml import get_obj_dimensions


# ==================== 旋转工具 ====================

def build_rotation_matrix(axis: str, degrees: float) -> np.ndarray:
    """构建绕指定轴旋转的 3x3 旋转矩阵。"""
    rad = math.radians(degrees)
    c, s = math.cos(rad), math.sin(rad)
    if axis.lower() == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis.lower() == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis.lower() == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    else:
        raise ValueError(f"未知旋转轴: {axis}，可选 x/y/z")


def rotate_obj_files(model_dir: Path, uid: str, axis: str, degrees: float):
    """
    旋转模型目录下所有 OBJ 文件的顶点和法线。

    Args:
        model_dir: 模型目录路径
        uid: 模型 UID
        axis: 旋转轴 ("x", "y", "z")
        degrees: 旋转角度（正负均可）
    """
    rot = build_rotation_matrix(axis, degrees)
    obj_files: List[Path] = []

    # 主 OBJ
    main_obj = model_dir / f"{uid}.obj"
    if main_obj.exists():
        obj_files.append(main_obj)

    # collision OBJ（在 uid 子目录下）
    collision_dir = model_dir / uid
    if collision_dir.is_dir():
        for f in collision_dir.glob("*.obj"):
            obj_files.append(f)

    logger.info(f"  旋转 {len(obj_files)} 个 OBJ 文件 (轴={axis}, 角度={degrees}°)")

    for obj_path in obj_files:
        _rotate_single_obj(obj_path, rot)


def _rotate_single_obj(obj_path: Path, rot: np.ndarray):
    """旋转单个 OBJ 文件中的顶点和法线。"""
    lines = []
    with open(obj_path, "r") as f:
        for line in f:
            if line.startswith("v ") and not line.startswith("vt ") and not line.startswith("vn "):
                parts = line.split()
                if len(parts) >= 4:
                    v = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                    v_rot = rot @ v
                    lines.append(f"v {v_rot[0]:.8f} {v_rot[1]:.8f} {v_rot[2]:.8f}\n")
                else:
                    lines.append(line)
            elif line.startswith("vn "):
                parts = line.split()
                if len(parts) >= 4:
                    n = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                    n_rot = rot @ n
                    lines.append(f"vn {n_rot[0]:.8f} {n_rot[1]:.8f} {n_rot[2]:.8f}\n")
                else:
                    lines.append(line)
            else:
                lines.append(line)

    with open(obj_path, "w") as f:
        f.writelines(lines)

    logger.info(f"    已旋转: {obj_path.name}")


# ==================== Bottom-align 工具 ====================

def get_obj_z_range(obj_path: Path) -> tuple:
    """读取 OBJ 文件的 Z 轴最小值和最大值。"""
    z_min, z_max = float('inf'), float('-inf')
    with open(obj_path) as f:
        for line in f:
            if line.startswith("v ") and not line.startswith("vt ") and not line.startswith("vn "):
                parts = line.split()
                if len(parts) >= 4:
                    z = float(parts[3])
                    z_min = min(z_min, z)
                    z_max = max(z_max, z)
    return z_min, z_max


def shift_obj_z(obj_path: Path, dz: float):
    """将 OBJ 文件所有顶点的 Z 坐标平移 dz。"""
    verts = []
    with open(obj_path) as f:
        for line in f:
            if line.startswith("v ") and not line.startswith("vt ") and not line.startswith("vn "):
                parts = line.split()
                if len(parts) >= 4:
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not verts:
        return
    arr = np.array(verts)
    arr[:, 2] += dz

    out = []
    idx = 0
    with open(obj_path) as f:
        for line in f:
            if line.startswith("v ") and not line.startswith("vt ") and not line.startswith("vn "):
                v = arr[idx]
                out.append(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
                idx += 1
            else:
                out.append(line)
    with open(obj_path, "w") as f:
        f.writelines(out)


def bottom_align_all(model_dir: Path, uid: str):
    """
    将所有 OBJ 文件的 Z 轴 bottom-align（Z_min=0）。
    以主 OBJ 的 Z_min 为基准，所有文件统一平移。
    """
    main_obj = model_dir / f"{uid}.obj"
    z_min, _ = get_obj_z_range(main_obj)

    if abs(z_min) < 0.001:
        logger.info(f"  Bottom-align: Z_min={z_min:.4f} 已接近0，无需调整")
        return 0.0

    offset = -z_min
    logger.info(f"  Bottom-align: Z 上移 {offset:.4f} (Z_min={z_min:.4f} → 0)")

    shift_obj_z(main_obj, offset)
    collision_dir = model_dir / uid
    if collision_dir.is_dir():
        for c in collision_dir.glob("*.obj"):
            shift_obj_z(c, offset)

    # 同步子目录中的 visual OBJ 副本
    sub_vis = collision_dir / f"{uid}.obj"
    if sub_vis.exists():
        shutil.copy2(str(main_obj), str(sub_vis))

    return offset


# ==================== XML 修正工具 ====================

def fix_xml_after_postprocess(model_dir: Path, uid: str, scale: float = None, mass: float = 0.02):
    """
    postprocess_model 之后，因为质心居中的问题，mesh 底部不在 Z=0。

    正确逻辑（三步）：
    1. 从 XML 读取实际 mesh scale
    2. 将所有 OBJ（主+collision）上移 half_height，使 Z_min=0
    3. 用正确的几何值重写 XML 中的 inertial/top_site/bottom_site

    Args:
        model_dir: 模型目录
        uid: 模型 UID
        scale: 废弃参数，保留兼容性，实际从 XML 读取
        mass: 物体质量
    """
    import re

    # 从 XML 读取实际 mesh scale（postprocess 写入的值）
    xml_path = model_dir / f"{uid}.xml"
    xml_content = xml_path.read_text(encoding='utf-8')
    scale_match = re.search(r'<mesh file="[^"]*\.obj" scale="([0-9.e+-]+)', xml_content)
    if scale_match:
        actual_scale = float(scale_match.group(1))
        logger.info(f"  从 XML 读取 mesh scale: {actual_scale:.6f}")
    else:
        actual_scale = 1.0
        logger.warning(f"  未找到 mesh scale，使用 1.0")

    main_obj = model_dir / f"{uid}.obj"
    dims = get_obj_dimensions(str(main_obj))

    z_min = dims['min_z']
    z_range = dims['z_range']
    z_max = dims['max_z']

    # mesh 高度（OBJ 坐标 × scale = 世界米）
    mesh_height = z_range * actual_scale
    half_height = mesh_height / 2
    top_height = z_max * actual_scale

    logger.info(f"  postprocess 后 OBJ Z: [{z_min:.4f}, {z_max:.4f}], Z_range={z_range:.4f}")
    logger.info(f"  mesh 高度: {mesh_height:.6f}m, 半高: {half_height:.6f}m")

    # Step 1: 如果 z_min < 0，上移所有 OBJ 使 Z_min=0
    if z_min < -0.001:
        dz_obj = -z_min
        logger.info(f"  上移 OBJ: dz={dz_obj:.4f} (Z_min={z_min:.4f} → 0)")
        shift_obj_z(main_obj, dz_obj)
        collision_dir = model_dir / uid
        if collision_dir.is_dir():
            for c in collision_dir.glob("*.obj"):
                shift_obj_z(c, dz_obj)
        sub_vis = collision_dir / f"{uid}.obj"
        if sub_vis.exists():
            shutil.copy2(str(main_obj), str(sub_vis))
    else:
        logger.info(f"  OBJ Z_min={z_min:.4f} 已在0附近，跳过上移")

    # Step 2: 重新从上移后的 OBJ 计算正确值
    dims2 = get_obj_dimensions(str(main_obj))
    z_range2 = dims2['z_range']
    mesh_h2 = z_range2 * actual_scale
    half2 = mesh_h2 / 2
    top2 = dims2['max_z'] * actual_scale

    logger.info(f"  上移后: Z=[{dims2['min_z']:.4f}, {dims2['max_z']:.4f}], top={top2:.6f}m")

    # Step 3: 重写 XML
    xml_path = model_dir / f"{uid}.xml"
    content = xml_path.read_text(encoding='utf-8')

    # 替换 inertial
    content = re.sub(
        r'<inertial[^>]*/>',
        f'<inertial pos="0 0 {half2:.6f}" mass="{mass}" diaginertia="{mass*0.04} {mass*0.04} {mass*0.025}"/>',
        content
    )
    logger.info(f"  inertial: pos=(0, 0, {half2:.6f})")

    # 替换 top_site: 在 mesh 顶部
    content = re.sub(
        r'<site name="top_site"[^>]*/>',
        f'<site name="top_site" pos="0 0 {top2:.6f}" size="0.01" rgba="1 0 0 0"/>',
        content
    )
    logger.info(f"  top_site: pos=(0, 0, {top2:.6f})")

    # 替换 bottom_site: 在 mesh 底部
    content = re.sub(
        r'<site name="bottom_site"[^>]*/>',
        '<site name="bottom_site" pos="0 0 0.0" size="0.01" rgba="0 1 0 0"/>',
        content
    )
    logger.info(f"  bottom_site: pos=(0, 0, 0.0)")

    # 替换 grasppoint: 三个点分布在 60%/80%/95% 高度处
    gp_heights = [mesh_h2 * 0.6, mesh_h2 * 0.8, mesh_h2 * 0.95]
    gp_pattern = re.compile(r'<site class="grasppoint"[^>]*/>')
    gp_matches = list(gp_pattern.finditer(content))
    # 从后往前替换，避免偏移量变化
    for i, m in reversed(list(enumerate(gp_matches[:3]))):
        h = gp_heights[i]
        replacement = f'<site class="grasppoint" pos="0.0000 0.0000 {h:.4f}"/>'
        content = content[:m.start()] + replacement + content[m.end():]
    logger.info(f"  grasppoints: pos=(0,0,{gp_heights[0]:.4f}), (0,0,{gp_heights[1]:.4f}), (0,0,{gp_heights[2]:.4f})")

    # 确保 body 没有 pos 属性
    content = re.sub(
        r'(<body name="[^"]*")[^>]*>',
        r'\1>',
        content
    )

    xml_path.write_text(content, encoding='utf-8')
    logger.info(f"  XML 已重写，body pos=[0,0,0]")


# ==================== Pipeline ====================

class LocalGLBPipeline:
    """本地 GLB 模型处理 Pipeline。"""

    def __init__(
        self,
        input_dir: str,
        keyword: str,
        output_dir: str = None,
        skip_existing: bool = True,
        merge_meshes: bool = True,
        recursive: bool = False,
        rotate_axis: str = None,
        rotate_degrees: float = 0,
        target_file: str = None,
    ):
        self.input_dir = Path(input_dir)
        self.keyword = keyword.lower().replace(" ", "_")
        self.merge_meshes = merge_meshes
        self.recursive = recursive
        self.skip_existing = skip_existing
        self.rotate_axis = rotate_axis
        self.rotate_degrees = rotate_degrees
        self.target_file = target_file

        # 默认输出目录
        if output_dir is None:
            output_dir = str(_PROJECT_ROOT / "VLABench" / "assets" / "review")
        self.base_dir = Path(output_dir) / self.keyword
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.stats = {
            'total': 0,
            'processed': 0,
            'failed': 0,
            'skipped': 0,
        }

        logger.info(f"输入目录: {self.input_dir}")
        logger.info(f"输出目录: {self.base_dir}")
        if self.rotate_axis:
            logger.info(f"旋转: 绕{self.rotate_axis.upper()}轴 {self.rotate_degrees}°")

    def scan_glb_files(self) -> Dict[str, Path]:
        """扫描输入目录中的 GLB 文件。"""
        if not self.input_dir.exists():
            raise FileNotFoundError(f"输入目录不存在: {self.input_dir}")

        pattern = "**/*.glb" if self.recursive else "*.glb"
        glb_files = list(self.input_dir.glob(pattern))

        if self.target_file:
            glb_files = [p for p in glb_files if p.stem == self.target_file]
            if not glb_files:
                raise FileNotFoundError(
                    f"在 {self.input_dir} 中未找到 {self.target_file}.glb（--file 指定）"
                )
            logger.info(f"--file 指定只处理: {self.target_file}.glb")

        if not glb_files:
            logger.warning(f"在 {self.input_dir} 中未找到 GLB 文件")
            return {}

        result = {}
        for glb_path in sorted(glb_files):
            uid = glb_path.stem
            if uid in result:
                uid = f"{glb_path.parent.name}_{uid}"
            result[uid] = glb_path

        logger.info(f"找到 {len(result)} 个 GLB 文件")
        for uid, path in result.items():
            logger.info(f"  {uid}: {path}")
        return result

    def process_single_glb(self, uid: str, glb_path: Path) -> bool:
        """处理单个本地 GLB 文件，走完整流水线。"""
        model_dir = self.base_dir / uid

        if self.skip_existing and (model_dir / "preview.png").exists():
            logger.info(f"跳过已存在: {uid}")
            self.stats['skipped'] += 1
            return True

        model_dir.mkdir(exist_ok=True)

        # 保存元数据和来源信息
        self._save_local_metadata(uid, glb_path, model_dir)
        self._save_local_source_info(uid, glb_path, model_dir)

        # 复制 GLB
        dest_glb = model_dir / f"{uid}.glb"
        shutil.copy2(glb_path, dest_glb)

        # 获取 AssetPipeline 实例复用方法
        pipeline = self._get_pipeline()

        # Step 1: GLB → OBJ
        from get_assets import glb_to_obj
        obj_path = model_dir / f"{uid}.obj"
        if not glb_to_obj(str(dest_glb), str(obj_path), self.merge_meshes):
            logger.error(f"GLB→OBJ 转换失败: {uid}")
            self.stats['failed'] += 1
            return False

        # Step 2: 旋转（如果指定）
        if self.rotate_axis and self.rotate_degrees != 0:
            logger.info(f"  [{uid}] 旋转: 绕{self.rotate_axis.upper()}轴 {self.rotate_degrees}°")
            rotate_obj_files(model_dir, uid, self.rotate_axis, self.rotate_degrees)

        # Step 3: OBJ → MJCF
        xml_path = pipeline.convert_obj_to_mjcf(model_dir)
        if not xml_path:
            logger.error(f"OBJ→MJCF 转换失败: {uid}")
            self.stats['failed'] += 1
            return False

        # 重命名为标准名称
        target_xml = model_dir / f"{uid}.xml"
        if xml_path != target_xml:
            xml_path.rename(target_xml)

        # 修复 XML 资源路径
        pipeline.fix_xml_asset_paths(target_xml, uid)

        # Step 4: 后处理（居中 + 缩放 + 物理属性）
        pipeline.postprocess_model(model_dir, uid, target_xml)

        # Step 5: Bottom-align + XML 修正（关键步骤！）
        # postprocess 会把质心居中到原点，导致 mesh 底部在 Z<0
        # 底部对齐：上移 OBJ 使 Z_min=0，并修正 XML 中的 inertial/site 值
        fix_xml_after_postprocess(model_dir, uid, mass=0.02)

        # Step 6: 渲染预览图（可选，跳过以避免无显示环境下的 OpenGL crash）
        # pipeline.render_model_preview(model_dir, uid)

        # Step 7: 验证
        pipeline.validate_model(model_dir, uid, target_xml)

        # Step 8: 尺寸修正（LLM）
        pipeline.fix_model_size(model_dir, uid, target_xml)

        # Step 9: size_fix 可能改变了 scale，重新同步 inertial/site/grasppoint
        size_report = model_dir / "size_report.json"
        if size_report.exists():
            import json as _json
            rpt = _json.loads(size_report.read_text())
            if rpt.get("action") == "rescaled":
                logger.info(f"  size_fix 改变了 scale，重新 bottom-align 并同步 XML 位置值...")
                bottom_align_all(model_dir, uid)
                fix_xml_after_postprocess(model_dir, uid, mass=0.02)

        self.stats['processed'] += 1
        logger.info(f"✓ 完成: {uid} -> {model_dir}")
        return True

    def _get_pipeline(self):
        """获取 AssetPipeline 实例复用方法。"""
        from get_assets import AssetPipeline
        pipeline = AssetPipeline.__new__(AssetPipeline)
        pipeline.keyword = self.keyword
        pipeline.search_terms = self.keyword.split("_")
        pipeline.merge_meshes = self.merge_meshes
        pipeline.base_dir = self.base_dir
        pipeline.skip_existing = self.skip_existing
        pipeline.stats = {}
        pipeline.context_profile = None
        return pipeline

    def _save_local_metadata(self, uid: str, glb_path: Path, model_dir: Path):
        metadata = {
            "uid": uid,
            "name": uid,
            "source": "local",
            "original_path": str(glb_path.resolve()),
            "keyword": self.keyword,
            "tags": [],
            "categories": [],
            "description": f"Local GLB model: {glb_path.name}",
            "rotation": {"axis": self.rotate_axis, "degrees": self.rotate_degrees} if self.rotate_axis else None,
        }
        (model_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))

    def _save_local_source_info(self, uid: str, glb_path: Path, model_dir: Path):
        lines = [
            f"Model UID: {uid}",
            f"Model Name: {uid}",
            "",
            "Source Information:",
            "=" * 60,
            "Source: Local File",
            f"Original Path: {glb_path.resolve()}",
            f"File Name: {glb_path.name}",
            f"Keyword/Category: {self.keyword}",
        ]
        if self.rotate_axis:
            lines.append(f"Rotation: {self.rotate_degrees}° around {self.rotate_axis.upper()} axis")
        lines += [
            "",
            "Processing:",
            "Pipeline: process_local_glb.py",
            "Steps: GLB->OBJ (trimesh) -> [rotate] -> MJCF (obj2mjcf) -> postprocess -> bottom-align -> validate -> size_fix",
        ]
        (model_dir / "sources.txt").write_text('\n'.join(lines), encoding='utf-8')

    def run(self):
        """运行完整批处理流程。"""
        logger.info("=" * 60)
        logger.info(f"本地 GLB 批处理: keyword={self.keyword}")
        if self.rotate_axis:
            logger.info(f"旋转参数: {self.rotate_degrees}° around {self.rotate_axis.upper()}")
        logger.info("=" * 60)

        glb_files = self.scan_glb_files()
        if not glb_files:
            logger.warning("没有可处理的 GLB 文件，退出")
            return

        self.stats['total'] = len(glb_files)

        for uid, glb_path in glb_files.items():
            logger.info(f"\n[{uid}] 处理: {glb_path.name}")
            try:
                self.process_single_glb(uid, glb_path)
            except Exception as e:
                logger.error(f"处理 {uid} 时出错: {e}")
                import traceback
                logger.error(traceback.format_exc())
                self.stats['failed'] += 1

        logger.info("=" * 60)
        logger.info("处理完成!")
        logger.info(f"  总数:   {self.stats['total']}")
        logger.info(f"  成功:   {self.stats['processed']}")
        logger.info(f"  失败:   {self.stats['failed']}")
        logger.info(f"  跳过:   {self.stats['skipped']}")
        logger.info(f"  输出:   {self.base_dir}")
        logger.info("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(
        description="本地 GLB 模型批处理脚本 — 完整流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 不旋转
  python process_local_glb.py --input_dir /path/to/glbs --keyword microscope

  # 绕X轴旋转+90度（常见：侧躺模型→竖立）
  python process_local_glb.py --input_dir /path/to/glbs --keyword pipettes_stand \\
      --rotate_axis x --rotate_degrees 90

  # 绕X轴旋转-90度
  python process_local_glb.py --input_dir /path/to/glbs --keyword model \\
      --rotate_axis x --rotate_degrees -90

  # 递归扫描
  python process_local_glb.py --input_dir /path/to/glbs --keyword lab_equip \\
      --recursive --rotate_axis x --rotate_degrees 90
        """
    )

    parser.add_argument('--input_dir', type=str, required=True,
                        help='包含 .glb 文件的输入目录')
    parser.add_argument('--keyword', type=str, required=True,
                        help='物体类别名称，用作输出子目录名')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出根目录（默认: VLABench/VLABench/assets/review）')
    parser.add_argument('--skip_existing', action='store_true',
                        help='跳过已存在 preview.png 的模型目录')
    parser.add_argument('--no_merge_meshes', action='store_true',
                        help='不合并 GLB 中的所有网格')
    parser.add_argument('--recursive', action='store_true',
                        help='递归扫描子目录中的 GLB 文件')
    parser.add_argument('--rotate_axis', type=str, default=None,
                        choices=['x', 'y', 'z'],
                        help='旋转轴（x/y/z），需配合 --rotate_degrees 使用')
    parser.add_argument('--rotate_degrees', type=float, default=0,
                        help='旋转角度（正负均可，如 90/-90/180），需配合 --rotate_axis 使用')
    parser.add_argument('--file', type=str, default=None,
                        help='只处理输入目录中文件名（不含扩展名）匹配此值的单个 GLB；'
                             '不指定则处理目录下所有 .glb 文件')

    return parser.parse_args()


def main():
    args = parse_args()

    # 参数校验
    if args.rotate_axis and args.rotate_degrees == 0:
        logger.warning("--rotate_axis 已指定但 --rotate_degrees 为 0，不执行旋转")

    pipeline = LocalGLBPipeline(
        input_dir=args.input_dir,
        keyword=args.keyword,
        output_dir=args.output_dir,
        skip_existing=args.skip_existing,
        merge_meshes=not args.no_merge_meshes,
        recursive=args.recursive,
        rotate_axis=args.rotate_axis,
        rotate_degrees=args.rotate_degrees,
        target_file=args.file,
    )

    pipeline.run()


if __name__ == "__main__":
    main()
