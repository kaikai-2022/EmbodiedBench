#!/usr/bin/env python3
"""
自动化模型资产下载与转换 Pipeline (使用 trimesh，无 Blender 依赖)
从 Objaverse 下载模型，转换为 MuJoCo MJCF 格式，并生成预览图

Usage:
    python get_assets.py --keyword tube --max_downloads 10
    python get_assets.py --keyword bottle --max_downloads 20 --output_dir ./assets/review
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

# ==================== 重要：HF镜像配置 ====================
# 必须在导入 objaverse 之前设置环境变量
# 如果环境中已经设置了 HF_ENDPOINT，优先使用环境变量
# 否则，检查是否在中国大陆，如果是则使用镜像
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
else:
    logger_temp = logging.getLogger(__name__)
    logger_temp.info(f"检测到 HF_ENDPOINT 环境变量: {os.environ['HF_ENDPOINT']}")
# =========================================================

# 可选依赖 - 在需要时才导入
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

# ==================== 上下文搜索配置 ====================
# 预定义的搜索上下文，用于优先筛选特定领域的模型
CONTEXT_PROFILES = {
    "lab": {
        "boost_categories": ["science-technology"],
        "penalize_categories": ["cultural-heritage-history", "art-abstract"],
        "boost_keywords": [
            "lab", "laboratory", "science", "scientific", "chemistry",
            "chemical", "experiment", "research", "glass", "glassware",
            "equipment", "medical", "biology", "physics", "beaker",
            "flask", "pipette", "microscope", "centrifuge", "tube",
            "syringe", "petri", "burette", "funnel", "mortar",
        ],
    },
    "home": {
        "boost_categories": ["furniture-home", "food-drink"],
        "penalize_categories": ["cultural-heritage-history", "weapons-military"],
        "boost_keywords": [
            "home", "furniture", "kitchen", "household", "domestic",
            "room", "indoor", "living", "table", "chair", "cup",
            "plate", "bowl", "spoon", "fork", "knife",
        ],
    },
}
# ===========================================================

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./get_assets.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def glb_to_obj(glb_path: str, obj_path: str, merge_meshes: bool = True) -> bool:
    """
    使用 trimesh 将 GLB 转换为 OBJ（无需 Blender）

    Args:
        glb_path: 输入的 GLB 文件路径
        obj_path: 输出的 OBJ 文件路径
        merge_meshes: 是否合并所有网格为单个物体
                     - True: 合并所有几何体（推荐，更简单）
                     - False: 保留场景结构（可能生成 .mtl）

    Returns:
        bool: 转换是否成功
    """
    try:
        import trimesh
    except ImportError:
        logger.error("trimesh 未安装，请运行: pip install trimesh")
        return False

    try:
        # 加载 GLB
        loaded = trimesh.load(glb_path, force='mesh' if merge_meshes else None)

        if isinstance(loaded, trimesh.Scene):
            if merge_meshes:
                # 合并所有几何体为单个 mesh
                meshes = list(loaded.geometry.values())
                if not meshes:
                    raise ValueError("GLB 中没有几何体")

                # 过滤掉空 mesh
                valid_meshes = [m for m in meshes if hasattr(m, 'vertices') and len(m.vertices) > 0]

                if not valid_meshes:
                    raise ValueError("GLB 中没有有效的几何体")

                combined = trimesh.util.concatenate(valid_meshes)
                combined.export(obj_path)
                logger.debug(f"合并了 {len(valid_meshes)} 个网格")
            else:
                # 导出整个 scene（会尝试保留材质）
                loaded.export(obj_path)
        elif isinstance(loaded, trimesh.Trimesh):
            # 已经是单个 mesh
            loaded.export(obj_path)
        else:
            raise ValueError(f"未知的加载类型: {type(loaded)}")

        logger.info(f"GLB → OBJ: {Path(glb_path).name} → {Path(obj_path).name}")
        return True

    except Exception as e:
        logger.error(f"trimesh 转换失败 {glb_path}: {e}")
        return False


def render_mjcf(xml_path: str, save_path: str, width: int = 640, height: int = 480) -> bool:
    """使用 MuJoCo headless 渲染预览图"""
    try:
        import mujoco
    except ImportError:
        logger.error("mujoco 未安装")
        return False

    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)

        # 初始化
        mujoco.mj_forward(model, data)

        # 渲染
        renderer = mujoco.Renderer(model, height=height, width=width)

        # 尝试使用默认相机或第一个相机
        if model.n_cam > 0:
            renderer.update_scene(data, camera=0)
        else:
            renderer.update_scene(data)

        img = renderer.render()

        if Image is None:
            logger.error("PIL 未安装，无法保存图片")
            return False

        Image.fromarray(img).save(save_path)
        renderer.close()

        logger.info(f"预览图: {save_path.name}")
        return True

    except Exception as e:
        logger.warning(f"渲染失败: {e}")
        return False


class AssetPipeline:
    """模型资产下载与转换 Pipeline (使用 trimesh，无 Blender)"""

    def __init__(
        self,
        keyword: str,
        max_downloads: int = 10,
        min_vertices: int = 400,
        output_dir: str = './assets/review',
        skip_existing: bool = True,
        merge_meshes: bool = True,
        context: Optional[str] = None
    ):
        self.keyword = keyword.lower().replace(" ", "_")  # 用于目录名
        self.search_terms = keyword.lower().split()       # 用于搜索匹配（保留多词）
        self.max_downloads = max_downloads
        self.min_vertices = min_vertices
        self.base_dir = Path(output_dir) / self.keyword
        self.skip_existing = skip_existing
        self.merge_meshes = merge_meshes

        # 加载上下文配置
        self.context_profile = None
        if context:
            if context in CONTEXT_PROFILES:
                self.context_profile = CONTEXT_PROFILES[context]
                logger.info(f"使用搜索上下文: {context}")
            else:
                logger.warning(f"未知上下文 '{context}'，可选: {list(CONTEXT_PROFILES.keys())}")

        # 统计信息
        self.stats = {
            'total': 0,
            'downloaded': 0,
            'converted': 0,
            'failed': 0,
            'skipped': 0
        }

        # 创建输出目录
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"输出目录: {self.base_dir}")

    def check_dependencies(self) -> bool:
        """检查必要的依赖"""
        logger.info("检查依赖...")

        # 检查 trimesh
        try:
            import trimesh
            logger.info(f"✓ trimesh: {trimesh.__version__}")
        except ImportError:
            logger.error("✗ trimesh 未安装 (pip install trimesh)")
            return False

        # 检查 obj2mjcf
        try:
            import obj2mjcf
            logger.info(f"✓ obj2mjcf: {obj2mjcf.__version__}")
        except ImportError:
            logger.error("✗ obj2mjcf 未安装 (pip install obj2mjcf)")
            return False

        # 检查 mujoco
        try:
            import mujoco
            logger.info(f"✓ mujoco: {mujoco.__version__}")
        except ImportError:
            logger.warning("✗ mujoco 未安装 (渲染功能不可用)")
            # 不返回 False，因为只是渲染不可用

        # 检查 objaverse
        try:
            import objaverse
            logger.info(f"✓ objaverse: {objaverse.__version__ if hasattr(objaverse, '__version__') else 'installed'}")
        except ImportError:
            logger.error("✗ objaverse 未安装 (pip install objaverse)")
            return False

        return True

    def search_objaverse(self) -> Tuple[list, dict]:
        """从 Objaverse 搜索匹配的模型"""
        logger.info("正在搜索 Objaverse...")

        try:
            from objaverse import load_annotations
        except ImportError:
            logger.error("objaverse 库未安装")
            return [], {}

        # 加载注解
        logger.info("加载 Objaverse 元数据（首次运行需要几分钟）...")
        try:
            annotations = load_annotations()
            logger.info(f"已加载 {len(annotations)} 个模型元数据")
        except Exception as e:
            logger.error(f"加载元数据失败: {e}")
            return [], {}

        # 搜索匹配
        candidates = []
        iterator = annotations.items()
        if tqdm is not None:
            iterator = tqdm(iterator, desc="搜索模型")

        for uid, ann in iterator:
            name = (ann.get("name") or "").lower()
            tags = [t.get("name", "").lower() for t in ann.get("tags", [])]
            description = (ann.get("description") or "").lower()

            if all(
                (term in name or
                 any(term in t for t in tags) or
                 term in description)
                for term in self.search_terms
            ):
                candidates.append((uid, ann))

        logger.info(f"找到 {len(candidates)} 个匹配 '{' '.join(self.search_terms)}' 的模型")

        # 排序：优先级规则
        # 1. 上下文类别加分/减分（如 science-technology 加分）
        # 2. 上下文关键词加分（tags/description 中的领域词）
        # 3. 名称中包含关键词的优先
        # 4. 名称完全匹配或开头匹配的优先
        # 5. 有名称的优先
        # 6. 社区质量评分（likeCount，同等条件下优先高赞模型）
        context_profile = self.context_profile

        def rank_model(item):
            uid, ann = item
            name = (ann.get("name") or "").lower()
            tags = [t.get("name", "").lower() for t in ann.get("tags", [])]
            categories = [c.get("name", "").lower() for c in ann.get("categories", [])]
            description = (ann.get("description") or "").lower()

            # --- 上下文评分 ---
            category_score = 0
            context_keyword_score = 0

            if context_profile:
                # 类别加分/减分
                for cat in categories:
                    if cat in context_profile["boost_categories"]:
                        category_score += 2
                    if cat in context_profile["penalize_categories"]:
                        category_score -= 2

                # 上下文关键词加分（在 name + tags + description 中搜索）
                all_text = name + " " + " ".join(tags) + " " + description
                hits = sum(1 for kw in context_profile["boost_keywords"] if kw in all_text)
                context_keyword_score = min(hits, 3)  # 上限 3 分

            # --- 原有评分 ---
            search_phrase = " ".join(self.search_terms)
            name_match = 3 if search_phrase in name else 0
            name_starts = 1 if name.startswith(search_phrase) else 0
            has_name = 1 if ann.get("name") else 0

            # --- 质量评分（likeCount）---
            like_count = ann.get("likeCount", 0) or 0

            return (category_score, context_keyword_score, name_match, name_starts, has_name, like_count)

        candidates.sort(key=rank_model, reverse=True)

        selected = candidates[:self.max_downloads]
        uids = [uid for uid, _ in selected]
        annotations_dict = {uid: ann for uid, ann in selected}

        self.stats['total'] = len(uids)
        return uids, annotations_dict

    def download_models(self, uids: list) -> dict:
        """从 Objaverse 下载模型"""
        logger.info(f"开始下载 {len(uids)} 个模型...")

        try:
            from objaverse import load_objects
        except ImportError:
            logger.error("objaverse 库未安装")
            return {}

        try:
            # 显示下载的 UID 信息
            logger.info(f"准备下载的模型 UID: {uids}")

            objects = load_objects(uids=uids, download_processes=6)
            logger.info(f"成功下载 {len(objects)} 个模型")

            # 显示下载结果
            for uid, path in objects.items():
                logger.debug(f"已下载: {uid} -> {path}")

            return objects
        except Exception as e:
            logger.error(f"下载失败: {e}")
            import traceback
            logger.error(f"详细错误:\n{traceback.format_exc()}")
            return {}

    def convert_obj_to_mjcf(self, obj_dir: Path) -> Optional[Path]:
        """
        使用 obj2mjcf 将 OBJ 转换为 MJCF

        注意：obj2mjcf 处理整个目录，不是单个文件
        """
        try:
            # 获取 obj2mjcf 的路径（从当前 Python 环境）
            import sys
            obj2mjcf_path = os.path.join(sys.prefix, 'bin', 'obj2mjcf')

            result = subprocess.run(
                [
                    obj2mjcf_path,
                    "--obj-dir", str(obj_dir),
                    "--save-mjcf",
                    "--overwrite",
                    "--add-free-joint",
                    "--decompose",  # 使用 CoACD 进行凸分解
                    "--verbose"
                ],
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )

            # 查找生成的 MJCF 文件
            # obj2mjcf 会在子目录中生成 xml（与 obj 文件同名）
            mjcf_files = list(obj_dir.glob("*/*.xml"))
            # ���果没找到，再在当前目录查找
            if not mjcf_files:
                mjcf_files = list(obj_dir.glob("*.xml"))

            if mjcf_files:
                # obj2mjcf 可能生成多个 xml，取最大的那个（通常是主文件）
                mjcf_files.sort(key=lambda p: p.stat().st_size, reverse=True)
                generated = mjcf_files[0]
                logger.debug(f"OBJ→MJCF: {generated.name}")
                return generated
            else:
                logger.warning("未找到生成的 MJCF 文件")
                if result.stderr:
                    logger.debug(f"stderr: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            logger.error("OBJ→MJCF 转换超时")
            return None
        except Exception as e:
            logger.error(f"OBJ→MJCF 转换异���: {e}")
            return None

    def process_model(self, uid: str, glb_path: Path, metadata: dict) -> bool:
        """处理单个模型"""
        model_dir = self.base_dir / uid

        # 检查是否跳过
        if self.skip_existing and (model_dir / "preview.png").exists():
            logger.info(f"跳过已存在: {uid}")
            self.stats['skipped'] += 1
            return True

        model_dir.mkdir(exist_ok=True)

        # 保存元数据
        (model_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2)
        )

        # 保存来源链接到 sources.txt
        self.save_source_info(uid, metadata, model_dir)

        # 复制 GLB
        shutil.copy2(glb_path, model_dir / f"{uid}.glb")

        # 转换为 OBJ (使用 trimesh)
        obj_path = model_dir / f"{uid}.obj"
        if not glb_to_obj(str(glb_path), str(obj_path), self.merge_meshes):
            self.stats['failed'] += 1
            return False

        # 转换为 MJCF (使用 obj2mjcf)
        xml_path = self.convert_obj_to_mjcf(model_dir)
        if not xml_path:
            self.stats['failed'] += 1
            return False

        # 重命名为标准名称
        target_xml = model_dir / f"{uid}.xml"
        if xml_path != target_xml:
            xml_path.rename(target_xml)
            xml_path = target_xml

        # 修复 XML 中的资源路径（贴图和网格）
        self.fix_xml_asset_paths(target_xml, uid)

        # [DEBUG] 打印 fix_xml_asset_paths 之后、postprocess 之前的 XML 内容片段
        logger.info(f"  [DEBUG get_assets] XML after fix_xml_asset_paths, before postprocess:")
        with open(target_xml, 'r') as f:
            content = f.read()
        # 检查是否有 <body> 标签
        if '<body' in content:
            logger.info(f"  [DEBUG get_assets] ✓ <body> found in XML before postprocess")
        else:
            logger.info(f"  [DEBUG get_assets] ✗ <body> NOT found in XML before postprocess")
        # 打印 worldbody 段
        import re
        wb_match = re.search(r'<worldbody>.*?</worldbody>', content, re.DOTALL)
        if wb_match:
            logger.info(f"  [DEBUG get_assets] worldbody section:\n{wb_match.group()[:500]}")

        # 后处理：几何中心归零 + 尺寸归一化 + 物理属性修复
        self.postprocess_model(model_dir, uid, target_xml)

        # [DEBUG] 打印 postprocess 之后的 XML 内容片段
        logger.info(f"  [DEBUG get_assets] XML after postprocess:")
        with open(target_xml, 'r') as f:
            content = f.read()
        if '<body' in content:
            logger.info(f"  [DEBUG get_assets] ✓ <body> found in XML after postprocess")
        else:
            logger.info(f"  [DEBUG get_assets] ✗ <body> NOT found in XML after postprocess")
        wb_match = re.search(r'<worldbody>.*?</worldbody>', content, re.DOTALL)
        if wb_match:
            logger.info(f"  [DEBUG get_assets] worldbody section after postprocess:\n{wb_match.group()[:500]}")

        # 渲染预览图
        self.render_model_preview(model_dir, uid)

        # [DEBUG] 检查渲染后 XML 是否被破坏
        logger.info(f"  [DEBUG get_assets] XML after render_model_preview:")
        with open(target_xml, 'r') as f:
            content = f.read()
        if '<body' in content:
            logger.info(f"  [DEBUG get_assets] ✓ <body> still intact after render")
        else:
            logger.info(f"  [DEBUG get_assets] ✗ <body> LOST after render!")

        # MuJoCo 加载验证
        self.validate_model(model_dir, uid, target_xml)

        # 朝向修正（基于 LLM 视觉判断）
        self.fix_model_orientation(model_dir, uid, target_xml)

        # 尺寸修正（基于 LLM 常识判断合理物理尺寸）
        self.fix_model_size(model_dir, uid, target_xml)

        self.stats['converted'] += 1
        logger.info(f"✓ 完成: {uid}")
        return True

    def postprocess_model(self, model_dir: Path, uid: str, target_xml: Path):
        """后处理：几何中心归零、尺寸归一化、物理属性修复"""
        try:
            # 添加项目根目录到 sys.path 以便导入 fix_obj2mjcf_xml
            project_root = Path(__file__).resolve().parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

            from fix_obj2mjcf_xml import postprocess_model as _postprocess

            report = _postprocess(
                model_dir=str(model_dir),
                uid=uid,
                mass=0.02,
                target_max_dim=0.15,
                remove_freejoint=True,
                add_grasppoints=True,
            )

            # 保存后处理报告
            report_path = model_dir / "postprocess_report.json"
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2)
            )
            logger.info(f"  后处理报告: {report_path.name}")

        except Exception as e:
            logger.warning(f"后处理失败 {uid}: {e}")
            import traceback
            logger.warning(f"后处理完整traceback:\n{traceback.format_exc()}")

    def validate_model(self, model_dir: Path, uid: str, target_xml: Path):
        """MuJoCo 加载验证"""
        try:
            scripts_dir = Path(__file__).resolve().parent
            tests_dir = scripts_dir.parent / "tests"
            for d in [scripts_dir, tests_dir]:
                if str(d) not in sys.path:
                    sys.path.insert(0, str(d))

            from validate_asset import validate_asset

            report = validate_asset(str(target_xml), render_preview=False,
                                    render_video=True)

            # 保存验证报告
            report_path = model_dir / "validation_report.json"
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2)
            )

            status = report.get('status', 'UNKNOWN')
            if status == 'FAIL':
                logger.warning(f"  模型验证失败: {uid}")
            elif status == 'WARN':
                logger.warning(f"  模型验证警告: {uid}")
            else:
                logger.info(f"  模型验证通过: {uid}")

        except ImportError:
            logger.debug("validate_asset 模块不可用，跳过验证")
        except Exception as e:
            logger.warning(f"验证失败 {uid}: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def fix_model_orientation(self, model_dir: Path, uid: str, target_xml: Path):
        """朝向修正：基于 LLM 视觉判断"""
        try:
            from orientation_fix import fix_orientation

            object_name = self.keyword
            report = fix_orientation(
                model_dir=str(model_dir),
                uid=uid,
                object_name=object_name,
                target_xml=str(target_xml),
            )

            # 保存朝向修正报告
            report_path = model_dir / "orientation_report.json"
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2)
            )

            if report.get("action") == "rotated":
                logger.info(f"  朝向已修正: {uid} (旋转 {report['rotation']})")
            elif report.get("action") == "no_change":
                logger.info(f"  朝向正确: {uid}")
            else:
                logger.info(f"  朝向检查: {report.get('action', 'unknown')}")

        except Exception as e:
            logger.warning(f"朝向修正失败 {uid}: {e}")
            import traceback
            logger.warning(traceback.format_exc())

    def fix_model_size(self, model_dir: Path, uid: str, target_xml: Path):
        """尺寸修正：基于 LLM 常识判断合理物理尺寸"""
        try:
            from size_fix import fix_size

            object_name = self.keyword
            report = fix_size(
                model_dir=str(model_dir),
                uid=uid,
                object_name=object_name,
                target_xml=str(target_xml),
            )

            # 保存尺寸修正报告
            report_path = model_dir / "size_report.json"
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2)
            )

            if report.get("action") == "rescaled":
                logger.info(f"  尺寸已修正: {uid} ({report['old_height_m']:.3f}m -> {report['new_height_m']:.3f}m)")
            else:
                logger.info(f"  尺寸检查: {report.get('action', 'unknown')}")

        except Exception as e:
            logger.warning(f"尺寸修正失败 {uid}: {e}")
            import traceback
            logger.warning(traceback.format_exc())

    def fix_xml_asset_paths(self, xml_path: Path, uid: str):
        """
        修复 XML 中的资源路径

        obj2mjcf 会在子目录中生成碰撞网格和贴图，但 XML 被移动到父目录后，
        需要更新资源路径以指向子目录

        Args:
            xml_path: XML 文件路径
            uid: 模型 UID（也是子目录名）
        """
        try:
            import re

            with open(xml_path, 'r', encoding='utf-8') as f:
                xml_content = f.read()

            # 修复贴图路径：default.png -> uid/default.png
            # 匹配 <texture ... file="default.png"/>
            xml_content = re.sub(
                r'(<texture[^>]+file=")default\.png(")',
                rf'\g<1>{uid}/default.png\g<2>',
                xml_content
            )

            # 修复碰撞网格路径：uid_collision_N.obj -> uid/uid_collision_N.obj
            # 匹配 <mesh file="uid_collision_N.obj"/>
            xml_content = re.sub(
                rf'(<mesh\s+file=")({uid}_collision_\d+\.obj)(")',
                rf'\g<1>{uid}/\g<2>\g<3>',
                xml_content
            )

            # 写回文件
            with open(xml_path, 'w', encoding='utf-8') as f:
                f.write(xml_content)

            logger.debug(f"✓ 修复资源路径: {xml_path.name}")

        except Exception as e:
            logger.warning(f"修复资源路径失败 {xml_path.name}: {e}")

    def save_source_info(self, uid: str, metadata: dict, model_dir: Path):
        """保存模型来源信息到 sources.txt"""
        source_file = model_dir / "sources.txt"

        lines = []
        lines.append(f"Model UID: {uid}")
        lines.append(f"Model Name: {metadata.get('name', 'Unknown')}")
        lines.append("")
        lines.append("Source Information:")
        lines.append("=" * 60)

        # 获取来源类型
        source = metadata.get('source', 'unknown')

        if source == 'sketchfab':
            # Sketchfab 模型
            sketchfab_uid = metadata.get('uid', '')
            sketchfab_id = sketchfab_uid.split('-')[-1] if '-' in sketchfab_uid else sketchfab_uid
            lines.append(f"Source: Sketchfab")
            lines.append(f"Sketchfab URL: https://sketchfab.com/3d-models/{sketchfab_id}")
            lines.append(f"Sketchfab UID: {sketchfab_uid}")

        elif source == 'thingiverse':
            # Thingiverse 模型
            thingiverse_id = metadata.get('thingiverse', {}).get('id', 'unknown')
            lines.append(f"Source: Thingiverse")
            lines.append(f"Thingiverse URL: https://www.thingiverse.com/thing:{thingiverse_id}")
            lines.append(f"Thingiverse ID: {thingiverse_id}")

        elif source == 'github':
            # GitHub 模型
            github_repo = metadata.get('github', {})
            lines.append(f"Source: GitHub")
            lines.append(f"Repository: {github_repo.get('repo', 'unknown')}")
            lines.append(f"GitHub URL: https://github.com/{github_repo.get('repo', '')}")
            lines.append(f"Commit: {github_repo.get('commit', 'unknown')}")

        else:
            # 其他来源或未知
            lines.append(f"Source: {source}")
            lines.append(f"Objaverse UID: {uid}")

        lines.append("")
        lines.append("HuggingFace Objaverse:")
        lines.append(f"Dataset: https://huggingface.co/datasets/allenai/objaverse")
        lines.append(f"Object Path: glbs/{uid[0:2]}/{uid}.glb")

        lines.append("")
        lines.append("Additional Metadata:")
        lines.append(f"Tags: {', '.join([t.get('name', '') for t in metadata.get('tags', [])[:5]])}")
        description = metadata.get('description', '')[:200]
        if description:
            lines.append(f"Description: {description}...")

        # 写入文件
        source_file.write_text('\n'.join(lines), encoding='utf-8')
        logger.debug(f"保存来源信息: {source_file.name}")

    def render_model_preview(self, model_dir: Path, uid: str):
        """为模型生成预览图"""
        preview_path = model_dir / "preview.png"

        # 尝试多种方式渲染预览图
        success = False

        # 方式 1: 优先使用 obj2mjcf 生成的完整 MJCF XML（包含贴图和材质）
        target_xml = model_dir / f"{uid}.xml"
        if target_xml.exists():
            success = self.render_preview_with_mjcf(str(target_xml), str(preview_path))

        # 方式 2: 如果 XML 渲染失败，回退到 OBJ 直接渲染（无贴图但更可靠）
        if not success:
            obj_path = model_dir / f"{uid}.obj"
            if obj_path.exists():
                success = self.render_preview_with_obj(str(obj_path), str(preview_path))

        if success:
            logger.debug(f"✓ 预览图已生成: {preview_path.name}")
        else:
            logger.warning(f"✗ 预览图生成失败: {uid}")

    def render_preview_with_mjcf(self, xml_path: str, save_path: str) -> bool:
        """使用 MJCF XML 渲染预览图"""
        try:
            import mujoco
            from PIL import Image
            import numpy as np
            import trimesh

            xml_dir = os.path.dirname(xml_path)
            xml_filename = os.path.basename(xml_path)
            uid = os.path.splitext(xml_filename)[0]

            original_dir = os.getcwd()
            # 在改变工作目录前，将 save_path 转换为绝对路径
            save_path = os.path.abspath(save_path)
            os.chdir(xml_dir)

            try:
                # 修复 XML 中的 collision mesh 路径
                with open(xml_filename, 'r') as f:
                    xml_content = f.read()

                # 将 collision mesh 路径添加子目录前缀
                import re
                fixed_xml = re.sub(
                    r'file="(' + uid + r'_collision_\d+\.obj)"',
                    r'file="' + uid + r'/\1"',
                    xml_content
                )

                # 写入临时修复的 XML
                temp_xml = 'temp_fixed.xml'
                with open(temp_xml, 'w') as f:
                    f.write(fixed_xml)

                try:
                    model = mujoco.MjModel.from_xml_path(temp_xml)
                    data = mujoco.MjData(model)
                    mujoco.mj_forward(model, data)

                    # 读取OBJ获取模型中心和大小，用于相机定位
                    obj_path = f'{uid}.obj'
                    if os.path.exists(obj_path):
                        mesh = trimesh.load(obj_path)
                        center = mesh.centroid
                        bbox_size = np.linalg.norm(mesh.extents)
                    else:
                        center = np.array([0, 0, 0])
                        bbox_size = 1.0

                    renderer = mujoco.Renderer(model, height=480, width=640)

                    # 设置相机对准原点，距离根据模型大小调整
                    camera = mujoco.MjvCamera()
                    camera.lookat[:] = [0, 0, 0]
                    camera.distance = bbox_size * 2.5
                    camera.azimuth = 45
                    camera.elevation = 0  # 水平视角

                    renderer.update_scene(data, camera=camera)
                    img = renderer.render()
                    Image.fromarray(img).save(save_path)
                    renderer.close()
                    return True
                finally:
                    # 清理临时文件
                    if os.path.exists(temp_xml):
                        os.remove(temp_xml)
            except Exception as e:
                logger.debug(f"MJCF 渲染失败: {e}")
                return False
            finally:
                os.chdir(original_dir)

        except ImportError:
            return False
        except Exception as e:
            logger.debug(f"MJCF 渲染异常: {e}")
            return False

    def render_preview_with_obj(self, obj_path: str, save_path: str) -> bool:
        """使用 OBJ 文件直接渲染预览图（创建临时 MJCF）"""
        try:
            import mujoco
            from PIL import Image
            import numpy as np
            import trimesh
            import os

            # 确保保存路径的目录存在
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            obj_dir = os.path.dirname(obj_path)
            obj_filename = os.path.basename(obj_path)
            uid = os.path.splitext(obj_filename)[0]

            # 使用 trimesh 读取 OBJ 获取真实的边界框和中心
            mesh = trimesh.load(obj_path)
            center = mesh.centroid
            bbox_size = np.linalg.norm(mesh.extents)

            # 创建简化的 MJCF，将模型移动到原点
            xml_content = f'''<mujoco model="{uid}">
  <asset>
    <mesh file="{obj_filename}"/>
  </asset>
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0 0 -1"/>
    <body name="{uid}" pos="{-center[0]} {-center[1]} {-center[2]}">
      <freejoint/>
      <geom mesh="{uid}" type="mesh" rgba="0.8 0.8 0.8 1"/>
    </body>
  </worldbody>
</mujoco>'''

            temp_xml = os.path.join(obj_dir, 'temp_render.xml')
            with open(temp_xml, 'w') as f:
                f.write(xml_content)

            try:
                original_dir = os.getcwd()
                # 在改变工作目录前，将 save_path 转换为绝对路径
                save_path = os.path.abspath(save_path)
                os.chdir(obj_dir)

                model = mujoco.MjModel.from_xml_path('temp_render.xml')
                data = mujoco.MjData(model)
                mujoco.mj_forward(model, data)

                # 创建渲染器并设置相机
                renderer = mujoco.Renderer(model, height=480, width=640)

                # 设置相机对准原点，距离根据模型大小调整
                camera = mujoco.MjvCamera()
                camera.lookat[:] = [0, 0, 0]
                camera.distance = bbox_size * 2.5
                camera.azimuth = 45
                camera.elevation = -20

                renderer.update_scene(data, camera=camera)
                img = renderer.render()
                Image.fromarray(img).save(save_path)
                renderer.close()

                return True
            finally:
                # 确保恢复工作目录和清理临时文件
                try:
                    os.chdir(original_dir)
                except Exception:
                    pass
                if os.path.exists(temp_xml):
                    os.remove(temp_xml)

        except Exception as e:
            import traceback
            logger.error(f"OBJ 渲染异常: {e}")
            logger.error(f"异常详情:\n{traceback.format_exc()}")
            return False

    def render_preview(self, xml_path: str, save_path: str) -> bool:
        """渲染预览图（封装失败不影响主流程）"""
        return render_mjcf(xml_path, save_path)

    def run(self):
        """运行完整的 Pipeline"""
        logger.info("=" * 60)
        logger.info(f"开始处理关键词: {self.keyword}")
        logger.info("=" * 60)

        # 检查依赖
        if not self.check_dependencies():
            logger.error("依赖检查失败，退出")
            return

        # 搜索模型
        uids, annotations = self.search_objaverse()
        if not uids:
            logger.warning("未找到匹配的模型")
            return

        # 下载模型
        downloaded = self.download_models(uids)
        if not downloaded:
            logger.error("下载失败")
            return

        self.stats['downloaded'] = len(downloaded)

        # 处理每个模型
        iterator = downloaded.items()
        if tqdm is not None:
            iterator = tqdm(iterator, desc="转换模型")

        for uid, glb_path in iterator:
            try:
                self.process_model(uid, glb_path, annotations.get(uid, {}))
            except Exception as e:
                logger.error(f"处理 {uid} 时出错: {e}")
                self.stats['failed'] += 1

        # 打印统计
        logger.info("=" * 60)
        logger.info("处理完成!")
        logger.info(f"  总数:     {self.stats['total']}")
        logger.info(f"  已下载:   {self.stats['downloaded']}")
        logger.info(f"  已转换:   {self.stats['converted']}")
        logger.info(f"  失败:     {self.stats['failed']}")
        logger.info(f"  跳过:     {self.stats['skipped']}")
        logger.info(f"  输出目录: {self.base_dir}")
        logger.info("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(
        description="自动化模型资产下载与转换 Pipeline (使用 trimesh，无 Blender 依赖)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python get_assets.py --keyword tube
  python get_assets.py --keyword bottle --max_downloads 20
  python get_assets.py --keyword "metal pipe" --output_dir ./assets/review
  python get_assets.py --keyword chair --merge-meshes  # 默认开启
        """
    )

    parser.add_argument(
        '--keyword',
        type=str,
        required=True,
        help='搜索关键词，如 tube, bottle, chair'
    )

    parser.add_argument(
        '--max_downloads',
        type=int,
        default=10,
        help='最大下载数量 (默认: 10)'
    )

    parser.add_argument(
        '--min_vertices',
        type=int,
        default=400,
        help='过滤顶点数太少的模型 (默认: 400)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./assets/review',
        help='输出根目录 (默认: ./assets/review)'
    )

    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='跳过已存在的模型文件夹'
    )

    parser.add_argument(
        '--merge_meshes',
        action='store_true',
        default=True,
        help='合并 GLB 中的所有网格为单个物体 (默认: True)'
    )

    parser.add_argument(
        '--no_merge_meshes',
        action='store_true',
        help='不合并网格，保留场景结构'
    )

    parser.add_argument(
        '--context',
        type=str,
        default=None,
        choices=list(CONTEXT_PROFILES.keys()),
        help=f'搜索上下文，优先筛选特定领域模型。可选: {list(CONTEXT_PROFILES.keys())}'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 处理 merge_meshes 参数
    merge_meshes = args.merge_meshes
    if args.no_merge_meshes:
        merge_meshes = False

    # 运行 Pipeline
    pipeline = AssetPipeline(
        keyword=args.keyword,
        max_downloads=args.max_downloads,
        min_vertices=args.min_vertices,
        output_dir=args.output_dir,
        skip_existing=args.skip_existing,
        merge_meshes=merge_meshes,
        context=args.context
    )

    pipeline.run()


if __name__ == "__main__":
    main()