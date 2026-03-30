"""
Asset Tools - 资产检查和下载工具
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# 同义词映射：LLM 常用名 -> name2class_xml 中的注册名
ASSET_SYNONYMS = {
    "test_tube": "tube",
    "test_tube_rack": "chemistry_tube_stand",
    "tube_rack": "chemistry_tube_stand",
    "tube_stand": "chemistry_tube_stand",
    "tube_holder": "chemistry_tube_stand",
    "bunsen_burner": "bunsen_burner",
    "erlenmeyer_flask": "flask",
    "conical_flask": "flask",
    "microscope_slide": "coverslip",
    "cutting_board": "cut_board",
    "box": "giftbox",
    "cardboard_box": "giftbox",
    "container_box": "giftbox",
}


def _register_downloaded_asset(canonical_name: str, xml_path: str):
    """
    将下载/发现的非内置资产动态注册到 name2class_xml 中。

    这样后续的 get_entity_config()、load_containers()、load_init_containers() 等
    都能通过 name2class_xml[name] 找到该资产，无需为每种非内置资产编写特殊处理逻辑。

    Args:
        canonical_name: 注册名（如 "bunsen_burner"）
        xml_path: 相对于 VLABENCH_ROOT/assets/ 的 XML 路径
    """
    try:
        from VLABench.tasks.components import CommonGraspedEntity
        from VLABench.configs.constant import name2class_xml

        if canonical_name not in name2class_xml:
            name2class_xml[canonical_name] = [CommonGraspedEntity, xml_path]
            logger.info(f"  → 动态注册到 name2class_xml: {canonical_name} -> {xml_path}")
        else:
            logger.debug(f"  → {canonical_name} 已在 name2class_xml 中，跳过注册")
    except ImportError as e:
        logger.warning(f"  ⚠ 动态注册失败（导入错误）: {e}")


def check_asset_exists(object_name: str) -> Dict:
    """
    检查指定物体的 MJCF 资产是否存在

    Args:
        object_name: 物体名称 (如 "microscope")

    Returns:
        {
            "found": bool,
            "xml_path": str or None,
            "class": str or None
        }
    """
    # 同义词映射：将 LLM 常用名映射到 name2class_xml 中的注册名
    canonical_name = ASSET_SYNONYMS.get(object_name, object_name)
    if canonical_name != object_name:
        logger.info(f"  → 同义词映射: {object_name} -> {canonical_name}")

    try:
        # 先导入 components 以避免循环导入
        import VLABench.tasks.components  # noqa: F401
        from VLABench.configs.constant import name2class_xml

        # 在 name2class_xml 中查找（先用映射后的名称，再用原始名称）
        lookup_name = canonical_name if canonical_name in name2class_xml else object_name
        if lookup_name in name2class_xml:
            class_type, xml_path = name2class_xml[lookup_name]
            xml_path_str = xml_path if isinstance(xml_path, str) else xml_path[0]

            logger.info(f"  ✓ 在配置中找到 {object_name} (as {lookup_name}): {xml_path_str}")

            return {
                "found": True,
                "xml_path": xml_path_str,
                "class": class_type.__name__,
                "builtin": True
            }
    except ImportError:
        logger.warning("  ⚠ VLABench.configs.constant 导入失败,跳过配置查找")
    except Exception as e:
        logger.warning(f"  ⚠ 配置查找失败: {e}")

    # 在文件系统中搜索（使用 canonical_name 和 object_name 两个名称都尝试）
    vlabench_root = os.environ.get('VLABENCH_ROOT')
    if not vlabench_root:
        logger.warning("  ⚠ VLABENCH_ROOT 未设置")
        return {"found": False, "xml_path": None, "class": None}

    asset_dir = Path(vlabench_root) / 'assets' / 'obj' / 'meshes'
    review_dir = Path(vlabench_root) / 'assets' / 'review'

    search_dirs = []
    if asset_dir.exists():
        search_dirs.append(asset_dir)
    if review_dir.exists():
        search_dirs.append(review_dir)

    if not search_dirs:
        logger.warning(f"  ⚠ 资产目录不存在: {asset_dir} 和 {review_dir}")
        return {"found": False, "xml_path": None, "class": None}

    # 搜索匹配的 XML 文件（尝试 canonical_name 和 object_name）
    search_names = [canonical_name] if canonical_name != object_name else [object_name]
    if canonical_name != object_name:
        search_names.append(object_name)

    # 同时搜索小写形式（review 目录中的文件夹通常是小写）
    for name in list(search_names):
        lower_name = name.lower()
        if lower_name not in search_names:
            search_names.append(lower_name)

    matches = []
    for search_name in search_names:
        for search_dir in search_dirs:
            # 搜索文件名匹配
            matches = list(search_dir.glob(f"**/*{search_name}*.xml"))
            if not matches:
                # 搜索目录名匹配（review 目录中 XML 文件名是哈希值，但目录名包含关键词）
                for subdir in search_dir.iterdir():
                    if subdir.is_dir() and search_name in subdir.name:
                        matches = list(subdir.glob("**/*.xml"))
                        if matches:
                            break
            if matches:
                logger.info(f"  → 文件系统搜索 '{search_name}' 在 {search_dir} 找到 {len(matches)} 个匹配")
                break
        if matches:
            break

    if matches:
        # 尝试找到一个可用的模型（纹理文件完整）
        for xml_file in matches:
            assets_root = Path(vlabench_root) / 'assets'
            xml_path = str(xml_file.relative_to(assets_root))

            # 检查纹理文件是否存在
            try:
                with open(xml_file, 'r', encoding='utf-8') as f:
                    xml_content = f.read()

                # 提取所有 texture file 引用
                import re
                texture_files = re.findall(r'<texture[^>]*file="([^"]+)"', xml_content)

                # 检查每个纹理文件是否存在
                all_textures_exist = True
                missing_textures = []

                for texture_file in texture_files:
                    # 纹理文件路径相对于 XML 文件所在目录
                    texture_path = xml_file.parent / texture_file
                    if not texture_path.exists():
                        all_textures_exist = False
                        missing_textures.append(texture_file)

                if all_textures_exist:
                    # 找到一个完整的模型
                    logger.info(f"  ✓ 在文件系统中找到 {object_name}: {xml_path}")

                    # 动态注册到 name2class_xml，使后续的 get_entity_config() 能找到
                    _register_downloaded_asset(canonical_name, xml_path)

                    return {
                        "found": True,
                        "xml_path": xml_path,
                        "class": "CommonGraspedEntity"
                    }
                else:
                    # 纹理缺失，尝试下一个模型
                    logger.warning(f"  ⚠ {xml_file.name} 纹理缺失: {missing_textures}，尝试其他模型...")

            except Exception as e:
                logger.warning(f"  ⚠ 检查 {xml_file.name} 时出错: {e}")
                continue

        # 所有匹配的模型都有问题
        logger.error(f"  ✗ 找到 {len(matches)} 个 {object_name} 模型，但都存在纹理缺失问题")
        logger.error(f"  建议: 使用 get_assets.py 下载新的模型")

        return {
            "found": False,
            "xml_path": None,
            "class": None,
            "error": f"所有 {object_name} 模型都存在纹理缺失问题"
        }

    logger.info(f"  ✗ 未找到 {object_name}")
    return {"found": False, "xml_path": None, "class": None}


def download_asset(keyword: str, max_downloads: int = 3) -> Dict:
    """
    调用 get_assets.py 下载指定关键词的模型

    Args:
        keyword: 搜索关键词 (如 "microscope")
        max_downloads: 最大下载数量

    Returns:
        {
            "success": bool,
            "downloaded_count": int,
            "assets": List[str],  # 下载的资产路径列表
            "error": str (optional)
        }
    """
    # 将下划线转换为空格，Objaverse 搜索使用自然语言
    search_keyword = keyword.replace("_", " ")
    logger.info(f"  开始下载资产: {keyword} (搜索关键词: {search_keyword})")

    vlabench_root = os.environ.get('VLABENCH_ROOT')
    if not vlabench_root:
        return {
            "success": False,
            "downloaded_count": 0,
            "assets": [],
            "error": "VLABENCH_ROOT 未设置"
        }

    # get_assets.py 在项目根目录的 scripts/ 下，不在 VLABench 包内
    project_root = Path(vlabench_root).parent  # VLABench/ -> workspace/VLABench/
    script_path = project_root / "scripts" / "get_assets.py"

    if not script_path.exists():
        return {
            "success": False,
            "downloaded_count": 0,
            "assets": [],
            "error": f"get_assets.py 不存在: {script_path}"
        }

    try:
        # 将资产下载到 VLABench 包内的 assets 目录
        # 因为 VLABench 会在 VLABENCH_ROOT/assets/ 下查找资产
        result = subprocess.run([
            "python", str(script_path),
            "--keyword", search_keyword,
            "--max_downloads", str(max_downloads),
            "--output_dir", "./VLABench/assets/review",
            "--skip_existing"
        ], capture_output=True, text=True, timeout=600, cwd=project_root)  # 在项目根目录运行

        if result.returncode == 0:
            # 解析输出,提取下载的资产信息
            output_dir = project_root / "VLABench" / "assets" / "review" / keyword.lower()

            # 后处理：修复 XML 文件中的路径引用
            # obj2mjcf 生成的结构是: uuid/uuid.xml 和 uuid/uuid/*.obj
            # 需要将 XML 中的 file="xxx.obj" 改为 file="uuid/xxx.obj"
            for xml_path in output_dir.glob("*/*.xml"):
                try:
                    with open(xml_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 获取 UUID (嵌套目录名)
                    uuid_dir = xml_path.stem  # 从 uuid.xml 获取 uuid

                    import re

                    # 1. 修改 <mesh file="xxx.obj"> 为 <mesh file="uuid/xxx.obj">
                    # 只修改 mesh 标签中不包含路径分隔符的 file 属性
                    def add_uuid_to_mesh(match):
                        full_tag = match.group(0)
                        filepath = match.group(1)
                        # 如果已经包含路径分隔符，不修改
                        if '/' in filepath or '\\' in filepath:
                            return full_tag
                        return f'<mesh file="{uuid_dir}/{filepath}"'

                    content = re.sub(r'<mesh file="([^"/\\]+)"', add_uuid_to_mesh, content)

                    # 2. 修复 texture 文件路径
                    # obj2mjcf 的行为不一致：
                    #   - 有时给 texture 添加 uuid 前缀: file="uuid/texture.png"
                    #   - 有时不添加: file="texture.png"
                    # 但 texture 文件总是在父目录，所以需要移除任何 uuid 前缀
                    # 注意：必须确保不跨越标签边界匹配

                    def fix_texture_file(match):
                        # match.group(1): texture 标签在 file 之前的部分
                        # match.group(2): file 属性值（可能带有 uuid/ 前缀）
                        # match.group(3): file 属性之后的部分
                        before_file = match.group(1)
                        file_value = match.group(2)
                        after_file = match.group(3)

                        # 移除可能的 uuid/ 前缀
                        if '/' in file_value:
                            file_value = file_value.split('/')[-1]

                        return f'<texture{before_file} file="{file_value}"{after_file}>'

                    # 匹配完整的 <texture ... file="xxx"> 标签（不跨越标签边界）
                    # 使用 [^<]* 而不是 [^>]* 来避免跨越到下一个标签
                    content = re.sub(
                        r'<texture([^<]*?) file="([^"]+)"([^<]*?)>',
                        fix_texture_file,
                        content
                    )

                    # 注意：不再移除 <body>/<freejoint>，也不再手动添加 mass/density
                    # get_assets.py 的 postprocess_model → fix_obj2mjcf_xml 已正确处理：
                    #   - <body> 包裹结构
                    #   - <inertial> 质量属性
                    #   - dm_control MJCF parser 要求 <inertial> 在 <body> 内

                    with open(xml_path, 'w', encoding='utf-8') as f:
                        f.write(content)

                    logger.info(f"  ✓ 修复 XML: {xml_path.name}")
                except Exception as e:
                    logger.warning(f"  ⚠ 跳过 XML 修复: {e}")

            assets = [x for x in output_dir.glob("*/*.xml")
                      if x.stem != "temp_render" and not x.stem.startswith("temp_")]

            logger.info(f"  ✓ 下载成功: {len(assets)} 个资产")

            return {
                "success": True,
                "downloaded_count": len(assets),
                "assets": [str(a) for a in assets]
            }
        else:
            logger.error(f"  ✗ 下载失败: {result.stderr}")
            return {
                "success": False,
                "downloaded_count": 0,
                "assets": [],
                "error": result.stderr
            }

    except subprocess.TimeoutExpired:
        logger.error(f"  ✗ 下载超时 (>10分钟)")
        return {
            "success": False,
            "downloaded_count": 0,
            "assets": [],
            "error": "下载超时"
        }
    except Exception as e:
        logger.error(f"  ✗ 下载异常: {e}")
        return {
            "success": False,
            "downloaded_count": 0,
            "assets": [],
            "error": str(e)
        }
