"""
Asset Tools - 资产检查和下载工具
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


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
    try:
        # 尝试导入 VLABench 配置
        from VLABench.configs.constant import name2class_xml

        # 在 name2class_xml 中查找
        if object_name in name2class_xml:
            class_type, xml_path = name2class_xml[object_name]
            xml_path_str = xml_path if isinstance(xml_path, str) else xml_path[0]

            logger.info(f"  ✓ 在配置中找到 {object_name}: {xml_path_str}")

            return {
                "found": True,
                "xml_path": xml_path_str,
                "class": class_type.__name__
            }
    except ImportError:
        logger.warning("  ⚠ VLABench.configs.constant 导入失败,跳过配置查找")
    except Exception as e:
        logger.warning(f"  ⚠ 配置查找失败: {e}")

    # 在文件系统中搜索
    vlabench_root = os.environ.get('VLABENCH_ROOT')
    if not vlabench_root:
        logger.warning("  ⚠ VLABENCH_ROOT 未设置")
        return {"found": False, "xml_path": None, "class": None}

    asset_dir = Path(vlabench_root) / 'assets' / 'obj' / 'meshes'

    if not asset_dir.exists():
        logger.warning(f"  ⚠ 资产目录不存在: {asset_dir}")
        return {"found": False, "xml_path": None, "class": None}

    # 搜索匹配的 XML 文件
    matches = list(asset_dir.glob(f"**/*{object_name}*.xml"))

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
    logger.info(f"  开始下载资产: {keyword}")

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
            "--keyword", keyword,
            "--max_downloads", str(max_downloads),
            "--output_dir", "./VLABench/assets/review",
            "--skip_existing"
        ], capture_output=True, text=True, timeout=600, cwd=project_root)  # 在项目根目录运行

        if result.returncode == 0:
            # 解析输出,提取下载的资产信息
            output_dir = project_root / "VLABench" / "assets" / "review" / keyword

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

                    # 重构 XML：移除 <body> 和 <freejoint/>，将 geom 直接放在 <worldbody> 下
                    # obj2mjcf 生成的 XML 包含 <body><freejoint/></body> 结构，
                    # 但 VLABench 期望直接在 <worldbody> 下使用 <geom>
                    lines = content.split('\n')
                    new_lines = []
                    in_worldbody = False
                    skip_until_worldbody_end = False

                    for i, line in enumerate(lines):
                        # 检测 <worldbody> 开始
                        if '<worldbody>' in line:
                            new_lines.append(line)
                            in_worldbody = True
                            continue
                        # 检测 </worldbody> 结束
                        if '</worldbody>' in line:
                            new_lines.append(line)
                            in_worldbody = False
                            skip_until_worldbody_end = False
                            continue
                        # 跳过 <body name=...> 行
                        if in_worldbody and '<body name=' in line:
                            continue
                        # 跳过 <freejoint/> 行
                        if in_worldbody and '<freejoint/>' in line:
                            continue
                        # 跳过 </body> 行
                        if in_worldbody and '</body>' in line:
                            continue
                        # 调整 geom 缩进（从 6 个空格改为 4 个）
                        if in_worldbody and '<geom' in line:
                            # 移除前导空格并添加 4 个空格
                            new_lines.append('    ' + line.lstrip())
                            continue
                        # 其他行正常添加
                        if not skip_until_worldbody_end:
                            new_lines.append(line)

                    content = '\n'.join(new_lines)

                    # 添加质量属性到 visual geom 元素
                    def add_mass_to_visual(match):
                        geom_tag = match.group(0)
                        if 'mass=' in geom_tag or 'density=' in geom_tag:
                            return geom_tag
                        if geom_tag.endswith('/>'):
                            return geom_tag.replace('/>', ' mass="1"/>')
                        else:
                            return geom_tag.replace('>', ' mass="1">')

                    content = re.sub(
                        r'<geom[^>]*class="visual"[^>]*/?>',
                        add_mass_to_visual,
                        content
                    )

                    # 添加密度属性到 collision geom 元素
                    def add_density_to_collision(match):
                        geom_tag = match.group(0)
                        if 'density=' in geom_tag or 'mass=' in geom_tag:
                            return geom_tag
                        if geom_tag.endswith('/>'):
                            return geom_tag.replace('/>', ' density="100"/>')
                        else:
                            return geom_tag.replace('>', ' density="100">')

                    content = re.sub(
                        r'<geom[^>]*class="collision"[^>]*/?>',
                        add_density_to_collision,
                        content
                    )

                    with open(xml_path, 'w', encoding='utf-8') as f:
                        f.write(content)

                    logger.info(f"  ✓ 修复 XML: {xml_path.name}")
                except Exception as e:
                    logger.warning(f"  ⚠ 跳过 XML 修复: {e}")

            assets = list(output_dir.glob("*/*.xml"))

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
