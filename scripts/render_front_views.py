#!/usr/bin/env python3
"""
渲染 VLABench assets/review 中所有模型的正面照。
用法:
    python scripts/render_front_views.py \\
        --input-dir  VLABench/assets/review \\
        --output-dir logs/asset_front_renders \\
        --azimuth 0 --elevation 0 --padding 1.25 \\
        --width 640 --height 480

输出:
    <output-dir>/<uid>.png          - 每模型一张渲染图
    <output-dir>/_render_summary.txt - 渲染结果汇总
"""

import argparse
import logging
import os
import re
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Render front-view PNGs for VLABench review assets.")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).parent.parent / "VLABench" / "assets" / "review",
        help="Directory containing review assets (default: VLABench/assets/review)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "logs" / "asset_front_renders",
        help="Directory to write PNG outputs (default: logs/asset_front_renders)",
    )
    p.add_argument("--azimuth", type=float, default=0.0,
                   help="Camera azimuth in degrees (default: 0 = front)")
    p.add_argument("--elevation", type=float, default=0.0,
                   help="Camera elevation in degrees (default: 0 = level)")
    p.add_argument("--padding", type=float, default=1.25,
                   help="Camera distance padding multiplier (default: 1.25)")
    p.add_argument("--width", type=int, default=640,
                   help="Output image width in pixels (default: 640)")
    p.add_argument("--height", type=int, default=480,
                   help="Output image height in pixels (default: 480)")
    p.add_argument("--white-bg", action="store_true",
                   help="Replace black background with white (default: False)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Asset discovery
# ---------------------------------------------------------------------------

def _find_model_files(entry: Path, uid: str):
    """Find the primary model file inside a review subdirectory.

    Returns (kind, resolved_path) or (None, None).
    The actual model files may live one level deeper (e.g.
    beaker_large/large_beaker/large_beaker.xml).
    """
    # 1. Look for MJCF XML
    for xml_path in entry.rglob(f"{uid}.xml"):
        return "mjcf", xml_path
    for xml_path in entry.rglob("*.xml"):
        return "mjcf", xml_path

    # 2. Look for GLB anywhere
    for glb_path in entry.rglob("*.glb"):
        return "glb", glb_path

    # 3. Look for OBJ (skip collision meshes)
    for obj_path in entry.rglob("*.obj"):
        if "_collision_" in obj_path.name or obj_path.name.startswith("."):
            continue
        return "obj", obj_path

    return None, None


def discover_assets(root: Path):
    assets = []
    seen_uids = set()
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        uid = entry.name
        if uid in ("validation_report.json", "__pycache__"):
            continue
        if not entry.is_dir() and not entry.suffix.lower() == ".glb":
            continue

        if entry.is_dir():
            kind, path = _find_model_files(entry, uid)
            if kind is not None:
                assets.append((uid, kind, path))
                seen_uids.add(uid)
        elif entry.suffix.lower() == ".glb":
            if uid not in seen_uids:
                assets.append((uid, "glb", entry))
                seen_uids.add(uid)
    return assets


# ---------------------------------------------------------------------------
# Camera helpers — compute bounds directly from MuJoCo data
# ---------------------------------------------------------------------------

def _compute_bounds_from_mujoco(model, data) -> tuple[np.ndarray, np.ndarray]:
    """Return (center, extents) of all geom AABBs in world space.

    Uses model.geom_size (static), model.geom_pos (local offset from parent
    body), data.geom_xpos (world center), and data.geom_xmat (world rotation).
    This accounts for mesh scale, body transforms, etc.
    """
    ngeom = model.ngeom
    if ngeom == 0:
        return np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])

    size = model.geom_size      # (ngeom, 3) — half-extents
    xpos = data.geom_xpos       # (ngeom, 3) — world centers
    xmat = data.geom_xmat       # (ngeom, 9) — world rotation

    # 8 corner signs for AABB
    signs = np.array([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
                      [ 1, -1, -1], [ 1, -1, 1], [ 1, 1, -1], [ 1, 1, 1]])

    all_corners = []
    for i in range(ngeom):
        R = xmat[i].reshape(3, 3)
        corners_local = signs * size[i]
        corners_world = (R @ corners_local.T).T + xpos[i]
        all_corners.append(corners_world)

    all_corners = np.concatenate(all_corners, axis=0)
    mins = all_corners.min(axis=0)
    maxs = all_corners.max(axis=0)
    center = (mins + maxs) / 2.0
    extents = maxs - mins
    return center, extents


def compute_camera_fit(extents, fov_deg=45.0, padding=1.25):
    """FOV-aware camera distance so the whole model fits in frame."""
    max_extent = float(np.max(extents))
    if max_extent < 1e-6:
        max_extent = 1.0
    fov_rad = np.deg2rad(fov_deg)
    return (max_extent / (2.0 * np.tan(fov_rad / 2.0))) * padding


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _render(model, data, out_png, args):
    """Render the model with a camera auto-fitted to its actual bounding box."""
    center, extents = _compute_bounds_from_mujoco(model, data)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    opts = mujoco.MjvOption()
    # Show only visual geoms (group 2) — collision geoms (group 3) are hidden by
    # default and don't contribute material color (they are gray boxes).
    opts.geomgroup[2] = 1
    # Enable alpha blending so semi-transparent materials (glass alpha < 1)
    # blend with the background instead of looking black.
    opts.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 1

    camera = mujoco.MjvCamera()
    camera.lookat[:] = center.astype(float)
    camera.distance = compute_camera_fit(extents, fov_deg=45.0, padding=args.padding)
    camera.azimuth = args.azimuth
    camera.elevation = args.elevation

    renderer.update_scene(data, camera=camera, scene_option=opts)
    img = renderer.render()

    # Replace black background with white if requested
    if args.white_bg:
        bg_mask = img.max(axis=2) < 5  # pixels where all RGB channels are near 0
        img[bg_mask] = [255, 255, 255]

    Image.fromarray(img).save(out_png)
    renderer.close()


def _strip_default_rgba(xml: str) -> str:
    """Remove rgba from <default class="visual"> so material colors take effect.

    The VLABench MJCF files have <default class="visual"> with rgba="0.5 0.5 0.5 1"
    which overrides the material's rgba, making all visual geoms gray.
    We strip this so the material-defined color is used instead.
    We also strip explicit rgba on <geom> elements within <default class="visual">.
    """
    # Remove rgba= from <default class="visual"> and its child <geom>
    result = re.sub(
        r'(<default\s+class="visual"[^>]*?)\s+rgba="[^"]*"',
        r'\1',
        xml, flags=re.DOTALL,
    )
    result = re.sub(
        r'(<default\s+class="visual"[^>]*>.*?<geom[^>]*?)\s+rgba="[^"]*"',
        r'\1',
        result, flags=re.DOTALL,
    )
    return result


def _apply_material_rgba_to_geom(model):
    """Copy material rgba to geom rgba so rendered geoms use material colors.

    MuJoCo sets geom_rgba to [0.5, 0.5, 0.5, 1] by default, overriding material
    colors. We fix this by copying mat_rgba to geom_rgba for each geom with a
    material (matid >= 0) in the visual group (group 2).

    Also: hide non-mesh visual geoms (ellipsoid flames, cylinder solutions, etc.)
    by setting their alpha to 0, so only the main mesh model is visible.
    """
    for i in range(model.ngeom):
        if model.geom_group[i] != 2:  # only visual geoms
            continue
        gtype = mujoco.mjtGeom(model.geom_type[i])
        # Only show mesh-type visual geoms; hide ellipsoids (flames), cylinders (solutions), etc.
        if gtype != mujoco.mjtGeom.mjGEOM_MESH:
            model.geom_rgba[i, 3] = 0.0  # set alpha to 0 (invisible)
            continue
        matid = model.geom_matid[i]
        if matid < 0:  # no material assigned
            continue
        # Copy material rgba to geom rgba
        model.geom_rgba[i] = model.mat_rgba[matid]


def render_mjcf(uid: str, xml_path: Path, out_png: Path, args) -> bool:
    """Render using an MJCF XML file (with collision-mesh path fix and material color fix)."""
    try:
        xml_dir = xml_path.parent
        original_dir = os.getcwd()
        out_png_abs = out_png.resolve()

        with open(xml_path, "r", encoding="utf-8") as fh:
            xml_content = fh.read()

        # 1. Fix bare collision mesh paths: add subdirectory prefix if missing.
        # Guard: already-prefixed paths are left alone to avoid double-prefixing.
        def _fix_collision(m):
            filename = m.group(1)
            return f'file="{uid}/{filename}"' if "/" not in filename else m.group(0)

        fixed_xml = re.sub(
            rf'file="({re.escape(uid)}(?:/[^"]*)?_collision_\d+\.obj)"',
            _fix_collision,
            xml_content,
        )

        # 2. Strip rgba from <default class="visual"> so material colors apply.
        fixed_xml = _strip_default_rgba(fixed_xml)

        temp_xml = xml_dir / "temp_render_front.xml"
        with open(temp_xml, "w", encoding="utf-8") as fh:
            fh.write(fixed_xml)

        try:
            os.chdir(xml_dir)
            model = mujoco.MjModel.from_xml_path(temp_xml.name)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)

            # 3. Apply material rgba to visual geoms (fixes default gray override)
            _apply_material_rgba_to_geom(model)

            _render(model, data, out_png_abs, args)
            return True
        finally:
            try:
                os.chdir(original_dir)
            except OSError:
                pass
            if temp_xml.exists():
                temp_xml.unlink()
    except Exception as e:
        logger.warning("  MJCF render failed for %s: %s", uid, e)
        return False


def render_obj(uid: str, obj_path: Path, out_png: Path, args) -> bool:
    """Render a standalone OBJ by wrapping it in a minimal MJCF."""
    try:
        obj_dir = obj_path.parent
        original_dir = os.getcwd()
        out_png_abs = out_png.resolve()

        # Load with trimesh to get centroid (for centering the MJCF body)
        mesh = trimesh.load(obj_path)
        centroid = mesh.centroid

        xml_content = f'''<mujoco model="{uid}">
  <asset>
    <mesh file="{obj_path.name}"/>
  </asset>
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0 0 -1"/>
    <body name="{uid}" pos="{-centroid[0]} {-centroid[1]} {-centroid[2]}">
      <freejoint/>
      <geom mesh="{uid}" type="mesh" rgba="180 180 180 255"/>
    </body>
  </worldbody>
</mujoco>'''

        temp_xml = obj_dir / "temp_render_front.xml"
        with open(temp_xml, "w", encoding="utf-8") as fh:
            fh.write(xml_content)

        try:
            os.chdir(obj_dir)
            model = mujoco.MjModel.from_xml_path(temp_xml.name)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            _render(model, data, out_png_abs, args)
            return True
        finally:
            try:
                os.chdir(original_dir)
            except OSError:
                pass
            if temp_xml.exists():
                temp_xml.unlink()
    except Exception as e:
        logger.warning("  OBJ render failed for %s: %s", uid, e)
        return False


def _sanitize_for_mjcf(name: str) -> str:
    """Strip characters (notably spaces) that MuJoCo's MJCF parser mishandles."""
    return "".join(c for c in name if c not in " \t/")


def _export_trimesh_as_obj(mesh: trimesh.Trimesh, out_path: Path, mesh_name: str) -> None:
    """Export a Trimesh to OBJ and prepend 'o <name>' so MuJoCo can reference it."""
    obj_str: str = mesh.export(file_type="obj")
    if not obj_str.startswith("o "):
        obj_str = f"o {mesh_name}\n{obj_str}"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(obj_str)


def render_glb(uid: str, glb_path: Path, out_png: Path, args) -> bool:
    """Render a GLB by loading it with trimesh, merging into one mesh,
    exporting to OBJ, then building a minimal MJCF.
    """
    try:
        glb_dir = glb_path.parent
        original_dir = os.getcwd()
        out_png_abs = out_png.resolve()
        safe_uid = _sanitize_for_mjcf(uid)

        loaded = trimesh.load(glb_path)
        # trimesh.Scene.to_geometry() returns an already-merged Trimesh
        if isinstance(loaded, trimesh.Scene):
            merged: trimesh.Trimesh = loaded.to_geometry()
        else:
            merged = loaded

        centroid = merged.centroid

        temp_obj = glb_dir / f"{safe_uid}_temp.obj"
        _export_trimesh_as_obj(merged, temp_obj, safe_uid)

        temp_xml = glb_dir / "temp_render_front.xml"
        xml_content = f'''<mujoco model="{safe_uid}">
  <asset>
    <mesh name="{safe_uid}" file="{temp_obj.name}"/>
  </asset>
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0 0 -1"/>
    <body name="{safe_uid}" pos="{-centroid[0]} {-centroid[1]} {-centroid[2]}">
      <freejoint/>
      <geom mesh="{safe_uid}" type="mesh" rgba="180 180 180 255"/>
    </body>
  </worldbody>
</mujoco>'''
        with open(temp_xml, "w", encoding="utf-8") as fh:
            fh.write(xml_content)

        try:
            os.chdir(glb_dir)
            model = mujoco.MjModel.from_xml_path(temp_xml.name)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            _render(model, data, out_png_abs, args)
            return True
        finally:
            try:
                os.chdir(original_dir)
            except OSError:
                pass
            for _path in (temp_xml, temp_obj):
                if _path.exists():
                    _path.unlink()
    except Exception as e:
        logger.warning("  GLB render failed for %s: %s", uid, e)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not args.input_dir.is_dir():
        logger.error("Input directory does not exist: %s", args.input_dir)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    assets = discover_assets(args.input_dir)
    logger.info("Found %d assets to render in %s", len(assets), args.input_dir)

    results = []  # (uid, kind, status, size, source)

    for uid, kind, path in assets:
        logger.info("Rendering %s (kind=%s) ...", uid, kind)
        out_png = args.output_dir / f"{uid}.png"

        if kind == "mjcf":
            success = render_mjcf(uid, path, out_png, args)
        elif kind == "obj":
            success = render_obj(uid, path, out_png, args)
        elif kind == "glb":
            success = render_glb(uid, path, out_png, args)
        else:
            success = False

        if success:
            try:
                img = Image.open(out_png)
                size_str = f"{img.width}x{img.height}"
            except Exception:
                size_str = "?"
            logger.info("  -> saved: %s", out_png)
            results.append((uid, kind, "PASS", size_str, str(path)))
        else:
            logger.error("  -> FAILED: %s", uid)
            results.append((uid, kind, "FAIL", "?", str(path)))

    passed = sum(1 for r in results if r[2] == "PASS")
    summary_path = args.output_dir / "_render_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(f"# Front-view render summary  |  {passed}/{len(results)} passed\n")
        fh.write(f"# Output dir: {args.output_dir}\n")
        fh.write(f"# Args: azimuth={args.azimuth}  elevation={args.elevation}  padding={args.padding}\n")
        fh.write(f"# {'uid':<30} | kind | status | size     | source\n")
        fh.write("#" * 120 + "\n")
        for uid, kind, status, size, src in results:
            fh.write(f"  {uid:<28} | {kind:<4} | {status:<6} | {size:<9} | {src}\n")

    logger.info("Done. %d/%d assets rendered. Summary: %s", passed, len(results), summary_path)


if __name__ == "__main__":
    main()
