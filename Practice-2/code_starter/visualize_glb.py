#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import open3d as o3d


def load_mesh(glb_path: Path) -> o3d.geometry.TriangleMesh:
    if not glb_path.exists():
        raise FileNotFoundError(f"File not found: {glb_path}")

    mesh = o3d.io.read_triangle_mesh(str(glb_path), enable_post_processing=True)
    if mesh.is_empty():
        raise ValueError(f"Open3D loaded an empty mesh from: {glb_path}")

    mesh.compute_vertex_normals()
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a GLB mesh in Open3D (interactive).")
    parser.add_argument("glb", type=str, help="Path to .glb file")
    parser.add_argument("--backface", action="store_true", help="Render back faces (see inside surfaces)")
    parser.add_argument("--no-light", action="store_true", help="Disable lighting (makes vertex colors look flat/clear)")
    args = parser.parse_args()

    glb_path = Path(args.glb).expanduser().resolve()
    mesh = load_mesh(glb_path)

    print("File:", glb_path)
    print("has_vertex_colors:", mesh.has_vertex_colors())
    print("has_textures:", mesh.has_textures())
    print("vertices:", len(mesh.vertices), "triangles:", len(mesh.triangles))

    vis = o3d.visualization.Visualizer()
    vis.create_window(f"Open3D GLB Viewer - {glb_path.name}", width=1280, height=720)
    vis.add_geometry(mesh)

    opt = vis.get_render_option()
    opt.mesh_show_back_face = bool(args.backface)
    if args.no_light:
        opt.light_on = False

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
