#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import cv2
import open3d as o3d


# ============================================================
# Part 1: Forward projection (TODO)
# ============================================================

def to_homogeneous(x: np.ndarray) -> np.ndarray:
    """
    Input:
      - x: (N,2) or (N,3)
    Output:
      - (N,3) if input (N,2)  -> append 1
      - (N,4) if input (N,3)  -> append 1
    """
    # TODO
    raise NotImplementedError


def from_homogeneous(x_h: np.ndarray) -> np.ndarray:
    """
    Input:
      - x_h: (N,3) or (N,4)
    Output:
      - (N,2) if input (N,3)
      - (N,3) if input (N,4)
    """
    # TODO
    raise NotImplementedError


def compose_extrinsic(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Return [R|t]
    Input:
      - R: (3,3)
      - t: (3,) or (3,1)
    Output:
      - Rt: (3,4)
    """
    # TODO
    raise NotImplementedError


def project_points(X_world: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Project 3D world points to pixel coordinates:

    Input:
      - X_world: (N,3)
      - K: (3,3)
      - R: (3,3)
      - t: (3,) or (3,1)
    Output:
      - uv: (N,2)
    """
    # TODO
    raise NotImplementedError


def load_bunny_points(n_points: int = 25000) -> np.ndarray:
    bunny = o3d.data.BunnyMesh() 
    mesh = o3d.io.read_triangle_mesh(bunny.path)
    pcd = mesh.sample_points_uniformly(number_of_points=int(n_points))
    X = np.asarray(pcd.points).astype(np.float64)

    # center + unit scale
    X = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.norm(X, axis=1).max()
    if s > 0:
        X = X / s
    return X


def render_points_depth_colored(uv: np.ndarray, z: np.ndarray, W: int, H: int, out_path: Path) -> None:
    img = np.zeros((H, W, 3), dtype=np.uint8)

    keep = z > 1e-6
    uv = uv[keep]
    z = z[keep]

    order = np.argsort(z)[::-1]
    uv = uv[order]
    z = z[order]

    zn = (z - z.min()) / (z.max() - z.min() + 1e-9)
    intensity = (255.0 * (1.0 - zn)).astype(np.uint8).reshape(-1, 1)
    colors = cv2.applyColorMap(intensity, cv2.COLORMAP_TURBO).reshape(-1, 3)

    for (u, v), c in zip(uv, colors):
        ui, vi = int(round(float(u))), int(round(float(v)))
        if 0 <= ui < W and 0 <= vi < H:
            img[vi, ui] = c

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def run_part1(figures_dir: Path, n_points: int, interactive: bool) -> None:
    Xw = load_bunny_points(n_points)

    # visualize
    if interactive:
        o3d.visualization.draw_geometries([o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Xw))])

    W, H = 960, 720
    fx = fy = 900.0
    cx, cy = W / 2.0, H / 2.0
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)

    R, _ = cv2.Rodrigues(np.array([0.35, -0.55, 0.10], dtype=np.float64))
    t = np.array([0.0, -0.05, 2.4], dtype=np.float64)

    uv = project_points(Xw, K, R, t)

    Xc = (R @ Xw.T).T + t.reshape(1, 3)
    z = Xc[:, 2]

    out_path = figures_dir / "part1_bunny_projection.png"
    render_points_depth_colored(uv, z, W, H, out_path)
    print(f"[OK] {out_path}")


# ============================================================
# Part 2: RGB-D back-projection (TODO)
# ============================================================

# change the params to match with the provided .txt
RGBD_WIDTH = 640
RGBD_HEIGHT = 480

K_RGBD = np.array([
    [1.0,   0.0,      1.0],
    [1.0,        1.0, 1.0],
    [1.0,          1.0,        1.0],
], dtype=np.float64)

DEPTH_SCALE = 1.0
STRIDE = 2


def unproject_depth_image_to_pointcloud(depth_m: np.ndarray, K: np.ndarray, rgb_bgr: np.ndarray):
    """
    TODO:
    - Use fx, fy, cx, cy from K
    - For pixels (u,v) with depth z>0:
        X = ?
        Y = ?
        Z = z
    - Return points (N,3) and colors (N,3)
    """
    raise NotImplementedError


def show_pointcloud(points: np.ndarray, colors_bgr: np.ndarray) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    c = colors_bgr.astype(np.float64) / 255.0
    c = c[:, ::-1]  # BGR -> RGB
    pcd.colors = o3d.utility.Vector3dVector(c)

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    o3d.visualization.draw_geometries([pcd, axes], window_name="RGB-D Point Cloud")


def run_part2(rgb_path: Path, depth_path: Path, interactive: bool) -> None:
    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    depth_m = depth_raw.astype(np.float64) / DEPTH_SCALE
    depth_m[depth_raw == 0] = 0.0

    points, colors = unproject_depth_image_to_pointcloud(depth_m, K_RGBD, rgb_bgr)
    if interactive:
        show_pointcloud(points, colors)


# ============================================================
# main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["1", "2"], required=True)

    ap.add_argument("--figures_dir", type=str, default="figures")
    ap.add_argument("--n_points", type=int, default=25000)

    ap.add_argument("--rgb", type=str, default="../data/rgbd/rgb.png")
    ap.add_argument("--depth", type=str, default="../data/rgbd/depth.png")
    ap.add_argument("--interactive", action="store_true", help="Enable interactive visualization")

    args = ap.parse_args()

    if args.part == "1":
        run_part1(Path(args.figures_dir), args.n_points, args.interactive)
    else:
        run_part2(Path(args.rgb), Path(args.depth), args.interactive)


if __name__ == "__main__":
    main()
