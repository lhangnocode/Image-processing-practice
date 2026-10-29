#!/usr/bin/env python3
"""
Part 3: Camera Calibration (single camera + chessboard)

Data
----
Put chessboard images here:
  data/chessboard/*.JPG

What you will implement
------------------
1) make_object_grid(pattern_size, square_size)
2) calibrate_from_chessboard(image_paths, pattern_size, square_size)

You MUST read OpenCV docs for:
- cv2.findChessboardCorners
- cv2.cornerSubPix
- cv2.calibrateCamera

Outputs (must be saved to figures/)
-----------------------------------
- figures/part3_corners.png
- figures/part3_calibration.txt
- figures/calib.npz

Run
---
python part3_calibration.py --data_dir data/chessboard --out_dir figures
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# ============================================================
# IO / Utils
# ============================================================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def list_images_sorted(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    paths = [p for p in folder.iterdir() if p.suffix.lower() in exts]
    paths.sort()
    if not paths:
        raise FileNotFoundError(f"No images found in: {folder}")
    return paths

def write_calibration_txt(
    out_path: Path,
    *,
    K: np.ndarray,
    dist: np.ndarray,
    image_size_wh: Tuple[int, int],
    pattern_size: Tuple[int, int],
    square_size: float,
    used_images: int,
    total_images: int,
    rms_reproj: float,
    mean_reproj_px: float,
) -> None:
    ensure_dir(out_path.parent)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Camera Calibration (single camera + chessboard)\n")
        f.write(f"pattern_size_inner_corners (cols,rows): {pattern_size}\n")
        f.write(f"square_size_m: {square_size}\n")
        f.write(f"image_size_wh: {image_size_wh}\n")
        f.write(f"used_images: {used_images} / {total_images}\n")
        f.write(f"opencv_rms_reprojection_error: {rms_reproj:.6f}\n")
        f.write(f"mean_reprojection_error_px: {mean_reproj_px:.6f}\n\n")

        f.write("K (intrinsic matrix):\n")
        f.write(np.array2string(K, precision=8, suppress_small=False) + "\n\n")

        f.write("dist (distortion coefficients):\n")
        f.write("  typical order starts: [k1, k2, p1, p2, k3, ...]\n")
        f.write(np.array2string(dist.reshape(-1), precision=8, suppress_small=False) + "\n")

def save_calib_npz(
    out_path: Path,
    *,
    K: np.ndarray,
    dist: np.ndarray,
    image_size_wh: Tuple[int, int],
    pattern_size: Tuple[int, int],
    square_size: float,
) -> None:
    ensure_dir(out_path.parent)
    np.savez(
        str(out_path),
        K=K.astype(np.float64),
        dist=dist.astype(np.float64),
        image_size=np.array(image_size_wh, dtype=np.int32),
        pattern_size=np.array(pattern_size, dtype=np.int32),
        square_size=np.array([square_size], dtype=np.float64),
    )

def save_corners_visualization(out_path: Path, vis_bgr: np.ndarray) -> None:
    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), vis_bgr)

def mean_reprojection_error(
    objpoints: List[np.ndarray],
    imgpoints: List[np.ndarray],
    rvecs: List[np.ndarray],
    tvecs: List[np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
) -> float:
    """Mean L2 reprojection error over all detected corners (pixels)."""
    total_err = 0.0
    total_n = 0
    for op, ip, rv, tv in zip(objpoints, imgpoints, rvecs, tvecs):
        proj, _ = cv2.projectPoints(op, rv, tv, K, dist)  # (N,1,2)
        proj = proj.reshape(-1, 2)
        ip2 = ip.reshape(-1, 2)
        err = np.linalg.norm(proj - ip2, axis=1)
        total_err += float(np.sum(err))
        total_n += int(err.size)
    return total_err / max(1, total_n)


def _grid_edges(pattern_size: Tuple[int, int]):
    """Yield index pairs for edges in the chessboard corner grid."""
    cols, rows = pattern_size
    # horizontal edges
    for r in range(rows):
        base = r * cols
        for c in range(cols - 1):
            i = base + c
            j = base + c + 1
            yield i, j
    # vertical edges
    for r in range(rows - 1):
        base = r * cols
        for c in range(cols):
            i = base + c
            j = base + c + cols
            yield i, j

def make_chessboard_visualization(
    img_bgr: np.ndarray,
    pattern_size: Tuple[int, int],
    corners_ref: np.ndarray,
    *,
    point_radius: int = 20,
    point_thickness: int = -1,
    line_thickness: int = 7,
) -> np.ndarray:
    """
    Create a clear visualization: big points + thick grid lines.

    corners_ref: expected shape (N,1,2) or (N,2), float.
    """
    vis = img_bgr.copy()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        

    pts = corners_ref.reshape(-1, 2)
    pts_i = np.round(pts).astype(np.int32)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    # Draw thick "shadow" lines first (black), then colored lines
    for i, j in _grid_edges(pattern_size):
        p1 = tuple(pts_i[i].tolist())
        p2 = tuple(pts_i[j].tolist())
        cv2.line(vis, p1, p2, (0, 0, 0), line_thickness + 2, cv2.LINE_AA)
        cv2.line(vis, p1, p2, (0, 0, 255), line_thickness, cv2.LINE_AA)

    # Draw points (shadow + colored)
    for (u, v) in pts_i:
        cv2.circle(vis, (int(u), int(v)), point_radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(vis, (int(u), int(v)), point_radius, (0, 255, 0), point_thickness, cv2.LINE_AA)

    # Small label
    cv2.putText(
        vis,
        f"Detected corners: {pts_i.shape[0]}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        f"Detected corners: {pts_i.shape[0]}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return vis



@dataclass
class CalibResult:
    K: np.ndarray
    dist: np.ndarray
    image_size_wh: Tuple[int, int]
    mean_err_px: float
    used_images: int


def make_object_grid(pattern_size: Tuple[int, int], square_size: float) -> np.ndarray:
    """
    TODO
    Return 3D coordinates of inner corners on the chessboard plane (Z=0).

    Requirements:
      - output shape: (N, 3) where N = cols*rows
      - dtype: float32
      - coordinates in meters using square_size
      - convention: x increases along columns, y increases along rows

    Example first few points for pattern_size=(3,2), square_size=s:
      (0,0,0), (s,0,0), (2s,0,0), (0,s,0), (s,s,0), (2s,s,0)
    """
    raise NotImplementedError


def calibrate_from_chessboard(
    image_paths: List[Path],
    pattern_size: Tuple[int, int],
    square_size: float,
    out_dir: Path,
) -> CalibResult:
    """
    TODO
    1) Detect corners with cv2.findChessboardCorners
    2) Refine with cv2.cornerSubPix
    3) Run cv2.calibrateCamera
    4) Compute mean reprojection error (pixels)
    5) Save required outputs via:
         - save_corners_visualization(...)
         - write_calibration_txt(...)
         - save_calib_npz(...)
    """
    ensure_dir(out_dir)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-6)

    objpoints: List[np.ndarray] = []
    imgpoints: List[np.ndarray] = []

    image_size_wh: Tuple[int, int] | None = None
    corners_vis_bgr: np.ndarray | None = None

    objp = make_object_grid(pattern_size, square_size)

    for p in image_paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if image_size_wh is None:
            h, w = gray.shape[:2]
            image_size_wh = (w, h)

        # TODO: detect corners
        ok, corners = None, None
        if not ok or corners is None:
            continue

        # TODO: refine corners
        corners_ref = None

        objpoints.append(objp.copy())
        imgpoints.append(corners_ref)

        # save one visualization
        if corners_vis_bgr is None:
            corners_vis_bgr = make_chessboard_visualization(img, pattern_size, corners_ref)

    if image_size_wh is None or len(objpoints) < 3:
        raise RuntimeError(
            "Not enough valid chessboard detections. "
        )

    # TODO: run calibration
    rms, K, dist, rvecs, tvecs = None, None, None, None, None

    # TODO: compute mean reprojection error
    mean_err_px = None

    # TODO: save outputs
    # if corners_vis_bgr is not None:
    #     save_corners_visualization(...)
    # write_calibration_txt(...)
    # save_calib_npz(...)

    raise NotImplementedError("Finish TODOs in calibrate_from_chessboard().")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="../data/chessboard")
    ap.add_argument("--out_dir", type=str, default="figures")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    image_paths = list_images_sorted(data_dir)

    # chessboard infor (DO NOT CHANGE)
    PATTERN_SIZE: Tuple[int, int] = (9, 6)  # (cols, rows) inner corners
    SQUARE_SIZE_M: float = 0.025            # meters

    res = calibrate_from_chessboard(image_paths, PATTERN_SIZE, SQUARE_SIZE_M, out_dir)

    print("Calibration finished")
    print("K:\n", res.K)
    print("dist:\n", res.dist.reshape(-1))
    print(f"mean reprojection error: {res.mean_err_px:.4f} px")
    print(f"used images: {res.used_images}")


if __name__ == "__main__":
    main()
