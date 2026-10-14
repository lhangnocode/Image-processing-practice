#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Regenerates every image in figures/ and every number quoted in Assignment_01.tex
by driving RawCameraPipeline (assign_code.py) and stopping it after the stage
each report section is about.

    python make_figures.py

Outputs:
    figures/*                 images included by the report
    campus_output.png/.jpg    final image (PNG + JPEG quality 95)
    outputs/campus_q*.jpg     final image at the other JPEG qualities
    figures/report_numbers.txt   everything printed during the run
"""

import os
import subprocess
import sys

import numpy as np
from skimage import color, img_as_ubyte, io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from assign_code import SUPPORTED_PATTERNS, RawCameraPipeline, detect_bayer_pattern

BLACK, WHITE = 2048, 15488
WB_MULTS = [1.858398, 1.0, 1.723633, 1.0]
M_XYZ_CAM = np.array([[6722, -635, -963],
                      [-4287, 12460, 2028],
                      [-908, 2162, 5668]], dtype=np.float64) / 10000.0
M_SRGB_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])

MANUAL_REGION = (2400, 2450, 2800, 2850)      # (y0, y1, x0, x1), same as the code default
DEMOSAIC_CROP = (2464, 2496, 3368, 3400)      # 32x32 px of sunlit, unclipped marble (Section 6)
EDGE_CROP = (2270, 2350, 3370, 3490)          # 80x120 px colour crop around it
JPEG_CROP = (2260, 2380, 3380, 3560)          # 120x180 px crop for Section 9
JPEG_QUALITIES = (95, 85, 75, 50, 25, 10)

FIG = "figures"
OUT = "outputs"


class Tee:
    """Mirror stdout (including the pipeline's own prints) into a log file."""

    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def log(msg=""):
    print(msg)


def new_pipeline(pattern="RGGB"):
    p = RawCameraPipeline("campus.tiff", BLACK, WHITE, WB_MULTS, pattern)
    p.load_raw_tiff()
    p.linearize()
    return p


def to_linear8(linear):
    """Render data exactly as it is at its stage: linear [0, 1] -> 8-bit, no gain, no gamma."""
    return img_as_ubyte(np.clip(linear, 0.0, 1.0))


def shrink(img, f):
    """Box-average downsampling by an integer factor."""
    h, w = img.shape[0] // f * f, img.shape[1] // f * f
    img = img[:h, :w]
    return img.reshape(h // f, f, w // f, f, *img.shape[2:]).mean(axis=(1, 3))


def enlarge(img, f):
    """Nearest-neighbour upsampling, so individual pixels stay visible."""
    return np.repeat(np.repeat(img, f, axis=0), f, axis=1)


def draw_box(img, y0, y1, x0, x1, t=3, rgb=(255, 0, 0)):
    img = img.copy()
    y0, x0 = max(y0, 0), max(x0, 0)
    for ys, xs in ((slice(y0, y0 + t), slice(x0, x1)), (slice(y1 - t, y1), slice(x0, x1)),
                   (slice(y0, y1), slice(x0, x0 + t)), (slice(y0, y1), slice(x1 - t, x1))):
        img[ys, xs] = rgb
    return img


def save(name, img, **kw):
    path = os.path.join(FIG, name)
    io.imsave(path, img, check_contrast=False, **kw)
    return path


def save_jpg(name, img):
    return save(name, img, quality=92)


def crop(img, region):
    y0, y1, x0, x1 = region
    return img[y0:y1, x0:x1]


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    sys.stdout = Tee(os.path.join(FIG, "report_numbers.txt"))

    # ------------------------------------------------------------------ Section 2
    base = new_pipeline()
    raw, lin = base.raw_image, base.linear_image.copy()
    h, w = lin.shape
    log("== Section 2 ==")
    log(f"width={w} height={h} min={raw.min()} max={raw.max()} "
        f"bits={np.log2(raw.max() + 1):.2f}")

    # ------------------------------------------------------------------ Section 4
    log("\n== Section 4: Bayer pattern from the CR2 bytes ==")
    detect_bayer_pattern("campus.CR2")
    try:
        info = subprocess.run(["./dcraw", "-i", "-v", "campus.CR2"],
                              capture_output=True, text=True, check=True).stdout
        log("dcraw -i -v campus.CR2:\n" + info)
    except (OSError, subprocess.CalledProcessError) as exc:
        log(f"dcraw not available: {exc}")

    # Reference run (RGGB, camera WB). Every figure up to Section 7 shows its data
    # linearly, as it is at the stage being discussed (no brightness, no gamma).
    ref = new_pipeline()
    ref.white_balance("camera")
    wb_mosaic = ref.linear_image.copy()
    cam_rgb = ref.demosaic_bilinear().copy()

    for pattern in SUPPORTED_PATTERNS:
        if pattern == "RGGB":
            rgb = cam_rgb
        else:
            p = new_pipeline(pattern)
            p.white_balance("camera")
            rgb = p.demosaic_bilinear()
        save_jpg(f"pattern_{pattern.lower()}.jpg", to_linear8(shrink(rgb, 4)))

    # ------------------------------------------------------------------ Section 5
    log("\n== Section 5: white balance ==")
    # Figures 2 and 3 are taken after the demosaicing step (Section 6), with no
    # color correction, brightness scaling or gamma.
    y0, y1, x0, x1 = MANUAL_REGION
    patch = lin[y0:y1, x0:x1]
    log(f"manual patch {MANUAL_REGION}: R={patch[0::2, 0::2].mean():.4f} "
        f"G={(patch[0::2, 1::2].mean() + patch[1::2, 0::2].mean()) / 2:.4f} "
        f"B={patch[1::2, 1::2].mean():.4f}")
    log(f"channel maxima (linear): R={lin[0::2, 0::2].max():.4f} "
        f"G={max(lin[0::2, 1::2].max(), lin[1::2, 0::2].max()):.4f} B={lin[1::2, 1::2].max():.4f}")

    for method in ("white_world", "gray_world", "camera", "manual"):
        p = new_pipeline()
        p.white_balance(method, manual_region=MANUAL_REGION if method == "manual" else None)
        balanced = p.demosaic_bilinear()
        save_jpg(f"wb_{method}.jpg", to_linear8(shrink(balanced, 4)))
        if method == "manual":
            # Figure 2: the patch on the Manual White Balance result, and the patch itself.
            wb_patch = p.linear_image[y0:y1, x0:x1]
            log(f"patch after manual WB: R={wb_patch[0::2, 0::2].mean():.4f} "
                f"G={(wb_patch[0::2, 1::2].mean() + wb_patch[1::2, 0::2].mean()) / 2:.4f} "
                f"B={wb_patch[1::2, 1::2].mean():.4f}")
            located = draw_box(to_linear8(shrink(balanced, 4)),
                               y0 // 4 - 1, y1 // 4 + 1, x0 // 4 - 1, x1 // 4 + 1, t=2)
            save_jpg("wb_patch_location.jpg", located)
            save("wb_patch_zoom.png", enlarge(to_linear8(crop(balanced, MANUAL_REGION)), 8))

    # ------------------------------------------------------------------ Section 6
    log("\n== Section 6: demosaicing ==")
    masks = RawCameraPipeline._build_bayer_masks(h, w, "RGGB")
    for i, name in enumerate(("red", "green", "blue")):
        ch_linear = np.zeros_like(cam_rgb)
        ch_linear[..., i] = cam_rgb[..., i]
        save_jpg(f"channel_{name}.jpg", to_linear8(shrink(ch_linear, 4)))
        log(f"{name}: measured sites {masks[i].mean():.0%}")

        # 32x32 crop before / after interpolation, drawn in the channel's own color.
        # Before: unmeasured sites are white, so the sampling grid is visible.
        measured = crop(masks[i], DEMOSAIC_CROP)
        before = np.full(measured.shape + (3,), 255, dtype=np.uint8)
        before[measured] = 0
        before[..., i][measured] = to_linear8(crop(wb_mosaic, DEMOSAIC_CROP))[measured]
        after = np.zeros(measured.shape + (3,), dtype=np.uint8)
        after[..., i] = to_linear8(crop(cam_rgb[..., i], DEMOSAIC_CROP))
        save(f"crop_sparse_{name}.png", enlarge(before, 12))
        save(f"crop_dense_{name}.png", enlarge(after, 12))
        log(f"{name}: crop {DEMOSAIC_CROP}, unmeasured pixels before={(~measured).sum()}, "
            f"zero-valued pixels after={(crop(cam_rgb[..., i], DEMOSAIC_CROP) == 0).sum()}")

    # ------------------------------------------------------------------ Section 7
    log("\n== Section 7: colour space correction ==")
    np.set_printoptions(precision=8, suppress=True)
    A = M_XYZ_CAM @ M_SRGB_XYZ
    A_inv = np.linalg.inv(A)
    M_normalized = np.linalg.inv(A / A.sum(axis=1, keepdims=True))
    log(f"A = M_sRGB->Cam:\n{A}\nrow sums of A: {A.sum(axis=1)}  reciprocals: {1 / A.sum(axis=1)}")
    log(f"inv(A) (no normalization):\n{A_inv}\nrow sums: {A_inv.sum(axis=1)}")
    log(f"rows normalized, then inverted:\n{M_normalized}\nrow sums: {M_normalized.sum(axis=1)}")

    small_cam = shrink(cam_rgb, 4)
    # Render before/after directly at this stage (pure linear, without Step 8 brightness/gamma)
    save_jpg("color_corr_before.jpg", to_linear8(small_cam))
    for name, M in (("cc_no_norm", A_inv), ("cc_with_norm", M_normalized)):
        save_jpg(f"{name}.jpg", to_linear8(small_cam @ M.T))

    ref.color_space_correction(M_XYZ_CAM)
    save_jpg("color_corr_after.jpg", to_linear8(shrink(ref.rgb_linear, 4)))

    # ------------------------------------------------------------------ Section 8
    log("\n== Section 8: brightness and gamma ==")
    mean_gray = color.rgb2gray(ref.rgb_linear).mean()
    scale = 0.25 / mean_gray
    log(f"mean gray before scaling={mean_gray:.4f} scale={scale:.4f}")
    bright_linear = np.clip(ref.rgb_linear * scale, 0, 1)
    save_jpg("linear_dark.jpg", img_as_ubyte(shrink(bright_linear, 4)))
    ref.brightness_and_gamma(target_mean=0.25)
    save_jpg("gamma_encoded.jpg", img_as_ubyte(shrink(ref.rgb_srgb, 4)))

    # ------------------------------------------------------------------ Section 9
    log("\n== Section 9: compression ==")
    res = ref.save_png_jpeg("campus_output", quality=95)
    png = io.imread(res["png_path"])
    png_gray = color.rgb2gray(png)
    log(f"PNG: {res['png_kb']:.2f} KB")
    save("jpeg_crop_png.png", enlarge(crop(png, JPEG_CROP), 5))
    cy0, cy1, cx0, cx1 = JPEG_CROP
    save_jpg("jpeg_crop_location.jpg",
             draw_box(img_as_ubyte(shrink(ref.rgb_srgb, 4)), cy0 // 4, cy1 // 4, cx0 // 4, cx1 // 4, t=3))
    for q in JPEG_QUALITIES:
        if q == 95:
            path = res["jpg_path"]
        else:
            path = os.path.join(OUT, f"campus_q{q}.jpg")
            io.imsave(path, png, quality=q, check_contrast=False)
        jpg = io.imread(path)
        kb = os.path.getsize(path) / 1024.0
        psnr = peak_signal_noise_ratio(png, jpg, data_range=255)
        ssim = structural_similarity(png_gray, color.rgb2gray(jpg), data_range=1.0)
        log(f"JPEG q={q}: {kb:.2f} KB ratio={res['png_kb'] / kb:.2f} PSNR={psnr:.2f} dB SSIM={ssim:.4f}")
        save(f"jpeg_crop_q{q}.png", enlarge(crop(jpg, JPEG_CROP), 5))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
