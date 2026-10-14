#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
INT6011 – Assignment 1
The Digital Camera Pipeline 

This file is structured to mirror the sections in the assignment PDF:

  2. Raw Image Extraction           → load_raw_tiff()
  3. Linearization                  → linearize()
  4. The Bayer Pattern              → _build_bayer_masks(), bayer_pattern
  5. White Balancing                → white_balance(...)
  6. Demosaicing                    → demosaic_bilinear()
  7. Color Space Correction         → color_space_correction(...)
  8. Brightness & Gamma Encoding    → brightness_and_gamma()
  9. Compression                    → save_png_jpeg(...)

Your job:
  • Fill in the TODOs using the instructions from the PDF.
  • Experiment with different Bayer patterns and WB methods, and report your observations.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve
from skimage import color, img_as_ubyte, io

SUPPORTED_PATTERNS = ("RGGB", "BGGR", "GRBG", "GBRG")


class RawCameraPipeline:
    """
    RAW → sRGB pipeline skeleton.

    You are expected to implement each stage according to the assignment text.
    """

    def __init__(self,
                 tiff_path: str,
                 black_level: int,
                 white_level: int,
                 wb_multipliers,
                 bayer_pattern: str = "RGGB"):
        """
        Parameters
        ----------
        tiff_path      : path to the TIFF produced 
        black_level    : "darkness" value from dcraw
        white_level    : "saturation" value from dcraw
        wb_multipliers : [r_scale, g_scale, b_scale, g_scale] from dcraw
        bayer_pattern  : one of 'RGGB', 'BGGR', 'GRBG', 'GBRG'
        """
        self.tiff_path = tiff_path
        self.black = float(black_level)
        self.white = float(white_level)
        self.wb_mults = np.array(wb_multipliers, dtype=np.float64)

        self.pattern = bayer_pattern.upper()
        if self.pattern not in SUPPORTED_PATTERNS:
            raise ValueError(f"Unsupported Bayer pattern: {self.pattern}")

        # Buffers for intermediate results
        self.raw_image = None      
        self.linear_image = None   
        self.rgb_linear = None     
        self.rgb_srgb = None       

    # ------------------------------------------------------------------
    # Section 2: Raw Image Extraction
    # ------------------------------------------------------------------
    def load_raw_tiff(self):
        """
        TODO (Section 2.2 – Generating the Linear TIFF):

        1. Load the TIFF using skimage.io.imread.
        2. Convert the array to double-precision floats (np.float64).
        3. Print:
           - width and height
           - min and max raw values
           - approximate bits per pixel (hint: log2(max_value + 1))
        4. Store the loaded image in self.raw_image.
        """
        raw = io.imread(self.tiff_path)
        self.raw_image = raw.astype(np.float64)

        h, w = self.raw_image.shape[:2]
        min_val = self.raw_image.min()
        max_val = self.raw_image.max()
        approx_bpp = np.log2(max_val + 1)

        print("--- Section 2: Raw Image Extraction ---")
        print(f"Image Width: {w}, Image Height: {h}")
        print(f"Min raw value: {min_val}, Max raw value: {max_val}")
        print(f"Approximate bits per pixel: {approx_bpp:.2f} (container: {raw.dtype})")
        return self.raw_image

    # ------------------------------------------------------------------
    # Section 3: Linearization
    # ------------------------------------------------------------------
    def linearize(self):
        """
        TODO (Section 3 – Linearization):

        Implement the formula to map raw pixel P_raw to P_linear:

            P_linear = (P_raw - black) / (white - black)

        with:
            - values <= black mapped to 0
            - values >= white mapped to 1
            - everything in between linearly scaled to [0,1]

        Steps:
        1. Use self.raw_image as input.
        2. Apply the formula.
        3. Clip the result to [0, 1].
        4. Store the result in self.linear_image.
        """
        if self.raw_image is None:
            raise ValueError("Raw image is not loaded. Call load_raw_tiff() first.")

        p_linear = (self.raw_image - self.black) / (self.white - self.black)
        self.linear_image = np.clip(p_linear, 0.0, 1.0)

        print("--- Section 3: Linearization ---")
        print(f"Linearized image range: [{self.linear_image.min():.4f}, {self.linear_image.max():.4f}]")
        print(f"Mean linear intensity: {self.linear_image.mean():.4f}")
        return self.linear_image

    # ------------------------------------------------------------------
    # Section 4: The Bayer Pattern
    # ------------------------------------------------------------------
    @staticmethod
    def _build_bayer_masks(h, w, pattern):
        """
        TODO (Section 4 – Pattern Phase Identification):

        Build boolean masks for where the R, G, and B pixels live
        in the Bayer mosaic.

        The Bayer tile is 2x2:
            pattern[0] pattern[1]
            pattern[2] pattern[3]

        Example for 'RGGB':

            R G
            G B

        Steps:
        1. Create three boolean arrays: mask_r, mask_g, mask_b of shape (h, w).
        2. Construct boolean masks, setting True where each color channel is located.
        3. Return mask_r, mask_g, mask_b.
        """
        pattern = pattern.upper()
        mask_r = np.zeros((h, w), dtype=bool)
        mask_g = np.zeros((h, w), dtype=bool)
        mask_b = np.zeros((h, w), dtype=bool)

        channel_masks = {'R': mask_r, 'G': mask_g, 'B': mask_b}

        # Slices for 2x2 grid positions:
        # (row_slice, col_slice)
        tile_positions = [
            (slice(0, h, 2), slice(0, w, 2)),  # (0, 0): pattern[0]
            (slice(0, h, 2), slice(1, w, 2)),  # (0, 1): pattern[1]
            (slice(1, h, 2), slice(0, w, 2)),  # (1, 0): pattern[2]
            (slice(1, h, 2), slice(1, w, 2)),  # (1, 1): pattern[3]
        ]

        for char, (row_slice, col_slice) in zip(pattern, tile_positions):
            if char in channel_masks:
                channel_masks[char][row_slice, col_slice] = True

        return mask_r, mask_g, mask_b

    # ------------------------------------------------------------------
    # Section 5: White Balancing
    # ------------------------------------------------------------------
    def white_balance(self, method='camera', manual_region=None):
        """
        Task 4: White Balance.

        Implements 4 strategies to compute scaling factors (s_r, s_g, s_b).
        The Green channel is usually kept as the baseline (s_g = 1).

        Args:
            method (str): One of 'camera', 'white_world', 'gray_world', 'manual'.
            manual_region (tuple): (y_start, y_end, x_start, x_end) for 'manual' method.

        Formulas:
            1. White World (Max-RGB):
               s_r = max(G) / max(R)
               s_b = max(G) / max(B)

            2. Gray World:
               s_r = mean(G) / mean(R)
               s_b = mean(G) / mean(B)

            3. Camera Presets:
               Use self.wb_mults [m1, m2, m3, m4].
               Normalize so Green is 1.0: s_r = m1/m2, s_b = m3/m2.

            4. Manual:
               Calculate means of R, G, B within the manual_region.
               s_r = mean(G_patch) / mean(R_patch)
               s_b = mean(G_patch) / mean(B_patch)
        """
        print(f"--- Section 5: White Balancing ({method}) ---")
        if self.linear_image is None:
            raise ValueError("Image must be linearized first.")
        
        img = self.linear_image.copy()
        h, w = img.shape
        
        # 1. Build R/G/B masks using self._build_bayer_masks()
        mask_r, mask_g, mask_b = self._build_bayer_masks(h, w, self.pattern)
        
        s_g = 1.0
        if method == 'white_world':
            max_r = img[mask_r].max()
            max_g = img[mask_g].max()
            max_b = img[mask_b].max()
            s_r = max_g / max_r if max_r > 0 else 1.0
            s_b = max_g / max_b if max_b > 0 else 1.0
        elif method == 'gray_world':
            mean_r = img[mask_r].mean()
            mean_g = img[mask_g].mean()
            mean_b = img[mask_b].mean()
            s_r = mean_g / mean_r if mean_r > 0 else 1.0
            s_b = mean_g / mean_b if mean_b > 0 else 1.0
        elif method == 'camera':
            m1, m2, m3 = self.wb_mults[0], self.wb_mults[1], self.wb_mults[2]
            s_r = m1 / m2
            s_b = m3 / m2
        elif method == 'manual':
            if manual_region is None:
                # Default region near campus pavement/neutral ground
                manual_region = (2400, 2450, 2800, 2850)
                print(f"Notice: No manual_region specified. Defaulting to patch: {manual_region}")
            
            y_start, y_end, x_start, x_end = manual_region
            patch = img[y_start:y_end, x_start:x_end]
            patch_mask_r = mask_r[y_start:y_end, x_start:x_end]
            patch_mask_g = mask_g[y_start:y_end, x_start:x_end]
            patch_mask_b = mask_b[y_start:y_end, x_start:x_end]
            
            mean_r_patch = patch[patch_mask_r].mean()
            mean_g_patch = patch[patch_mask_g].mean()
            mean_b_patch = patch[patch_mask_b].mean()
            
            s_r = mean_g_patch / mean_r_patch if mean_r_patch > 0 else 1.0
            s_b = mean_g_patch / mean_b_patch if mean_b_patch > 0 else 1.0
        else:
            raise ValueError(f"Unknown white balance method: {method}")

        print(f"Applied WB scaling factors -> s_r: {s_r:.4f}, s_g: {s_g:.4f}, s_b: {s_b:.4f}")

        # 3. Apply scalars to the pixels in 'img'
        img[mask_r] *= s_r
        img[mask_b] *= s_b

        # 4. Clip values to [0, 1] and update self.linear_image
        self.linear_image = np.clip(img, 0.0, 1.0)
        return self.linear_image
    
    # ------------------------------------------------------------------
    # Section 6: Demosaicing (Bilinear Interpolation)
    # ------------------------------------------------------------------
    def demosaic_bilinear(self):
        """
        TODO (Section 6 – Demosaicing):

        Implement bilinear demosaicing using convolution as in the figure
        and text (Fig. "Demosaicing Logic").

        Use self.linear_image (white-balanced mosaic) and produce a full
        H x W x 3 RGB image in self.rgb_linear.

        Hints:
        - Use _build_bayer_masks(...) to get mask_r, mask_g, mask_b.
        - For Green:
            Known at mask_g locations.
            At R/B locations, interpolate from 4 cross neighbors (up/down/left/right).
              You can use a cross-shaped kernel, e.g.:

                  [0, 1, 0]
                  [1, 0, 1] / 4
                  [0, 1, 0]

        - For Red at Blue:
            Interpolate from 4 diagonal neighbors (kernel with diagonals).

        - For Red/Blue at Green:
            Interpolate from horizontal/vertical neighbors (only 2 neighbors),
              so be careful with normalization.

        - Mirror padding (mode="mirror") is acceptable.
        """
        print("--- Section 6: Demosaicing (Bilinear Interpolation) ---")
        if self.linear_image is None:
            raise ValueError("Linearized image not found. Run linearize() and white_balance() first.")

        img = self.linear_image
        h, w = img.shape
        mask_r, mask_g, mask_b = self._build_bayer_masks(h, w, self.pattern)

        # Averaging kernels: the centre weight is 0, so a kernel only ever
        # combines the neighbours of a pixel, and its weights sum to 1.
        k_cross = np.array([[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]], dtype=np.float64) / 4.0   # up/down/left/right
        k_diag = np.array([[1, 0, 1],
                           [0, 0, 0],
                           [1, 0, 1]], dtype=np.float64) / 4.0    # 4 diagonals
        k_horiz = np.array([[0, 0, 0],
                            [1, 0, 1],
                            [0, 0, 0]], dtype=np.float64) / 2.0   # left/right
        k_vert = np.array([[0, 1, 0],
                           [0, 0, 0],
                           [0, 1, 0]], dtype=np.float64) / 2.0    # up/down

        # Green: measured at G sites; at R/B sites all 4 cross neighbours are G.
        g_sparse = img * mask_g
        g_channel = np.where(mask_g, img, convolve(g_sparse, k_cross, mode="mirror"))

        def interpolate_red_or_blue(mask_c, mask_opposite):
            """Fill one of R/B. `mask_opposite` marks the sites of the other one."""
            sparse = img * mask_c
            # A G site has this colour either left/right (same row) or up/down.
            rows_with_c = mask_c.any(axis=1, keepdims=True)
            g_same_row = mask_g & rows_with_c
            g_other_row = mask_g & ~rows_with_c

            channel = sparse.copy()                      # measured samples kept as is
            channel[mask_opposite] = convolve(sparse, k_diag, mode="mirror")[mask_opposite]
            channel[g_same_row] = convolve(sparse, k_horiz, mode="mirror")[g_same_row]
            channel[g_other_row] = convolve(sparse, k_vert, mode="mirror")[g_other_row]
            return channel

        r_channel = interpolate_red_or_blue(mask_r, mask_b)
        b_channel = interpolate_red_or_blue(mask_b, mask_r)

        self.rgb_linear = np.stack([r_channel, g_channel, b_channel], axis=-1)
        self.rgb_linear = np.clip(self.rgb_linear, 0.0, 1.0)

        print(f"Demosaiced RGB image shape: {self.rgb_linear.shape}")
        return self.rgb_linear

    # ------------------------------------------------------------------
    # Section 7: Color Space Correction
    # ------------------------------------------------------------------
    def color_space_correction(self, M_xyz_to_cam: np.ndarray):
        """
        TODO (Section 7 – Color Space Correction):

        Convert from camera RGB to sRGB using CIE XYZ as an intermediate space.

        Given:
          - M_xyz_to_cam : M_{XYZ→Cam}, camera-specific (from dcraw/adobe_coeff).
          - M_srgb_to_xyz: fixed standard sRGB→XYZ matrix.

        You must:
        1. Compute M_srgb_to_cam = M_xyz_to_cam · M_srgb_to_xyz.
        2. Invert it to get M_cam_to_srgb.
        3. For each pixel in self.rgb_linear (camera RGB), multiply by M_cam_to_srgb.
        4. Store the result back into self.rgb_linear (now in sRGB linear space).
        5. Clip values to [0, 1].

        Note:
        - You can reshape the image to (N, 3) for a single matrix multiplication.
        """
        print("--- Section 7: Color Space Correction ---")
        if self.rgb_linear is None:
            raise ValueError("Demosaiced image not found. Run demosaic_bilinear() first.")

        # Fixed standard sRGB -> XYZ matrix (IEC 61966-2-1)
        M_srgb_to_xyz = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ], dtype=np.float64)

        # 1. Compute M_srgb_to_cam = M_xyz_to_cam · M_srgb_to_xyz
        M_srgb_to_cam = M_xyz_to_cam @ M_srgb_to_xyz

        # Row-normalize BEFORE inverting: each row of M_srgb_to_cam then sums to 1,
        # so sRGB white (1,1,1) maps to camera (1,1,1) -- which is what "white"
        # is in the already white-balanced camera image.
        M_srgb_to_cam = M_srgb_to_cam / M_srgb_to_cam.sum(axis=1, keepdims=True)

        # 2. Invert to get M_cam_to_srgb (its rows also sum to 1)
        M_cam_to_srgb = np.linalg.inv(M_srgb_to_cam)

        print("Computed Transformation Matrix M_{Cam->sRGB}:")
        print(np.round(M_cam_to_srgb, 8))

        # 3. For each pixel in self.rgb_linear, multiply by M_cam_to_srgb
        h, w, c = self.rgb_linear.shape
        flat_rgb = self.rgb_linear.reshape(-1, 3)
        corrected_flat = flat_rgb @ M_cam_to_srgb.T

        # 4 & 5. Store back into self.rgb_linear and clip
        self.rgb_linear = np.clip(corrected_flat.reshape(h, w, c), 0.0, 1.0)
        return self.rgb_linear

    # ------------------------------------------------------------------
    # Section 8: Brightness and Gamma Encoding
    # ------------------------------------------------------------------
    def brightness_and_gamma(self, target_mean: float = 0.25):
        """
        TODO (Section 8 – Brightness and Gamma):

        Step 1: Brightness Scaling
        - Convert self.rgb_linear to grayscale using color.rgb2gray.
        - Compute the current mean gray level.
        - Compute a scale factor so that the mean gray becomes target_mean (~0.25).
        - Multiply the RGB image by this factor and clip to [0, 1].

        Step 2: sRGB Gamma Encoding
        - For each pixel/channel C_linear in [0,1], apply:

            if C_linear <= 0.0031308:
                C_nonlin = 12.92 * C_linear
            else:
                C_nonlin = 1.055 * C_linear^(1/2.4) - 0.055

        - Store the gamma-encoded result in self.rgb_srgb.
        """
        print("--- Section 8: Brightness and Gamma Encoding ---")
        if self.rgb_linear is None:
            raise ValueError("Linear RGB image not found.")

        # Step 1: Brightness Scaling
        gray = color.rgb2gray(self.rgb_linear)
        current_mean = gray.mean()
        if current_mean > 0:
            scale_factor = target_mean / current_mean
        else:
            scale_factor = 1.0

        print(f"Current mean gray level: {current_mean:.4f}")
        print(f"Brightness scaling factor: {scale_factor:.4f} (target: {target_mean})")

        img_bright = np.clip(self.rgb_linear * scale_factor, 0.0, 1.0)

        # Step 2: sRGB Gamma Encoding
        mask = img_bright <= 0.0031308
        c_nonlin = np.where(
            mask,
            12.92 * img_bright,
            1.055 * (np.maximum(img_bright, 0.0) ** (1.0 / 2.4)) - 0.055
        )

        self.rgb_srgb = np.clip(c_nonlin, 0.0, 1.0)
        print(f"Final sRGB mean intensity: {self.rgb_srgb.mean():.4f}")
        return self.rgb_srgb

    # ------------------------------------------------------------------
    # Section 9: Compression
    # ------------------------------------------------------------------
    def save_png_jpeg(self, prefix: str, quality: int = 95):
        """
        TODO (Section 9 – Compression):

        1. Convert self.rgb_srgb (float [0,1]) to 8-bit using img_as_ubyte.
        2. Save as:
            prefix + ".png"  (lossless)
            prefix + ".jpg"  (with given quality, e.g., 95)
        3. Use os.path.getsize(...) to get file sizes.
        4. Print:
            - PNG size in KB
            - JPEG size in KB
            - Compression ratio = PNG_size / JPEG_size

        For the report:
        - You can call this multiple times with different JPEG quality values
          and visually inspect which quality is "indistinguishable" from PNG.
        """
        print(f"--- Section 9: Compression (JPEG quality = {quality}) ---")
        if self.rgb_srgb is None:
            raise ValueError("sRGB image not found. Call brightness_and_gamma() first.")

        # 1. Convert to 8-bit uint8
        img_8bit = img_as_ubyte(np.clip(self.rgb_srgb, 0.0, 1.0))

        png_path = f"{prefix}.png"
        jpg_path = f"{prefix}.jpg"

        # 2. Save PNG and JPEG
        io.imsave(png_path, img_8bit)
        io.imsave(jpg_path, img_8bit, quality=quality)

        # 3. Get file sizes in bytes
        png_size = os.path.getsize(png_path)
        jpg_size = os.path.getsize(jpg_path)

        png_kb = png_size / 1024.0
        jpg_kb = jpg_size / 1024.0
        ratio = png_size / jpg_size if jpg_size > 0 else 0.0

        # 4. Print metrics
        print(f"Saved: {png_path} ({png_kb:.2f} KB)")
        print(f"Saved: {jpg_path} ({jpg_kb:.2f} KB)")
        print(f"Compression Ratio (PNG / JPEG): {ratio:.2f}x")

        return {
            "png_path": png_path,
            "jpg_path": jpg_path,
            "png_kb": png_kb,
            "jpg_kb": jpg_kb,
            "compression_ratio": ratio
        }
    
    def show(self, title="Final Developed Image"):
        """
        Optional helper: show the final sRGB image with matplotlib.

        You can use this to visually compare different Bayer patterns and
        white balance methods.
        """
        if self.rgb_srgb is None:
            raise RuntimeError("Call brightness_and_gamma() first.")
        plt.figure(figsize=(6, 6))
        plt.imshow(self.rgb_srgb)
        plt.axis("off")
        plt.title(title)
        plt.show()


if __name__ == "__main__":
    # === Values extracted from: dcraw -4 -d -v -w -T campus.CR2 ===
    BLACK = 2048
    WHITE = 15488
    WB_MULTS = [1.858398, 1.0, 1.723633, 1.0]

    # Supported Bayer patterns: 'RGGB', 'BGGR', 'GRBG', 'GBRG'
    BAYER_PATTERN = "RGGB"

    # Example XYZ→Camera matrix (from dcraw/adobe_coeff, for Canon EOS 5D Mark III)
    # Values: { 6722,-635,-963,-4287,12460,2028,-908,2162,5668 } / 10000
    M_XYZ_CAM = np.array([
        [6722,  -635,  -963],
        [-4287, 12460,  2028],
        [ -908,  2162,  5668]
    ], dtype=np.float64) / 10000.0

    pipeline = RawCameraPipeline(
        tiff_path="campus.tiff",
        black_level=BLACK,
        white_level=WHITE,
        wb_multipliers=WB_MULTS,
        bayer_pattern=BAYER_PATTERN
    )

    # Full pipeline execution
    pipeline.load_raw_tiff()
    pipeline.linearize()
    pipeline.white_balance(method="camera")   # Options: "camera", "gray_world", "white_world", "manual"
    pipeline.demosaic_bilinear()
    pipeline.color_space_correction(M_XYZ_CAM)
    pipeline.brightness_and_gamma(target_mean=0.25)
    pipeline.save_png_jpeg("campus_output", quality=95)
    # pipeline.show()  # Uncomment if running in an interactive graphical environment


# ----------------------------------------------------------------------
# Section 4.2.2: automatic Bayer phase detection from the CR2 bytes
# (not part of the pipeline; run it with
#   python -c "from assign_code import detect_bayer_pattern; detect_bayer_pattern('campus.CR2')")
# ----------------------------------------------------------------------
def detect_bayer_pattern(cr2_path, verbose=True):
    """
    Find the Bayer phase of a Canon CR2 file by reading its bytes directly,
    the way a RAW decoder does (no imaging library involved):

      1. TIFF/CR2 header  -> byte order, IFD0 offset, raw IFD offset.
      2. IFDs              -> camera make/model, any CFAPattern tag.
      3. Raw IFD           -> lossless-JPEG SOF3 header: full sensor size, bit depth.
      4. MakerNote 0x00E0  -> SensorInfo: the camera's own crop borders.
      5. Sensor tile shifted by the crop origin (mod 2) -> pattern at pixel (0, 0).

    Returns a dict; "pattern" is the phase of the TIFF produced by "dcraw -D".
    """

    # Default Bayer code that dcraw uses when the file names no CFA layout
    # (dcraw.c: "if (filters == UINT_MAX) filters = 0x94949494;"). Each 2-bit
    # field is a color index: 0 = R, 1 = G, 2 = B.
    DEFAULT_FILTERS = 0x94949494

    # Margins dcraw cuts from a Canon sensor of a given size (dcraw.c, table
    # "canon[][11]": {raw_width, raw_height, left_margin, top_margin, ...}).
    # The TIFF written by "dcraw -D" starts at this corner.
    DCRAW_CANON_MARGINS = {(5920, 3950): (122, 80)}
    with open(cr2_path, "rb") as f:
        data = f.read()

    # 1. Header: "II" = little-endian (Intel), "MM" = big-endian (Motorola).
    order = {b"II": "little", b"MM": "big"}[data[0:2]]

    def uint(offset, size):
        return int.from_bytes(data[offset:offset + size], order)

    if uint(2, 2) != 42 or data[8:10] != b"CR":
        raise ValueError("not a TIFF-based Canon CR2 file")
    ifd0_offset, raw_ifd_offset = uint(4, 4), uint(12, 4)

    type_size = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1}

    def read_ifd(offset):
        """Return {tag: (type, count, value_offset)} and the next IFD offset."""
        entries = {}
        for i in range(uint(offset, 2)):
            entry = offset + 2 + 12 * i
            tag, typ, count = uint(entry, 2), uint(entry + 2, 2), uint(entry + 4, 4)
            # Values of up to 4 bytes are stored in the entry itself.
            inline = type_size.get(typ, 1) * count <= 4
            entries[tag] = (typ, count, entry + 8 if inline else uint(entry + 8, 4))
        return entries, uint(offset + 2 + 12 * uint(offset, 2), 4)

    def ascii(entries, tag):
        _, count, at = entries[tag]
        return data[at:at + count].rstrip(b"\0").decode("ascii")

    # 2. IFD chain (IFD0 = preview + metadata, ..., last = raw data), EXIF, MakerNote.
    ifds, offset = [], ifd0_offset
    while offset:
        entries, offset = read_ifd(offset)
        ifds.append(entries)
    ifd0 = ifds[0]
    exif, _ = read_ifd(uint(ifd0[0x8769][2], 4))
    makernote, _ = read_ifd(exif[0x927C][2])
    make, model = ascii(ifd0, 0x010F), ascii(ifd0, 0x0110)

    # A CFA layout would be stored in TIFF/EP tag 0x828E or EXIF tag 0xA302.
    has_cfa_tag = any(t in d for d in ifds + [exif, makernote] for t in (0x828E, 0xA302))
    if has_cfa_tag:
        raise NotImplementedError("file carries an explicit CFAPattern tag")

    # 3. Raw IFD -> lossless JPEG stream; its SOF3 marker gives the sensor size.
    raw_ifd, _ = read_ifd(raw_ifd_offset)
    pos = uint(raw_ifd[0x0111][2], 4) + 2     # StripOffsets, then skip SOI (FF D8)
    while data[pos:pos + 2] != b"\xff\xc3":   # walk markers until SOF3
        pos += 2 + int.from_bytes(data[pos + 2:pos + 4], "big")
    sof = pos + 4                             # JPEG headers are always big-endian
    bit_depth = data[sof]
    lines = int.from_bytes(data[sof + 1:sof + 3], "big")
    samples = int.from_bytes(data[sof + 3:sof + 5], "big")
    components = data[sof + 5]
    sensor_w, sensor_h = samples * components, lines

    # 4. Canon SensorInfo (MakerNote tag 0x00E0, int16 array): camera crop borders.
    _, _, at = makernote[0x00E0]
    info = [uint(at + 2 * k, 2) for k in range(9)]
    camera_left, camera_top = info[5], info[6]

    # 5. No CFA tag: the sensor tile at sensor pixel (0, 0) is dcraw's default code.
    #    Color index of (row, col) = filters >> ((((row << 1) & 14) + (col & 1)) << 1) & 3
    def tile_at(row, col):
        return "RGB"[(DEFAULT_FILTERS >> ((((row << 1) & 14) + (col & 1)) << 1)) & 3]

    def pattern_from(top, left):
        return "".join(tile_at(top + r, left + c) for r in (0, 1) for c in (0, 1))

    left, top = DCRAW_CANON_MARGINS[(sensor_w, sensor_h)]
    result = {
        "make": make, "model": model, "model_id": uint(makernote[0x0010][2], 4),
        "byte_order": order, "cfa_tag": has_cfa_tag,
        "sensor_size": (sensor_w, sensor_h), "bit_depth": bit_depth,
        "sensor_tile": pattern_from(0, 0),
        "camera_crop": (camera_left, camera_top), "camera_pattern": pattern_from(camera_top, camera_left),
        "dcraw_crop": (left, top), "pattern": pattern_from(top, left),
    }
    if verbose:
        print("--- Section 4.2.2: Bayer phase from the CR2 bytes ---")
        print(f"{make} / {model} (ModelID {result['model_id']:#x}), byte order {order}")
        print(f"CFAPattern tag present: {has_cfa_tag}")
        print(f"Sensor: {sensor_w} x {sensor_h}, {bit_depth}-bit (SOF3: {lines} lines x "
              f"{samples} samples x {components} components)")
        print(f"Sensor tile at (0,0): {result['sensor_tile']}")
        print(f"Camera crop origin (left, top) = {result['camera_crop']} -> {result['camera_pattern']}")
        print(f"dcraw crop origin (left, top)  = {result['dcraw_crop']} -> {result['pattern']}")
    return result
