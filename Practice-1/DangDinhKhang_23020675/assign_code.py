#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
INT6011 – Assignment 1
The Digital Camera Pipeline 

This file is structured to mirror the sections in the assignment PDF:

  2. Raw Image Extraction           → load_raw_tiff()
  3. Linearization                  → linearize()
  4. The Bayer Pattern              → _build_bayer_masks(), bayer_pattern
                                       detect_bayer_pattern_from_cr2()
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

    def detect_bayer_pattern_from_cr2(self, cr2_path: str):
        """
        Section 4.2.2 – read the Bayer pattern from the bytes of the Canon CR2
        file (no imaging or metadata library) and use it as self.pattern.

          1. CR2 header -> byte order, offsets of IFD0 and of the RAW IFD (IFD#3).
          2. IFD0, EXIF -> camera make/model; look for a CFAPattern tag.
          3. RAW IFD    -> lossless-JPEG SOF3 marker: full sensor width x height.
          4. CFA code   -> a CR2 has no CFA tag: Canon's code 0x94949494 (as in dcraw)
                           gives the 2x2 tile at sensor pixel (0, 0).
          5. Crop       -> dcraw cuts (left, top) border pixels before writing the
                           TIFF; the tile moves by that offset modulo 2.
        """
        # 2-bit color index (0 = R, 1 = G, 2 = B) per tile position, used by dcraw
        # for Canon EOS cameras such as the 5D Mark III, the value it assigns when the
        # file states no layout ("filters = 0x94949494" in dcraw.c). Some other Canon
        # models (older PowerShots, EOS D2000C) use different codes.
        CANON_FILTERS = 0x94949494
        # Border cut by dcraw for this sensor size: (width, height) -> (left, top)
        # (dcraw.c, table canon[]).
        DCRAW_CANON_MARGINS = {(5920, 3950): (122, 80)}

        with open(cr2_path, "rb") as f:
            data = f.read()

        def hexbytes(offset, size):
            return " ".join(f"{b:02x}" for b in data[offset:offset + size])

        # 1. Header: "II" = little-endian (Intel), "MM" = big-endian (Motorola).
        order = {b"II": "little", b"MM": "big"}.get(data[0:2])

        def uint(offset, size):
            return int.from_bytes(data[offset:offset + size], order)

        if order is None or uint(2, 2) != 42 or data[8:10] != b"CR":
            raise ValueError(f"{cr2_path} is not a Canon CR2 file")
        ifd0_offset, raw_ifd_offset = uint(4, 4), uint(12, 4)

        def read_ifd(offset):
            """Return {tag: position of the 4-byte value field} and the next IFD offset.
            IFD = count (2 bytes) + 12-byte entries (tag, type, count, value/offset)."""
            n = uint(offset, 2)
            entries = {uint(e, 2): e + 8 for e in range(offset + 2, offset + 2 + 12 * n, 12)}
            return entries, uint(offset + 2 + 12 * n, 4)

        def string_at(offset):
            return data[offset:data.index(b"\0", offset)].decode("ascii")

        # 2. IFD chain (IFD0 ... IFD3) and the EXIF sub-IFD; a CFA layout would be
        #    stored in tag 0x828E (TIFF/EP) or 0xA302 (EXIF).
        ifds, offset = [], ifd0_offset
        while offset:
            entries, offset = read_ifd(offset)
            ifds.append(entries)
        ifd0 = ifds[0]
        exif, _ = read_ifd(uint(ifd0[0x8769], 4))
        make, model = string_at(uint(ifd0[0x010F], 4)), string_at(uint(ifd0[0x0110], 4))
        if any(tag in d for d in ifds + [exif] for tag in (0x828E, 0xA302)):
            raise ValueError("the file records its CFA layout in a CFAPattern tag")

        # 3. RAW IFD -> StripOffsets -> lossless JPEG; walk its markers to SOF3.
        raw_ifd, _ = read_ifd(raw_ifd_offset)
        pos = uint(raw_ifd[0x0111], 4) + 2          # skip SOI (FF D8)
        while data[pos:pos + 2] != b"\xff\xc3":
            pos += 2 + int.from_bytes(data[pos + 2:pos + 4], "big")
        sof3 = pos                                  # JPEG markers are big-endian
        bit_depth = data[sof3 + 4]
        lines = int.from_bytes(data[sof3 + 5:sof3 + 7], "big")
        samples = int.from_bytes(data[sof3 + 7:sof3 + 9], "big")
        components = data[sof3 + 9]
        sensor_w, sensor_h = samples * components, lines

        # 4. Color of sensor pixel (row, col) from the CFA code.
        def sensor_color(row, col):
            return "RGB"[(CANON_FILTERS >> ((((row << 1) & 14) + (col & 1)) << 1)) & 3]

        # 5. Pixel (0, 0) of the TIFF is sensor pixel (top, left).
        if (sensor_w, sensor_h) not in DCRAW_CANON_MARGINS:
            raise ValueError(f"no dcraw margins known for a {sensor_w}x{sensor_h} sensor")
        left, top = DCRAW_CANON_MARGINS[(sensor_w, sensor_h)]
        sensor_tile = "".join(sensor_color(r, c) for r in (0, 1) for c in (0, 1))
        pattern = "".join(sensor_color(top + r, left + c) for r in (0, 1) for c in (0, 1))

        print("--- Section 4.2.2: Bayer Pattern from the CR2 Bytes ---")
        print(f"Header: {hexbytes(0, 16)}")
        print(f"  byte order: {order}, IFD0 at {ifd0_offset:#x}, RAW IFD at {raw_ifd_offset:#x}")
        print(f"Camera: {make} / {model}")
        print(f"CFAPattern tag (0x828E / 0xA302): not present -> CFA code {CANON_FILTERS:#010x}")
        print(f"SOF3 at {sof3:#x}: {hexbytes(sof3, 10)}")
        print(f"  {bit_depth}-bit, {lines} lines x {samples} samples x {components} components"
              f" -> sensor {sensor_w} x {sensor_h}")
        print(f"Sensor tile at (0, 0): {sensor_tile}")
        print(f"Crop origin (left, top) = ({left}, {top}) -> Bayer pattern: {pattern}")

        if pattern != self.pattern:
            print(f"Bayer pattern changed from {self.pattern} to {pattern}")
        self.pattern = pattern
        return pattern

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
        wb_multipliers=WB_MULTS
    )

    # Full pipeline execution
    pipeline.detect_bayer_pattern_from_cr2("campus.CR2")   # Section 4.2.2: sets the Bayer pattern
    pipeline.load_raw_tiff()
    pipeline.linearize()
    pipeline.white_balance(method="camera")   # Options: "camera", "gray_world", "white_world", "manual"
    pipeline.demosaic_bilinear()
    pipeline.color_space_correction(M_XYZ_CAM)
    pipeline.brightness_and_gamma(target_mean=0.25)
    pipeline.save_png_jpeg("campus_output", quality=95)
    # pipeline.show()  # Uncomment if running in an interactive graphical environment
