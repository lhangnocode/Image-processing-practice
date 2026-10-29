# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Coursework for INT3404 Image Processing, Assignment 1 ("The Digital Camera Pipeline"): develop the Canon EOS 5D Mark III RAW file `campus.CR2` into an sRGB image by hand, and write it up. The spec is `INT3404_Assignment_01.pdf`; its section numbers (2–9) are the organising scheme for both the code and the report.

Two deliverables live side by side and must agree with each other:

- `assign_code.py` — the instructor-provided skeleton (`RawCameraPipeline`), with the TODOs filled in.
- `Assignment_01.tex` + `reference_01.bib` + `figures/` — the report.

The git repository root is the parent directory; this project lives in `Practice-1/`. There are no tests and no lint configuration.

## Commands

```bash
# Environment (uv-created venv, Python 3.10 — the spec recommends 3.10.11)
source myvenv/bin/activate
pip install -r requirements.txt          # or: uv pip install -r requirements.txt

# Run the pipeline (reads campus.tiff from the CWD, writes campus_output.{png,jpg})
python assign_code.py

# Regenerate every report figure and number (~30 s): figures/, outputs/, campus_output.*,
# and figures/report_numbers.txt (the values quoted in the .tex)
python make_figures.py

# Compile the report (tectonic fetches packages and runs BibTeX itself)
tectonic Assignment_01.tex

# Regenerate the pipeline input from the RAW file
./dcraw -4 -d -v -w -T campus.CR2        # reconnaissance: prints black/white levels and WB multipliers
./dcraw -4 -D -T campus.CR2              # writes campus.tiff: 16-bit, un-demosaiced, no WB

# Rebuild dcraw if the binary is missing (it links only libm)
gcc -o dcraw -O4 dcraw.c -lm -DNODEPS
```

Both `dcraw` invocations write `campus.tiff`. The pipeline needs the `-D` output (the raw mosaic); the `-d` reconnaissance run overwrites it with a scaled, white-balanced version, so always run `-D` last.

`tectonic` is the only LaTeX engine installed locally (`pdflatex`, `latexmk`, `bibtex` are absent). The report must also stay compilable with pdfLaTeX on Overleaf (the spec's template lives there), so keep the source ASCII apart from what is already present; there the sequence is pdflatex → bibtex → pdflatex ×2 (`\bibliographystyle{IEEEtran}`).

## Pipeline architecture

`RawCameraPipeline` is a stateful, strictly ordered pipeline. Each stage reads one buffer on `self` and writes the next; stages raise `ValueError` if their input buffer is `None`.

| Stage | Reads | Writes |
|---|---|---|
| `load_raw_tiff()` | `campus.tiff` | `raw_image` (H×W float64, raw counts) |
| `linearize()` | `raw_image` | `linear_image` (H×W mosaic in [0,1]) |
| `white_balance(method, manual_region)` | `linear_image` | `linear_image` (**in place**) |
| `demosaic_bilinear()` | `linear_image` | `rgb_linear` (H×W×3, camera RGB) |
| `color_space_correction(M_xyz_to_cam)` | `rgb_linear` | `rgb_linear` (**in place**, now linear sRGB) |
| `brightness_and_gamma(target_mean)` | `rgb_linear` | `rgb_srgb` (gamma-encoded) |
| `save_png_jpeg(prefix, quality)` | `rgb_srgb` | `<prefix>.png`, `<prefix>.jpg` |

Things that follow from this and are easy to get wrong:

- **White balance happens on the mosaic, before demosaicing**, by scaling the R and B sites selected with `_build_bayer_masks()`. Green is the fixed baseline (`s_g = 1`).
- **`white_balance()` and `color_space_correction()` overwrite their input.** Calling either twice compounds the effect. To compare WB methods or Bayer patterns, build a fresh `RawCameraPipeline` per variant (or call `linearize()` again to reset `linear_image`).
- **The Bayer pattern is a constructor argument (default `RGGB`), overwritten by `detect_bayer_pattern_from_cr2()`**, used by both white balance and demosaicing; the string maps row-major onto the 2×2 tile. `RGGB` is the correct phase for this file.
- **`M_srgb_to_cam` is row-normalised before it is inverted** (as dcraw's `cam_xyz_coeff` does), so white-balanced neutrals stay neutral. This goes beyond the formula in the spec; the report's "Row Normalization" subsection states this and shows the result with and without it — keep code and that subsection in agreement.
- `detect_bayer_pattern_from_cr2(cr2_path)` (Section 4.2.2) is a pipeline method: it parses the CR2 bytes with no imaging/metadata library (header → IFD0/EXIF → no CFAPattern tag → IFD#3 → SOF3 sensor size → Canon code `0x94949494` shifted by dcraw's crop `(122, 80)`; it also reads the camera's own crop `(140, 96)` from MakerNote SensorInfo, tag `0x00E0`, for comparison only, since `campus.tiff` uses dcraw's crop) and sets `self.pattern`. `__main__` calls it before `load_raw_tiff()` instead of hard-coding the pattern (the old standalone `detect_bayer_pattern()` and `BAYER_PATTERN` were removed at the user's request).
- The default `manual_region` `(2400, 2450, 2800, 2850)` is `(y_start, y_end, x_start, x_end)` in full-resolution mosaic coordinates (image is 5796×3870) and points at the shaded part of the white marble plaque, just below the English inscription.
- Demosaicing uses four zero-centre averaging kernels (cross, diagonal, horizontal, vertical) via `scipy.ndimage.convolve` on mask-zeroed channels, each result used only at the site type it applies to. `mode="mirror"` matters: it preserves the Bayer phase across the border, `reflect` would not.

The calibration constants in the `__main__` block (`BLACK=2048`, `WHITE=15488`, `WB_MULTS`, `M_XYZ_CAM`) come from the `dcraw` reconnaissance output and the `adobe_coeff` table in `dcraw.c`; they are specific to this camera and file.

## Report ↔ code coupling

`make_figures.py` drives `RawCameraPipeline`, stopping after the stage each report section is about, and writes all images plus `figures/report_numbers.txt`. The text itself is not generated, so after a pipeline change rerun it and propagate by hand:

- `Assignment_01.tex` contains condensed copies of every pipeline method in `lstlisting` blocks (prints and guards stripped). Editing a method in `assign_code.py` means updating its listing.
- The report hard-codes numbers produced by running the code: image dimensions and raw min/max, the WB multiplier table, the `M_cam_to_srgb` matrix, the brightness scale factor, and the PNG/JPEG size table.
- Every figure in Sections 4–7 is rendered linearly (`to_linear8`: ×255, no brightness, no gamma), as the reviewer asked. Figures 2 and 3 (white balance) are taken after demosaicing, as the user asked; each caption names the step.
- The crop/patch coordinates at the top of `make_figures.py` are quoted in the report text and captions.
- The report must keep its "Statement of AI Usage" section accurate.

`report_image_processing.pdf` is not this report: it is another student's submission kept as a reference for structure only. Do not copy its text, figures or patch choices.

## Submission

Per the spec: zip the final source, the result images and the PDF report as `FullName_StudentID.zip` for the UET Portal. Large working files in the directory (`campus.CR2`, `campus.tiff`, the ~340 MB `report_image_processing.pdf`, `myvenv/`, `dcraw*`, `campus.CR2.pp3`) are worth checking before packaging.
