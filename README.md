# Leaf Picture Processing

OpenCV-based tooling for processing photos that contain two leaves, separating them from the background, and measuring pale or damaged areas on each leaf.

The current main workflow is interactive:

1. Choose a folder that contains the input pictures.
2. Step through the pictures one by one.
3. Adjust the mask and spot thresholds in the GUI for the current image.
4. Accept, skip, or stop.
5. Save split leaf images, processed previews, and a CSV summary.

## Main script

Use [`picture_processing_oulema.py`](./picture_processing_oulema.py).

What it does:

- opens a folder picker if no input folder is given on the command line
- iterates through all supported images in that folder
- opens a calibration GUI before each image is processed
- segments the two largest green leaf contours
- saves the two leaf cutouts as `*_leaf1.jpg` and `*_leaf2.jpg`
- detects bright or pale spots on each extracted leaf using a grayscale threshold
- writes the measurements to a CSV file in a timestamped results folder

## Requirements

Install:

```bash
pip install opencv-python numpy
```

The script also uses standard-library modules such as `argparse`, `csv`, `pathlib`, `subprocess`, and `time`.

## How To Run

### Pick the folder interactively

```bash
python picture_processing_oulema.py
```

This opens a folder picker and lets you choose the folder with leaf pictures.

### Pass the folder on the command line

```bash
python picture_processing_oulema.py ./pictures_rect
```

### Optional starting values

You can also provide initial values for the sliders:

```bash
python picture_processing_oulema.py ./pictures_rect --lh 36 --ls 25 --lv 25 --uh 86 --us 255 --uv 255 --epsilon 10 --spot-threshold 115
```

## Build A Windows Executable

If `PyInstaller` is available in the environment, build the executable with:

```powershell
.\build_exe.ps1
```

The generated executable will be:

```text
dist\leaf-picture-processing.exe
```

## GUI Workflow

For each picture, the script opens:

- `Threshold Setup`: the main preview window
- `Controls`: the slider window

The main preview shows:

- the original image
- the original image with detected contours and `Leaf 1` / `Leaf 2` labels
- the binary leaf mask
- per-leaf spot preview panels with leaf area, spot area, and spot percentage

Keyboard controls:

- `Enter` or `q`: accept the current values and process the image
- `s`: skip the current image
- `Esc`: stop the full session

## Slider Guide

These sliders mainly control two stages:

- leaf masking: deciding which pixels belong to the leaf
- spot detection: deciding which bright pixels on the leaf count as damage or pale tissue

### Min Hue

- Lower boundary of the accepted green color range.
- Raise it if yellow or brown background starts being included in the leaf mask.
- Lower it if green leaf edges disappear.

### Min Saturation

- Minimum color strength for a pixel to be considered leaf.
- Raise it to reject pale paper background, gray shadows, or weak reflections.
- Lower it if faded or washed-out green leaf regions are being lost.

### Min Brightness

- Minimum brightness for a pixel to be considered leaf.
- Raise it if dark background regions are leaking into the mask.
- Lower it if darker leaf tissue is being removed.

### Max Hue

- Upper boundary of the accepted green color range.
- Raise it if yellow-green leaf areas should still belong to the mask.
- Lower it if warm non-leaf tones are entering the mask.

### Max Saturation

- Maximum color strength allowed for the leaf mask.
- In many cases this can stay high.
- Lower it only if very saturated artifacts or reflections are incorrectly accepted.

### Max Brightness

- Maximum brightness allowed for the leaf mask.
- Keep it high if the image contains strong lighting or glossy leaves.
- Lower it if very bright glare should be excluded from the contour.

### Contour Smoothness

- Controls how strongly the detected leaf outline is simplified.
- Higher values make the contour smoother and less detailed.
- Lower it if the outline becomes too rough or important shape details disappear.
- Raise it if the contour is noisy and jagged.

### Spot Threshold

- Controls how bright a pixel must be to count as a pale spot on the leaf.
- Lower it to detect more light or pale areas.
- Raise it if too much healthy texture is being marked as damaged.

## Practical Tuning Tips

- If the mask is too small, first lower `Min Saturation` or `Min Brightness`.
- If the mask is too large, first raise `Min Saturation` or `Min Brightness`.
- If leaf edges are missing, widen the hue range by lowering `Min Hue` or raising `Max Hue`.
- If the contour is noisy, increase `Contour Smoothness`.
- If too many pale regions are marked, raise `Spot Threshold`.
- If too few pale regions are marked, lower `Spot Threshold`.

## Output

Each run creates a new timestamped results folder such as:

- `results-2026-04-20-15-38-01`

Inside that folder you will find:

- a CSV summary file
- extracted leaves
- processed preview images
- threshold-based spot images

Typical filenames:

- `results-YYYY-MM-DD-HH-MM-SS/results-YYYY-MM-DD-HH-MM-SS.csv`
- `image_leaf1.jpg`
- `image_leaf2.jpg`
- `image_leaf1_processing.jpg`
- `image_leaf1-green_object.jpg`

## CSV Measurements

The measurements are pixel-based, not physical units.

Each CSV row contains:

- image name
- total leaf area in pixels
- detected spot area in pixels
- spotted area percentage
- threshold used for spot detection

## Current Limitations

- Input images are expected to contain two main leaves.
- Leaf extraction depends strongly on the selected HSV thresholds.
- Spot detection is based on grayscale brightness, so lighting still matters.
- Results are measured in pixels and are not calibrated to real-world area.
- Very unusual backgrounds or overlapping leaves may still need manual tuning.
