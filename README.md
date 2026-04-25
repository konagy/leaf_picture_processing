# Leaf Picture Processing

OpenCV-based tooling for processing photos that contain two leaves, separating them from the background, and measuring pale or damaged areas on each leaf.

The current main workflow is an application-style interactive GUI:

1. Open a folder that contains the input pictures.
2. Select any picture from the file list.
3. Adjust the mask and spot thresholds while the preview updates in the same window.
4. Process, skip, stop, or jump to the next pending file.
5. Resume the same work later from the saved session.

## Main script

Use [`picture_processing_oulema.py`](./picture_processing_oulema.py).

What it does:

- opens a folder picker if no input folder is given on the command line
- shows all supported images in a file list with pending/done/skipped/error status
- opens a single-window calibration GUI with preview, sliders, actions, and the file list together
- resumes unfinished work from the latest matching results folder
- segments the two largest green leaf contours
- saves the two leaf cutouts as `*_leaf1.jpg` and `*_leaf2.jpg`
- detects bright or pale spots on each extracted leaf using a grayscale threshold
- writes the measurements to a CSV file in a timestamped results folder

## Code Layout

- `picture_processing_oulema.py`: command-line entrypoint and legacy batch loop
- `leaf_processing_config.py`: shared constants and slider names
- `leaf_processing_core.py`: CSV, session, folder, and resume helpers
- `leaf_processing_image.py`: OpenCV image-processing and preview functions
- `leaf_processing_cv_app.py`: single-window OpenCV application shell
- `leaf_processing_tk_app.py`: native Tkinter application shell when Tkinter is available

## Requirements

Install:

```bash
pip install opencv-python numpy
```

The native desktop shell uses Python's standard `tkinter` module when it is available. If the bundled Python does not include `tkinter`, the script automatically falls back to a single-window OpenCV application with the same file list, sliders, and session resume behavior.

## How To Run

### Pick the folder interactively

```bash
python picture_processing_oulema.py
```

This opens the application. Use `Open Folder` to choose the folder with leaf pictures.

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

The main window contains:

- the full file list with processing status
- the current picture preview and leaf/spot processing preview
- the HSV, contour, and spot-threshold sliders
- action buttons for processing, skipping, selecting the next pending file, and stopping

The preview shows:

- the original image
- the original image with detected contours and `Leaf 1` / `Leaf 2` labels
- each extracted leaf with leaf area, spot area, spot percentage, and hole count
- each leaf's threshold-based spot preview

The preview is arranged as:

- row 1: original image and detected leaf contours
- row 2: `Leaf 1` and `Leaf 1 Spots`
- row 3: `Leaf 2` and `Leaf 2 Spots`

Keyboard controls:

- `Enter`: process the selected image
- `s`: skip the current image
- `n`: select the next pending image
- `o`: open a different folder
- `0`: reset preview zoom
- `Esc`: save and stop the session

Preview controls:

- mouse wheel: zoom in and out around the cursor
- left mouse drag: pan while zoomed in
- double-click: reset preview zoom

The application window uses a sharper preview source for zooming than the compact saved preview image, so zooming in the GUI keeps more detail visible while the processing still runs on the original input image.

You can click any file in the list, including already processed files, to review or reprocess it.

## Resume Workflow

Each results folder now includes a session file:

```text
leaf-processing-session.json
```

When the same input folder is opened again, the application automatically resumes the newest matching session that still has pending or error files. Done and skipped files stay marked in the list, and the last slider values are restored.

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
- `leaf-processing-session.json`
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
- total detected hole/spot area in pixels
- spotted area percentage, based only on the total hole/spot area
- number of detected holes/spots
- individual hole/spot areas in pixels, stored in `Hole areas [pixels]` as a semicolon-separated list
- slider settings used for that row: `lh`, `ls`, `lv`, `uh`, `us`, `uv`, `epsilon`, and `spot_threshold`

When an older resumable session is continued, the application upgrades the CSV header before appending new rows.

## Current Limitations

- Input images are expected to contain two main leaves.
- Leaf extraction depends strongly on the selected HSV thresholds.
- Spot detection is based on grayscale brightness, so lighting still matters.
- Results are measured in pixels and are not calibrated to real-world area.
- Very unusual backgrounds or overlapping leaves may still need manual tuning.
