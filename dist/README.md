# Leaf Picture Processing Executable

This folder contains the packaged Windows executable for interactive leaf picture processing.

Main file:

- `leaf-picture-processing.exe`

## What It Does

The executable helps you process photos that contain two leaves.

For each picture it:

1. lets you choose or confirm the leaf mask with sliders
2. shows the detected `Leaf 1` and `Leaf 2` contours
3. detects pale or damaged spots on each leaf
4. saves output images and a CSV summary

## How To Run

Double-click:

```text
leaf-picture-processing.exe
```

Or run it from PowerShell:

```powershell
.\leaf-picture-processing.exe
```

If you run it without arguments, it opens a folder picker so you can choose the folder that contains the leaf pictures.

You can also pass the input folder directly:

```powershell
.\leaf-picture-processing.exe .\pictures_rect
```

## Optional Arguments

You can provide starting values for the sliders:

```powershell
.\leaf-picture-processing.exe .\pictures_rect --lh 36 --ls 25 --lv 25 --uh 86 --us 255 --uv 255 --epsilon 10 --spot-threshold 115
```

Useful options:

- `--output-root`: choose where the timestamped results folder will be created. If you do not set this, the results folder is created in the current working directory.
- `--lh`: initial minimum hue for the green mask. Lower values include more yellow-green tones. Raise it if warm background colors are entering the leaf mask.
- `--ls`: initial minimum saturation for the green mask. Raise it to reject gray or washed-out background pixels. Lower it if pale leaf tissue is being excluded.
- `--lv`: initial minimum brightness for the green mask. Raise it to remove dark shadow regions from the mask. Lower it if darker leaf areas disappear.
- `--uh`: initial maximum hue for the green mask. Raise it if more yellow-green leaf parts should be included. Lower it if non-leaf warm tones are leaking in.
- `--us`: initial maximum saturation for the green mask. This usually stays high. Lower it only if highly saturated artifacts should be excluded.
- `--uv`: initial maximum brightness for the green mask. Keep it high for bright or glossy pictures. Lower it if glare should not count as part of the leaf.
- `--epsilon`: initial contour smoothness. Higher values simplify the leaf outline more strongly. Lower it if the contour becomes too rough or loses important shape details.
- `--spot-threshold`: initial brightness threshold for pale-spot detection. Lower it to detect more pale or damaged regions. Raise it if too many normal leaf areas are marked as spots.

To see the built-in help:

```powershell
.\leaf-picture-processing.exe --help
```

## GUI Windows

For each image, the program opens:

- `Threshold Setup`: main preview window
- `Controls`: slider window

The main preview shows:

- the original image
- the original image with detected contours and large `Leaf 1` / `Leaf 2` labels
- the binary leaf mask
- per-leaf spot preview panels with measurements

## Keyboard Controls

- `Enter` or `q`: accept current values and process the image
- `s`: skip the current image
- `Esc`: stop the session

## Slider Meaning

### Min Hue

- Lower limit of the green color range.
- Lower it if parts of the leaf are missing.
- Raise it if yellow or brown background enters the mask.

### Min Saturation

- Rejects pale gray or weak-color pixels from the leaf mask.
- Lower it if faded leaf tissue disappears.
- Raise it if background gets included.

### Min Brightness

- Rejects dark pixels from the leaf mask.
- Lower it if dark leaf regions are missing.
- Raise it if shadows are entering the mask.

### Max Hue

- Upper limit of the green color range.
- Raise it if yellow-green leaf parts should be included.
- Lower it if non-leaf warm tones enter the mask.

### Max Saturation

- Upper saturation limit for accepted leaf pixels.
- Usually this can stay high.

### Max Brightness

- Upper brightness limit for accepted leaf pixels.
- Lower it only if strong glare should be excluded.

### Contour Smoothness

- Controls how much the contour is simplified.
- Raise it for smoother outlines.
- Lower it if too much leaf shape detail is lost.

### Spot Threshold

- Controls how bright a region must be to count as a pale spot.
- Lower it to detect more spots.
- Raise it to reduce false positives.

## Output

Each run creates a timestamped results folder in the selected output location, for example:

```text
results-2026-04-20-16-00-00
```

Typical output files:

- `results-YYYY-MM-DD-HH-MM-SS.csv`
- `image_leaf1.jpg`
- `image_leaf2.jpg`
- `image_leaf1_processing.jpg`
- `image_leaf1-green_object.jpg`

## Notes

- The tool expects two main leaves in each image.
- Results are measured in pixels, not physical units.
- Lighting and background quality still affect the result, so manual tuning may be needed.
