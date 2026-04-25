from __future__ import annotations

APP_TITLE = "Leaf Picture Processing"
SESSION_FILE_NAME = "leaf-processing-session.json"
SESSION_VERSION = 1
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PREVIEW_DEBOUNCE_SECONDS = 0.08
PREVIEW_ZOOM_MIN = 1.0
PREVIEW_ZOOM_MAX = 8.0
PREVIEW_ZOOM_FACTOR = 1.22
PREVIEW_BACKGROUND_COLOR = (16, 24, 32)
TOP_PANEL_WIDTH = 560
TOP_PANEL_HEIGHT = 336
BOTTOM_PANEL_WIDTH = 1008
BOTTOM_PANEL_HEIGHT = 461
TRACKBAR_NAMES = {
    "lh": "Min Hue",
    "ls": "Min Saturation",
    "lv": "Min Brightness",
    "uh": "Max Hue",
    "us": "Max Saturation",
    "uv": "Max Brightness",
    "epsilon": "Contour Smoothness",
    "spot_threshold": "Spot Threshold",
}
SLIDER_CONFIG = (
    ("lh", TRACKBAR_NAMES["lh"], 0, 179),
    ("ls", TRACKBAR_NAMES["ls"], 0, 255),
    ("lv", TRACKBAR_NAMES["lv"], 0, 255),
    ("uh", TRACKBAR_NAMES["uh"], 0, 179),
    ("us", TRACKBAR_NAMES["us"], 0, 255),
    ("uv", TRACKBAR_NAMES["uv"], 0, 255),
    ("epsilon", TRACKBAR_NAMES["epsilon"], 0, 100),
    ("spot_threshold", TRACKBAR_NAMES["spot_threshold"], 0, 255),
)
CSV_SLIDER_COLUMNS = tuple(key for key, _, _, _ in SLIDER_CONFIG)
CSV_BASE_COLUMNS = (
    "Image name",
    "Area of leaf [pixels]",
    "Area of spots [pixels]",
    "Percentage of spots [%]",
    "Number of holes [pcs]",
    "Hole areas [pixels]",
)
CSV_DEPRECATED_COLUMNS = {"Threshold"}
STATUS_LABELS = {
    "pending": "Pending",
    "done": "Done",
    "skipped": "Skipped",
    "error": "Error",
}
