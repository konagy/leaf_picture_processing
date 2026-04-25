from __future__ import annotations

import argparse
import base64
import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ModuleNotFoundError:
    tk = None
    filedialog = None
    messagebox = None
    ttk = None

import cv2
import numpy as np


APP_TITLE = "Leaf Picture Processing"
SESSION_FILE_NAME = "leaf-processing-session.json"
SESSION_VERSION = 1
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PREVIEW_DEBOUNCE_SECONDS = 0.08
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


def is_deprecated_csv_column(column: str) -> bool:
    return column in CSV_DEPRECATED_COLUMNS or (
        column.startswith("Hole ") and column.endswith(" area [pixels]")
    )


def csv_header_for(existing_header: Optional[Sequence[str]] = None) -> Tuple[str, ...]:
    existing_header = tuple(existing_header or ())
    known_columns = {*CSV_BASE_COLUMNS, *CSV_SLIDER_COLUMNS}
    unknown_columns = tuple(
        column for column in existing_header if column not in known_columns and not is_deprecated_csv_column(column)
    )
    return (*CSV_BASE_COLUMNS, *CSV_SLIDER_COLUMNS, *unknown_columns)


CSV_HEADER = csv_header_for()


def initialize_csv(file_path: Path) -> None:
    with file_path.open(mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(csv_header_for())


def ensure_csv_header(file_path: Path) -> List[str]:
    if not file_path.exists():
        initialize_csv(file_path)
        return list(csv_header_for())

    with file_path.open(mode="r", newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))

    if not rows:
        initialize_csv(file_path)
        return list(csv_header_for())

    header = rows[0]
    upgraded_header = list(csv_header_for(header))
    if header == upgraded_header:
        return header

    index_by_column = {column: index for index, column in enumerate(header)}
    upgraded_rows = [upgraded_header]
    for row in rows[1:]:
        upgraded_row = []
        for column in upgraded_header:
            source_index = index_by_column.get(column)
            upgraded_row.append(row[source_index] if source_index is not None and source_index < len(row) else "")
        upgraded_rows.append(upgraded_row)

    with file_path.open(mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(upgraded_rows)

    return upgraded_header


def append_to_csv(file_path: Path, data: Sequence[object]) -> None:
    header = ensure_csv_header(file_path)
    row = list(data)
    if len(row) < len(header):
        row.extend([""] * (len(header) - len(row)))

    with file_path.open(mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(row)


def slider_values_for_csv(slider_values: Optional[dict], threshold: object = "") -> List[object]:
    values: List[object] = []
    for key in CSV_SLIDER_COLUMNS:
        if slider_values and key in slider_values:
            values.append(slider_values[key])
        elif key == "spot_threshold":
            values.append(threshold)
        else:
            values.append("")
    return values


def measurement_csv_row(
    image_name: str,
    all_area: object,
    spot_area: object,
    spot_percentage: object,
    threshold: object,
    slider_values: Optional[dict],
    hole_areas: Optional[Sequence[int]] = None,
) -> Tuple[object, ...]:
    hole_area_values = list(hole_areas or [])
    return (
        image_name,
        all_area,
        spot_area,
        spot_percentage,
        len(hole_area_values),
        ";".join(str(area) for area in hole_area_values),
        *slider_values_for_csv(slider_values, threshold),
    )


def error_csv_row(
    image_name: str,
    error: Exception,
    slider_values: Optional[dict],
) -> Tuple[object, ...]:
    threshold = slider_values.get("spot_threshold", "") if slider_values else ""
    return (
        image_name,
        "ERROR",
        str(error),
        "",
        "",
        "",
        *slider_values_for_csv(slider_values, threshold),
    )


def current_timestamp() -> str:
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


def create_results_paths(output_root: Path) -> Tuple[Path, Path]:
    result_path = output_root / f"results-{current_timestamp()}"
    result_path.mkdir(parents=True, exist_ok=True)
    csv_file_path = result_path / f"{result_path.name}.csv"
    initialize_csv(csv_file_path)
    return result_path, csv_file_path


def list_images(folder: Path) -> List[Path]:
    return sorted(
        [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS]
    )


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def path_key(path: Path) -> str:
    return str(path.resolve()).casefold()


def session_path_for(results_folder: Path) -> Path:
    return results_folder / SESSION_FILE_NAME


def status_label(status: str) -> str:
    return STATUS_LABELS.get(status, status.title())


def make_file_entry(file_path: Path) -> dict:
    return {
        "path": str(file_path.resolve()),
        "name": file_path.name,
        "status": "pending",
        "last_values": None,
        "processed_at": None,
        "error": "",
    }


def session_has_open_work(session_data: dict) -> bool:
    return any(item.get("status", "pending") in {"pending", "error"} for item in session_data.get("files", []))


def normalize_session_data(
    session_data: dict,
    session_path: Path,
    input_folder: Path,
    initial_values: dict,
) -> dict:
    results_folder = session_path.parent.resolve()
    csv_file_path = Path(session_data.get("csv_file_path") or (results_folder / f"{results_folder.name}.csv"))
    ensure_csv_header(csv_file_path)

    session_data.setdefault("version", SESSION_VERSION)
    session_data["input_folder"] = str(input_folder.resolve())
    session_data["results_folder"] = str(results_folder)
    session_data["csv_file_path"] = str(csv_file_path.resolve())
    session_data.setdefault("created_at", now_iso())
    session_data.setdefault("current_values", initial_values.copy())
    session_data.setdefault("files", [])

    known_paths = set()
    for item in session_data["files"]:
        file_path = Path(item.get("path", ""))
        item["path"] = str(file_path.resolve())
        item.setdefault("name", file_path.name)
        item.setdefault("status", "pending")
        item.setdefault("last_values", None)
        item.setdefault("processed_at", None)
        item.setdefault("error", "")
        known_paths.add(path_key(file_path))

    for file_path in list_images(input_folder):
        if path_key(file_path) not in known_paths:
            session_data["files"].append(make_file_entry(file_path))

    session_data["updated_at"] = now_iso()
    return session_data


def write_session_data(session_path: Path, session_data: dict) -> None:
    session_data["updated_at"] = now_iso()
    temp_path = session_path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(session_data, indent=2), encoding="utf-8")
    temp_path.replace(session_path)


def find_resume_session(input_folder: Path, output_root: Path) -> Optional[Tuple[Path, dict]]:
    input_key = path_key(input_folder)
    candidates: List[Tuple[float, Path, dict]] = []

    for session_path in output_root.glob(f"results-*/{SESSION_FILE_NAME}"):
        try:
            session_data = json.loads(session_path.read_text(encoding="utf-8"))
            session_input = Path(session_data.get("input_folder", ""))
        except (OSError, json.JSONDecodeError):
            continue

        if path_key(session_input) == input_key and session_has_open_work(session_data):
            candidates.append((session_path.stat().st_mtime, session_path, session_data))

    if not candidates:
        return None

    _, session_path, session_data = max(candidates, key=lambda item: item[0])
    return session_path, session_data


def create_session(input_folder: Path, output_root: Path, initial_values: dict) -> Tuple[Path, dict]:
    files = list_images(input_folder)
    if not files:
        raise FileNotFoundError(f"No supported images were found in: {input_folder}")

    results_folder, csv_file_path = create_results_paths(output_root)
    session_data = {
        "version": SESSION_VERSION,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "input_folder": str(input_folder.resolve()),
        "results_folder": str(results_folder.resolve()),
        "csv_file_path": str(csv_file_path.resolve()),
        "current_values": initial_values.copy(),
        "selected_path": str(files[0].resolve()),
        "files": [make_file_entry(file_path) for file_path in files],
    }
    session_path = session_path_for(results_folder)
    write_session_data(session_path, session_data)
    return session_path, session_data


def session_summary(session_data: Optional[dict]) -> str:
    if not session_data:
        return "No folder selected"

    counts: Dict[str, int] = {"pending": 0, "done": 0, "skipped": 0, "error": 0}
    for item in session_data.get("files", []):
        counts[item.get("status", "pending")] = counts.get(item.get("status", "pending"), 0) + 1

    total = len(session_data.get("files", []))
    return (
        f"{total} files | "
        f"{counts.get('done', 0)} done | "
        f"{counts.get('pending', 0)} pending | "
        f"{counts.get('skipped', 0)} skipped | "
        f"{counts.get('error', 0)} error"
    )


def pick_input_folder() -> Path:
    print("Opening folder picker...")
    if tk is not None and filedialog is not None:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory(title="Select Folder With Leaf Pictures")
        root.destroy()
    else:
        script = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$dialog = New-Object System.Windows.Forms.FolderBrowserDialog; "
            "$dialog.Description = 'Select Folder With Leaf Pictures'; "
            "$dialog.UseDescriptionForTitle = $true; "
            "if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { "
            "[Console]::Write($dialog.SelectedPath) }"
        )
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            check=False,
        )
        folder = completed.stdout.strip()

    if not folder:
        raise ValueError("No input folder was selected.")
    return Path(folder).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a folder, iterate through pictures, and set thresholds for each picture."
    )
    parser.add_argument("input_folder", nargs="?", help="Folder containing the input images.")
    parser.add_argument(
        "--output-root",
        default=".",
        help="Root folder where the timestamped results folder will be created.",
    )
    parser.add_argument("--lh", type=int, default=36, help="Initial lower hue.")
    parser.add_argument("--ls", type=int, default=25, help="Initial lower saturation.")
    parser.add_argument("--lv", type=int, default=25, help="Initial lower value.")
    parser.add_argument("--uh", type=int, default=86, help="Initial upper hue.")
    parser.add_argument("--us", type=int, default=255, help="Initial upper saturation.")
    parser.add_argument("--uv", type=int, default=255, help="Initial upper value.")
    parser.add_argument(
        "--epsilon",
        type=int,
        default=10,
        help="Initial contour approximation factor in ten-thousandths.",
    )
    parser.add_argument(
        "--spot-threshold",
        type=int,
        default=115,
        help="Initial grayscale threshold for spots.",
    )
    parser.add_argument(
        "--legacy-opencv",
        action="store_true",
        help="Use the old OpenCV preview and separate Controls window.",
    )
    return parser.parse_args()


def clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


def resize_to_fit(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(max_width / max(width, 1), max_height / max(height, 1), 1.0)
    if scale >= 1.0:
        return image.copy()
    return cv2.resize(
        image,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def approximate_contour(contour: np.ndarray, epsilon_factor: float) -> np.ndarray:
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)


def segment_two_leaves(
    image: np.ndarray,
    hsv_image: np.ndarray,
    lower_green: np.ndarray,
    upper_green: np.ndarray,
    epsilon_factor: float,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    approx_contours = [approximate_contour(contour, epsilon_factor) for contour in contours]

    leaf_masks: List[np.ndarray] = []
    leaf_images: List[np.ndarray] = []
    for contour in approx_contours:
        contour_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
        leaf_image = cv2.bitwise_and(image, image, mask=contour_mask)
        cv2.drawContours(leaf_image, [contour], -1, color=0, thickness=6)
        leaf_masks.append(contour_mask)
        leaf_images.append(leaf_image)

    return mask, contours, approx_contours, leaf_images


def place_on_black_background(image: np.ndarray) -> np.ndarray:
    background_color = np.array([79, 152, 243], dtype=np.uint8)
    return np.where(image == 0, background_color, image)


def place_on_gray_background(image: np.ndarray) -> np.ndarray:
    background_color = np.array([160, 160, 160], dtype=np.uint8)
    return np.where(image == 0, background_color, image)


def connected_spot_areas(spot_mask: np.ndarray) -> List[int]:
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        spot_mask.astype(np.uint8),
        connectivity=8,
    )
    areas = [
        int(stats[index, cv2.CC_STAT_AREA])
        for index in range(1, component_count)
        if int(stats[index, cv2.CC_STAT_AREA]) > 0
    ]
    return sorted(areas, reverse=True)


def detect_spots(leaf_image: np.ndarray, threshold: int) -> Tuple[np.ndarray, int, int, float, List[int]]:
    gray_image_cv = cv2.cvtColor(leaf_image, cv2.COLOR_BGR2GRAY)
    _, white_areas_cv = cv2.threshold(gray_image_cv, threshold, 250, cv2.THRESH_BINARY)
    output_image_cv = cv2.bitwise_and(leaf_image, leaf_image, mask=white_areas_cv)

    spot_pixels = np.any(output_image_cv != [0, 0, 0], axis=-1)
    hole_areas = connected_spot_areas(spot_pixels)
    spot_area = sum(hole_areas)
    leaf_pixels = np.any(leaf_image != [0, 0, 0], axis=-1)
    all_area = int(np.count_nonzero(leaf_pixels))
    percentage = (spot_area / all_area) * 100.0 if all_area else 0.0
    return output_image_cv, all_area, spot_area, percentage, hole_areas


def pad_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    pad_total = height - image.shape[0]
    pad_top = pad_total // 2
    pad_bottom = pad_total - pad_top
    return cv2.copyMakeBorder(
        image,
        pad_top,
        pad_bottom,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def contour_label_position(contour: np.ndarray) -> Tuple[int, int]:
    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
        return center_x, center_y

    x, y, width, height = cv2.boundingRect(contour)
    return x + width // 2, y + height // 2


def draw_leaf_labels(image: np.ndarray, contours: Sequence[np.ndarray]) -> None:
    for index, contour in enumerate(contours, start=1):
        center_x, center_y = contour_label_position(contour)
        label = f"Leaf {index}"
        cv2.putText(
            image,
            label,
            (center_x - 140, center_y + 20),
            cv2.FONT_ITALIC,
            4.0,
            (0, 255, 255),
            12,
            cv2.LINE_AA,
        )

def build_calibration_preview(
    image: np.ndarray,
    hsv_image: np.ndarray,
    image_name: str,
    lower_green: np.ndarray,
    upper_green: np.ndarray,
    epsilon_factor: float,
    spot_threshold: int,
) -> np.ndarray:
    _, _, approx_contours, leaf_images = segment_two_leaves(image, hsv_image, lower_green, upper_green, epsilon_factor)

    picSize = 0.6
    original_panel = resize_to_fit(image, TOP_PANEL_WIDTH, TOP_PANEL_HEIGHT)

    contour_overlay = image.copy()
    if approx_contours:
        cv2.drawContours(contour_overlay, approx_contours, -1, (0, 255, 255), 20)
        draw_leaf_labels(contour_overlay, approx_contours)
    contour_panel = resize_to_fit(contour_overlay, TOP_PANEL_WIDTH, TOP_PANEL_HEIGHT)
    cv2.putText(contour_panel, "Original + Leaf Contours", (20, 35), cv2.FONT_ITALIC, picSize*1.0, (50, 144, 66), 2)

    combined_panels: List[np.ndarray] = []
    spot_panels: List[np.ndarray] = []
    for index, leaf_image in enumerate(leaf_images, start=1):
        spots_image, all_area, spot_area, percentage, hole_areas = detect_spots(leaf_image, spot_threshold)
        combined = cv2.addWeighted(place_on_black_background(leaf_image), 0.75, spots_image, 0.95, 0.0)
        combined_panel = resize_to_fit(combined, BOTTOM_PANEL_WIDTH, BOTTOM_PANEL_HEIGHT)
        cv2.putText(combined_panel, f"Leaf {index}", (20, 35), cv2.FONT_ITALIC, picSize*1.0, (0, 0, 0), 2)
        cv2.putText(combined_panel, f"Leaf area: {all_area}", (20, 70), cv2.FONT_ITALIC, picSize*0.8, (0, 0, 0), 2)
        cv2.putText(combined_panel, f"Spot area: {spot_area}", (20, 105), cv2.FONT_ITALIC, picSize*0.8, (0, 0, 0), 2)
        cv2.putText(combined_panel, f"Spot %: {percentage:.2f}", (20, 140), cv2.FONT_ITALIC, picSize*0.8, (0, 0, 0), 2)
        cv2.putText(combined_panel, f"Holes: {len(hole_areas)}", (20, 175), cv2.FONT_ITALIC, picSize*0.8, (0, 0, 0), 2)
        combined_panels.append(combined_panel)

        spot_panel = resize_to_fit(place_on_gray_background(spots_image), BOTTOM_PANEL_WIDTH, BOTTOM_PANEL_HEIGHT)
        cv2.putText(spot_panel, f"Leaf {index} Spots", (20, 35), cv2.FONT_ITALIC, picSize*1.0, (50, 144, 66), 2)
        cv2.putText(spot_panel, f"Threshold: {spot_threshold}", (20, 70), cv2.FONT_ITALIC, picSize*0.8, (50, 144, 66), 2)
        spot_panels.append(spot_panel)

    leaf_panels = combined_panels + spot_panels

    if not leaf_panels:
        empty = np.zeros((240, 700, 3), dtype=np.uint8)
        cv2.putText(empty, "Could not isolate two leaves with current HSV values.", (20, 120), cv2.FONT_ITALIC, picSize*0.9, (0, 0, 255), 2)
        leaf_panels = [empty]

    top_row_height = max(original_panel.shape[0], contour_panel.shape[0])
    top_row = cv2.hconcat(
        [
            pad_to_height(original_panel, top_row_height),
            pad_to_height(contour_panel, top_row_height),
        ]
    )

    bottom_row_height = max(panel.shape[0] for panel in leaf_panels)
    bottom_row = cv2.hconcat([pad_to_height(panel, bottom_row_height) for panel in leaf_panels])

    canvas_width = max(top_row.shape[1], bottom_row.shape[1])
    if top_row.shape[1] < canvas_width:
        top_row = cv2.copyMakeBorder(top_row, 0, 0, 0, canvas_width - top_row.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
    if bottom_row.shape[1] < canvas_width:
        bottom_row = cv2.copyMakeBorder(bottom_row, 0, 0, 0, canvas_width - bottom_row.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))

    preview = cv2.vconcat([top_row, bottom_row])
    preview = resize_to_fit(preview, 1650, 950)
    cv2.putText(
        preview,
        f"File: {image_name}",
        (20, 40),
        cv2.FONT_ITALIC,
        picSize*0.9,
        (50, 144, 66),
        2,
    )
    cv2.putText(
        preview,
        "Enter: process | s: skip image | Esc: save and stop",
        (20, preview.shape[0] - 20),
        cv2.FONT_ITALIC,
        0.8,
        (255, 255, 255),
        2,
    )
    return preview


def calibrate_image(
    image_path: Path,
    initial_values: dict,
) -> Tuple[dict, str]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def nothing(_: int) -> None:
        return None

    cv2.namedWindow("Threshold Setup", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Threshold Setup", 1600, 950)
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 480, 360)

    cv2.createTrackbar(TRACKBAR_NAMES["lh"], "Controls", initial_values["lh"], 179, nothing)
    cv2.createTrackbar(TRACKBAR_NAMES["ls"], "Controls", initial_values["ls"], 255, nothing)
    cv2.createTrackbar(TRACKBAR_NAMES["lv"], "Controls", initial_values["lv"], 255, nothing)
    cv2.createTrackbar(TRACKBAR_NAMES["uh"], "Controls", initial_values["uh"], 179, nothing)
    cv2.createTrackbar(TRACKBAR_NAMES["us"], "Controls", initial_values["us"], 255, nothing)
    cv2.createTrackbar(TRACKBAR_NAMES["uv"], "Controls", initial_values["uv"], 255, nothing)
    cv2.createTrackbar(TRACKBAR_NAMES["epsilon"], "Controls", initial_values["epsilon"], 100, nothing)
    cv2.createTrackbar(TRACKBAR_NAMES["spot_threshold"], "Controls", initial_values["spot_threshold"], 255, nothing)

    current_values = initial_values.copy()
    previous_values = None
    pending_values = None
    last_change_time = 0.0
    preview = None
    action = "accept"

    while True:
        current_values = {
            "lh": cv2.getTrackbarPos(TRACKBAR_NAMES["lh"], "Controls"),
            "ls": cv2.getTrackbarPos(TRACKBAR_NAMES["ls"], "Controls"),
            "lv": cv2.getTrackbarPos(TRACKBAR_NAMES["lv"], "Controls"),
            "uh": cv2.getTrackbarPos(TRACKBAR_NAMES["uh"], "Controls"),
            "us": cv2.getTrackbarPos(TRACKBAR_NAMES["us"], "Controls"),
            "uv": cv2.getTrackbarPos(TRACKBAR_NAMES["uv"], "Controls"),
            "epsilon": cv2.getTrackbarPos(TRACKBAR_NAMES["epsilon"], "Controls"),
            "spot_threshold": cv2.getTrackbarPos(TRACKBAR_NAMES["spot_threshold"], "Controls"),
        }

        if current_values != pending_values:
            pending_values = current_values.copy()
            last_change_time = time.monotonic()

        should_refresh = preview is None or (
            pending_values is not None
            and pending_values != previous_values
            and (time.monotonic() - last_change_time) >= PREVIEW_DEBOUNCE_SECONDS
        )

        if should_refresh:
            try:
                preview = build_calibration_preview(
                    image=image,
                    hsv_image=hsv_image,
                    image_name=image_path.name,
                    lower_green=np.array([pending_values["lh"], pending_values["ls"], pending_values["lv"]]),
                    upper_green=np.array([pending_values["uh"], pending_values["us"], pending_values["uv"]]),
                    epsilon_factor=pending_values["epsilon"] / 10000.0,
                    spot_threshold=pending_values["spot_threshold"],
                )
            except Exception as error:
                preview = resize_to_fit(image, 1650, 950)
                cv2.putText(preview, f"File: {image_path.name}", (20, 40), cv2.FONT_ITALIC, 0.9, (255, 255, 255), 2)
                cv2.putText(preview, str(error), (20, 80), cv2.FONT_ITALIC, 0.9, (0, 0, 255), 2)
            previous_values = pending_values.copy()

        cv2.imshow("Threshold Setup", preview)
        key = cv2.waitKey(15) & 0xFF
        if key == 13:       # if key in (13, ord("q"))
            action = "accept"
            break
        if key == ord("s"):
            action = "skip"
            break
        if key == 27:
            action = "stop"
            break

    cv2.destroyAllWindows()
    return current_values, action


def save_processing_preview(original: np.ndarray, processed: np.ndarray, output_path: Path) -> None:
    left = resize_to_fit(original, 900, 700)
    right = resize_to_fit(processed, 900, 700)
    height = max(left.shape[0], right.shape[0])
    left = pad_to_height(left, height)
    right = pad_to_height(right, height)
    combined = cv2.hconcat([left, right])
    cv2.imwrite(str(output_path), combined)


def splitting_images(
    image_path: Path,
    results_folder: Path,
    lower_green: np.ndarray,
    upper_green: np.ndarray,
    epsilon_factor: float,
) -> List[Tuple[Path, np.ndarray]]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    _, _, approx_contours, leaf_images = segment_two_leaves(image, hsv_image, lower_green, upper_green, epsilon_factor)
    if len(approx_contours) < 2 or len(leaf_images) < 2:
        raise ValueError("Could not find two leaves with the selected HSV thresholds.")

    processed_files: List[Tuple[Path, np.ndarray]] = []
    for index, leaf_image in enumerate(leaf_images, start=1):
        output_path = results_folder / f"{image_path.stem}_leaf{index}.jpg"
        cv2.imwrite(str(output_path), place_on_black_background(leaf_image))
        processed_files.append((output_path, leaf_image))

    return processed_files


def picture_processing_from_image(
    leaf_image: np.ndarray,
    image_name: str,
    threshold: int,
    results_folder: Path,
    csv_file_path: Path,
    plotting: bool,
    slider_values: Optional[dict] = None,
) -> None:
    filename_id = Path(image_name).stem
    output_image_cv, all_area, spot_area, spot_percentage, hole_areas = detect_spots(leaf_image, threshold)

    save_processing_preview(
        original=place_on_black_background(leaf_image),
        processed=place_on_gray_background(output_image_cv),
        output_path=results_folder / f"{filename_id}_processing.jpg",
    )
    cv2.imwrite(str(results_folder / f"{filename_id}-green_object.jpg"), place_on_gray_background(output_image_cv))

    append_to_csv(
        csv_file_path,
        measurement_csv_row(
            image_name=image_name,
            all_area=all_area,
            spot_area=spot_area,
            spot_percentage=round(spot_percentage, 4),
            threshold=threshold,
            slider_values=slider_values,
            hole_areas=hole_areas,
        ),
    )

    if plotting:
        print(f"  {image_name}: leaf area={all_area}, spot area={spot_area}, holes={len(hole_areas)}, percentage={spot_percentage:.4f}, threshold={threshold}")


def picture_processing(
    image_path: Path,
    threshold: int,
    results_folder: Path,
    csv_file_path: Path,
    plotting: bool,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    picture_processing_from_image(
        leaf_image=image,
        image_name=image_path.name,
        threshold=threshold,
        results_folder=results_folder,
        csv_file_path=csv_file_path,
        plotting=plotting,
        slider_values={"spot_threshold": threshold},
    )


def cv_image_to_photo(image: np.ndarray) -> tk.PhotoImage:
    if len(image.shape) == 2:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width = rgb_image.shape[:2]
    ppm_bytes = f"P6\n{width} {height}\n255\n".encode("ascii") + rgb_image.tobytes()
    encoded = base64.b64encode(ppm_bytes).decode("ascii")
    return tk.PhotoImage(data=encoded, format="PPM")


def make_message_preview(title: str, lines: Sequence[str]) -> np.ndarray:
    canvas = np.full((720, 1100, 3), (34, 42, 51), dtype=np.uint8)
    cv2.putText(canvas, title, (44, 84), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (236, 240, 241), 2, cv2.LINE_AA)
    y = 142
    for line in lines:
        cv2.putText(canvas, line[:92], (48, y), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (190, 202, 214), 1, cv2.LINE_AA)
        y += 38
    return canvas


class LeafProcessingApp:
    def __init__(self, root: tk.Tk, input_folder: Optional[Path], output_root: Path, initial_values: dict) -> None:
        self.root = root
        self.output_root = output_root
        self.initial_values = initial_values.copy()
        self.session_path: Optional[Path] = None
        self.session_data: Optional[dict] = None
        self.selected_path: Optional[str] = None
        self.loaded_image_path: Optional[str] = None
        self.loaded_image: Optional[np.ndarray] = None
        self.loaded_hsv_image: Optional[np.ndarray] = None
        self.preview_image: Optional[np.ndarray] = None
        self.tk_preview: Optional[tk.PhotoImage] = None
        self.preview_after_id: Optional[str] = None
        self.render_after_id: Optional[str] = None
        self.slider_vars: Dict[str, tk.DoubleVar] = {}
        self.value_vars: Dict[str, tk.StringVar] = {}
        self.slider_widgets: List[ttk.Scale] = []
        self.action_buttons: List[ttk.Button] = []
        self.tree_item_by_path: Dict[str, str] = {}
        self.path_by_tree_item: Dict[str, str] = {}
        self.loading_values = False
        self.busy = False

        self.folder_var = tk.StringVar(value="No folder selected")
        self.results_var = tk.StringVar(value="")
        self.summary_var = tk.StringVar(value=session_summary(None))
        self.current_file_var = tk.StringVar(value="Select a folder to begin")
        self.status_var = tk.StringVar(value="Ready")

        self._configure_window()
        self._build_layout()
        self._apply_values_to_sliders(self.initial_values)
        self._show_empty_preview(
            "Leaf Picture Processing",
            ["Choose a picture folder to start or pass one on the command line."],
        )

        if input_folder:
            self.load_folder(input_folder)

        self.root.protocol("WM_DELETE_WINDOW", self.stop_and_save)
        self.root.bind("<Return>", lambda _: self.process_selected())
        self.root.bind("<Escape>", lambda _: self.stop_and_save())
        self.root.bind("s", lambda _: self.skip_selected())
        self.root.bind("<Control-o>", lambda _: self.choose_folder())

    def _configure_window(self) -> None:
        self.root.title(APP_TITLE)
        self.root.geometry("1480x900")
        self.root.minsize(1120, 720)
        self.root.configure(bg="#edf1f5")
        self.root.option_add("*Font", "Segoe UI 10")

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("App.TFrame", background="#edf1f5")
        style.configure("Surface.TFrame", background="#ffffff")
        style.configure("Panel.TFrame", background="#f8fafc")
        style.configure("Title.TLabel", background="#ffffff", foreground="#16202a", font=("Segoe UI", 16, "bold"))
        style.configure("Heading.TLabel", background="#ffffff", foreground="#16202a", font=("Segoe UI", 11, "bold"))
        style.configure("Muted.TLabel", background="#ffffff", foreground="#5e6b78")
        style.configure("PanelMuted.TLabel", background="#f8fafc", foreground="#5e6b78")
        style.configure("Current.TLabel", background="#edf1f5", foreground="#16202a", font=("Segoe UI", 13, "bold"))
        style.configure("Status.TLabel", background="#edf1f5", foreground="#536170")
        style.configure("Primary.TButton", padding=(12, 8))
        style.configure("App.TButton", padding=(10, 7))
        style.configure("Treeview", rowheight=28, fieldbackground="#ffffff", background="#ffffff")
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))

    def _build_layout(self) -> None:
        self.root.grid_columnconfigure(0, minsize=340)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, minsize=370)
        self.root.grid_rowconfigure(0, weight=1)

        sidebar = ttk.Frame(self.root, style="Surface.TFrame", padding=(16, 16, 16, 16))
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(14, 8), pady=14)
        sidebar.grid_rowconfigure(5, weight=1)
        sidebar.grid_columnconfigure(0, weight=1)

        ttk.Label(sidebar, text=APP_TITLE, style="Title.TLabel").grid(row=0, column=0, sticky="ew")
        ttk.Label(sidebar, textvariable=self.summary_var, style="Muted.TLabel").grid(row=1, column=0, sticky="ew", pady=(4, 14))

        self.open_button = ttk.Button(sidebar, text="Open Folder", style="Primary.TButton", command=self.choose_folder)
        self.open_button.grid(row=2, column=0, sticky="ew", pady=(0, 14))

        ttk.Label(sidebar, text="Input", style="Heading.TLabel").grid(row=3, column=0, sticky="ew")
        ttk.Label(sidebar, textvariable=self.folder_var, style="Muted.TLabel", wraplength=300).grid(row=4, column=0, sticky="ew", pady=(3, 12))

        tree_frame = ttk.Frame(sidebar, style="Surface.TFrame")
        tree_frame.grid(row=5, column=0, sticky="nsew")
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        self.file_tree = ttk.Treeview(tree_frame, columns=("status", "file"), show="headings", selectmode="browse")
        self.file_tree.heading("status", text="Status")
        self.file_tree.heading("file", text="File")
        self.file_tree.column("status", width=82, minwidth=70, stretch=False)
        self.file_tree.column("file", width=220, minwidth=160, stretch=True)
        self.file_tree.grid(row=0, column=0, sticky="nsew")
        self.file_tree.bind("<<TreeviewSelect>>", self._on_tree_selection)
        self.file_tree.tag_configure("pending", foreground="#475569")
        self.file_tree.tag_configure("done", foreground="#15803d")
        self.file_tree.tag_configure("skipped", foreground="#a16207")
        self.file_tree.tag_configure("error", foreground="#b91c1c")

        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.file_tree.yview)
        tree_scroll.grid(row=0, column=1, sticky="ns")
        self.file_tree.configure(yscrollcommand=tree_scroll.set)

        ttk.Label(sidebar, text="Results", style="Heading.TLabel").grid(row=6, column=0, sticky="ew", pady=(14, 0))
        ttk.Label(sidebar, textvariable=self.results_var, style="Muted.TLabel", wraplength=300).grid(row=7, column=0, sticky="ew", pady=(3, 0))

        main = ttk.Frame(self.root, style="App.TFrame", padding=(8, 14, 8, 14))
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_rowconfigure(1, weight=1)
        main.grid_columnconfigure(0, weight=1)

        ttk.Label(main, textvariable=self.current_file_var, style="Current.TLabel").grid(row=0, column=0, sticky="ew", pady=(0, 7))

        preview_shell = tk.Frame(main, bg="#101820", bd=0, highlightthickness=0)
        preview_shell.grid(row=1, column=0, sticky="nsew")
        preview_shell.grid_rowconfigure(0, weight=1)
        preview_shell.grid_columnconfigure(0, weight=1)

        self.preview_label = tk.Label(preview_shell, bg="#101820", fg="#dbe4ee", text="", anchor="center")
        self.preview_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.preview_label.bind("<Configure>", lambda _: self._schedule_render())

        ttk.Label(main, textvariable=self.status_var, style="Status.TLabel").grid(row=2, column=0, sticky="ew", pady=(8, 0))

        controls = ttk.Frame(self.root, style="Surface.TFrame", padding=(16, 16, 16, 16))
        controls.grid(row=0, column=2, sticky="nsew", padx=(8, 14), pady=14)
        controls.grid_columnconfigure(0, weight=1)

        ttk.Label(controls, text="Thresholds", style="Title.TLabel").grid(row=0, column=0, sticky="ew")
        ttk.Label(
            controls,
            text="Tune the selected picture, then process it. Values carry forward to the next pending file.",
            style="Muted.TLabel",
            wraplength=320,
        ).grid(row=1, column=0, sticky="ew", pady=(4, 12))

        slider_container = ttk.Frame(controls, style="Surface.TFrame")
        slider_container.grid(row=2, column=0, sticky="ew")
        slider_container.grid_columnconfigure(0, weight=1)
        self._build_sliders(slider_container)

        actions = ttk.Frame(controls, style="Surface.TFrame")
        actions.grid(row=3, column=0, sticky="ew", pady=(18, 0))
        actions.grid_columnconfigure(0, weight=1)

        process_button = ttk.Button(actions, text="Process Selected", style="Primary.TButton", command=self.process_selected)
        next_button = ttk.Button(actions, text="Next Pending", style="App.TButton", command=self.select_next_pending)
        skip_button = ttk.Button(actions, text="Skip Selected", style="App.TButton", command=self.skip_selected)
        stop_button = ttk.Button(actions, text="Save And Stop", style="App.TButton", command=self.stop_and_save)

        process_button.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        next_button.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        skip_button.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        stop_button.grid(row=3, column=0, sticky="ew")
        self.action_buttons = [process_button, next_button, skip_button, stop_button]
        self._update_action_states()

    def _build_sliders(self, parent: ttk.Frame) -> None:
        for row, (key, label, minimum, maximum) in enumerate(SLIDER_CONFIG):
            value_var = tk.StringVar(value="")
            slider_var = tk.DoubleVar(value=self.initial_values.get(key, minimum))
            self.slider_vars[key] = slider_var
            self.value_vars[key] = value_var

            slider_row = ttk.Frame(parent, style="Surface.TFrame")
            slider_row.grid(row=row, column=0, sticky="ew", pady=(0, 10))
            slider_row.grid_columnconfigure(0, weight=1)

            label_row = ttk.Frame(slider_row, style="Surface.TFrame")
            label_row.grid(row=0, column=0, sticky="ew")
            label_row.grid_columnconfigure(0, weight=1)
            ttk.Label(label_row, text=label, style="Muted.TLabel").grid(row=0, column=0, sticky="w")
            ttk.Label(label_row, textvariable=value_var, style="Muted.TLabel").grid(row=0, column=1, sticky="e")

            scale = ttk.Scale(
                slider_row,
                from_=minimum,
                to=maximum,
                orient="horizontal",
                variable=slider_var,
                command=self._on_slider_changed,
            )
            scale.grid(row=1, column=0, sticky="ew", pady=(3, 0))
            self.slider_widgets.append(scale)

    def choose_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select Folder With Leaf Pictures")
        if folder:
            self.load_folder(Path(folder))

    def load_folder(self, input_folder: Path) -> None:
        input_folder = input_folder.resolve()
        if not input_folder.exists() or not input_folder.is_dir():
            messagebox.showerror(APP_TITLE, f"Folder does not exist:\n{input_folder}")
            return

        if not list_images(input_folder):
            messagebox.showerror(APP_TITLE, f"No supported images were found in:\n{input_folder}")
            return

        self._save_session()
        self.output_root.mkdir(parents=True, exist_ok=True)

        resume = find_resume_session(input_folder, self.output_root)
        if resume:
            session_path, session_data = resume
            session_data = normalize_session_data(session_data, session_path, input_folder, self.initial_values)
            status_text = f"Resumed {session_path.parent.name}"
        else:
            session_path, session_data = create_session(input_folder, self.output_root, self.initial_values)
            status_text = f"Created {session_path.parent.name}"

        self.session_path = session_path
        self.session_data = session_data
        self.loaded_image_path = None
        self.loaded_image = None
        self.loaded_hsv_image = None
        write_session_data(session_path, session_data)

        self._refresh_context_labels()
        self._populate_file_tree()
        self._apply_values_to_sliders(session_data.get("current_values", self.initial_values))

        start_path = self._starting_selection()
        if start_path:
            self.select_path(start_path)
        else:
            self._show_empty_preview("No Images", ["The selected folder does not contain supported image files."])

        self.status_var.set(status_text)
        self._update_action_states()

    def _starting_selection(self) -> Optional[str]:
        if not self.session_data:
            return None

        paths = {item["path"] for item in self.session_data.get("files", [])}
        selected_path = self.session_data.get("selected_path")
        if selected_path in paths:
            return selected_path

        for status in ("pending", "error", "done", "skipped"):
            for item in self.session_data.get("files", []):
                if item.get("status", "pending") == status:
                    return item["path"]
        return None

    def _populate_file_tree(self) -> None:
        for item_id in self.file_tree.get_children():
            self.file_tree.delete(item_id)

        self.tree_item_by_path.clear()
        self.path_by_tree_item.clear()
        if not self.session_data:
            return

        for index, item in enumerate(self.session_data.get("files", []), start=1):
            status = item.get("status", "pending")
            tree_id = f"file-{index}"
            self.tree_item_by_path[item["path"]] = tree_id
            self.path_by_tree_item[tree_id] = item["path"]
            self.file_tree.insert(
                "",
                "end",
                iid=tree_id,
                values=(status_label(status), item.get("name", Path(item["path"]).name)),
                tags=(status,),
            )

    def _update_file_row(self, item: dict) -> None:
        tree_id = self.tree_item_by_path.get(item["path"])
        if not tree_id:
            return

        status = item.get("status", "pending")
        self.file_tree.item(
            tree_id,
            values=(status_label(status), item.get("name", Path(item["path"]).name)),
            tags=(status,),
        )

    def _on_tree_selection(self, _: object = None) -> None:
        selection = self.file_tree.selection()
        if not selection:
            return

        path = self.path_by_tree_item.get(selection[0])
        if not path:
            return

        self.selected_path = path
        if self.session_data is not None:
            self.session_data["selected_path"] = path

        item = self._file_item(path)
        if not item:
            return

        values = item.get("last_values") or self.session_data.get("current_values", self.initial_values)
        self._apply_values_to_sliders(values)
        self.current_file_var.set(item.get("name", Path(path).name))

        status = item.get("status", "pending")
        if status == "error" and item.get("error"):
            self.status_var.set(f"{status_label(status)}: {item['error']}")
        else:
            self.status_var.set(f"{status_label(status)} file selected")

        self._schedule_preview()
        self._save_session()

    def _file_item(self, path: str) -> Optional[dict]:
        if not self.session_data:
            return None

        for item in self.session_data.get("files", []):
            if item.get("path") == path:
                return item
        return None

    def select_path(self, path: str) -> None:
        tree_id = self.tree_item_by_path.get(path)
        if not tree_id:
            return

        self.file_tree.selection_set(tree_id)
        self.file_tree.focus(tree_id)
        self.file_tree.see(tree_id)
        self._on_tree_selection()

    def select_next_pending(self) -> bool:
        if not self.session_data:
            return False

        files = self.session_data.get("files", [])
        if not files:
            return False

        current_index = -1
        for index, item in enumerate(files):
            if item.get("path") == self.selected_path:
                current_index = index
                break

        for offset in range(1, len(files) + 1):
            item = files[(current_index + offset) % len(files)]
            if item.get("status", "pending") in {"pending", "error"}:
                self.select_path(item["path"])
                return True

        self.status_var.set("All files are done or skipped.")
        return False

    def _on_slider_changed(self, _: object = None) -> None:
        self._refresh_slider_labels()
        if not self.loading_values:
            self._schedule_preview()

    def _read_slider_values(self) -> dict:
        values = {}
        for key, _, minimum, maximum in SLIDER_CONFIG:
            value = int(round(self.slider_vars[key].get()))
            values[key] = clamp_int(value, minimum, maximum)
        return values

    def _apply_values_to_sliders(self, values: dict) -> None:
        self.loading_values = True
        try:
            for key, _, minimum, maximum in SLIDER_CONFIG:
                value = clamp_int(values.get(key, self.initial_values.get(key, minimum)), minimum, maximum)
                self.slider_vars[key].set(value)
        finally:
            self.loading_values = False

        self._refresh_slider_labels()
        self._schedule_preview()

    def _refresh_slider_labels(self) -> None:
        for key, _, minimum, maximum in SLIDER_CONFIG:
            value = clamp_int(int(round(self.slider_vars[key].get())), minimum, maximum)
            self.value_vars[key].set(str(value))

    def _schedule_preview(self) -> None:
        if self.preview_after_id:
            self.root.after_cancel(self.preview_after_id)
        self.preview_after_id = self.root.after(int(PREVIEW_DEBOUNCE_SECONDS * 1000), self._update_preview)

    def _update_preview(self) -> None:
        self.preview_after_id = None
        if not self.session_data or not self.selected_path:
            return

        image_path = Path(self.selected_path)
        values = self._read_slider_values()

        try:
            self._load_image_data(image_path)
            self.preview_image = build_calibration_preview(
                image=self.loaded_image,
                hsv_image=self.loaded_hsv_image,
                image_name=image_path.name,
                lower_green=np.array([values["lh"], values["ls"], values["lv"]]),
                upper_green=np.array([values["uh"], values["us"], values["uv"]]),
                epsilon_factor=values["epsilon"] / 10000.0,
                spot_threshold=values["spot_threshold"],
            )
        except Exception as error:
            self.preview_image = make_message_preview(
                image_path.name,
                ["Preview could not be built.", str(error)],
            )

        self._schedule_render()

    def _load_image_data(self, image_path: Path) -> None:
        image_key = str(image_path.resolve())
        if self.loaded_image_path == image_key and self.loaded_image is not None and self.loaded_hsv_image is not None:
            return

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        self.loaded_image_path = image_key
        self.loaded_image = image
        self.loaded_hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def _show_empty_preview(self, title: str, lines: Sequence[str]) -> None:
        self.preview_image = make_message_preview(title, lines)
        self._schedule_render()

    def _schedule_render(self) -> None:
        if self.render_after_id:
            self.root.after_cancel(self.render_after_id)
        self.render_after_id = self.root.after(20, self._render_preview)

    def _render_preview(self) -> None:
        self.render_after_id = None
        if self.preview_image is None:
            return

        width = max(self.preview_label.winfo_width() - 12, 320)
        height = max(self.preview_label.winfo_height() - 12, 240)
        display_image = resize_to_fit(self.preview_image, width, height)
        self.tk_preview = cv_image_to_photo(display_image)
        self.preview_label.configure(image=self.tk_preview, text="")

    def process_selected(self) -> None:
        if self.busy or not self.session_data or not self.selected_path:
            return

        item = self._file_item(self.selected_path)
        if not item:
            return

        if item.get("status") == "done":
            should_reprocess = messagebox.askyesno(
                APP_TITLE,
                "This file is already marked done.\n\nReprocess it and append new CSV rows?",
            )
            if not should_reprocess:
                return

        values = self._read_slider_values()
        image_path = Path(item["path"])
        results_folder = Path(self.session_data["results_folder"])
        csv_file_path = Path(self.session_data["csv_file_path"])

        self._set_busy(True)
        self.status_var.set(f"Processing {item.get('name', image_path.name)}...")
        self.root.update_idletasks()

        try:
            lower_green = np.array([values["lh"], values["ls"], values["lv"]])
            upper_green = np.array([values["uh"], values["us"], values["uv"]])
            epsilon_factor = values["epsilon"] / 10000.0

            processed_files = splitting_images(
                image_path=image_path,
                results_folder=results_folder,
                lower_green=lower_green,
                upper_green=upper_green,
                epsilon_factor=epsilon_factor,
            )
            for processed_file, leaf_image in processed_files:
                picture_processing_from_image(
                    leaf_image=leaf_image,
                    image_name=processed_file.name,
                    threshold=values["spot_threshold"],
                    results_folder=results_folder,
                    csv_file_path=csv_file_path,
                    plotting=True,
                    slider_values=values,
                )

            item["status"] = "done"
            item["last_values"] = values.copy()
            item["processed_at"] = now_iso()
            item["error"] = ""
            self.session_data["current_values"] = values.copy()
            self._update_file_row(item)
            self._refresh_context_labels()
            self._save_session()

            selected_next = self.select_next_pending()
            if selected_next:
                self.status_var.set(f"Processed {item.get('name', image_path.name)}. Next pending file selected.")
            else:
                self.status_var.set(f"Processed {item.get('name', image_path.name)}. All files are done or skipped.")
        except Exception as error:
            item["status"] = "error"
            item["last_values"] = values.copy()
            item["processed_at"] = now_iso()
            item["error"] = str(error)
            append_to_csv(csv_file_path, error_csv_row(image_path.name, error, values))
            self._update_file_row(item)
            self._refresh_context_labels()
            self._save_session()
            self.status_var.set(f"Error while processing {image_path.name}: {error}")
            messagebox.showerror(APP_TITLE, f"Could not process {image_path.name}:\n\n{error}")
        finally:
            self._set_busy(False)

    def skip_selected(self) -> None:
        if self.busy or not self.session_data or not self.selected_path:
            return

        item = self._file_item(self.selected_path)
        if not item:
            return

        item["status"] = "skipped"
        item["last_values"] = self._read_slider_values()
        item["processed_at"] = now_iso()
        item["error"] = ""
        self._update_file_row(item)
        self._refresh_context_labels()
        self._save_session()

        selected_next = self.select_next_pending()
        if selected_next:
            self.status_var.set(f"Skipped {item.get('name', Path(item['path']).name)}. Next pending file selected.")
        else:
            self.status_var.set(f"Skipped {item.get('name', Path(item['path']).name)}. All files are done or skipped.")

    def stop_and_save(self) -> None:
        self._save_session()
        self.root.destroy()

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        self._update_action_states()

    def _update_action_states(self) -> None:
        has_session = bool(self.session_data and self.session_data.get("files"))
        action_state = "normal" if has_session and not self.busy else "disabled"
        stop_state = "normal" if has_session else "disabled"

        for button in self.action_buttons:
            button.configure(state=stop_state if button.cget("text") == "Save And Stop" else action_state)

        for scale in self.slider_widgets:
            if has_session and not self.busy:
                scale.state(["!disabled"])
            else:
                scale.state(["disabled"])

    def _refresh_context_labels(self) -> None:
        if not self.session_data:
            self.folder_var.set("No folder selected")
            self.results_var.set("")
            self.summary_var.set(session_summary(None))
            return

        self.folder_var.set(self.session_data.get("input_folder", ""))
        self.results_var.set(self.session_data.get("results_folder", ""))
        self.summary_var.set(session_summary(self.session_data))

    def _save_session(self) -> None:
        if not self.session_path or not self.session_data:
            return

        self.session_data["current_values"] = self._read_slider_values()
        if self.selected_path:
            self.session_data["selected_path"] = self.selected_path
        write_session_data(self.session_path, self.session_data)


class OpenCvProcessingApp:
    WIDTH = 1600
    HEIGHT = 920
    SIDEBAR_WIDTH = 340
    CONTROLS_WIDTH = 360
    HEADER_HEIGHT = 64
    STATUS_HEIGHT = 42
    ROW_HEIGHT = 28
    WINDOW_NAME = APP_TITLE

    def __init__(self, input_folder: Optional[Path], output_root: Path, initial_values: dict) -> None:
        self.input_folder = input_folder.resolve() if input_folder else None
        self.output_root = output_root
        self.initial_values = initial_values.copy()
        self.values = initial_values.copy()
        self.session_path: Optional[Path] = None
        self.session_data: Optional[dict] = None
        self.selected_index = -1
        self.loaded_image_path: Optional[str] = None
        self.loaded_image: Optional[np.ndarray] = None
        self.loaded_hsv_image: Optional[np.ndarray] = None
        self.preview_image: Optional[np.ndarray] = None
        self.preview_dirty = True
        self.last_change_time = 0.0
        self.running = True
        self.busy = False
        self.dragging_slider: Optional[str] = None
        self.file_scroll = 0
        self.file_row_hitboxes: List[Tuple[Tuple[int, int, int, int], int]] = []
        self.slider_hitboxes: Dict[str, Tuple[int, int, int, int, int, int]] = {}
        self.button_hitboxes: Dict[str, Tuple[int, int, int, int]] = {}
        self.status_text = "Ready"

    def run(self) -> None:
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, self.WIDTH, self.HEIGHT)
        cv2.setMouseCallback(self.WINDOW_NAME, self._on_mouse)

        if self.input_folder:
            self.load_folder(self.input_folder)
        else:
            self.status_text = "Open a folder to begin. Click Open Folder or press O."

        try:
            while self.running:
                now = time.monotonic()
                if self.preview_dirty and (now - self.last_change_time) >= PREVIEW_DEBOUNCE_SECONDS:
                    self._update_preview()

                canvas = self._draw_canvas()
                cv2.imshow(self.WINDOW_NAME, canvas)
                key = cv2.waitKeyEx(30)
                if key != -1:
                    self._handle_key(key)
        finally:
            self._save_session()
            try:
                cv2.destroyWindow(self.WINDOW_NAME)
            except cv2.error:
                pass

    def load_folder(self, input_folder: Path) -> None:
        input_folder = input_folder.resolve()
        if not input_folder.exists() or not input_folder.is_dir():
            self.status_text = f"Folder does not exist: {input_folder}"
            return

        if not list_images(input_folder):
            self.status_text = f"No supported images found in: {input_folder}"
            return

        self._save_session()
        self.input_folder = input_folder
        self.output_root.mkdir(parents=True, exist_ok=True)

        resume = find_resume_session(input_folder, self.output_root)
        if resume:
            session_path, session_data = resume
            session_data = normalize_session_data(session_data, session_path, input_folder, self.initial_values)
            self.status_text = f"Resumed {session_path.parent.name}"
        else:
            session_path, session_data = create_session(input_folder, self.output_root, self.initial_values)
            self.status_text = f"Created {session_path.parent.name}"

        self.session_path = session_path
        self.session_data = session_data
        self.values = session_data.get("current_values", self.initial_values).copy()
        self.loaded_image_path = None
        self.loaded_image = None
        self.loaded_hsv_image = None
        write_session_data(session_path, session_data)

        start_index = self._starting_index()
        self._select_index(start_index if start_index is not None else 0)

    def _starting_index(self) -> Optional[int]:
        if not self.session_data:
            return None

        selected_path = self.session_data.get("selected_path")
        files = self.session_data.get("files", [])
        for index, item in enumerate(files):
            if item.get("path") == selected_path:
                return index

        for status in ("pending", "error", "done", "skipped"):
            for index, item in enumerate(files):
                if item.get("status", "pending") == status:
                    return index
        return None

    def _selected_item(self) -> Optional[dict]:
        if not self.session_data:
            return None
        files = self.session_data.get("files", [])
        if 0 <= self.selected_index < len(files):
            return files[self.selected_index]
        return None

    def _select_index(self, index: int) -> None:
        if not self.session_data or not self.session_data.get("files"):
            return

        files = self.session_data["files"]
        self.selected_index = clamp_int(index, 0, len(files) - 1)
        item = files[self.selected_index]
        self.values = (item.get("last_values") or self.session_data.get("current_values", self.initial_values)).copy()
        self.session_data["selected_path"] = item["path"]
        self.loaded_image_path = None
        self.preview_dirty = True
        self.last_change_time = 0.0
        self._keep_selection_visible()
        self._save_session()
        self.status_text = f"{status_label(item.get('status', 'pending'))}: {item.get('name', Path(item['path']).name)}"

    def _keep_selection_visible(self) -> None:
        visible_rows = self._visible_file_rows()
        if self.selected_index < self.file_scroll:
            self.file_scroll = self.selected_index
        elif self.selected_index >= self.file_scroll + visible_rows:
            self.file_scroll = max(0, self.selected_index - visible_rows + 1)

    def _visible_file_rows(self) -> int:
        return max(5, (self.HEIGHT - 290) // self.ROW_HEIGHT)

    def _update_preview(self) -> None:
        self.preview_dirty = False
        item = self._selected_item()
        if not item:
            self.preview_image = make_message_preview("Leaf Picture Processing", ["Open a folder to begin."])
            return

        image_path = Path(item["path"])
        try:
            self._load_image_data(image_path)
            self.preview_image = build_calibration_preview(
                image=self.loaded_image,
                hsv_image=self.loaded_hsv_image,
                image_name=image_path.name,
                lower_green=np.array([self.values["lh"], self.values["ls"], self.values["lv"]]),
                upper_green=np.array([self.values["uh"], self.values["us"], self.values["uv"]]),
                epsilon_factor=self.values["epsilon"] / 10000.0,
                spot_threshold=self.values["spot_threshold"],
            )
        except Exception as error:
            self.preview_image = make_message_preview(image_path.name, ["Preview could not be built.", str(error)])

    def _load_image_data(self, image_path: Path) -> None:
        image_key = str(image_path.resolve())
        if self.loaded_image_path == image_key and self.loaded_image is not None and self.loaded_hsv_image is not None:
            return

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        self.loaded_image_path = image_key
        self.loaded_image = image
        self.loaded_hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def process_selected(self) -> None:
        if self.busy or not self.session_data:
            return

        item = self._selected_item()
        if not item:
            return

        image_path = Path(item["path"])
        results_folder = Path(self.session_data["results_folder"])
        csv_file_path = Path(self.session_data["csv_file_path"])

        self.busy = True
        self.status_text = f"Processing {image_path.name}..."
        cv2.imshow(self.WINDOW_NAME, self._draw_canvas())
        cv2.waitKey(1)

        try:
            lower_green = np.array([self.values["lh"], self.values["ls"], self.values["lv"]])
            upper_green = np.array([self.values["uh"], self.values["us"], self.values["uv"]])
            epsilon_factor = self.values["epsilon"] / 10000.0

            processed_files = splitting_images(
                image_path=image_path,
                results_folder=results_folder,
                lower_green=lower_green,
                upper_green=upper_green,
                epsilon_factor=epsilon_factor,
            )
            for processed_file, leaf_image in processed_files:
                picture_processing_from_image(
                    leaf_image=leaf_image,
                    image_name=processed_file.name,
                    threshold=self.values["spot_threshold"],
                    results_folder=results_folder,
                    csv_file_path=csv_file_path,
                    plotting=True,
                    slider_values=self.values,
                )

            item["status"] = "done"
            item["last_values"] = self.values.copy()
            item["processed_at"] = now_iso()
            item["error"] = ""
            self.session_data["current_values"] = self.values.copy()
            self._save_session()
            if self.select_next_pending():
                self.status_text = f"Processed {image_path.name}. Next pending file selected."
            else:
                self.status_text = f"Processed {image_path.name}. All files are done or skipped."
        except Exception as error:
            item["status"] = "error"
            item["last_values"] = self.values.copy()
            item["processed_at"] = now_iso()
            item["error"] = str(error)
            append_to_csv(csv_file_path, error_csv_row(image_path.name, error, self.values))
            self._save_session()
            self.status_text = f"Error processing {image_path.name}: {error}"
        finally:
            self.busy = False

    def skip_selected(self) -> None:
        if self.busy or not self.session_data:
            return

        item = self._selected_item()
        if not item:
            return

        item["status"] = "skipped"
        item["last_values"] = self.values.copy()
        item["processed_at"] = now_iso()
        item["error"] = ""
        self._save_session()
        skipped_name = item.get("name", Path(item["path"]).name)
        if self.select_next_pending():
            self.status_text = f"Skipped {skipped_name}. Next pending file selected."
        else:
            self.status_text = f"Skipped {skipped_name}. All files are done or skipped."

    def select_next_pending(self) -> bool:
        if not self.session_data:
            return False

        files = self.session_data.get("files", [])
        if not files:
            return False

        for offset in range(1, len(files) + 1):
            index = (self.selected_index + offset) % len(files)
            if files[index].get("status", "pending") in {"pending", "error"}:
                self._select_index(index)
                return True

        return False

    def open_folder(self) -> None:
        try:
            folder = pick_input_folder()
        except ValueError as error:
            self.status_text = str(error)
            return
        self.load_folder(folder)

    def _handle_key(self, key: int) -> None:
        if key in (13, 10):
            self.process_selected()
        elif key in (27,):
            self.running = False
        elif key in (ord("s"), ord("S")):
            self.skip_selected()
        elif key in (ord("n"), ord("N")):
            if not self.select_next_pending():
                self.status_text = "All files are done or skipped."
        elif key in (ord("o"), ord("O")):
            self.open_folder()
        elif key in (2424832, 81):
            self._select_index(max(0, self.selected_index - 1))
        elif key in (2555904, 83):
            if self.session_data:
                self._select_index(min(len(self.session_data.get("files", [])) - 1, self.selected_index + 1))

    def _on_mouse(self, event: int, x: int, y: int, flags: int, _: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            for action, rect in self.button_hitboxes.items():
                if self._point_in_rect(x, y, rect):
                    self._run_button_action(action)
                    return

            for key, hitbox in self.slider_hitboxes.items():
                x1, y1, x2, y2, _, _ = hitbox
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.dragging_slider = key
                    self._set_slider_from_x(key, x)
                    return

            for rect, index in self.file_row_hitboxes:
                if self._point_in_rect(x, y, rect):
                    self._select_index(index)
                    return

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging_slider:
            self._set_slider_from_x(self.dragging_slider, x)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_slider = None
            self._save_session()
        elif event == cv2.EVENT_MOUSEWHEEL and self.session_data:
            direction = -1 if flags > 0 else 1
            max_scroll = max(0, len(self.session_data.get("files", [])) - self._visible_file_rows())
            self.file_scroll = clamp_int(self.file_scroll + direction * 3, 0, max_scroll)

    def _run_button_action(self, action: str) -> None:
        if action == "open":
            self.open_folder()
        elif action == "process":
            self.process_selected()
        elif action == "next":
            if not self.select_next_pending():
                self.status_text = "All files are done or skipped."
        elif action == "skip":
            self.skip_selected()
        elif action == "stop":
            self.running = False

    def _set_slider_from_x(self, key: str, x: int) -> None:
        hitbox = self.slider_hitboxes.get(key)
        if not hitbox:
            return

        x1, _, x2, _, minimum, maximum = hitbox
        ratio = (clamp_int(x, x1, x2) - x1) / max(1, x2 - x1)
        self.values[key] = clamp_int(round(minimum + ratio * (maximum - minimum)), minimum, maximum)
        self.preview_dirty = True
        self.last_change_time = time.monotonic()

    def _draw_canvas(self) -> np.ndarray:
        canvas = np.full((self.HEIGHT, self.WIDTH, 3), (237, 241, 245), dtype=np.uint8)
        self.file_row_hitboxes.clear()
        self.slider_hitboxes.clear()
        self.button_hitboxes.clear()

        self._draw_sidebar(canvas)
        self._draw_preview_area(canvas)
        self._draw_controls(canvas)
        self._draw_status_bar(canvas)
        return canvas

    def _draw_sidebar(self, canvas: np.ndarray) -> None:
        x1, y1, x2, y2 = 14, 14, self.SIDEBAR_WIDTH - 10, self.HEIGHT - 14
        self._fill_rect(canvas, (x1, y1, x2, y2), (255, 255, 255))
        self._draw_text(canvas, APP_TITLE, x1 + 18, y1 + 36, 0.72, (26, 32, 42), 2)

        summary = session_summary(self.session_data)
        self._draw_text(canvas, summary, x1 + 18, y1 + 66, 0.45, (92, 107, 120), 1, max_width=280)

        self._draw_button(canvas, "open", "Open Folder", (x1 + 18, y1 + 86, x2 - 18, y1 + 124), primary=True)

        folder_text = str(self.input_folder) if self.input_folder else "No folder selected"
        self._draw_text(canvas, "Input", x1 + 18, y1 + 154, 0.48, (26, 32, 42), 1)
        self._draw_text(canvas, folder_text, x1 + 18, y1 + 178, 0.43, (92, 107, 120), 1, max_width=282)

        list_top = y1 + 214
        list_bottom = self.HEIGHT - 118
        self._draw_text(canvas, "Files", x1 + 18, list_top - 14, 0.48, (26, 32, 42), 1)

        files = self.session_data.get("files", []) if self.session_data else []
        visible_rows = max(1, (list_bottom - list_top) // self.ROW_HEIGHT)
        max_scroll = max(0, len(files) - visible_rows)
        self.file_scroll = clamp_int(self.file_scroll, 0, max_scroll)

        status_colors = {
            "pending": (85, 96, 108),
            "done": (64, 130, 74),
            "skipped": (29, 125, 178),
            "error": (39, 39, 190),
        }
        for row, index in enumerate(range(self.file_scroll, min(len(files), self.file_scroll + visible_rows))):
            item = files[index]
            row_y = list_top + row * self.ROW_HEIGHT
            rect = (x1 + 12, row_y, x2 - 12, row_y + self.ROW_HEIGHT - 3)
            selected = index == self.selected_index
            self._fill_rect(canvas, rect, (224, 238, 231) if selected else (250, 252, 253))
            status = item.get("status", "pending")
            cv2.circle(canvas, (rect[0] + 13, row_y + 13), 5, status_colors.get(status, (85, 96, 108)), -1)
            self._draw_text(canvas, item.get("name", Path(item["path"]).name), rect[0] + 26, row_y + 18, 0.43, (31, 41, 55), 1, max_width=240)
            self.file_row_hitboxes.append((rect, index))

        if len(files) > visible_rows:
            scroll_note = f"{self.file_scroll + 1}-{min(len(files), self.file_scroll + visible_rows)} of {len(files)}"
            self._draw_text(canvas, scroll_note, x1 + 18, list_bottom + 22, 0.42, (92, 107, 120), 1)

        results = self.session_data.get("results_folder", "") if self.session_data else ""
        self._draw_text(canvas, "Results", x1 + 18, self.HEIGHT - 74, 0.48, (26, 32, 42), 1)
        self._draw_text(canvas, results, x1 + 18, self.HEIGHT - 50, 0.43, (92, 107, 120), 1, max_width=282)

    def _draw_preview_area(self, canvas: np.ndarray) -> None:
        x1 = self.SIDEBAR_WIDTH + 8
        y1 = 14
        x2 = self.WIDTH - self.CONTROLS_WIDTH - 8
        y2 = self.HEIGHT - 14
        current = self._selected_item()
        title = current.get("name", "No file selected") if current else "No file selected"
        self._draw_text(canvas, title, x1 + 10, y1 + 34, 0.62, (26, 32, 42), 2, max_width=x2 - x1 - 20)

        panel = (x1, y1 + self.HEADER_HEIGHT, x2, y2 - self.STATUS_HEIGHT)
        self._fill_rect(canvas, panel, (16, 24, 32))

        preview = self.preview_image
        if preview is None:
            preview = make_message_preview("Leaf Picture Processing", ["Open a folder to begin."])

        max_width = panel[2] - panel[0] - 20
        max_height = panel[3] - panel[1] - 20
        display = resize_to_fit(preview, max_width, max_height)
        top = panel[1] + (max_height - display.shape[0]) // 2 + 10
        left = panel[0] + (max_width - display.shape[1]) // 2 + 10
        canvas[top : top + display.shape[0], left : left + display.shape[1]] = display

    def _draw_controls(self, canvas: np.ndarray) -> None:
        x1 = self.WIDTH - self.CONTROLS_WIDTH + 10
        y1 = 14
        x2 = self.WIDTH - 14
        y2 = self.HEIGHT - 14
        self._fill_rect(canvas, (x1, y1, x2, y2), (255, 255, 255))
        self._draw_text(canvas, "Thresholds", x1 + 18, y1 + 36, 0.72, (26, 32, 42), 2)
        self._draw_text(canvas, "Tune values, then process the selected file.", x1 + 18, y1 + 66, 0.43, (92, 107, 120), 1, max_width=300)

        slider_y = y1 + 104
        for key, label, minimum, maximum in SLIDER_CONFIG:
            self._draw_text(canvas, label, x1 + 18, slider_y, 0.45, (65, 75, 88), 1)
            self._draw_text(canvas, str(self.values.get(key, minimum)), x2 - 58, slider_y, 0.45, (65, 75, 88), 1)

            line_x1 = x1 + 18
            line_x2 = x2 - 24
            line_y = slider_y + 22
            cv2.line(canvas, (line_x1, line_y), (line_x2, line_y), (203, 213, 225), 4, cv2.LINE_AA)
            value = clamp_int(self.values.get(key, minimum), minimum, maximum)
            ratio = (value - minimum) / max(1, maximum - minimum)
            knob_x = int(line_x1 + ratio * (line_x2 - line_x1))
            cv2.line(canvas, (line_x1, line_y), (knob_x, line_y), (52, 128, 89), 4, cv2.LINE_AA)
            cv2.circle(canvas, (knob_x, line_y), 8, (52, 128, 89), -1, cv2.LINE_AA)
            self.slider_hitboxes[key] = (line_x1, line_y - 14, line_x2, line_y + 14, minimum, maximum)
            slider_y += 70

        button_top = y2 - 198
        self._draw_button(canvas, "process", "Process Selected", (x1 + 18, button_top, x2 - 18, button_top + 42), primary=True)
        self._draw_button(canvas, "next", "Next Pending", (x1 + 18, button_top + 52, x2 - 18, button_top + 92), primary=False)
        self._draw_button(canvas, "skip", "Skip Selected", (x1 + 18, button_top + 102, x2 - 18, button_top + 142), primary=False)
        self._draw_button(canvas, "stop", "Save And Stop", (x1 + 18, button_top + 152, x2 - 18, button_top + 192), primary=False)

    def _draw_status_bar(self, canvas: np.ndarray) -> None:
        x1 = self.SIDEBAR_WIDTH + 8
        y1 = self.HEIGHT - 52
        x2 = self.WIDTH - self.CONTROLS_WIDTH - 8
        self._draw_text(canvas, self.status_text, x1 + 10, y1 + 26, 0.48, (83, 97, 112), 1, max_width=x2 - x1 - 20)
        hint = "Enter: process | S: skip | N: next | O: open folder | Esc: save and stop"
        self._draw_text(canvas, hint, x1 + 10, y1 + 48, 0.42, (83, 97, 112), 1, max_width=x2 - x1 - 20)

    def _draw_button(self, canvas: np.ndarray, action: str, label: str, rect: Tuple[int, int, int, int], primary: bool) -> None:
        color = (52, 128, 89) if primary else (241, 245, 249)
        text_color = (255, 255, 255) if primary else (31, 41, 55)
        border = (52, 128, 89) if primary else (203, 213, 225)
        self._fill_rect(canvas, rect, color)
        cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), border, 1, cv2.LINE_AA)
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_x = rect[0] + (rect[2] - rect[0] - text_size[0]) // 2
        text_y = rect[1] + (rect[3] - rect[1] + text_size[1]) // 2
        self._draw_text(canvas, label, text_x, text_y, 0.5, text_color, 1)
        self.button_hitboxes[action] = rect

    @staticmethod
    def _fill_rect(canvas: np.ndarray, rect: Tuple[int, int, int, int], color: Tuple[int, int, int]) -> None:
        x1, y1, x2, y2 = rect
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness=cv2.FILLED)

    @staticmethod
    def _point_in_rect(x: int, y: int, rect: Tuple[int, int, int, int]) -> bool:
        return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]

    @staticmethod
    def _draw_text(
        canvas: np.ndarray,
        text: str,
        x: int,
        y: int,
        scale: float,
        color: Tuple[int, int, int],
        thickness: int,
        max_width: Optional[int] = None,
    ) -> None:
        safe_text = str(text)
        if max_width:
            while len(safe_text) > 4:
                text_size, _ = cv2.getTextSize(safe_text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
                if text_size[0] <= max_width:
                    break
                safe_text = safe_text[:-4] + "..."
        cv2.putText(canvas, safe_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    def _save_session(self) -> None:
        if not self.session_path or not self.session_data:
            return

        self.session_data["current_values"] = self.values.copy()
        item = self._selected_item()
        if item:
            self.session_data["selected_path"] = item["path"]
        write_session_data(self.session_path, self.session_data)


def run_opencv_application(input_folder: Optional[Path], output_root: Path, initial_values: dict) -> None:
    app = OpenCvProcessingApp(input_folder=input_folder, output_root=output_root, initial_values=initial_values)
    app.run()


def run_application(input_folder: Optional[Path], output_root: Path, initial_values: dict) -> None:
    if tk is None:
        print("Tkinter is not available in this Python environment.")
        print("Using the single-window OpenCV application instead.")
        run_opencv_application(input_folder=input_folder, output_root=output_root, initial_values=initial_values)
        return

    root = tk.Tk()
    LeafProcessingApp(root, input_folder=input_folder, output_root=output_root, initial_values=initial_values)
    root.mainloop()


def process_folder(input_folder: Path, output_root: Path, initial_values: dict, plotting: bool) -> None:
    files = list_images(input_folder)
    if not files:
        raise FileNotFoundError(f"No supported images were found in: {input_folder}")

    output_root.mkdir(parents=True, exist_ok=True)
    resume = find_resume_session(input_folder, output_root)
    if resume:
        session_path, session_data = resume
        session_data = normalize_session_data(session_data, session_path, input_folder, initial_values)
        print(f"Resuming session: {session_path.parent.name}")
    else:
        session_path, session_data = create_session(input_folder, output_root, initial_values)
        print(f"Created session: {session_path.parent.name}")

    results_folder = Path(session_data["results_folder"])
    csv_file_path = Path(session_data["csv_file_path"])
    print(f"Input folder:  {input_folder}")
    print(f"Results folder: {results_folder}")
    print(f"CSV file:       {csv_file_path}")

    current_values = session_data.get("current_values", initial_values).copy()

    for item in session_data.get("files", []):
        file_path = Path(item["path"])
        if item.get("status") in {"done", "skipped"}:
            print(f"Already {item['status']}: {file_path.name}")
            continue

        print(f"Processing: {file_path.name}")
        current_values, action = calibrate_image(file_path, current_values)
        item["last_values"] = current_values.copy()
        item["processed_at"] = now_iso()

        if action == "skip":
            print("  Skipped by user.")
            item["status"] = "skipped"
            item["error"] = ""
            session_data["current_values"] = current_values.copy()
            write_session_data(session_path, session_data)
            continue
        if action == "stop":
            print("Processing stopped by user.")
            session_data["current_values"] = current_values.copy()
            write_session_data(session_path, session_data)
            break

        lower_green = np.array([current_values["lh"], current_values["ls"], current_values["lv"]])
        upper_green = np.array([current_values["uh"], current_values["us"], current_values["uv"]])
        epsilon_factor = current_values["epsilon"] / 10000.0

        try:
            processed_files = splitting_images(
                image_path=file_path,
                results_folder=results_folder,
                lower_green=lower_green,
                upper_green=upper_green,
                epsilon_factor=epsilon_factor,
            )
            for processed_file, leaf_image in processed_files:
                picture_processing_from_image(
                    leaf_image=leaf_image,
                    image_name=processed_file.name,
                    threshold=current_values["spot_threshold"],
                    results_folder=results_folder,
                    csv_file_path=csv_file_path,
                    plotting=plotting,
                    slider_values=current_values,
                )
            item["status"] = "done"
            item["error"] = ""
            session_data["current_values"] = current_values.copy()
            write_session_data(session_path, session_data)
            print("##########################################")
        except Exception as error:
            print("  ERROR:", error)
            item["status"] = "error"
            item["error"] = str(error)
            session_data["current_values"] = current_values.copy()
            append_to_csv(csv_file_path, error_csv_row(file_path.name, error, current_values))
            write_session_data(session_path, session_data)

    print("Processing done.")


def main() -> None:
    args = parse_args()

    output_root = Path(args.output_root).resolve()
    initial_values = {
        "lh": clamp_int(args.lh, 0, 179),
        "ls": clamp_int(args.ls, 0, 255),
        "lv": clamp_int(args.lv, 0, 255),
        "uh": clamp_int(args.uh, 0, 179),
        "us": clamp_int(args.us, 0, 255),
        "uv": clamp_int(args.uv, 0, 255),
        "epsilon": clamp_int(args.epsilon, 0, 100),
        "spot_threshold": clamp_int(args.spot_threshold, 0, 255),
    }

    if args.legacy_opencv:
        if args.input_folder:
            input_folder = Path(args.input_folder).resolve()
        else:
            input_folder = pick_input_folder()

        process_folder(
            input_folder=input_folder,
            output_root=output_root,
            initial_values=initial_values,
            plotting=True,
        )
        return

    input_folder = Path(args.input_folder).resolve() if args.input_folder else None
    run_application(
        input_folder=input_folder,
        output_root=output_root,
        initial_values=initial_values,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}")
