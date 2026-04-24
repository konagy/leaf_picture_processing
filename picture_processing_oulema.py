import argparse
import csv
import subprocess
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
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


def initialize_csv(file_path: Path) -> None:
    with file_path.open(mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Image name",
                "Area of leaf [pixels]",
                "Area of spots [pixels]",
                "Percentage of spots [%]",
                "Threshold",
            ]
        )


def append_to_csv(file_path: Path, data: Sequence[object]) -> None:
    with file_path.open(mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(data)


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


def pick_input_folder() -> Path:
    print("Opening folder picker...")
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory(title="Select Folder With Leaf Pictures")
        root.destroy()
    except ModuleNotFoundError:
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
    lower_green: np.ndarray,
    upper_green: np.ndarray,
    epsilon_factor: float,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
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
    return np.where(image == 0, np.zeros_like(image), image)


def detect_spots(leaf_image: np.ndarray, threshold: int) -> Tuple[np.ndarray, int, int, float]:
    gray_image_cv = cv2.cvtColor(leaf_image, cv2.COLOR_BGR2GRAY)
    _, white_areas_cv = cv2.threshold(gray_image_cv, threshold, 250, cv2.THRESH_BINARY)
    output_image_cv = cv2.bitwise_and(leaf_image, leaf_image, mask=white_areas_cv)

    spot_pixels = np.any(output_image_cv != [0, 0, 0], axis=-1)
    spot_area = int(np.count_nonzero(spot_pixels))
    leaf_pixels = np.any(leaf_image != [0, 0, 0], axis=-1)
    all_area = int(np.count_nonzero(leaf_pixels))
    percentage = (spot_area / all_area) * 100.0 if all_area else 0.0
    return output_image_cv, all_area, spot_area, percentage


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
            cv2.FONT_HERSHEY_SIMPLEX,
            3.0,
            (0, 0, 0),
            12,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label,
            (center_x - 140, center_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            3.0,
            (0, 255, 255),
            6,
            cv2.LINE_AA,
        )


def build_calibration_preview(
    image: np.ndarray,
    image_name: str,
    lower_green: np.ndarray,
    upper_green: np.ndarray,
    epsilon_factor: float,
    spot_threshold: int,
) -> np.ndarray:
    mask, _, approx_contours, leaf_images = segment_two_leaves(image, lower_green, upper_green, epsilon_factor)

    original_panel = resize_to_fit(image, 700, 420)
    mask_panel = cv2.cvtColor(resize_to_fit(mask, 700, 420), cv2.COLOR_GRAY2BGR)
    cv2.putText(mask_panel, "Leaf Mask", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    contour_overlay = image.copy()
    if approx_contours:
        cv2.drawContours(contour_overlay, approx_contours, -1, (0, 255, 255), 5)
        draw_leaf_labels(contour_overlay, approx_contours)
    contour_panel = resize_to_fit(contour_overlay, 700, 420)
    cv2.putText(contour_panel, "Original + Leaf Contours", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    leaf_panels: List[np.ndarray] = []
    for index, leaf_image in enumerate(leaf_images, start=1):
        spots_image, all_area, spot_area, percentage = detect_spots(leaf_image, spot_threshold)
        combined = cv2.addWeighted(place_on_black_background(leaf_image), 0.75, spots_image, 0.95, 0.0)
        panel = resize_to_fit(combined, 700, 320)
        cv2.putText(panel, f"Leaf {index}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(panel, f"Leaf area: {all_area}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(panel, f"Spot area: {spot_area}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(panel, f"Spot %: {percentage:.2f}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        leaf_panels.append(panel)

    if not leaf_panels:
        empty = np.zeros((240, 700, 3), dtype=np.uint8)
        cv2.putText(empty, "Could not isolate two leaves with current HSV values.", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        leaf_panels = [empty]

    top_row_height = max(original_panel.shape[0], contour_panel.shape[0], mask_panel.shape[0])
    top_row = cv2.hconcat(
        [
            pad_to_height(original_panel, top_row_height),
            pad_to_height(contour_panel, top_row_height),
            pad_to_height(mask_panel, top_row_height),
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
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        preview,
        "Enter/q: accept | s: skip image | Esc: stop session",
        (20, preview.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
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
    cv2.createTrackbar(TRACKBAR_NAMES["epsilon"], "Controls", initial_values["epsilon"], 1000, nothing)
    cv2.createTrackbar(TRACKBAR_NAMES["spot_threshold"], "Controls", initial_values["spot_threshold"], 255, nothing)

    current_values = initial_values.copy()
    previous_values = None
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

        if current_values != previous_values or preview is None:
            try:
                preview = build_calibration_preview(
                    image=image,
                    image_name=image_path.name,
                    lower_green=np.array([current_values["lh"], current_values["ls"], current_values["lv"]]),
                    upper_green=np.array([current_values["uh"], current_values["us"], current_values["uv"]]),
                    epsilon_factor=current_values["epsilon"] / 10000.0,
                    spot_threshold=current_values["spot_threshold"],
                )
            except Exception as error:
                preview = resize_to_fit(image, 1650, 950)
                cv2.putText(preview, f"File: {image_path.name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(preview, str(error), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            previous_values = current_values.copy()

        cv2.imshow("Threshold Setup", preview)
        key = cv2.waitKey(15) & 0xFF
        if key in (13, ord("q")):
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
) -> List[Path]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    _, _, approx_contours, leaf_images = segment_two_leaves(image, lower_green, upper_green, epsilon_factor)
    if len(approx_contours) < 2 or len(leaf_images) < 2:
        raise ValueError("Could not find two leaves with the selected HSV thresholds.")

    processed_files: List[Path] = []
    for index, leaf_image in enumerate(leaf_images, start=1):
        output_path = results_folder / f"{image_path.stem}_leaf{index}.jpg"
        cv2.imwrite(str(output_path), place_on_black_background(leaf_image))
        processed_files.append(output_path)

    return processed_files


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

    filename_id = image_path.stem
    output_image_cv, all_area, spot_area, spot_percentage = detect_spots(image, threshold)

    save_processing_preview(
        original=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        processed=cv2.cvtColor(output_image_cv, cv2.COLOR_BGR2RGB),
        output_path=results_folder / f"{filename_id}_processing.jpg",
    )
    cv2.imwrite(str(results_folder / f"{filename_id}-green_object.jpg"), output_image_cv)

    append_to_csv(csv_file_path, (image_path.name, all_area, spot_area, round(spot_percentage, 4), threshold))

    if plotting:
        print(f"  {image_path.name}: leaf area={all_area}, spot area={spot_area}, percentage={spot_percentage:.4f}, threshold={threshold}")


def process_folder(input_folder: Path, output_root: Path, initial_values: dict, plotting: bool) -> None:
    files = list_images(input_folder)
    if not files:
        raise FileNotFoundError(f"No supported images were found in: {input_folder}")

    results_folder, csv_file_path = create_results_paths(output_root)

    print(f"Input folder:  {input_folder}")
    print(f"Results folder: {results_folder}")
    print(f"CSV file:       {csv_file_path}")

    current_values = initial_values.copy()

    for file_path in files:
        print(f"Processing: {file_path.name}")
        current_values, action = calibrate_image(file_path, current_values)
        if action == "skip":
            print("  Skipped by user.")
            continue
        if action == "stop":
            print("Processing stopped by user.")
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
            for processed_file in processed_files:
                picture_processing(
                    image_path=processed_file,
                    threshold=current_values["spot_threshold"],
                    results_folder=results_folder,
                    csv_file_path=csv_file_path,
                    plotting=plotting,
                )
            print("##########################################")
        except Exception as error:
            print("  ERROR:", error)
            append_to_csv(csv_file_path, (file_path.name, "ERROR", str(error), "", ""))

    print("Processing done.")


def main() -> None:
    args = parse_args()

    if args.input_folder:
        input_folder = Path(args.input_folder).resolve()
    else:
        input_folder = pick_input_folder()

    output_root = Path(args.output_root).resolve()
    initial_values = {
        "lh": clamp_int(args.lh, 0, 179),
        "ls": clamp_int(args.ls, 0, 255),
        "lv": clamp_int(args.lv, 0, 255),
        "uh": clamp_int(args.uh, 0, 179),
        "us": clamp_int(args.us, 0, 255),
        "uv": clamp_int(args.uv, 0, 255),
        "epsilon": clamp_int(args.epsilon, 0, 1000),
        "spot_threshold": clamp_int(args.spot_threshold, 0, 255),
    }

    process_folder(
        input_folder=input_folder,
        output_root=output_root,
        initial_values=initial_values,
        plotting=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}")
