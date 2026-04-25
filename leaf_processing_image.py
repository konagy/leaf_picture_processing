from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from leaf_processing_config import (
    BOTTOM_PANEL_HEIGHT,
    BOTTOM_PANEL_WIDTH,
    PREVIEW_BACKGROUND_COLOR,
    PREVIEW_DEBOUNCE_SECONDS,
    PREVIEW_ZOOM_FACTOR,
    PREVIEW_ZOOM_MAX,
    PREVIEW_ZOOM_MIN,
    SHARP_PREVIEW_MAX_HEIGHT,
    SHARP_PREVIEW_MAX_WIDTH,
    SHARP_PREVIEW_PANEL_HEIGHT,
    SHARP_PREVIEW_PANEL_WIDTH,
    TOP_PANEL_HEIGHT,
    TOP_PANEL_WIDTH,
    TRACKBAR_NAMES,
)
from leaf_processing_core import append_to_csv, clamp_int, measurement_csv_row

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


def preview_fit_scale(image: np.ndarray, viewport_width: int, viewport_height: int) -> float:
    height, width = image.shape[:2]
    return min(viewport_width / max(width, 1), viewport_height / max(height, 1), 1.0)


def preview_scaled_size(image: np.ndarray, viewport_width: int, viewport_height: int, zoom: float) -> Tuple[int, int]:
    height, width = image.shape[:2]
    scale = preview_fit_scale(image, viewport_width, viewport_height) * zoom
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def clamp_preview_pan(
    pan_x: int,
    pan_y: int,
    scaled_width: int,
    scaled_height: int,
    viewport_width: int,
    viewport_height: int,
) -> Tuple[int, int]:
    max_pan_x = max(0, scaled_width - viewport_width)
    max_pan_y = max(0, scaled_height - viewport_height)
    return clamp_int(pan_x, 0, max_pan_x), clamp_int(pan_y, 0, max_pan_y)


def render_preview_view(
    image: np.ndarray,
    viewport_width: int,
    viewport_height: int,
    zoom: float,
    pan_x: int,
    pan_y: int,
) -> Tuple[np.ndarray, int, int]:
    viewport_width = max(1, viewport_width)
    viewport_height = max(1, viewport_height)
    zoom = max(PREVIEW_ZOOM_MIN, min(PREVIEW_ZOOM_MAX, zoom))
    scaled_width, scaled_height = preview_scaled_size(image, viewport_width, viewport_height, zoom)
    pan_x, pan_y = clamp_preview_pan(pan_x, pan_y, scaled_width, scaled_height, viewport_width, viewport_height)

    interpolation = cv2.INTER_AREA if scaled_width < image.shape[1] or scaled_height < image.shape[0] else cv2.INTER_LINEAR
    scaled = cv2.resize(image, (scaled_width, scaled_height), interpolation=interpolation)
    view = np.full((viewport_height, viewport_width, 3), PREVIEW_BACKGROUND_COLOR, dtype=np.uint8)

    if scaled_width <= viewport_width:
        src_x = 0
        dst_x = (viewport_width - scaled_width) // 2
        copy_width = scaled_width
    else:
        src_x = pan_x
        dst_x = 0
        copy_width = viewport_width

    if scaled_height <= viewport_height:
        src_y = 0
        dst_y = (viewport_height - scaled_height) // 2
        copy_height = scaled_height
    else:
        src_y = pan_y
        dst_y = 0
        copy_height = viewport_height

    view[dst_y : dst_y + copy_height, dst_x : dst_x + copy_width] = scaled[
        src_y : src_y + copy_height,
        src_x : src_x + copy_width,
    ]
    return view, pan_x, pan_y


def zoom_preview_at(
    image: np.ndarray,
    viewport_width: int,
    viewport_height: int,
    zoom: float,
    pan_x: int,
    pan_y: int,
    cursor_x: int,
    cursor_y: int,
    direction: int,
) -> Tuple[float, int, int]:
    if direction == 0:
        return zoom, pan_x, pan_y

    old_zoom = max(PREVIEW_ZOOM_MIN, min(PREVIEW_ZOOM_MAX, zoom))
    factor = PREVIEW_ZOOM_FACTOR if direction > 0 else 1.0 / PREVIEW_ZOOM_FACTOR
    new_zoom = max(PREVIEW_ZOOM_MIN, min(PREVIEW_ZOOM_MAX, old_zoom * factor))
    if abs(new_zoom - old_zoom) < 0.001:
        return old_zoom, pan_x, pan_y

    old_width, old_height = preview_scaled_size(image, viewport_width, viewport_height, old_zoom)
    pan_x, pan_y = clamp_preview_pan(pan_x, pan_y, old_width, old_height, viewport_width, viewport_height)
    old_left = (viewport_width - old_width) // 2 if old_width <= viewport_width else 0
    old_top = (viewport_height - old_height) // 2 if old_height <= viewport_height else 0
    image_x = clamp_int(pan_x + cursor_x - old_left, 0, old_width)
    image_y = clamp_int(pan_y + cursor_y - old_top, 0, old_height)
    ratio_x = image_x / max(old_width, 1)
    ratio_y = image_y / max(old_height, 1)

    new_width, new_height = preview_scaled_size(image, viewport_width, viewport_height, new_zoom)
    new_pan_x = int(round(ratio_x * new_width - cursor_x))
    new_pan_y = int(round(ratio_y * new_height - cursor_y))
    new_pan_x, new_pan_y = clamp_preview_pan(
        new_pan_x,
        new_pan_y,
        new_width,
        new_height,
        viewport_width,
        viewport_height,
    )
    return new_zoom, new_pan_x, new_pan_y


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


def pad_to_width(image: np.ndarray, width: int) -> np.ndarray:
    if image.shape[1] == width:
        return image
    pad_total = width - image.shape[1]
    return cv2.copyMakeBorder(
        image,
        0,
        0,
        0,
        pad_total,
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
    top_panel_width: int = TOP_PANEL_WIDTH,
    top_panel_height: int = TOP_PANEL_HEIGHT,
    bottom_panel_width: int = BOTTOM_PANEL_WIDTH,
    bottom_panel_height: int = BOTTOM_PANEL_HEIGHT,
    max_preview_width: Optional[int] = 1650,
    max_preview_height: Optional[int] = 950,
) -> np.ndarray:
    _, _, approx_contours, leaf_images = segment_two_leaves(image, hsv_image, lower_green, upper_green, epsilon_factor)

    picSize = 0.6
    original_panel = resize_to_fit(image, top_panel_width, top_panel_height)

    contour_overlay = image.copy()
    if approx_contours:
        cv2.drawContours(contour_overlay, approx_contours, -1, (0, 255, 255), 5)
        draw_leaf_labels(contour_overlay, approx_contours)
    contour_panel = resize_to_fit(contour_overlay, top_panel_width, top_panel_height)
    cv2.putText(contour_panel, "Original + Leaf Contours", (20, 35), cv2.FONT_ITALIC, picSize*1.0, (50, 144, 66), 2)

    combined_panels: List[np.ndarray] = []
    spot_panels: List[np.ndarray] = []
    for index, leaf_image in enumerate(leaf_images, start=1):
        spots_image, all_area, spot_area, percentage, hole_areas = detect_spots(leaf_image, spot_threshold)
        combined = cv2.addWeighted(place_on_black_background(leaf_image), 0.75, spots_image, 0.95, 0.0)
        combined_panel = resize_to_fit(combined, bottom_panel_width, bottom_panel_height)
        cv2.putText(combined_panel, f"Leaf {index}", (20, 35), cv2.FONT_ITALIC, picSize*1.0, (0, 0, 0), 2)
        cv2.putText(combined_panel, f"Leaf area: {all_area}", (20, 70), cv2.FONT_ITALIC, picSize*0.8, (0, 0, 0), 2)
        cv2.putText(combined_panel, f"Spot area: {spot_area}", (20, 105), cv2.FONT_ITALIC, picSize*0.8, (0, 0, 0), 2)
        cv2.putText(combined_panel, f"Spot %: {percentage:.2f}", (20, 140), cv2.FONT_ITALIC, picSize*0.8, (0, 0, 0), 2)
        cv2.putText(combined_panel, f"Holes: {len(hole_areas)}", (20, 175), cv2.FONT_ITALIC, picSize*0.8, (0, 0, 0), 2)
        combined_panels.append(combined_panel)

        spot_panel = resize_to_fit(place_on_gray_background(spots_image), bottom_panel_width, bottom_panel_height)
        cv2.putText(spot_panel, f"Leaf {index} Spots", (20, 35), cv2.FONT_ITALIC, picSize*1.0, (50, 144, 66), 2)
        cv2.putText(spot_panel, f"Threshold: {spot_threshold}", (20, 70), cv2.FONT_ITALIC, picSize*0.8, (50, 144, 66), 2)
        spot_panels.append(spot_panel)

    top_row_height = max(original_panel.shape[0], contour_panel.shape[0])
    top_row = cv2.hconcat(
        [
            pad_to_height(original_panel, top_row_height),
            pad_to_height(contour_panel, top_row_height),
        ]
    )

    leaf_rows: List[np.ndarray] = []
    if not combined_panels:
        empty = np.zeros((240, 700, 3), dtype=np.uint8)
        cv2.putText(empty, "Could not isolate two leaves with current HSV values.", (20, 120), cv2.FONT_ITALIC, picSize*0.9, (0, 0, 255), 2)
        leaf_rows.append(empty)
    else:
        for combined_panel, spot_panel in zip(combined_panels, spot_panels):
            row_height = max(combined_panel.shape[0], spot_panel.shape[0])
            leaf_rows.append(
                cv2.hconcat(
                    [
                        pad_to_height(combined_panel, row_height),
                        pad_to_height(spot_panel, row_height),
                    ]
                )
            )

    preview_rows = [top_row, *leaf_rows]
    canvas_width = max(row.shape[1] for row in preview_rows)
    preview_rows = [pad_to_width(row, canvas_width) for row in preview_rows]

    preview = cv2.vconcat(preview_rows)
    if max_preview_width is not None and max_preview_height is not None:
        preview = resize_to_fit(preview, max_preview_width, max_preview_height)
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


def build_sharp_calibration_preview(
    image: np.ndarray,
    hsv_image: np.ndarray,
    image_name: str,
    lower_green: np.ndarray,
    upper_green: np.ndarray,
    epsilon_factor: float,
    spot_threshold: int,
) -> np.ndarray:
    return build_calibration_preview(
        image=image,
        hsv_image=hsv_image,
        image_name=image_name,
        lower_green=lower_green,
        upper_green=upper_green,
        epsilon_factor=epsilon_factor,
        spot_threshold=spot_threshold,
        top_panel_width=SHARP_PREVIEW_PANEL_WIDTH,
        top_panel_height=SHARP_PREVIEW_PANEL_HEIGHT,
        bottom_panel_width=SHARP_PREVIEW_PANEL_WIDTH,
        bottom_panel_height=SHARP_PREVIEW_PANEL_HEIGHT,
        max_preview_width=SHARP_PREVIEW_MAX_WIDTH,
        max_preview_height=SHARP_PREVIEW_MAX_HEIGHT,
    )


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

def make_message_preview(title: str, lines: Sequence[str]) -> np.ndarray:
    canvas = np.full((720, 1100, 3), (34, 42, 51), dtype=np.uint8)
    cv2.putText(canvas, title, (44, 84), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (236, 240, 241), 2, cv2.LINE_AA)
    y = 142
    for line in lines:
        cv2.putText(canvas, line[:92], (48, y), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (190, 202, 214), 1, cv2.LINE_AA)
        y += 38
    return canvas
