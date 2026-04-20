import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class Settings:
    processing_max_side: int = 1800
    background_border_width: int = 40
    min_delta_e: float = 8.0
    min_hole_area: int = 40
    edge_smooth_window: int = 81
    band_padding: int = 25
    support_fraction: float = 0.35
    save_outputs: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Alternative leaf processor for pictures with two elongated leaves on a bright "
            "background. It estimates the intact leaf from smoothed edge envelopes."
        )
    )
    parser.add_argument("input_folder", help="Folder with input images.")
    parser.add_argument("--output-root", default=".", help="Root directory for the timestamped results folder.")
    parser.add_argument("--max-side", type=int, default=1800, help="Maximum side length used during processing.")
    parser.add_argument("--min-hole-area", type=int, default=40, help="Ignore missing regions smaller than this.")
    parser.add_argument("--no-outputs", action="store_true", help="Disable preview image export.")
    return parser.parse_args()


def current_timestamp() -> str:
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


def create_results_paths(output_root: Path) -> Tuple[Path, Path]:
    stamp = current_timestamp()
    results_dir = output_root / f"results-envelope-{stamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"results-envelope-{stamp}.csv"
    return results_dir, csv_path


def initialize_csv(csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Image name",
                "Source image",
                "Leaf index",
                "Area of leaf [pixels]",
                "Area of holes [pixels]",
                "Percentage of holes [%]",
                "Number of holes [pcs]",
                "Hole areas [pixels]",
                "Status",
            ]
        )


def append_csv_row(csv_path: Path, row: Sequence[object]) -> None:
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(row)


def list_images(folder: Path) -> List[Path]:
    return sorted([path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS])


def resize_for_processing(image: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    largest_side = max(height, width)
    if largest_side <= max_side:
        return image.copy(), 1.0

    scale = max_side / largest_side
    resized = cv2.resize(
        image,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def border_sample_mask(shape: Tuple[int, int], border_width: int) -> np.ndarray:
    height, width = shape
    border = max(5, min(border_width, min(height, width) // 4))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[:border, :] = 255
    mask[-border:, :] = 255
    mask[:, :border] = 255
    mask[:, -border:] = 255
    return mask


def estimate_background_lab(image_bgr: np.ndarray, settings: Settings) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    border_mask = border_sample_mask(image_bgr.shape[:2], settings.background_border_width)
    border_pixels = lab[border_mask > 0]
    if border_pixels.size == 0:
        return np.array([255.0, 128.0, 128.0], dtype=np.float32)

    lightness = border_pixels[:, 0]
    keep = border_pixels[lightness >= np.percentile(lightness, 60)]
    if keep.size == 0:
        keep = border_pixels
    return np.median(keep, axis=0)


def detect_tissue_mask(image_bgr: np.ndarray, settings: Settings) -> np.ndarray:
    blurred = cv2.GaussianBlur(image_bgr, (5, 5), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    background_lab = estimate_background_lab(blurred, settings)
    delta = lab - background_lab.reshape(1, 1, 3)
    delta_e = np.sqrt(np.sum(delta * delta, axis=2))

    blue = blurred[:, :, 0].astype(np.float32)
    green = blurred[:, :, 1].astype(np.float32)
    red = blurred[:, :, 2].astype(np.float32)
    total = blue + green + red + 1.0
    green_ratio = green / total
    red_ratio = red / total
    blue_ratio = blue / total
    exg = (2 * green) - red - blue

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    hue = hsv[:, :, 0]

    tissue = (
        (delta_e >= settings.min_delta_e)
        & (hue >= 20)
        & (hue <= 110)
        & (saturation >= 12)
        & (value >= 30)
        & (value <= 248)
        & (green >= red + 6)
        & (green >= blue - 2)
        & (
            ((green_ratio >= 0.34) & ((green_ratio - red_ratio) >= 0.02))
            | (exg >= 10)
        )
    ).astype(np.uint8) * 255

    tissue = cv2.morphologyEx(
        tissue,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    tissue = cv2.morphologyEx(
        tissue,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    return tissue


def scale_contour(contour: np.ndarray, inv_scale: float) -> np.ndarray:
    scaled = contour.astype(np.float32) * inv_scale
    return np.round(scaled).astype(np.int32)


def contour_center(contour: np.ndarray) -> Tuple[float, float]:
    x, y, w, h = cv2.boundingRect(contour)
    return x + (w / 2.0), y + (h / 2.0)


def contiguous_segments(active: np.ndarray) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    start = None
    for index, is_active in enumerate(active):
        if is_active and start is None:
            start = index
        elif not is_active and start is not None:
            segments.append((start, index))
            start = None
    if start is not None:
        segments.append((start, len(active)))
    return segments


def offset_contour(contour: np.ndarray, x_offset: int = 0, y_offset: int = 0) -> np.ndarray:
    shifted = contour.copy()
    shifted[:, 0, 0] += x_offset
    shifted[:, 0, 1] += y_offset
    return shifted


def extract_band_contours(mask: np.ndarray, axis: str, padding: int) -> List[np.ndarray]:
    if axis == "rows":
        counts = np.count_nonzero(mask, axis=1)
        threshold = max(40, int(mask.shape[1] * 0.08))
    else:
        counts = np.count_nonzero(mask, axis=0)
        threshold = max(40, int(mask.shape[0] * 0.08))

    segments = contiguous_segments(counts >= threshold)
    segments = sorted(segments, key=lambda segment: segment[1] - segment[0], reverse=True)[:2]
    contours_out: List[np.ndarray] = []

    if len(segments) < 2:
        return contours_out

    for start, end in sorted(segments, key=lambda segment: segment[0]):
        if axis == "rows":
            band_top = max(0, start - padding)
            band_bottom = min(mask.shape[0], end + padding)
            band = mask[band_top:band_bottom, :]
            contours, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            largest = max(contours, key=cv2.contourArea)
            contours_out.append(offset_contour(largest, y_offset=band_top))
        else:
            band_left = max(0, start - padding)
            band_right = min(mask.shape[1], end + padding)
            band = mask[:, band_left:band_right]
            contours, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            largest = max(contours, key=cv2.contourArea)
            contours_out.append(offset_contour(largest, x_offset=band_left))

    return contours_out


def find_leaf_contours(image_bgr: np.ndarray, settings: Settings) -> Tuple[List[np.ndarray], np.ndarray]:
    processing_image, scale = resize_for_processing(image_bgr, settings.processing_max_side)
    tissue_small = detect_tissue_mask(processing_image, settings)
    bridged = cv2.morphologyEx(
        tissue_small,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (41, 9)),
    )
    bridged_transpose = cv2.morphologyEx(
        tissue_small,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 41)),
    )

    candidates = [
        extract_band_contours(bridged, "rows", settings.band_padding),
        extract_band_contours(bridged_transpose, "cols", settings.band_padding),
    ]

    def candidate_score(contour_list: List[np.ndarray]) -> float:
        if len(contour_list) != 2:
            return -1.0
        areas = [cv2.contourArea(contour) for contour in contour_list]
        if min(areas) <= 0:
            return -1.0
        balance = min(areas) / max(areas)
        return float(sum(areas) * (0.25 + 0.75 * balance))

    filtered = max(candidates, key=candidate_score)
    if len(filtered) < 2:
        raise ValueError("Could not isolate two leaf contours.")

    inv_scale = 1.0 / scale
    contours_full = [scale_contour(contour, inv_scale) for contour in filtered]
    first = contour_center(contours_full[0])
    second = contour_center(contours_full[1])
    if abs(first[0] - second[0]) >= abs(first[1] - second[1]):
        contours_full = sorted(contours_full, key=lambda contour: contour_center(contour)[0])
    else:
        contours_full = sorted(contours_full, key=lambda contour: contour_center(contour)[1])

    full_tissue = detect_tissue_mask(image_bgr, settings)
    return contours_full, full_tissue


def crop_to_mask(image: np.ndarray, mask: np.ndarray, padding: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Empty mask after cropping.")

    x1 = max(int(xs.min()) - padding, 0)
    y1 = max(int(ys.min()) - padding, 0)
    x2 = min(int(xs.max()) + padding + 1, image.shape[1])
    y2 = min(int(ys.max()) + padding + 1, image.shape[0])
    return image[y1:y2, x1:x2].copy(), mask[y1:y2, x1:x2].copy()


def estimate_leaf_angle(mask: np.ndarray) -> float:
    points_y, points_x = np.where(mask > 0)
    if len(points_x) < 2:
        return 0.0
    points = np.column_stack((points_x.astype(np.float32), points_y.astype(np.float32)))
    _, eigenvectors = cv2.PCACompute(points, mean=None)
    axis = eigenvectors[0]
    return math.degrees(math.atan2(float(axis[1]), float(axis[0])))


def rotate_image_and_mask(image: np.ndarray, mask: np.ndarray, angle_degrees: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    height, width = mask.shape
    center = (width / 2.0, height / 2.0)
    rotation = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    cos_v = abs(rotation[0, 0])
    sin_v = abs(rotation[0, 1])
    new_width = int((height * sin_v) + (width * cos_v))
    new_height = int((height * cos_v) + (width * sin_v))
    rotation[0, 2] += (new_width / 2.0) - center[0]
    rotation[1, 2] += (new_height / 2.0) - center[1]

    rotated_image = cv2.warpAffine(
        image,
        rotation,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    rotated_mask = cv2.warpAffine(
        mask,
        rotation,
        (new_width, new_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rotated_image, rotated_mask, rotation, (width, height)


def unrotate_mask(mask: np.ndarray, rotation_matrix: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
    inverse = cv2.invertAffineTransform(rotation_matrix)
    original_width, original_height = original_size
    return cv2.warpAffine(
        mask,
        inverse,
        (original_width, original_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def interpolate_missing(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    xs = np.arange(len(values), dtype=np.float32)
    if not np.any(valid):
        return values.astype(np.float32)
    known_x = xs[valid]
    known_y = values[valid].astype(np.float32)
    filled = np.interp(xs, known_x, known_y)
    return filled.astype(np.float32)


def smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    window = max(5, window)
    if window % 2 == 0:
        window += 1
    array = values.reshape(1, -1).astype(np.float32)
    return cv2.GaussianBlur(array, (window, 1), 0).reshape(-1)


def reconstruct_leaf_from_envelope(actual_mask: np.ndarray, settings: Settings) -> np.ndarray:
    angle = estimate_leaf_angle(actual_mask)
    _, rotated_mask, rotation_matrix, original_size = rotate_image_and_mask(
        np.dstack([actual_mask, actual_mask, actual_mask]),
        actual_mask,
        -angle,
    )

    ys, xs = np.where(rotated_mask > 0)
    if len(xs) == 0:
        return actual_mask.copy()

    observed_widths_all = []
    for x in range(rotated_mask.shape[1]):
        column_ys = np.flatnonzero(rotated_mask[:, x] > 0)
        if column_ys.size:
            observed_widths_all.append(float(column_ys.max() - column_ys.min() + 1))

    median_leaf_width = float(np.median(observed_widths_all)) if observed_widths_all else 0.0
    repair_kernel_height = max(7, int(round(median_leaf_width * 0.22)))
    if repair_kernel_height % 2 == 0:
        repair_kernel_height += 1

    # Repair bites that touch the outer edge of the leaf without globally pushing
    # the boundary outward. This is much less aggressive than the previous max-envelope.
    repaired_rotated_mask = cv2.morphologyEx(
        rotated_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, repair_kernel_height)),
    )

    x_start = int(xs.min())
    x_end = int(xs.max())
    width = repaired_rotated_mask.shape[1]
    height = repaired_rotated_mask.shape[0]

    top = np.full(width, np.nan, dtype=np.float32)
    bottom = np.full(width, np.nan, dtype=np.float32)
    observed_width = np.zeros(width, dtype=np.float32)

    for x in range(x_start, x_end + 1):
        column_ys = np.flatnonzero(repaired_rotated_mask[:, x] > 0)
        if column_ys.size:
            top[x] = float(column_ys.min())
            bottom[x] = float(column_ys.max())
            observed_width[x] = float(column_ys.max() - column_ys.min() + 1)

    valid = ~np.isnan(top)
    top_filled = interpolate_missing(np.nan_to_num(top, nan=0.0), valid)
    bottom_filled = interpolate_missing(np.nan_to_num(bottom, nan=0.0), valid)

    top_smooth = smooth_1d(top_filled, settings.edge_smooth_window)
    bottom_smooth = smooth_1d(bottom_filled, settings.edge_smooth_window)

    width_support = observed_width[valid]
    reference_width = float(np.median(width_support)) if width_support.size else 0.0
    min_supported_width = max(6.0, reference_width * settings.support_fraction)
    support_mask = observed_width >= min_supported_width
    support_segments = contiguous_segments(support_mask)
    if support_segments:
        longest_segment = max(support_segments, key=lambda segment: segment[1] - segment[0])
        fill_start, fill_end = longest_segment
    else:
        fill_start, fill_end = x_start, x_end + 1

    reconstructed_rotated = np.zeros_like(rotated_mask)
    for x in range(fill_start, fill_end):
        y1 = int(np.clip(round(top_smooth[x]), 0, height - 1))
        y2 = int(np.clip(round(bottom_smooth[x]), 0, height - 1))
        if y2 >= y1:
            reconstructed_rotated[y1 : y2 + 1, x] = 255

    reconstructed_rotated = cv2.bitwise_or(reconstructed_rotated, repaired_rotated_mask)
    reconstructed_rotated[:, :fill_start] = repaired_rotated_mask[:, :fill_start]
    reconstructed_rotated[:, fill_end:] = repaired_rotated_mask[:, fill_end:]
    reconstructed = unrotate_mask(reconstructed_rotated, rotation_matrix, original_size)
    reconstructed = cv2.bitwise_or(reconstructed, actual_mask)
    return reconstructed


def measure_missing_regions(reconstructed_mask: np.ndarray, actual_mask: np.ndarray, min_hole_area: int) -> Tuple[np.ndarray, List[int]]:
    missing = cv2.subtract(reconstructed_mask, actual_mask)
    output = np.zeros_like(missing)
    areas: List[int] = []

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(missing, connectivity=8)
    for component_index in range(1, component_count):
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        if area < min_hole_area:
            continue
        component_mask = np.where(labels == component_index, 255, 0).astype(np.uint8)
        output = cv2.bitwise_or(output, component_mask)
        areas.append(area)

    areas.sort(reverse=True)
    return output, areas


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if component_count <= 1:
        return mask

    largest_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest_index, 255, 0).astype(np.uint8)


def create_preview(leaf_image: np.ndarray, actual_mask: np.ndarray, reconstructed_mask: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
    preview = leaf_image.copy()
    observed_contours, _ = cv2.findContours(actual_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    reconstructed_contours, _ = cv2.findContours(reconstructed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    missing_contours, _ = cv2.findContours(missing_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    missing_overlay = np.zeros_like(preview)
    missing_overlay[:, :, 2] = missing_mask
    preview = cv2.addWeighted(preview, 1.0, missing_overlay, 0.35, 0.0)
    cv2.drawContours(preview, observed_contours, -1, (0, 255, 255), 2)
    cv2.drawContours(preview, reconstructed_contours, -1, (255, 200, 0), 2)
    cv2.drawContours(preview, missing_contours, -1, (0, 0, 255), 2)
    return preview


def process_leaf(
    image: np.ndarray,
    full_tissue_mask: np.ndarray,
    contour: np.ndarray,
    image_stem: str,
    image_name: str,
    leaf_index: int,
    results_dir: Path,
    settings: Settings,
) -> List[object]:
    contour_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
    actual_mask_full = cv2.bitwise_and(full_tissue_mask, contour_mask)

    leaf_image, actual_mask = crop_to_mask(image, actual_mask_full)
    actual_mask = (actual_mask > 0).astype(np.uint8) * 255
    actual_mask = keep_largest_component(actual_mask)

    reconstructed = reconstruct_leaf_from_envelope(actual_mask, settings)
    missing_mask, missing_areas = measure_missing_regions(reconstructed, actual_mask, settings.min_hole_area)

    leaf_area = int(cv2.countNonZero(reconstructed))
    hole_area = int(cv2.countNonZero(missing_mask))
    hole_percentage = (hole_area / leaf_area * 100.0) if leaf_area else 0.0

    leaf_name = f"{image_stem}_leaf{leaf_index}"
    if settings.save_outputs:
        preview = create_preview(leaf_image, actual_mask, reconstructed, missing_mask)
        cv2.imwrite(str(results_dir / f"{leaf_name}_observed_mask.png"), actual_mask)
        cv2.imwrite(str(results_dir / f"{leaf_name}_reconstructed_mask.png"), reconstructed)
        cv2.imwrite(str(results_dir / f"{leaf_name}_missing.png"), missing_mask)
        cv2.imwrite(str(results_dir / f"{leaf_name}_preview.png"), preview)

    return [
        leaf_name,
        image_name,
        leaf_index,
        leaf_area,
        hole_area,
        round(hole_percentage, 4),
        len(missing_areas),
        "; ".join(str(value) for value in missing_areas),
        "OK",
    ]


def process_folder(input_folder: Path, output_root: Path, settings: Settings) -> None:
    images = list_images(input_folder)
    if not images:
        raise FileNotFoundError(f"No supported images found in {input_folder}")

    results_dir, csv_path = create_results_paths(output_root)
    initialize_csv(csv_path)

    print(f"Input folder:  {input_folder}")
    print(f"Results folder: {results_dir}")
    print(f"CSV file:       {csv_path}")

    for image_path in images:
        print(f"Processing: {image_path.name}")
        image = cv2.imread(str(image_path))
        if image is None:
            append_csv_row(csv_path, [image_path.name, image_path.name, "", "", "", "", "", "", "ERROR: could not load image"])
            print("  ERROR: Could not load image.")
            continue

        try:
            contours, full_tissue_mask = find_leaf_contours(image, settings)
            if settings.save_outputs:
                cv2.imwrite(str(results_dir / f"{image_path.stem}_tissue_mask.png"), full_tissue_mask)

            for leaf_index, contour in enumerate(contours, start=1):
                row = process_leaf(
                    image=image,
                    full_tissue_mask=full_tissue_mask,
                    contour=contour,
                    image_stem=image_path.stem,
                    image_name=image_path.name,
                    leaf_index=leaf_index,
                    results_dir=results_dir,
                    settings=settings,
                )
                append_csv_row(csv_path, row)
                print(
                    f"  {row[0]}: leaf area={row[3]}, holes={row[4]}, "
                    f"percentage={row[5]}%, count={row[6]}"
                )
        except Exception as error:
            append_csv_row(csv_path, [image_path.name, image_path.name, "", "", "", "", "", "", f"ERROR: {error}"])
            print(f"  ERROR: {error}")

    print("Processing done.")


def main() -> None:
    args = parse_args()
    settings = Settings(
        processing_max_side=args.max_side,
        min_hole_area=args.min_hole_area,
        save_outputs=not args.no_outputs,
    )

    process_folder(
        input_folder=Path(args.input_folder).resolve(),
        output_root=Path(args.output_root).resolve(),
        settings=settings,
    )


if __name__ == "__main__":
    main()
