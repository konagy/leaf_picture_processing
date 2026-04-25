from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

try:
    import tkinter as tk
    from tkinter import filedialog
except ModuleNotFoundError:
    tk = None
    filedialog = None

from leaf_processing_config import (
    CSV_BASE_COLUMNS,
    CSV_DEPRECATED_COLUMNS,
    CSV_SLIDER_COLUMNS,
    SESSION_FILE_NAME,
    SESSION_VERSION,
    STATUS_LABELS,
    SUPPORTED_EXTENSIONS,
)

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

def clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))
