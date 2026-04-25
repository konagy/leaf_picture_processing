from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from leaf_processing_core import (
    append_to_csv,
    clamp_int,
    create_session,
    error_csv_row,
    find_resume_session,
    list_images,
    normalize_session_data,
    now_iso,
    pick_input_folder,
    write_session_data,
)
from leaf_processing_cv_app import run_opencv_application
from leaf_processing_image import calibrate_image, picture_processing_from_image, splitting_images
from leaf_processing_tk_app import LeafProcessingApp, tk

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
