from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from leaf_processing_config import (
    APP_TITLE,
    PREVIEW_DEBOUNCE_SECONDS,
    PREVIEW_ZOOM_MIN,
    SLIDER_CONFIG,
)
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
    session_summary,
    status_label,
    write_session_data,
)
from leaf_processing_image import (
    build_sharp_calibration_preview,
    clamp_preview_pan,
    make_message_preview,
    picture_processing_from_image,
    preview_scaled_size,
    render_preview_view,
    splitting_images,
    zoom_preview_at,
)

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
        self.preview_zoom = PREVIEW_ZOOM_MIN
        self.preview_pan_x = 0
        self.preview_pan_y = 0
        self.preview_viewport_rect: Optional[Tuple[int, int, int, int]] = None
        self.preview_drag_start: Optional[Tuple[int, int]] = None
        self.preview_drag_start_pan: Tuple[int, int] = (0, 0)
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

        # testing todo
        p = Path(input_folder)
        print("INPUT:", repr(input_folder))
        print("PATH:", p)
        print("RESOLVED:", p.resolve())
        print("CWD:", Path.cwd())
        print("EXISTS:", p.exists())
        print("IS_DIR:", p.is_dir())
        #
        
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
        self._reset_preview_zoom()
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
            self.preview_image = build_sharp_calibration_preview(
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
        elif key in (ord("0"),):
            self._reset_preview_zoom()
            self.status_text = "Preview zoom reset."
        elif key in (2424832, 81):
            self._select_index(max(0, self.selected_index - 1))
        elif key in (2555904, 83):
            if self.session_data:
                self._select_index(min(len(self.session_data.get("files", [])) - 1, self.selected_index + 1))

    def _on_mouse(self, event: int, x: int, y: int, flags: int, _: object) -> None:
        if event == cv2.EVENT_LBUTTONDBLCLK and self._point_in_preview(x, y):
            self._reset_preview_zoom()
            self.status_text = "Preview zoom reset."
            return

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

            if self._point_in_preview(x, y):
                self.preview_drag_start = (x, y)
                self.preview_drag_start_pan = (self.preview_pan_x, self.preview_pan_y)
                return

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging_slider:
            self._set_slider_from_x(self.dragging_slider, x)
        elif event == cv2.EVENT_MOUSEMOVE and self.preview_drag_start:
            self._drag_preview_to(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_slider = None
            self.preview_drag_start = None
            self._save_session()
        elif event == cv2.EVENT_MOUSEWHEEL and self.session_data:
            if self._point_in_preview(x, y):
                self._zoom_preview_at(x, y, 1 if flags > 0 else -1)
            else:
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

    def _reset_preview_zoom(self) -> None:
        self.preview_zoom = PREVIEW_ZOOM_MIN
        self.preview_pan_x = 0
        self.preview_pan_y = 0
        self.preview_drag_start = None

    def _point_in_preview(self, x: int, y: int) -> bool:
        return self.preview_viewport_rect is not None and self._point_in_rect(x, y, self.preview_viewport_rect)

    def _zoom_preview_at(self, x: int, y: int, direction: int) -> None:
        if self.preview_image is None or self.preview_viewport_rect is None:
            return

        left, top, right, bottom = self.preview_viewport_rect
        self.preview_zoom, self.preview_pan_x, self.preview_pan_y = zoom_preview_at(
            self.preview_image,
            max(1, right - left),
            max(1, bottom - top),
            self.preview_zoom,
            self.preview_pan_x,
            self.preview_pan_y,
            clamp_int(x - left, 0, max(1, right - left)),
            clamp_int(y - top, 0, max(1, bottom - top)),
            direction,
        )
        self.status_text = f"Preview zoom: {self.preview_zoom:.1f}x"

    def _drag_preview_to(self, x: int, y: int) -> None:
        if self.preview_image is None or self.preview_viewport_rect is None or self.preview_drag_start is None:
            return

        left, top, right, bottom = self.preview_viewport_rect
        start_x, start_y = self.preview_drag_start
        delta_x = x - start_x
        delta_y = y - start_y
        viewport_width = max(1, right - left)
        viewport_height = max(1, bottom - top)
        scaled_width, scaled_height = preview_scaled_size(
            self.preview_image,
            viewport_width,
            viewport_height,
            self.preview_zoom,
        )
        self.preview_pan_x, self.preview_pan_y = clamp_preview_pan(
            self.preview_drag_start_pan[0] - delta_x,
            self.preview_drag_start_pan[1] - delta_y,
            scaled_width,
            scaled_height,
            viewport_width,
            viewport_height,
        )

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

        self.preview_viewport_rect = (panel[0] + 10, panel[1] + 10, panel[2] - 10, panel[3] - 10)
        left, top, right, bottom = self.preview_viewport_rect
        display, self.preview_pan_x, self.preview_pan_y = render_preview_view(
            preview,
            max(1, right - left),
            max(1, bottom - top),
            self.preview_zoom,
            self.preview_pan_x,
            self.preview_pan_y,
        )
        canvas[top:bottom, left:right] = display

        if self.preview_zoom > PREVIEW_ZOOM_MIN + 0.01:
            label = f"{self.preview_zoom:.1f}x"
            self._fill_rect(canvas, (right - 78, top + 12, right - 18, top + 38), (31, 41, 55))
            self._draw_text(canvas, label, right - 67, top + 31, 0.48, (226, 232, 240), 1)

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
        hint = "Enter: process | Wheel: zoom | Drag: pan | Double-click/0: reset | Esc: save and stop"
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
