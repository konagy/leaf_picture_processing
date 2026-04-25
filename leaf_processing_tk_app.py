from __future__ import annotations

import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

from leaf_processing_config import (
    APP_TITLE,
    PREVIEW_DEBOUNCE_SECONDS,
    PREVIEW_ZOOM_MIN,
    SLIDER_CONFIG,
)
from leaf_processing_core import (
    append_to_csv,
    create_session,
    error_csv_row,
    find_resume_session,
    list_images,
    normalize_session_data,
    now_iso,
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

def cv_image_to_photo(image: np.ndarray) -> tk.PhotoImage:
    if len(image.shape) == 2:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width = rgb_image.shape[:2]
    ppm_bytes = f"P6\n{width} {height}\n255\n".encode("ascii") + rgb_image.tobytes()
    encoded = base64.b64encode(ppm_bytes).decode("ascii")
    return tk.PhotoImage(data=encoded, format="PPM")

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
        self.preview_zoom = PREVIEW_ZOOM_MIN
        self.preview_pan_x = 0
        self.preview_pan_y = 0
        self.preview_drag_start: Optional[Tuple[int, int]] = None
        self.preview_drag_start_pan: Tuple[int, int] = (0, 0)
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
        self.preview_label.bind("<MouseWheel>", self._on_preview_mouse_wheel)
        self.preview_label.bind("<Button-4>", self._on_preview_mouse_wheel)
        self.preview_label.bind("<Button-5>", self._on_preview_mouse_wheel)
        self.preview_label.bind("<ButtonPress-1>", self._on_preview_press)
        self.preview_label.bind("<B1-Motion>", self._on_preview_drag)
        self.preview_label.bind("<ButtonRelease-1>", self._on_preview_release)
        self.preview_label.bind("<Double-Button-1>", self._on_preview_double_click)

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

        self._reset_preview_zoom(schedule=False)
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
            self.preview_image = build_sharp_calibration_preview(
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
        display_image, self.preview_pan_x, self.preview_pan_y = render_preview_view(
            self.preview_image,
            width,
            height,
            self.preview_zoom,
            self.preview_pan_x,
            self.preview_pan_y,
        )
        self.tk_preview = cv_image_to_photo(display_image)
        self.preview_label.configure(image=self.tk_preview, text="")

    def _reset_preview_zoom(self, schedule: bool = True) -> None:
        self.preview_zoom = PREVIEW_ZOOM_MIN
        self.preview_pan_x = 0
        self.preview_pan_y = 0
        self.preview_drag_start = None
        if schedule:
            self._schedule_render()

    def _on_preview_mouse_wheel(self, event: object) -> str:
        if self.preview_image is None:
            return "break"

        direction = 1 if getattr(event, "delta", 0) > 0 or getattr(event, "num", None) == 4 else -1
        width = max(self.preview_label.winfo_width() - 12, 320)
        height = max(self.preview_label.winfo_height() - 12, 240)
        self.preview_zoom, self.preview_pan_x, self.preview_pan_y = zoom_preview_at(
            self.preview_image,
            width,
            height,
            self.preview_zoom,
            self.preview_pan_x,
            self.preview_pan_y,
            int(getattr(event, "x", width // 2)),
            int(getattr(event, "y", height // 2)),
            direction,
        )
        self.status_var.set(f"Preview zoom: {self.preview_zoom:.1f}x")
        self._schedule_render()
        return "break"

    def _on_preview_press(self, event: object) -> str:
        self.preview_drag_start = (int(getattr(event, "x", 0)), int(getattr(event, "y", 0)))
        self.preview_drag_start_pan = (self.preview_pan_x, self.preview_pan_y)
        return "break"

    def _on_preview_drag(self, event: object) -> str:
        if self.preview_image is None or self.preview_drag_start is None:
            return "break"

        start_x, start_y = self.preview_drag_start
        delta_x = int(getattr(event, "x", start_x)) - start_x
        delta_y = int(getattr(event, "y", start_y)) - start_y
        width = max(self.preview_label.winfo_width() - 12, 320)
        height = max(self.preview_label.winfo_height() - 12, 240)
        scaled_width, scaled_height = preview_scaled_size(self.preview_image, width, height, self.preview_zoom)
        self.preview_pan_x, self.preview_pan_y = clamp_preview_pan(
            self.preview_drag_start_pan[0] - delta_x,
            self.preview_drag_start_pan[1] - delta_y,
            scaled_width,
            scaled_height,
            width,
            height,
        )
        self._schedule_render()
        return "break"

    def _on_preview_release(self, _: object) -> str:
        self.preview_drag_start = None
        return "break"

    def _on_preview_double_click(self, _: object) -> str:
        self._reset_preview_zoom()
        self.status_var.set("Preview zoom reset")
        return "break"

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
