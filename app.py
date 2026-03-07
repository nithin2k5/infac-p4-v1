"""
InFac P4 — Industrial Inspection System
Single-page live camera detection application using OpenCV + Tkinter + Roboflow.
"""

import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import numpy as np
import time
import threading
import queue
import base64
import io
import os

from ultralytics import YOLO
from PIL import Image, ImageTk
from datetime import datetime
from ui.theme import Colors, Fonts, Dimensions, configure_styles
from ui.components import StyledButton, ToggleSwitch


# ═════════════════════════════════════════════════════════
#  MODEL CONFIGURATION
# ═════════════════════════════════════════════════════════

MODEL_PATH = "weights-2.pt"


class InFacApp(tk.Tk):
    """Single-page industrial inspection application with live camera feed."""

    def __init__(self):
        
        super().__init__()

        # ── Window Setup ─────────────────────────────────
        self.title("InFac P4 — Industrial Inspection System")
        self.configure(bg=Colors.BG_DARKEST)
        self.minsize(1200, 750)

        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        win_w = max(1280, int(screen_w * 0.85))
        win_h = max(800, int(screen_h * 0.85))
        x = (screen_w - win_w) // 2
        y = (screen_h - win_h) // 2
        self.geometry(f"{win_w}x{win_h}+{x}+{y}")

        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

        configure_styles(self)

        # ── State ────────────────────────────────────────
        self.cap = None
        self.is_running = False
        self.is_detecting = False
        self.is_video_file = False
        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_time = time.time()
        self.fps_frame_count = 0
        self.current_detections = []       # latest detections from Roboflow
        self.detection_log_items = []      # log widgets in right panel
        self.total_inspected = 0
        self.total_defects = 0
        self.all_confidences = []          # for avg confidence calc
        self.confidence_threshold = 0.65
        self.current_frame = None
        self._photo_ref = None             # prevent GC
        self._inference_lock = threading.Lock()
        self._inference_busy = False       # throttle: one request at a time
        self._detect_interval = 0          # frame counter for inference spacing
        self._inference_fps = 0.0
        self._last_inference_time = 0
        self.frame_queue = queue.Queue(maxsize=1)
        self.capture_thread = None
        self.solder_history = []           # to smooth out flickering detections

        # Load Local Model
        # Loads the YOLO weights from the project folder.
        try:
            self.model = YOLO(MODEL_PATH)
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

        # ── Build UI ─────────────────────────────────────
        self._build_ui()

        # ── Protocol ─────────────────────────────────────
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ═════════════════════════════════════════════════════
    #  UI CONSTRUCTION
    # ═════════════════════════════════════════════════════

    def _build_ui(self):
        """Build the single-page layout."""

        # ── Top Bar ──────────────────────────────────────
        topbar = tk.Frame(self, bg=Colors.BG_DARKEST, height=50)
        topbar.pack(fill="x")
        topbar.pack_propagate(False)

        # Brand
        brand = tk.Frame(topbar, bg=Colors.BG_DARKEST)
        brand.pack(side="left", padx=20)
        tk.Label(brand, text="🏭", font=("Segoe UI", 18),
                 bg=Colors.BG_DARKEST, fg=Colors.PRIMARY).pack(side="left", padx=(0, 10))
        tk.Label(brand, text="InFac P4", font=("Segoe UI", 14, "bold"),
                 bg=Colors.BG_DARKEST, fg=Colors.TEXT_PRIMARY).pack(side="left")
        tk.Label(brand, text="Industrial Inspection System", font=Fonts.SMALL,
                 bg=Colors.BG_DARKEST, fg=Colors.TEXT_MUTED).pack(side="left", padx=(12, 0))

        # Right side status
        right = tk.Frame(topbar, bg=Colors.BG_DARKEST)
        right.pack(side="right", padx=20)

        self.cam_status_label = tk.Label(right, text="● Camera Disconnected",
                                          font=Fonts.SMALL_BOLD,
                                          bg=Colors.BG_DARKEST, fg=Colors.DANGER)
        self.cam_status_label.pack(side="left", padx=(0, 20))

        self.model_status_label = tk.Label(right, text="● Model Idle",
                                            font=Fonts.SMALL_BOLD,
                                            bg=Colors.BG_DARKEST, fg=Colors.TEXT_MUTED)
        self.model_status_label.pack(side="left", padx=(0, 20))

        self.time_label = tk.Label(right, text="", font=Fonts.MONO_SMALL,
                                    bg=Colors.BG_DARKEST, fg=Colors.TEXT_SECONDARY)
        self.time_label.pack(side="left")
        self._update_clock()

        # Separator
        tk.Frame(self, bg=Colors.BORDER, height=1).pack(fill="x")

        # ── Main Body ────────────────────────────────────
        body = tk.Frame(self, bg=Colors.BG_DARK)
        body.pack(fill="both", expand=True)

        # Left: Camera + Controls
        left_panel = tk.Frame(body, bg=Colors.BG_DARK)
        left_panel.pack(side="left", fill="both", expand=True, padx=(16, 8), pady=16)

        self._build_camera_section(left_panel)
        self._build_controls_bar(left_panel)
        self._build_stats_row(left_panel)

        # Right: Settings + Detection Log
        right_panel = tk.Frame(body, bg=Colors.BG_CARD, width=340)
        right_panel.pack(side="right", fill="y", padx=(8, 16), pady=16)
        right_panel.pack_propagate(False)

        self._build_right_panel(right_panel)

    def _build_camera_section(self, parent):
        """Camera feed area with header."""
        cam_header = tk.Frame(parent, bg=Colors.BG_DARK)
        cam_header.pack(fill="x", pady=(0, 8))

        tk.Label(cam_header, text="📷  Live Camera Feed", font=Fonts.SUBHEADING,
                 bg=Colors.BG_DARK, fg=Colors.TEXT_PRIMARY).pack(side="left")

        info_frame = tk.Frame(cam_header, bg=Colors.BG_DARK)
        info_frame.pack(side="right")

        self.fps_label = tk.Label(info_frame, text="FPS: --", font=Fonts.MONO_SMALL,
                                   bg=Colors.BG_DARK, fg=Colors.SUCCESS)
        self.fps_label.pack(side="left", padx=(0, 16))

        self.res_label = tk.Label(info_frame, text="-- × --", font=Fonts.MONO_SMALL,
                                   bg=Colors.BG_DARK, fg=Colors.TEXT_MUTED)
        self.res_label.pack(side="left", padx=(0, 16))

        self.frame_label = tk.Label(info_frame, text="Frame: 0", font=Fonts.MONO_SMALL,
                                     bg=Colors.BG_DARK, fg=Colors.TEXT_MUTED)
        self.frame_label.pack(side="left")

        cam_container = tk.Frame(parent, bg="#0a0a0a", highlightbackground=Colors.BORDER,
                                  highlightthickness=1)
        cam_container.pack(fill="both", expand=True)

        self.camera_canvas = tk.Canvas(cam_container, bg="#0a0a0a",
                                        highlightthickness=0, cursor="crosshair")
        self.camera_canvas.pack(fill="both", expand=True)
        self.camera_canvas.bind("<Configure>", self._draw_placeholder)

    def _build_controls_bar(self, parent):
        """Control buttons bar."""
        controls = tk.Frame(parent, bg=Colors.BG_MEDIUM)
        controls.pack(fill="x", pady=(8, 8))

        inner = tk.Frame(controls, bg=Colors.BG_MEDIUM)
        inner.pack(pady=10, padx=16)

        self.start_btn = StyledButton(
            inner, text="▶  Start Camera", bg_color=Colors.SUCCESS_DIM,
            hover_color=Colors.SUCCESS, width=150, height=40,
            command=self._toggle_camera, font=Fonts.BUTTON)
        self.start_btn.pack(side="left", padx=4)

        self.test_btn = StyledButton(
            inner, text="🧪  Test", bg_color=Colors.INFO,
            hover_color="#3b9eff", width=110, height=40,
            command=self._test_detect, font=Fonts.BUTTON)
        self.test_btn.pack(side="left", padx=4)

        StyledButton(
            inner, text="📂  Upload Media", bg_color=Colors.BG_CARD,
            hover_color=Colors.BG_CARD_HOVER, width=150, height=40,
            command=self._upload_media, font=Fonts.BUTTON).pack(side="left", padx=4)

        StyledButton(
            inner, text="🔄  Reset Stats", bg_color=Colors.BG_CARD,
            hover_color=Colors.BG_CARD_HOVER, width=130, height=40,
            command=self._reset_stats, font=Fonts.BUTTON).pack(side="left", padx=4)

    def _build_stats_row(self, parent):
        """Stats cards row below controls."""
        stats = tk.Frame(parent, bg=Colors.BG_DARK)
        stats.pack(fill="x")

        cards_data = [
            ("🔬", "Total Inspected", "total_inspected_val", "0", Colors.PRIMARY),
            ("⚠️", "Defects Found", "defects_val", "0", Colors.DANGER),
            ("✅", "Pass Rate", "pass_rate_val", "100.0%", Colors.SUCCESS),
            ("🎯", "Avg Confidence", "avg_conf_val", "-- %", Colors.WARNING),
            ("⚡", "FPS", "fps_stat_val", "--", Colors.INFO),
        ]

        self.stat_labels = {}
        for i, (icon, label, key, default, color) in enumerate(cards_data):
            card = tk.Frame(stats, bg=Colors.BG_CARD)
            card.pack(side="left", fill="x", expand=True,
                      padx=(0 if i == 0 else 4, 0 if i == len(cards_data)-1 else 4))

            inner = tk.Frame(card, bg=Colors.BG_CARD, padx=12, pady=10)
            inner.pack(fill="both", expand=True)

            top = tk.Frame(inner, bg=Colors.BG_CARD)
            top.pack(fill="x")
            tk.Label(top, text=icon, font=("Segoe UI", 12),
                     bg=Colors.BG_CARD, fg=color).pack(side="left")
            tk.Label(top, text=label, font=Fonts.TINY,
                     bg=Colors.BG_CARD, fg=Colors.TEXT_MUTED).pack(side="left", padx=(6, 0))

            accent = tk.Frame(inner, bg=color, height=2)
            accent.pack(fill="x", pady=(6, 4))

            val = tk.Label(inner, text=default, font=("Segoe UI", 18, "bold"),
                           bg=Colors.BG_CARD, fg=Colors.TEXT_PRIMARY)
            val.pack(anchor="w")
            self.stat_labels[key] = val

    def _build_right_panel(self, parent):
        """Right sidebar with settings and detection log."""

        # ── Detection Settings ───────────────────────────
        tk.Label(parent, text="⚙️  Detection Settings", font=Fonts.SUBHEADING,
                 bg=Colors.BG_CARD, fg=Colors.TEXT_PRIMARY).pack(
            anchor="w", padx=16, pady=(16, 12))

        # Confidence threshold
        thresh_frame = tk.Frame(parent, bg=Colors.BG_CARD)
        thresh_frame.pack(fill="x", padx=16, pady=(0, 8))

        top = tk.Frame(thresh_frame, bg=Colors.BG_CARD)
        top.pack(fill="x")
        tk.Label(top, text="Confidence Threshold", font=Fonts.SMALL_BOLD,
                 bg=Colors.BG_CARD, fg=Colors.TEXT_PRIMARY).pack(side="left")
        self.conf_display = tk.Label(top, text="65%", font=Fonts.MONO_SMALL,
                                      bg=Colors.BG_CARD, fg=Colors.PRIMARY)
        self.conf_display.pack(side="right")

        self.conf_var = tk.DoubleVar(value=0.65)
        ttk.Scale(thresh_frame, from_=0.1, to=1.0, variable=self.conf_var,
                  orient="horizontal",
                  command=self._on_threshold_change).pack(fill="x", pady=(4, 0))

        # Camera source
        cam_frame = tk.Frame(parent, bg=Colors.BG_CARD)
        cam_frame.pack(fill="x", padx=16, pady=(4, 8))
        tk.Label(cam_frame, text="Camera Source", font=Fonts.SMALL_BOLD,
                 bg=Colors.BG_CARD, fg=Colors.TEXT_PRIMARY).pack(anchor="w")
        self.cam_combo = ttk.Combobox(cam_frame, values=["Camera 0", "Camera 1", "Camera 2"],
                                       state="readonly", font=Fonts.SMALL)
        self.cam_combo.set("Camera 0")
        self.cam_combo.pack(fill="x", pady=(4, 0))

        # Result Indicators (PASS / NG)
        sep0 = tk.Frame(parent, bg=Colors.BORDER, height=1)
        sep0.pack(fill="x", padx=16, pady=8)

        result_frame = tk.Frame(parent, bg=Colors.BG_CARD)
        result_frame.pack(fill="x", padx=16, pady=(4, 4))

        # Using a frame for each to simulate a button look with padding and background
        self.pass_frame = tk.Frame(result_frame, bg=Colors.BG_MEDIUM, bd=1, relief="ridge")
        self.pass_frame.pack(side="left", fill="x", expand=True, padx=(0, 4), ipady=8)
        self.pass_label = tk.Label(self.pass_frame, text="PASS", font=Fonts.SUBHEADING,
                                   bg=Colors.BG_MEDIUM, fg=Colors.TEXT_MUTED)
        self.pass_label.pack(expand=True)

        self.ng_frame = tk.Frame(result_frame, bg=Colors.BG_MEDIUM, bd=1, relief="ridge")
        self.ng_frame.pack(side="left", fill="x", expand=True, padx=(4, 0), ipady=8)
        self.ng_label = tk.Label(self.ng_frame, text="NG", font=Fonts.SUBHEADING,
                                 bg=Colors.BG_MEDIUM, fg=Colors.TEXT_MUTED)
        self.ng_label.pack(expand=True)

        # Separator
        tk.Frame(parent, bg=Colors.BORDER, height=1).pack(fill="x", padx=16, pady=8)

        # ── Detection Log ────────────────────────────────
        log_header = tk.Frame(parent, bg=Colors.BG_CARD)
        log_header.pack(fill="x", padx=16, pady=(0, 8))
        tk.Label(log_header, text="📋  Detection Log", font=Fonts.SUBHEADING,
                 bg=Colors.BG_CARD, fg=Colors.TEXT_PRIMARY).pack(side="left")
        self.log_count_label = tk.Label(log_header, text="0 items", font=Fonts.TINY,
                                         bg=Colors.BG_CARD, fg=Colors.TEXT_MUTED)
        self.log_count_label.pack(side="right")

        log_container = tk.Frame(parent, bg=Colors.BG_CARD)
        log_container.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.log_canvas = tk.Canvas(log_container, bg=Colors.BG_CARD,
                                     highlightthickness=0)
        log_scrollbar = ttk.Scrollbar(log_container, orient="vertical",
                                       command=self.log_canvas.yview)
        self.log_frame = tk.Frame(self.log_canvas, bg=Colors.BG_CARD)

        self.log_frame.bind("<Configure>",
            lambda e: self.log_canvas.configure(scrollregion=self.log_canvas.bbox("all")))
        self.log_canvas.create_window((0, 0), window=self.log_frame, anchor="nw")
        self.log_canvas.configure(yscrollcommand=log_scrollbar.set)

        self.log_canvas.pack(side="left", fill="both", expand=True)
        log_scrollbar.pack(side="right", fill="y")

        self.empty_label = tk.Label(self.log_frame,
                                     text="No detections yet.\nStart camera & detection\nto see results here.",
                                     font=Fonts.SMALL, bg=Colors.BG_CARD,
                                     fg=Colors.TEXT_MUTED, justify="center")
        self.empty_label.pack(pady=40)

    # ═════════════════════════════════════════════════════
    #  CAMERA OPERATIONS
    # ═════════════════════════════════════════════════════

    def _toggle_camera(self):
        if getattr(self, "is_paused", False):
            # Resume from paused state
            self.is_paused = False
            self.is_detecting = True  # Auto-resume detection
            self.cam_status_label.configure(text="● Camera Connected", fg=Colors.SUCCESS)
            self.model_status_label.configure(text="● Detecting", fg=Colors.SUCCESS)
            self.start_btn.itemconfig(self.start_btn._text_id, text="⏹  Stop Camera")
            self.start_btn.bg_color = Colors.DANGER_DIM
            self.start_btn.hover_color = Colors.DANGER
            self.start_btn.itemconfig(self.start_btn._bg_id, fill=Colors.DANGER_DIM)
            self.camera_canvas.unbind("<Configure>")
        elif self.is_running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        cam_idx = int(self.cam_combo.get().replace("Camera ", ""))

        # Show connecting status & disable button while opening
        self.cam_status_label.configure(text="● Connecting...", fg=Colors.WARNING)
        self.start_btn.itemconfig(self.start_btn._text_id, text="⏳  Connecting...")
        self.update_idletasks()

        # Open camera in background thread to avoid freezing the UI
        thread = threading.Thread(
            target=self._open_camera_thread, args=(cam_idx,), daemon=True)
        thread.start()

    def _open_camera_thread(self, cam_idx):
        """Open camera in a background thread for faster, non-blocking init."""
        import platform
        # Use DirectShow on Windows for much faster camera initialization
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(cam_idx)

        # Lower resolution to speed up preprocessing and inference. Match model resolution 416x416.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 416)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 416)
        # Minimize the buffer so we always get the latest frame
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            self.after(0, self._on_camera_open_failed)
            return

        # Pre-read one frame to fully initialise the pipeline
        cap.read()

        self.after(0, self._on_camera_opened, cap)

    def _on_camera_open_failed(self):
        """Called on main thread when camera fails to open."""
        self.cam_status_label.configure(text="● Camera Error", fg=Colors.DANGER)
        self.start_btn.itemconfig(self.start_btn._text_id, text="▶  Start Camera")
        self.start_btn.bg_color = Colors.SUCCESS_DIM
        self.start_btn.hover_color = Colors.SUCCESS
        self.start_btn.itemconfig(self.start_btn._bg_id, fill=Colors.SUCCESS_DIM)

    def _on_camera_opened(self, cap):
        """Called on main thread once the camera is ready."""
        self.cap = cap

        # Clear any uploaded image state
        self._static_frame = None
        self._static_predictions = []
        self.camera_canvas.unbind("<Configure>")

        self.is_running = True
        self.is_detecting = True  # Start detection automatically
        self.cam_status_label.configure(text="● Camera Connected", fg=Colors.SUCCESS)
        self.model_status_label.configure(text="● Detecting", fg=Colors.SUCCESS)

        self.start_btn.itemconfig(self.start_btn._text_id, text="⏹  Stop Camera")
        self.start_btn.bg_color = Colors.DANGER_DIM
        self.start_btn.hover_color = Colors.DANGER
        self.start_btn.itemconfig(self.start_btn._bg_id, fill=Colors.DANGER_DIM)

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.res_label.configure(text=f"{w} × {h}")

        self.last_fps_time = time.time()
        self.fps_frame_count = 0
        self._detect_interval = 0

        self.capture_thread = threading.Thread(target=self._camera_capture_loop, daemon=True)
        self.capture_thread.start()

        self._update_frame()

    def _camera_capture_loop(self):
        """Continuously read frames in the background to avoid blocking the UI."""
        while getattr(self, "is_running", False) and self.cap and self.cap.isOpened():
            if getattr(self, "is_video_file", False):
                time.sleep(0.05)
                continue
            
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
                
            if not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)

    def _stop_camera(self):
        self.is_running = False
        self.is_detecting = False
        self.is_video_file = False
        self.is_paused = False
        if self.cap:
            self.cap.release()
            self.cap = None

        self.cam_status_label.configure(text="● Camera Disconnected", fg=Colors.DANGER)
        self.model_status_label.configure(text="● Model Idle", fg=Colors.TEXT_MUTED)

        self.start_btn.itemconfig(self.start_btn._text_id, text="▶  Start Camera")
        self.start_btn.bg_color = Colors.SUCCESS_DIM
        self.start_btn.hover_color = Colors.SUCCESS
        self.start_btn.itemconfig(self.start_btn._bg_id, fill=Colors.SUCCESS_DIM)

        self.fps_label.configure(text="FPS: --")
        self.frame_label.configure(text="Frame: 0")
        self.frame_count = 0
        self.current_detections = []

        self.camera_canvas.delete("all")
        self.camera_canvas.bind("<Configure>", self._draw_placeholder)
        self._draw_placeholder()
        self._update_result_indicators([])

    def _update_frame(self):
        if not self.is_running or not self.cap:
            return

        if getattr(self, "is_video_file", False):
            ret, frame = self.cap.read()
            if not ret:
                self._stop_camera()
                self.cam_status_label.configure(text="● Video Ended", fg=Colors.TEXT_MUTED)
                return
        else:
            try:
                frame = self.frame_queue.get_nowait()
            except queue.Empty:
                self.after(10, self._update_frame)
                return

        if getattr(self, "is_paused", False):
            self.after(15, self._update_frame)
            return

        self.frame_count += 1
        self.fps_frame_count += 1
        self.current_frame = frame.copy()

        # Calculate FPS
        now = time.time()
        elapsed = now - self.last_fps_time
        if elapsed >= 1.0:
            self.fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.last_fps_time = now
            self.fps_label.configure(text=f"FPS: {self.fps:.1f}")
            self.stat_labels["fps_stat_val"].configure(text=f"{self.fps:.0f}")

        self.frame_label.configure(text=f"Frame: {self.frame_count:,}")

        # ── Trigger Roboflow inference ───────────────────
        if self.is_detecting:
            self._detect_interval += 1
            # Run local inference as fast as possible without overlapping
            if self._detect_interval >= 2 and not self._inference_busy:
                self._detect_interval = 0
                self._run_inference_async(frame.copy())

        # ── Draw frame with detection overlays ───────────
        display_frame = frame.copy()

        if self.is_detecting:
            h_frame, w_frame = display_frame.shape[:2]

            # Draw bounding boxes from latest detections
            with self._inference_lock:
                dets = list(self.current_detections)

            for det in dets:
                x1 = int(det["x"] - det["width"] / 2)
                y1 = int(det["y"] - det["height"] / 2)
                x2 = int(det["x"] + det["width"] / 2)
                y2 = int(det["y"] + det["height"] / 2)
                conf = det["confidence"]
                cls_name = det["class"]

                # Vivid colors for visibility
                if conf >= 0.8:
                    color_bgr = (0, 255, 0)       # bright green
                elif conf >= 0.5:
                    color_bgr = (0, 255, 255)     # bright yellow
                else:
                    color_bgr = (0, 0, 255)       # bright red

                # Thick bounding box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color_bgr, 3)

                # Bold corner accents
                corner_len = min(20, min(x2-x1, y2-y1) // 3)
                for cx, cy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                                        (x1, y2, 1, -1), (x2, y2, -1, -1)]:
                    cv2.line(display_frame, (cx, cy), (cx + corner_len*dx, cy), color_bgr, 4)
                    cv2.line(display_frame, (cx, cy), (cx, cy + corner_len*dy), color_bgr, 4)

                # Large label background + text
                label_text = f"{cls_name} {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(display_frame, (x1, y1 - th - 14), (x1 + tw + 14, y1), color_bgr, -1)
                cv2.putText(display_frame, label_text, (x1 + 7, y1 - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

            # Crosshair
            cx, cy = w_frame // 2, h_frame // 2
            cv2.line(display_frame, (cx-20, cy), (cx+20, cy), (88, 166, 255), 1)
            cv2.line(display_frame, (cx, cy-20), (cx, cy+20), (88, 166, 255), 1)

            # Timestamp
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            cv2.putText(display_frame, ts, (10, h_frame - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (138, 148, 158), 1)

            # DETECTING indicator
            cv2.circle(display_frame, (w_frame - 30, 25), 8, (0, 0, 255), -1)
            cv2.putText(display_frame, "DETECTING", (w_frame - 130, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Convert to Tkinter image
        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        canvas_w = self.camera_canvas.winfo_width()
        canvas_h = self.camera_canvas.winfo_height()
        if canvas_w > 10 and canvas_h > 10:
            img_w, img_h = img.size
            scale = min(canvas_w / img_w, canvas_h / img_h)
            new_w, new_h = int(img_w * scale), int(img_h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        photo = ImageTk.PhotoImage(image=img)
        self._photo_ref = photo
        self.camera_canvas.delete("all")
        self.camera_canvas.create_image(canvas_w // 2, canvas_h // 2,
                                         image=photo, anchor="center")

        self.after(15, self._update_frame)

    # ═════════════════════════════════════════════════════
    #  ROBOFLOW INFERENCE
    # ═════════════════════════════════════════════════════

    def _run_inference_async(self, frame):
        """Send a frame to Roboflow in a background thread."""
        self._inference_busy = True
        thread = threading.Thread(target=self._do_inference, args=(frame,), daemon=True)
        thread.start()

    def _do_inference(self, frame):
        """Perform local offline inference (runs in background thread)."""
        if self.model is None:
            self.after(0, lambda: self.model_status_label.configure(
                text="● Model not loaded", fg=Colors.DANGER))
            self._inference_busy = False
            return

        try:
            predictions = self._infer_with_roi(frame)

            with self._inference_lock:
                self.current_detections = predictions

            # Schedule UI update on the main thread
            self.after(0, self._on_inference_result, predictions)

        except Exception as e:
            print(f"Inference error: {e}")
            self.after(0, lambda: self.model_status_label.configure(
                text="● Inference Error", fg=Colors.DANGER))
        finally:
            self._inference_busy = False

    def _on_inference_result(self, predictions):
        """Handle inference results on the main thread — update stats & log."""
        if not self.is_detecting:
            return

        # ── Proceed with inspection ────────
        solder_preds = [p for p in predictions if p["class"].lower() == "solder"]
        solder_count = len(solder_preds)

        self.model_status_label.configure(text="● Detecting", fg=Colors.SUCCESS)

        self._update_result_indicators(predictions)

        # Note: We purposely do NOT log to the UI or update stats here during live feed.
        # Logging and stats belong only to the Test snapshot flow.

    # ═════════════════════════════════════════════════════
    #  CONTROLS
    # ═════════════════════════════════════════════════════

    def _upload_media(self):
        """Open a file dialog to upload an image or video for detection testing."""
        filetypes = [
            ("Media files", "*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mov *.mkv"),
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        ]
        filepath = filedialog.askopenfilename(title="Select Media for Detection",
                                              filetypes=filetypes)
        if not filepath:
            return

        # Stop live camera if running
        if self.is_running:
            self._stop_camera()

        ext = filepath.lower().split('.')[-1]
        is_video = ext in ['mp4', 'avi', 'mov', 'mkv']

        if is_video:
            self._start_video_file(filepath)
        else:
            self._load_static_image(filepath)

    def _start_video_file(self, filepath):
        """Start playing and detecting from a video file."""
        self.cap = cv2.VideoCapture(filepath)
        if not self.cap.isOpened():
            self.model_status_label.configure(text="● Failed to load video", fg=Colors.DANGER)
            return

        self._static_frame = None
        self._static_predictions = []
        self.camera_canvas.unbind("<Configure>")

        self.is_running = True
        self.is_video_file = True
        filename = filepath.replace("/", "\\").split("\\")[-1]
        self.cam_status_label.configure(text=f"● Video: {filename}", fg=Colors.PRIMARY)

        self.start_btn.itemconfig(self.start_btn._text_id, text="⏹  Stop Video")
        self.start_btn.bg_color = Colors.DANGER_DIM
        self.start_btn.hover_color = Colors.DANGER
        self.start_btn.itemconfig(self.start_btn._bg_id, fill=Colors.DANGER_DIM)

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.res_label.configure(text=f"{w} × {h}")

        self.last_fps_time = time.time()
        self.fps_frame_count = 0
        self._detect_interval = 0
        self._update_frame()

    def _load_static_image(self, filepath):
        """Load and run inference on a single static image."""
        frame = cv2.imread(filepath)
        if frame is None:
            self.model_status_label.configure(text="● Failed to load image", fg=Colors.DANGER)
            return

        self.current_frame = frame.copy()
        self._static_frame = frame.copy()       # store for resize redraw
        self._static_predictions = []            # store for resize redraw

        # Unbind placeholder so uploaded image persists on resize
        self.camera_canvas.unbind("<Configure>")
        self.camera_canvas.bind("<Configure>", self._redraw_static)

        # Show image dimensions
        h_img, w_img = frame.shape[:2]
        self.res_label.configure(text=f"{w_img} × {h_img}")
        self.cam_status_label.configure(text="● Image Loaded", fg=Colors.PRIMARY)
        self.model_status_label.configure(text="● Running inference...", fg=Colors.WARNING)
        self.update_idletasks()

        # Display the image immediately (no detections yet)
        self._display_static_frame(frame.copy(), [])

        # Run local offline inference
        if self.model is None:
            self.model_status_label.configure(text="● Model not loaded", fg=Colors.DANGER)
            return

        try:
            predictions = self._infer_with_roi(frame)

            self._static_predictions = predictions  # store for resize redraw
            self._display_static_frame(frame.copy(), predictions)

            # ── PCB Presence Guard (static image) ─────
            self._update_result_indicators(predictions)
            solder_preds = [p for p in predictions if p["class"].lower() == "solder"]
            solder_count = len(solder_preds)
            filename = filepath.replace("/", "\\").split("\\")[-1]

            if solder_count == 0:
                # No PCB detected in the uploaded image
                self.model_status_label.configure(
                    text="● No PCB detected in image", fg=Colors.WARNING)
                return

            self.total_inspected += 1
            self.stat_labels["total_inspected_val"].configure(text=str(self.total_inspected))

            for pred in solder_preds:
                self.all_confidences.append(pred["confidence"])

            if solder_count >= 2:
                self.model_status_label.configure(
                    text=f"● PASS — {solder_count} solder(s)", fg=Colors.SUCCESS)
                self._add_log_entry(
                    "✅ PASS",
                    f"Image: {filename}  •  {solder_count} solder(s)",
                    Colors.SUCCESS,
                    f"{max((p['confidence'] for p in solder_preds), default=0):.0%}"
                )
            else:
                missing = 2 - solder_count
                self.total_defects += 1
                self.model_status_label.configure(
                    text=f"● NG — {missing} solder(s) missing", fg=Colors.DANGER)
                self._add_log_entry(
                    "❌ NG",
                    f"Image: {filename}  •  {missing} solder(s) missing",
                    Colors.DANGER,
                    f"{solder_count}/2"
                )

            # Update stats
            self.stat_labels["defects_val"].configure(text=str(self.total_defects))
            if self.total_inspected > 0:
                pass_rate = (1 - self.total_defects / self.total_inspected) * 100
                self.stat_labels["pass_rate_val"].configure(text=f"{pass_rate:.1f}%")
            if self.all_confidences:
                avg_c = sum(self.all_confidences) / len(self.all_confidences)
                self.stat_labels["avg_conf_val"].configure(text=f"{avg_c:.1%}")
                
        except Exception as e:
            print(f"Static inference error: {e}")
            self.model_status_label.configure(text="● Inference Error", fg=Colors.DANGER)

    def _redraw_static(self, event=None):
        """Redraw the uploaded image on canvas resize."""
        if hasattr(self, '_static_frame') and self._static_frame is not None:
            self._display_static_frame(self._static_frame.copy(),
                                        self._static_predictions)

    def _display_static_frame(self, frame, predictions):
        """Display a static image on the canvas with detection overlays."""
        for det in predictions:
            x1 = int(det["x"] - det["width"] / 2)
            y1 = int(det["y"] - det["height"] / 2)
            x2 = int(det["x"] + det["width"] / 2)
            y2 = int(det["y"] + det["height"] / 2)
            conf = det["confidence"]
            cls_name = det["class"]

            color_bgr = (0, 255, 0) if conf >= 0.8 else (0, 255, 255) if conf >= 0.5 else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 3)
            corner_len = min(20, min(x2-x1, y2-y1) // 3)
            for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(frame, (cx, cy), (cx+corner_len*dx, cy), color_bgr, 4)
                cv2.line(frame, (cx, cy), (cx, cy+corner_len*dy), color_bgr, 4)

            label_text = f"{cls_name} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1-th-14), (x1+tw+14, y1), color_bgr, -1)
            cv2.putText(frame, label_text, (x1+7, y1-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        canvas_w = self.camera_canvas.winfo_width()
        canvas_h = self.camera_canvas.winfo_height()
        if canvas_w > 10 and canvas_h > 10:
            img_w, img_h = img.size
            scale = min(canvas_w / img_w, canvas_h / img_h)
            new_w, new_h = int(img_w * scale), int(img_h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        photo = ImageTk.PhotoImage(image=img)
        self._photo_ref = photo
        self.camera_canvas.delete("all")
        self.camera_canvas.create_image(canvas_w // 2, canvas_h // 2,
                                         image=photo, anchor="center")

    def _test_detect(self):
        """Take a snapshot from the live camera and run detection on it."""
        if self.current_frame is None:
            return

        # Grab a copy of the current frame
        frame = self.current_frame.copy()

        # Pause live camera update so the static result stays on screen
        if self.is_running:
            self.is_paused = True
            if self.is_detecting:
                self.is_detecting = False
                self.model_status_label.configure(text="● Model Paused", fg=Colors.WARNING)
                with self._inference_lock:
                    self.current_detections = []

            # Change start button to Resume Camera
            self.start_btn.itemconfig(self.start_btn._text_id, text="▶  Resume Camera")
            self.start_btn.bg_color = Colors.SUCCESS_DIM
            self.start_btn.hover_color = Colors.SUCCESS
            self.start_btn.itemconfig(self.start_btn._bg_id, fill=Colors.SUCCESS_DIM)

        # Use the same static-image display pipeline
        self._static_frame = frame.copy()
        self._static_predictions = []
        self.camera_canvas.unbind("<Configure>")
        self.camera_canvas.bind("<Configure>", self._redraw_static)

        h_img, w_img = frame.shape[:2]
        self.res_label.configure(text=f"{w_img} × {h_img}")
        self.cam_status_label.configure(text="● Test Snapshot", fg=Colors.INFO)
        self.model_status_label.configure(text="● Running inference...", fg=Colors.WARNING)
        self.update_idletasks()

        # Show the raw frame immediately
        self._display_static_frame(frame.copy(), [])

        # Run inference in background to keep UI responsive
        def _run():
            if self.model is None:
                self.after(0, lambda: self.model_status_label.configure(
                    text="● Model not loaded", fg=Colors.DANGER))
                return
            try:
                predictions = self._infer_with_roi(frame)
                self.after(0, self._on_test_result, frame, predictions)
            except Exception as e:
                print(f"Test inference error: {e}")
                self.after(0, lambda: self.model_status_label.configure(
                    text="● Inference Error", fg=Colors.DANGER))

        threading.Thread(target=_run, daemon=True).start()

    def _on_test_result(self, frame, predictions):
        """Display test detection results on the main thread."""
        solder_preds = [p for p in predictions if p["class"].lower() == "solder"]
        solder_count = len(solder_preds)

        self._static_predictions = predictions
        self._display_static_frame(frame.copy(), predictions)
        self._update_result_indicators(predictions)

        # Increment inspection count on Test
        self.total_inspected += 1
        self.stat_labels["total_inspected_val"].configure(text=str(self.total_inspected))

        timestamp = datetime.now().strftime("%H:%M:%S")

        if solder_count == 2:
            self.model_status_label.configure(text="● Test PASS", fg=Colors.SUCCESS)
            self._add_log_entry(
                "✅ PASS",
                f"{timestamp}  •  {solder_count} solder(s)",
                Colors.SUCCESS,
                f"{max((p['confidence'] for p in solder_preds), default=0):.0%}" if solder_preds else "--"
            )
        else:
            self.model_status_label.configure(text="● Test NG", fg=Colors.DANGER)
            self.total_defects += 1
            if solder_count < 2:
                missing = 2 - solder_count
                detail_msg = f"{timestamp}  •  {missing} solder(s) missing"
            else:
                extra = solder_count - 2
                detail_msg = f"{timestamp}  •  {extra} extra solder(s)"

            self._add_log_entry(
                "❌ NG",
                detail_msg,
                Colors.DANGER,
                f"{solder_count}/2"
            )

        # Record confidences and update stats
        for pred in solder_preds:
            self.all_confidences.append(pred["confidence"])

        self.stat_labels["defects_val"].configure(text=str(self.total_defects))

        if self.total_inspected > 0:
            pass_rate = (1 - self.total_defects / self.total_inspected) * 100
            self.stat_labels["pass_rate_val"].configure(text=f"{pass_rate:.1f}%")
        
        if self.all_confidences:
            avg_conf = sum(self.all_confidences) / len(self.all_confidences)
            self.stat_labels["avg_conf_val"].configure(text=f"{avg_conf:.1%}")

        # Log each detection individually (except the generic 'pcb' class)
        for pred in predictions:
            cls_name = pred["class"]
            if cls_name.lower() == "pcb":
                continue

            conf = pred["confidence"]
            color = Colors.SUCCESS if conf >= 0.8 else (Colors.WARNING if conf >= 0.5 else Colors.DANGER)
            self._add_log_entry(
                f"🧪 {cls_name}",
                f"{timestamp}  •  Test snapshot",
                color,
                f"{conf:.0%}"
            )

    def _reset_stats(self):
        self.total_inspected = 0
        self.total_defects = 0
        self.all_confidences.clear()
        self.detection_log_items.clear()
        self.stat_labels["total_inspected_val"].configure(text="0")
        self.stat_labels["defects_val"].configure(text="0")
        self.stat_labels["pass_rate_val"].configure(text="100.0%")
        self.stat_labels["avg_conf_val"].configure(text="-- %")

        for widget in self.log_frame.winfo_children():
            widget.destroy()
        self.empty_label = tk.Label(self.log_frame,
                                     text="No detections yet.\nStart camera & detection\nto see results here.",
                                     font=Fonts.SMALL, bg=Colors.BG_CARD,
                                     fg=Colors.TEXT_MUTED, justify="center")
        self.empty_label.pack(pady=40)
        self.log_count_label.configure(text="0 items")
        self._update_result_indicators([])

    # ═════════════════════════════════════════════════════
    #  DETECTION LOG
    # ═════════════════════════════════════════════════════

    def _add_log_entry(self, label, detail, color, confidence):
        """Add an entry to the detection log panel."""
        if hasattr(self, 'empty_label') and self.empty_label.winfo_exists():
            self.empty_label.destroy()

        item = tk.Frame(self.log_frame, bg=Colors.BG_MEDIUM)
        item.pack(fill="x", pady=2, padx=2)

        indicator = tk.Frame(item, width=3, bg=color)
        indicator.pack(side="left", fill="y")

        info = tk.Frame(item, bg=Colors.BG_MEDIUM)
        info.pack(side="left", fill="x", expand=True, padx=10, pady=6)

        top_row = tk.Frame(info, bg=Colors.BG_MEDIUM)
        top_row.pack(fill="x")
        tk.Label(top_row, text=label, font=Fonts.SMALL_BOLD,
                 bg=Colors.BG_MEDIUM, fg=color).pack(side="left")
        tk.Label(top_row, text=confidence, font=Fonts.MONO_SMALL,
                 bg=Colors.BG_MEDIUM, fg=Colors.TEXT_SECONDARY).pack(side="right")

        tk.Label(info, text=detail, font=Fonts.TINY,
                 bg=Colors.BG_MEDIUM, fg=Colors.TEXT_MUTED).pack(anchor="w")

        self.detection_log_items.append(item)
        self.log_count_label.configure(text=f"{len(self.detection_log_items)} items")

        # Keep log manageable — remove oldest if > 100
        if len(self.detection_log_items) > 100:
            old = self.detection_log_items.pop(0)
            old.destroy()

        self.log_canvas.update_idletasks()
        self.log_canvas.yview_moveto(1.0)

    # ═════════════════════════════════════════════════════
    #  HELPERS
    # ═════════════════════════════════════════════════════

    def _infer_with_roi(self, frame):
        """Infer on full frame and apply NMS (ROI suffix kept for method compatibility)."""
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        result = results[0]

        raw_predictions = []
        if len(result.boxes) > 0:
            for box in result.boxes:
                # Ultralytics boxes: xywh (center X, center Y, width, height) format
                x, y, w, h = box.xywh[0].cpu().numpy()
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                class_name = result.names[cls_id]

                raw_predictions.append({
                    "x": float(x),
                    "y": float(y),
                    "width": float(w),
                    "height": float(h),
                    "class": class_name.lower(),
                    "confidence": confidence
                })
                
        return self._apply_nms(raw_predictions, iou_threshold=0.3)
        
    def _apply_nms(self, predictions, iou_threshold=0.3):
        """Apply Non-Maximum Suppression to filter out overlapping bounding boxes."""
        if not predictions:
            return []
            
        # Group by class
        by_class = {}
        for p in predictions:
            c = p["class"]
            if c not in by_class:
                by_class[c] = []
            by_class[c].append(p)
            
        final_preds = []
        
        for cls_name, preds in by_class.items():
            # Sort by confidence descending
            preds.sort(key=lambda x: x["confidence"], reverse=True)
            
            keep = []
            for p in preds:
                # Convert to x1, y1, x2, y2 for easier IoU calc
                x1_a = p["x"] - p["width"] / 2
                y1_a = p["y"] - p["height"] / 2
                x2_a = p["x"] + p["width"] / 2
                y2_a = p["y"] + p["height"] / 2
                area_a = p["width"] * p["height"]
                
                overlap = False
                for k in keep:
                    x1_b = k["x"] - k["width"] / 2
                    y1_b = k["y"] - k["height"] / 2
                    x2_b = k["x"] + k["width"] / 2
                    y2_b = k["y"] + k["height"] / 2
                    area_b = k["width"] * k["height"]
                    
                    # Calculate intersection
                    inter_x1 = max(x1_a, x1_b)
                    inter_y1 = max(y1_a, y1_b)
                    inter_x2 = min(x2_a, x2_b)
                    inter_y2 = min(y2_a, y2_b)
                    
                    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                        union_area = area_a + area_b - inter_area
                        iou = inter_area / union_area
                        
                        if iou > iou_threshold:
                            overlap = True
                            break
                            
                if not overlap:
                    keep.append(p)
                    
            final_preds.extend(keep)
            
        return final_preds

    def _update_result_indicators(self, predictions):
        if not hasattr(self, 'pass_frame') or not hasattr(self, 'ng_frame'):
            return
            
        pcb_preds = [p for p in predictions if p["class"].lower() == "pcb"]
        solder_preds = [p for p in predictions if p["class"].lower() == "solder"]
        
        pcb_detected = len(pcb_preds) > 0
        current_solder_count = len(solder_preds)
        
        # Keep a rolling window of the last 5 frames to prevent flickering
        if getattr(self, "is_detecting", False) and not getattr(self, "is_paused", False):
            self.solder_history.append(current_solder_count)
            if len(self.solder_history) > 5:
                self.solder_history.pop(0)
            
            # Use the highest number of solders seen in the last 5 frames 
            # (since missing a detection is common, but hallucinating a fake perfect solder is rare)
            effective_solder_count = max(self.solder_history) if self.solder_history else current_solder_count
        else:
            effective_solder_count = current_solder_count
            self.solder_history = []
        
        if pcb_detected or effective_solder_count > 0:
            if effective_solder_count >= 2:
                # Glow PASS
                self.pass_frame.configure(bg=Colors.SUCCESS)
                self.pass_label.configure(bg=Colors.SUCCESS, fg=Colors.BG_DARKEST)
                self.ng_frame.configure(bg=Colors.BG_MEDIUM)
                self.ng_label.configure(bg=Colors.BG_MEDIUM, fg=Colors.TEXT_MUTED)
            else:
                # Glow NG
                self.pass_frame.configure(bg=Colors.BG_MEDIUM)
                self.pass_label.configure(bg=Colors.BG_MEDIUM, fg=Colors.TEXT_MUTED)
                self.ng_frame.configure(bg=Colors.DANGER)
                self.ng_label.configure(bg=Colors.DANGER, fg=Colors.TEXT_PRIMARY)
        else:
            # Reset
            self.pass_frame.configure(bg=Colors.BG_MEDIUM)
            self.pass_label.configure(bg=Colors.BG_MEDIUM, fg=Colors.TEXT_MUTED)
            self.ng_frame.configure(bg=Colors.BG_MEDIUM)
            self.ng_label.configure(bg=Colors.BG_MEDIUM, fg=Colors.TEXT_MUTED)

    def _on_threshold_change(self, value):
        pct = int(float(value) * 100)
        self.confidence_threshold = float(value)
        self.conf_display.configure(text=f"{pct}%")

    def _update_clock(self):
        now = datetime.now().strftime("%H:%M:%S")
        self.time_label.configure(text=now)
        self.after(1000, self._update_clock)

    def _draw_placeholder(self, event=None):
        c = self.camera_canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 50 or h < 50:
            return

        for x in range(0, w, 40):
            c.create_line(x, 0, x, h, fill="#1a1a2e", width=1)
        for y in range(0, h, 40):
            c.create_line(0, y, w, y, fill="#1a1a2e", width=1)

        c.create_text(w//2, h//2 - 30, text="📷", font=("Segoe UI", 36), fill=Colors.TEXT_MUTED)
        c.create_text(w//2, h//2 + 20, text="Camera feed will appear here",
                      font=Fonts.BODY, fill=Colors.TEXT_MUTED)
        c.create_text(w//2, h//2 + 45, text='Click "Start Camera" to begin',
                      font=Fonts.SMALL, fill=Colors.TEXT_MUTED)

    def _on_close(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.destroy()
