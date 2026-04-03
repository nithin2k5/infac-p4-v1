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
import subprocess

from core.camera import CameraManager
from core.inference import InferenceEngine
from core.inspection import InspectionManager
from PIL import Image, ImageTk
from datetime import datetime
from ui.theme import Colors, Fonts, Dimensions, configure_styles
from ui.components import StyledButton, ToggleSwitch


# ═════════════════════════════════════════════════════════
#  MODEL CONFIGURATION (Roboflow hosted — see core/inference.py)
# ═════════════════════════════════════════════════════════



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
        self.camera = CameraManager()
        self.inference = InferenceEngine()
        self.inspection = InspectionManager()
        
        self.inspection.on_log_result = self._add_log_entry
        self.inspection.on_stats_update = self._update_stats_ui

        self.is_detecting = False
        self.current_detections = []
        self.detection_log_items = []
        self.confidence_threshold = 0.65
        self._photo_ref = None

        threading.Thread(target=self.inference.load_model, daemon=True).start()

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

        self.settings_btn = StyledButton(
            inner, text="⚙️  Cam Settings", bg_color=Colors.BG_CARD,
            hover_color=Colors.BG_CARD_HOVER, width=150, height=40,
            command=self._open_cam_settings, font=Fonts.BUTTON)
        self.settings_btn.pack(side="left", padx=4)

        StyledButton(
            inner, text="🔄  Reset Stats", bg_color=Colors.BG_CARD,
            hover_color=Colors.BG_CARD_HOVER, width=130, height=40,
            command=self._reset_stats, font=Fonts.BUTTON).pack(side="left", padx=4)

    def _open_cam_settings(self):
        if self.camera.is_running:
            self.camera.open_settings_dialog()

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

        # Auto-Inspect
        auto_frame = tk.Frame(parent, bg=Colors.BG_CARD)
        auto_frame.pack(fill="x", padx=16, pady=(4, 8))
        tk.Label(auto_frame, text="Auto-Inspect", font=Fonts.SMALL_BOLD, bg=Colors.BG_CARD, fg=Colors.TEXT_PRIMARY).pack(side="left")
        self.auto_var = tk.BooleanVar(value=False)
        auto_cb = ttk.Checkbutton(auto_frame, variable=self.auto_var, command=self._on_auto_toggle)
        auto_cb.pack(side="right")
        
        # ROI Crop slider (Vertical height crop)
        roi_frame = tk.Frame(parent, bg=Colors.BG_CARD)
        roi_frame.pack(fill="x", padx=16, pady=(0, 8))
        tk.Label(roi_frame, text="ROI Zoom (Center)", font=Fonts.SMALL_BOLD, bg=Colors.BG_CARD, fg=Colors.TEXT_PRIMARY).pack(anchor="w")
        self.roi_var = tk.DoubleVar(value=1.0)
        ttk.Scale(roi_frame, from_=0.3, to=1.0, variable=self.roi_var, orient="horizontal").pack(fill="x", pady=(4, 0))

        # Detection Type
        det_type_frame = tk.Frame(parent, bg=Colors.BG_CARD)
        det_type_frame.pack(fill="x", padx=16, pady=(0, 8))
        tk.Label(det_type_frame, text="Detection Type", font=Fonts.SMALL_BOLD,
                 bg=Colors.BG_CARD, fg=Colors.TEXT_PRIMARY).pack(anchor="w")
        self.det_type_var = tk.StringVar(value="Type 1 (2 solders)")
        self.det_type_combo = ttk.Combobox(
            det_type_frame,
            textvariable=self.det_type_var,
            values=["Type 1 (2 solders)", "Type 2 (3 solders)"],
            state="readonly",
            font=Fonts.SMALL,
        )
        self.det_type_combo.pack(fill="x", pady=(4, 0))
        self.det_type_combo.bind("<<ComboboxSelected>>", self._on_det_type_change)

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
        if self.camera.is_paused:
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
            self.camera.is_paused = False
            self.is_detecting = True
        elif self.camera.is_running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        cam_idx = int(self.cam_combo.get().replace("Camera ", ""))
        self.cam_status_label.configure(text="● Connecting...", fg=Colors.WARNING)
        self.start_btn.itemconfig(self.start_btn._text_id, text="⏳  Connecting...")
        self.update_idletasks()
        
        self.camera.start_camera(cam_idx, self._on_camera_opened, self._on_camera_open_failed)

    def _on_camera_open_failed(self):
        self.after(0, lambda: self.cam_status_label.configure(text="● Camera Error", fg=Colors.DANGER))
        self.after(0, lambda: self.start_btn.itemconfig(self.start_btn._text_id, text="▶  Start Camera"))
        self.start_btn.bg_color = Colors.SUCCESS_DIM
        self.start_btn.hover_color = Colors.SUCCESS
        self.start_btn.itemconfig(self.start_btn._bg_id, fill=Colors.SUCCESS_DIM)

    def _on_camera_opened(self, w, h):
        self._static_frame = None
        self._static_predictions = []
        self.camera_canvas.unbind("<Configure>")
        
        self.is_detecting = True
        self.cam_status_label.configure(text="● Camera Connected", fg=Colors.SUCCESS)
        self.model_status_label.configure(text="● Detecting", fg=Colors.SUCCESS)
        
        self.start_btn.itemconfig(self.start_btn._text_id, text="⏹  Stop Camera")
        self.start_btn.bg_color = Colors.DANGER_DIM
        self.start_btn.hover_color = Colors.DANGER
        self.start_btn.itemconfig(self.start_btn._bg_id, fill=Colors.DANGER_DIM)
        
        self.res_label.configure(text=f"{w} × {h}")
        self._detect_interval = 0
        self._inference_busy = False
        
        self._update_frame()

    def _stop_camera(self):
        self.camera.is_running = False
        self.is_detecting = False
        self.camera.stop()

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
        frame, is_end = self.camera.read_frame()
        
        if is_end:
            self._stop_camera()
            self.cam_status_label.configure(text="● Video Ended", fg=Colors.TEXT_MUTED)
            return

        if frame is None:
            if self.camera.is_running:
                self.after(10, self._update_frame)
            return

        self.current_frame = frame.copy()
        
        self.fps_label.configure(text=f"FPS: {self.camera.fps:.1f}")
        self.stat_labels["fps_stat_val"].configure(text=f"{self.camera.fps:.0f}")
        self.frame_label.configure(text=f"Frame: {self.camera.frame_count:,}")

        # Provide ROI to inference based on slider
        roi_scale = self.roi_var.get()
        h_frame, w_frame = frame.shape[:2]
        new_w, new_h = int(w_frame * roi_scale), int(h_frame * roi_scale)
        rx, ry = (w_frame - new_w) // 2, (h_frame - new_h) // 2
        
        roi = None if roi_scale >= 0.99 else (rx, ry, new_w, new_h)

        if self.is_detecting:
            self._detect_interval += 1
            if self._detect_interval >= 2 and not getattr(self, '_inference_busy', False):
                self._detect_interval = 0
                self._inference_busy = True
                
                # Run inference in background
                def _bg_infer():
                    preds = self.inference.infer(frame.copy(), self.confidence_threshold, roi)
                    self.after(0, self._on_live_inference_result, preds)
                
                threading.Thread(target=_bg_infer, daemon=True).start()

        display_frame = frame.copy()

        # Draw ROI Box if scaled
        if roi:
            cv2.rectangle(display_frame, (rx, ry), (rx + new_w, ry + new_h), (255, 255, 255), 2, cv2.LINE_DASH)
            cv2.putText(display_frame, "ROI Mask", (rx + 5, ry + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if self.is_detecting:
            dets = list(self.current_detections)
            for det in dets:
                x1 = int(det["x"] - det["width"] / 2)
                y1 = int(det["y"] - det["height"] / 2)
                x2 = int(det["x"] + det["width"] / 2)
                y2 = int(det["y"] + det["height"] / 2)
                conf = det["confidence"]
                cls_name = det["class"]
                color_bgr = (0, 255, 0) if conf >= 0.8 else (0, 255, 255) if conf >= 0.5 else (0, 0, 255)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color_bgr, 3)
                
                # Bold corner accents
                corner_len = min(20, min(max(1, x2-x1), max(1, y2-y1)) // 3)
                for cx, cy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                                        (x1, y2, 1, -1), (x2, y2, -1, -1)]:
                    cv2.line(display_frame, (cx, cy), (cx + corner_len*dx, cy), color_bgr, 4)
                    cv2.line(display_frame, (cx, cy), (cx, cy + corner_len*dy), color_bgr, 4)

                label_text = f"{cls_name} {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(display_frame, (x1, y1 - th - 14), (x1 + tw + 14, y1), color_bgr, -1)
                cv2.putText(display_frame, label_text, (x1 + 7, y1 - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

            cx, cy = w_frame // 2, h_frame // 2
            cv2.line(display_frame, (cx-20, cy), (cx+20, cy), (88, 166, 255), 1)
            cv2.line(display_frame, (cx, cy-20), (cx, cy+20), (88, 166, 255), 1)

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            cv2.putText(display_frame, ts, (10, h_frame - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (138, 148, 158), 1)

            cv2.circle(display_frame, (w_frame - 30, 25), 8, (0, 0, 255), -1)
            cv2.putText(display_frame, "DETECTING", (w_frame - 130, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

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

    def _on_live_inference_result(self, predictions):
        self._inference_busy = False
        if not self.is_detecting:
            return
            
        self.current_detections = predictions
        
        # Pass to inspection logic for glowing PASS/FAIL indicators and Auto-Inspect
        pcb_detected, solder_count = self.inspection.process_live_frame(predictions)
        
        required = self.inspection.required_solders
        if pcb_detected and solder_count >= required:
            self.pass_frame.configure(bg=Colors.SUCCESS)
            self.pass_label.configure(bg=Colors.SUCCESS, fg=Colors.BG_DARKEST)
            self.ng_frame.configure(bg=Colors.BG_MEDIUM)
            self.ng_label.configure(bg=Colors.BG_MEDIUM, fg=Colors.TEXT_MUTED)
            self.model_status_label.configure(text="● PASS - Detected", fg=Colors.SUCCESS)
        elif pcb_detected and solder_count < required:
            self.pass_frame.configure(bg=Colors.BG_MEDIUM)
            self.pass_label.configure(bg=Colors.BG_MEDIUM, fg=Colors.TEXT_MUTED)
            self.ng_frame.configure(bg=Colors.DANGER)
            self.ng_label.configure(bg=Colors.DANGER, fg=Colors.TEXT_PRIMARY)
            self.model_status_label.configure(text="● NG - Inspecting", fg=Colors.DANGER)
        else:
            self.pass_frame.configure(bg=Colors.BG_MEDIUM)
            self.pass_label.configure(bg=Colors.BG_MEDIUM, fg=Colors.TEXT_MUTED)
            self.ng_frame.configure(bg=Colors.BG_MEDIUM)
            self.ng_label.configure(bg=Colors.BG_MEDIUM, fg=Colors.TEXT_MUTED)
            self.model_status_label.configure(text="● Detecting", fg=Colors.SUCCESS)

    def _on_det_type_change(self, event=None):
        is_type2 = "Type 2" in self.det_type_var.get()
        self.inspection.required_solders = 3 if is_type2 else 2

        # Update inference model version dynamically
        version = "11" if is_type2 else "9"
        self.model_status_label.configure(text=f"● Loading Model v{version}...", fg=Colors.WARNING)

        def _update_model():
            success = self.inference.set_model_version(version)
            if success:
                self.after(0, lambda: self.model_status_label.configure(
                    text="● Detecting" if getattr(self, 'is_detecting', False) else "● Model Ready",
                    fg=Colors.SUCCESS if getattr(self, 'is_detecting', False) else Colors.TEXT_MUTED))
            else:
                self.after(0, lambda: self.model_status_label.configure(
                    text="● Model Load Failed", fg=Colors.DANGER))

        threading.Thread(target=_update_model, daemon=True).start()

    def _on_auto_toggle(self):
        self.inspection.auto_inspect_enabled = self.auto_var.get()

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
        if self.camera.is_running:
            self._stop_camera()

        ext = filepath.lower().split('.')[-1]
        is_video = ext in ['mp4', 'avi', 'mov', 'mkv']

        if is_video:
            self._start_video_file(filepath)
        else:
            self._load_static_image(filepath)

    def _start_video_file(self, filepath):
        w, h = self.camera.start_video(filepath)
        if w is None:
            self.model_status_label.configure(text="● Failed to load video", fg=Colors.DANGER)
            return

        self._static_frame = None
        self._static_predictions = []
        self.camera_canvas.unbind("<Configure>")

        self.is_detecting = True
        filename = filepath.replace("/", "\\").split("\\")[-1]
        self.cam_status_label.configure(text=f"● Video: {filename}", fg=Colors.PRIMARY)

        self.start_btn.itemconfig(self.start_btn._text_id, text="⏹  Stop Video")
        self.start_btn.bg_color = Colors.DANGER_DIM
        self.start_btn.hover_color = Colors.DANGER
        self.start_btn.itemconfig(self.start_btn._bg_id, fill=Colors.DANGER_DIM)

        self.res_label.configure(text=f"{w} × {h}")
        self._detect_interval = 0
        self._update_frame()
    def _load_static_image(self, filepath):
        frame = cv2.imread(filepath)
        if frame is None:
            self.model_status_label.configure(text="● Failed to load image", fg=Colors.DANGER)
            return

        self.current_frame = frame.copy()
        self._static_frame = frame.copy()
        self._static_predictions = []

        self.camera_canvas.unbind("<Configure>")
        self.camera_canvas.bind("<Configure>", self._redraw_static)

        h_img, w_img = frame.shape[:2]
        self.res_label.configure(text=f"{w_img} × {h_img}")
        self.cam_status_label.configure(text="● Image Loaded", fg=Colors.PRIMARY)
        self.model_status_label.configure(text="● Running inference...", fg=Colors.WARNING)
        self.update_idletasks()

        self._display_static_frame(frame.copy(), [])

        def _bg_static():
            if not self.inference.is_loaded():
                self.after(0, lambda: self.model_status_label.configure(text="● Model not loaded", fg=Colors.DANGER))
                return
            try:
                # Use current ROI slider value
                roi_scale = self.roi_var.get()
                roi = None if roi_scale >= 0.99 else ((w_img - int(w_img * roi_scale)) // 2, (h_img - int(h_img * roi_scale)) // 2, int(w_img * roi_scale), int(h_img * roi_scale))

                predictions = self.inference.infer(frame, self.confidence_threshold, roi)
                self.after(0, self._on_static_result, frame.copy(), filepath, predictions)
            except Exception as e:
                print(e)
                self.after(0, lambda: self.model_status_label.configure(text="● Inference Error", fg=Colors.DANGER))
                
        threading.Thread(target=_bg_static, daemon=True).start()
        
    def _on_static_result(self, frame, filepath, predictions):
        self._static_predictions = predictions
        self._display_static_frame(frame.copy(), predictions)
        
        filename = filepath.replace("/", "\\").split("\\")[-1]
        self.inspection.process_test_snapshot(predictions, filename)
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
        if self.current_frame is None:
            return

        frame = self.current_frame.copy()

        if self.camera.is_running:
            self.camera.is_paused = True
            if self.is_detecting:
                self.is_detecting = False
                self.model_status_label.configure(text="● Model Paused", fg=Colors.WARNING)
                self.current_detections = []

            self.start_btn.itemconfig(self.start_btn._text_id, text="▶  Resume Camera")
            self.start_btn.bg_color = Colors.SUCCESS_DIM
            self.start_btn.hover_color = Colors.SUCCESS
            self.start_btn.itemconfig(self.start_btn._bg_id, fill=Colors.SUCCESS_DIM)

        self._static_frame = frame.copy()
        self._static_predictions = []
        self.camera_canvas.unbind("<Configure>")
        self.camera_canvas.bind("<Configure>", self._redraw_static)

        h_img, w_img = frame.shape[:2]
        self.res_label.configure(text=f"{w_img} × {h_img}")
        self.cam_status_label.configure(text="● Test Snapshot", fg=Colors.INFO)
        self.model_status_label.configure(text="● Running inference...", fg=Colors.WARNING)
        self.update_idletasks()

        self._display_static_frame(frame.copy(), [])

        def _run():
            if not self.inference.is_loaded():
                self.after(0, lambda: self.model_status_label.configure(text="● Model not loaded", fg=Colors.DANGER))
                return
            try:
                roi_scale = self.roi_var.get()
                roi = None if roi_scale >= 0.99 else ((w_img - int(w_img * roi_scale)) // 2, (h_img - int(h_img * roi_scale)) // 2, int(w_img * roi_scale), int(h_img * roi_scale))

                predictions = self.inference.infer(frame, self.confidence_threshold, roi)
                self.after(0, self._on_test_result, frame, predictions)
            except Exception as e:
                print(f"Test inference error: {e}")
                self.after(0, lambda: self.model_status_label.configure(text="● Inference Error", fg=Colors.DANGER))

        threading.Thread(target=_run, daemon=True).start()

    def _on_test_result(self, frame, predictions):
        self._static_predictions = predictions
        self._display_static_frame(frame.copy(), predictions)
        self.inspection.process_test_snapshot(predictions, "Test snapshot")

    def _reset_stats(self):
        self.inspection.reset_stats()
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

    def _update_stats_ui(self):
        stats = self.inspection.get_stats()
        self.stat_labels["total_inspected_val"].configure(text=str(stats["inspected"]))
        self.stat_labels["defects_val"].configure(text=str(stats["defects"]))
        self.stat_labels["pass_rate_val"].configure(text=f"{stats['pass_rate']:.1f}%")
        self.stat_labels["avg_conf_val"].configure(text=f"{stats['avg_conf']:.1%}")

    def _play_ng_alarm(self):
        """Play an alarm sound when NG is detected (non-blocking)."""
        threading.Thread(
            target=lambda: subprocess.run(
                ["afplay", "/System/Library/Sounds/Basso.aiff"],
                capture_output=True
            ),
            daemon=True
        ).start()

    def _add_log_entry(self, label, detail, color, confidence):
        """Add an entry to the detection log panel."""
        if "NG" in label:
            self._play_ng_alarm()
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
        self.camera.is_running = False
        self.camera.stop()
        self.destroy()
