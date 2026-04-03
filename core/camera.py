import cv2
import threading
import queue
import time
import platform

class CameraManager:
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.is_video_file = False
        self.is_paused = False
        self.frame_queue = queue.Queue(maxsize=1)
        self.capture_thread = None

        self.fps = 0.0
        self.fps_frame_count = 0
        self.last_fps_time = time.time()
        self.frame_count = 0
        self.filename = ""
        
    def start_camera(self, cam_idx, on_success, on_fail):
        """Open camera in a background thread."""
        self.filename = f"Camera {cam_idx}"
        self.is_video_file = False
        thread = threading.Thread(
            target=self._open_camera_thread, args=(cam_idx, on_success, on_fail), daemon=True)
        thread.start()

    def _open_camera_thread(self, cam_idx, on_success, on_fail):
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(cam_idx)

        # Force MJPG codec to prevent lag and slow framerates on Windows
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FPS, 40)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 416)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 416)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Force manual focus (Autofocus OFF)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        
        # ⚠️ HARDCODED FOCUS VALUE
        # Note: This is usually between 0 and 255,, or a specific increment (like multiples of 5).
        # You will need to change this number until the image is perfectly sharp.
        FOCUS_VALUE = 40  
        cap.set(cv2.CAP_PROP_FOCUS, FOCUS_VALUE)

        if not cap.isOpened():
            on_fail()
            return

        cap.read() # Pre-read
        self.cap = cap
        self.is_running = True
        self.is_paused = False
        
        # Reset stats
        self.fps_frame_count = 0
        self.last_fps_time = time.time()
        self.frame_count = 0
        
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        on_success(width, height)

    def start_video(self, filepath):
        self.cap = cv2.VideoCapture(filepath)
        if not self.cap.isOpened():
            return None, None

        self.is_running = True
        self.is_video_file = True
        self.is_paused = False
        self.filename = filepath.replace("/", "\\").split("\\")[-1]
        
        self.fps_frame_count = 0
        self.last_fps_time = time.time()
        self.frame_count = 0
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    def stop(self):
        self.is_running = False
        self.is_video_file = False
        self.is_paused = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def open_settings_dialog(self):
        """Open the native DirectShow settings dialog on Windows."""
        if self.cap and self.is_running and platform.system() == "Windows":
            # Must be called from the same process, often will pause the capture thread
            self.cap.set(cv2.CAP_PROP_SETTINGS, 1)

    def _capture_loop(self):
        while self.is_running and self.cap and self.cap.isOpened():
            if self.is_video_file:
                # Delay for video to not run too fast in capture loop
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

    def read_frame(self):
        """Returns (frame, is_video_end). frame is None if no frame available."""
        if not self.is_running or not self.cap or self.is_paused:
            return None, False

        if self.is_video_file:
            ret, frame = self.cap.read()
            if not ret:
                return None, True # Video ended
            self._update_fps()
            return frame, False
        else:
            try:
                frame = self.frame_queue.get_nowait()
                self._update_fps()
                return frame, False
            except queue.Empty:
                return None, False

    def _update_fps(self):
        self.frame_count += 1
        self.fps_frame_count += 1
        now = time.time()
        elapsed = now - self.last_fps_time
        if elapsed >= 1.0:
            self.fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.last_fps_time = now
