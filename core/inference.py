import cv2
import base64
import requests
import numpy as np


# ═════════════════════════════════════════════════════════
#  ROBOFLOW HOSTED INFERENCE
# ═════════════════════════════════════════════════════════

ROBOFLOW_API_KEY = "cf7X6JDorlmwhw6aqKUK"
ROBOFLOW_MODEL = "p4-kbph4"
ROBOFLOW_VERSION = "9"
ROBOFLOW_URL = (
    f"https://detect.roboflow.com/{ROBOFLOW_MODEL}/{ROBOFLOW_VERSION}"
)


class InferenceEngine:
    """Inference engine using Roboflow hosted API."""

    def __init__(self, model_path=None):
        # model_path kept for API compatibility but unused with hosted inference
        self.api_key = ROBOFLOW_API_KEY
        self.api_url = ROBOFLOW_URL
        self._loaded = False

    def set_model_version(self, version):
        """Update inference model version and reload."""
        self.api_url = f"https://detect.roboflow.com/{ROBOFLOW_MODEL}/{version}"
        self._loaded = False
        return self.load_model()

    def load_model(self):
        """Verify API connectivity by sending a small test request."""
        try:
            # Create a tiny test image to verify the API key works
            test_img = np.zeros((64, 64, 3), dtype=np.uint8)
            _, buf = cv2.imencode(".jpg", test_img)
            img_b64 = base64.b64encode(buf).decode("utf-8")

            resp = requests.post(
                self.api_url,
                params={
                    "api_key": self.api_key,
                    "confidence": 50,
                },
                data=img_b64,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=15,
            )
            if resp.status_code == 200:
                self._loaded = True
                print("Roboflow API connected successfully.")
                return True
            else:
                print(f"Roboflow API error: {resp.status_code} — {resp.text}")
                self._loaded = False
                return False
        except Exception as e:
            print(f"Roboflow API connection error: {e}")
            self._loaded = False
            return False

    def is_loaded(self):
        return self._loaded

    def infer(self, frame, confidence_threshold=0.40, roi=None):
        """Run inference via Roboflow hosted API."""
        if not self._loaded:
            return []

        # ROI cropping
        if roi is not None:
            rx, ry, rw, rh = roi
            img_h, img_w = frame.shape[:2]
            y1 = max(0, int(ry))
            y2 = min(img_h, int(ry + rh))
            x1 = max(0, int(rx))
            x2 = min(img_w, int(rx + rw))
            inference_frame = frame[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1
        else:
            inference_frame = frame
            offset_x, offset_y = 0, 0

        # Encode frame as JPEG → base64
        _, buf = cv2.imencode(".jpg", inference_frame)
        img_b64 = base64.b64encode(buf).decode("utf-8")

        try:
            resp = requests.post(
                self.api_url,
                params={
                    "api_key": self.api_key,
                    "confidence": int(confidence_threshold * 100),
                },
                data=img_b64,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=10,
            )
            if resp.status_code != 200:
                print(f"Roboflow inference error: {resp.status_code}")
                return []

            data = resp.json()
        except Exception as e:
            print(f"Roboflow inference request failed: {e}")
            return []

        # Parse Roboflow response → unified prediction format
        raw_predictions = []
        for pred in data.get("predictions", []):
            raw_predictions.append({
                "x": float(pred["x"]) + offset_x,
                "y": float(pred["y"]) + offset_y,
                "width": float(pred["width"]),
                "height": float(pred["height"]),
                "class": pred["class"].lower(),
                "confidence": float(pred["confidence"]),
            })

        return self.apply_nms(raw_predictions, iou_threshold=0.45)

    # ─────────────────────────────────────────────────────
    #  NMS (unchanged)
    # ─────────────────────────────────────────────────────

    def apply_nms(self, predictions, iou_threshold=0.45):
        """Apply Non-Maximum Suppression to filter overlapping boxes."""
        if not predictions:
            return []

        by_class = {}
        for p in predictions:
            by_class.setdefault(p["class"], []).append(p)

        final_preds = []

        for cls_name, preds in by_class.items():
            preds.sort(key=lambda x: x["confidence"], reverse=True)
            keep = []
            for p in preds:
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
