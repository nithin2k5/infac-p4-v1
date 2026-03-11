import os
import cv2
from ultralytics import YOLO

class InferenceEngine:
    def __init__(self, model_path="weights-2.pt"):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        try:
            self.model = YOLO(self.model_path)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            return False

    def is_loaded(self):
        return self.model is not None

    def infer(self, frame, confidence_threshold=0.40, roi=None):
        if not self.is_loaded():
            return []

        # ROI Cropping functionality
        if roi is not None:
            rx, ry, rw, rh = roi
            img_h, img_w = frame.shape[:2]
            
            # Ensure ROI is within bounds
            y1 = max(0, int(ry))
            y2 = min(img_h, int(ry + rh))
            x1 = max(0, int(rx))
            x2 = min(img_w, int(rx + rw))
            
            inference_frame = frame[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1
        else:
            inference_frame = frame
            offset_x, offset_y = 0, 0

        # Run inference
        results = self.model(inference_frame, conf=confidence_threshold, iou=0.45, verbose=False)
        result = results[0]

        raw_predictions = []
        if len(result.boxes) > 0:
            for box in result.boxes:
                # Ultralytics boxes: xywh (center X, center Y, width, height) format
                bx, by, bw, bh = box.xywh[0].cpu().numpy()
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                class_name = result.names[cls_id]

                # --- Strict Detection Rules removed ---

                # Adjust coordinates back to the original full-frame space
                raw_predictions.append({
                    "x": float(bx) + offset_x,
                    "y": float(by) + offset_y,
                    "width": float(bw),
                    "height": float(bh),
                    "class": class_name.lower(),
                    "confidence": confidence
                })
                
        return self.apply_nms(raw_predictions, iou_threshold=0.45)

    def apply_nms(self, predictions, iou_threshold=0.45):
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
