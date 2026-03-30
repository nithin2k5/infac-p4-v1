class InspectionManager:
    def __init__(self):
        self.total_inspected = 0
        self.total_defects = 0
        self.all_confidences = []
        
        self.required_solders = 2  # set by app based on detection type

        # For auto-inspection state
        self.auto_inspect_enabled = False
        self.state = "WAITING" # WAITING, INSPECTING, COOLDOWN
        self.inspection_frames = 0
        self.max_solder_seen = 0
        self.solder_history = []
        
        # Callbacks
        self.on_log_result = None # signature: (status, detail, color, confidence)
        self.on_stats_update = None # empty signature

    def reset_stats(self):
        self.total_inspected = 0
        self.total_defects = 0
        self.all_confidences.clear()
        self.reset_auto_state()
        if self.on_stats_update:
            self.on_stats_update()

    def get_stats(self):
        pass_rate = 100.0
        if self.total_inspected > 0:
            pass_rate = (1 - self.total_defects / self.total_inspected) * 100
            
        avg_conf = 0.0
        if self.all_confidences:
            avg_conf = sum(self.all_confidences) / len(self.all_confidences)
            
        return {
            "inspected": self.total_inspected,
            "defects": self.total_defects,
            "pass_rate": pass_rate,
            "avg_conf": avg_conf
        }

    def process_test_snapshot(self, predictions, filename="Test snapshot", timestamp=None):
        """Processes a single manual snapshot result."""
        pcb_detected = any(p["class"].lower() == "pcb" for p in predictions)
        solder_preds = [p for p in predictions if p["class"].lower() == "solder"]
        solder_count = len(solder_preds)
        req = self.required_solders

        self.total_inspected += 1

        if timestamp is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        max_conf = max((p['confidence'] for p in solder_preds), default=0)

        if pcb_detected and solder_count >= req:
            status_text = "✅ PASS"
            detail = f"{timestamp}  •  {solder_count} solder(s)  •  {filename}"
            color = "#28a745" # Colors.SUCCESS equivalent
            conf_str = f"{max_conf:.0%}" if solder_preds else "--"
        else:
            self.total_defects += 1
            status_text = "❌ NG"
            if not pcb_detected:
                detail = f"{timestamp}  •  No PCB detected  •  {filename}"
            else:
                missing = req - solder_count
                detail = f"{timestamp}  •  {missing} solder(s) missing  •  {filename}"
            color = "#dc3545" # Colors.DANGER
            conf_str = f"{solder_count}/{req}"

        for pred in solder_preds:
            self.all_confidences.append(pred["confidence"])

        if self.on_stats_update:
            self.on_stats_update()

        if self.on_log_result:
            self.on_log_result(status_text, detail, color, conf_str)
            
        return pcb_detected and solder_count >= req

    # Continuous Auto-Inspection Logic
    def reset_auto_state(self):
        self.state = "WAITING"
        self.inspection_frames = 0
        self.max_solder_seen = 0
        self.solder_history.clear()

    def process_live_frame(self, predictions):
        """Evaluates live frames for stabilizing visual results and Auto-Inspection mode."""
        pcb_detected = any(p["class"].lower() == "pcb" for p in predictions)
        solder_preds = [p for p in predictions if p["class"].lower() == "solder"]
        solder_count = len(solder_preds)

        # Keep a rolling window of the last 5 frames to prevent flickering in UI
        self.solder_history.append(solder_count)
        if len(self.solder_history) > 5:
            self.solder_history.pop(0)

        effective_solder_count = max(self.solder_history) if self.solder_history else 0

        # Auto-Inspection Logic
        if self.auto_inspect_enabled:
            if self.state == "WAITING":
                if pcb_detected or effective_solder_count > 0:
                    self.state = "INSPECTING"
                    self.inspection_frames = 0
                    self.max_solder_seen = effective_solder_count
            elif self.state == "INSPECTING":
                self.inspection_frames += 1
                self.max_solder_seen = max(self.max_solder_seen, effective_solder_count)
                
                # If we've looked at it for ~15 frames (approx 0.5 sec at 30 fps)
                if self.inspection_frames >= 15:
                    self._trigger_auto_log(self.max_solder_seen, predictions)
                    self.state = "COOLDOWN"
                    self.inspection_frames = 0
            elif self.state == "COOLDOWN":
                # Wait until PCB is completely gone before allowing another inspection
                if not pcb_detected and effective_solder_count == 0:
                    self.inspection_frames += 1
                    # Require 15 completely empty frames to confirm it left
                    if self.inspection_frames >= 15:
                        self.reset_auto_state()
                else:
                    self.inspection_frames = 0 # reset empty frame counter if seen again

        return pcb_detected, effective_solder_count

    def _trigger_auto_log(self, max_solder_count, predictions):
        self.total_inspected += 1
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        pcb_detected = any(p["class"].lower() == "pcb" for p in predictions)
        solder_preds = [p for p in predictions if p["class"].lower() == "solder"]
        max_conf = max((p['confidence'] for p in solder_preds), default=0)
        req = self.required_solders

        if pcb_detected and max_solder_count >= req:
            status_text = "✅ PASS"
            detail = f"{timestamp}  •  {max_solder_count} solder(s)  •  Auto"
            color = "#28a745"
            conf_str = f"{max_conf:.0%}" if solder_preds else "--"
        else:
            self.total_defects += 1
            status_text = "❌ NG"
            if not pcb_detected:
                detail = f"{timestamp}  •  No PCB detected  •  Auto"
            else:
                missing = req - max_solder_count
                detail = f"{timestamp}  •  {missing} solder(s) missing  •  Auto"
            color = "#dc3545"
            conf_str = f"{max_solder_count}/{req}"

        for pred in solder_preds:
            self.all_confidences.append(pred["confidence"])

        if self.on_stats_update:
            self.on_stats_update()
        if self.on_log_result:
            self.on_log_result(status_text, detail, color, conf_str)
