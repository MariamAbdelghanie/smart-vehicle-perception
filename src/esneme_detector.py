import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional

UPPER_LIP   = [13, 312, 82]
LOWER_LIP   = [14, 311, 87]
LEFT_CORNER = 78
RIGHT_CORNER= 308

@dataclass
class EsnemeEvent:
    frame_id: int
    zaman_saniye: float
    confidence_score: float

class EsnemeDetector:
    def __init__(self, fps=25.0, mar_threshold=0.47, min_frames=7, cooldown_sec=5.0):
        self.fps = fps
        self.mar_threshold = mar_threshold
        self.min_frames = min_frames
        self.cooldown_sec = cooldown_sec
        self._open_count = 0
        self._last_event = -999.0
        self._mar_history = deque(maxlen=30)
        self._face_mesh = None
        self._init_mp()

    def _init_mp(self):
        try:
            import mediapipe as mp
            try:
                from mediapipe.python.solutions.face_mesh import FaceMesh
                cls = FaceMesh
            except:
                cls = mp.solutions.face_mesh.FaceMesh
            self._face_mesh = cls(
                static_image_mode=False, max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.35,
                min_tracking_confidence=0.35,
            )
            print("  [Esneme] MediaPipe hazir")
        except Exception as e:
            print(f"  [Esneme] Devre disi: {e}")

    def _mar(self, lm, h, w):
        def pt(i): return np.array([lm[i].x*w, lm[i].y*h])
        v = sum(np.linalg.norm(pt(UPPER_LIP[i])-pt(LOWER_LIP[i])) for i in range(3))
        d = np.linalg.norm(pt(LEFT_CORNER)-pt(RIGHT_CORNER))
        return float(v/(2.0*d)) if d > 1e-6 else 0.0

    def update(self, frame_id: int, driver_crop: np.ndarray,
               yolo_cf: float = 0.0) -> Optional[EsnemeEvent]:
        t = frame_id / self.fps
        if t - self._last_event < self.cooldown_sec:
            return None
        mar = 0.0
        mp_open = False
        if self._face_mesh is not None and driver_crop is not None and driver_crop.size > 0:
            try:
                rgb = cv2.cvtColor(driver_crop, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                result = self._face_mesh.process(rgb)
                if result.multi_face_landmarks:
                    mar = self._mar(result.multi_face_landmarks[0].landmark, h, w)
                    mp_open = mar >= self.mar_threshold
            except:
                pass
        self._mar_history.append(mar)
        if mp_open:
            self._open_count += 1
        elif yolo_cf > 0.25:
            self._open_count += 1
        else:
            self._open_count = max(0, self._open_count - 2)
        if self._open_count < self.min_frames:
            return None
        avg = float(np.mean([m for m in self._mar_history if m > 0] or [self.mar_threshold]))
        if mp_open and yolo_cf > 0.25:
            cf = round(min(0.97, 0.70+(avg-self.mar_threshold)/0.3*0.20+yolo_cf*0.07), 2)
        elif mp_open:
            cf = round(min(0.90, 0.58+(avg-self.mar_threshold)/0.3*0.28), 2)
        else:
            cf = round(max(0.52, yolo_cf*0.92), 2)
        self._last_event = t
        self._open_count = 0
        return EsnemeEvent(frame_id, round(t, 2), cf)

def make_esneme_json(event: EsnemeEvent) -> dict:
    return {"zaman_saniye": event.zaman_saniye, "kategori": "sofor_eylemi",
            "etiket": "esneme", "confidence_score": event.confidence_score}