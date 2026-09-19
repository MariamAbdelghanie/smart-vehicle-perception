import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional

NOSE_TIP    = 1
LEFT_CHEEK  = 234
RIGHT_CHEEK = 454


@dataclass
class HeadPoseEvent:
    frame_id: int
    zaman_saniye: float
    etiket: str
    confidence_score: float
    source: str


class HeadPoseDetector:
    def __init__(self,
                 fps: float = 25.0,
                 yaw_side_threshold: float = 0.28,
                 yaw_back_threshold: float = 0.55,
                 min_frames: int = 5,
                 face_lost_min_frames: int = 8,
                 cooldown_sec: float = 4.0):
        self.fps            = fps
        self.yaw_side       = yaw_side_threshold
        self.yaw_back       = yaw_back_threshold
        self.min_frames     = min_frames
        self.face_lost_min  = face_lost_min_frames
        self.cooldown_sec   = cooldown_sec

        self._side_count                = 0
        self._face_lost_count           = 0
        self._was_face_visible_recently = False
        self._last_event   = {"etrafa_bakinma": -999.0, "arkaya_bakma": -999.0}
        self._yaw_history  = deque(maxlen=15)

        self._fm = None
        self._init_mp()

    def _init_mp(self):
        try:
            import mediapipe as mp
            try:
                from mediapipe.python.solutions.face_mesh import FaceMesh
                cls = FaceMesh
            except Exception:
                cls = mp.solutions.face_mesh.FaceMesh
            self._fm = cls(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.35,
                min_tracking_confidence=0.35,
            )
            print("  [HeadPose] FaceMesh ready")
        except Exception as e:
            print(f"  [HeadPose] Disabled: {e}")

    def _yaw_ratio(self, landmarks, w: int) -> float:
        def x(i):
            return landmarks[i].x * w
        nose, lc, rc = x(NOSE_TIP), x(LEFT_CHEEK), x(RIGHT_CHEEK)
        span = max(rc - lc, 1e-6)
        return float((nose - (lc + rc) / 2.0) / span)

    def update(self,
               frame_id: int,
               driver_crop: Optional[np.ndarray],
               yolo_etrafa_cf: float = 0.0,
               yolo_arkaya_cf: float = 0.0) -> Optional[HeadPoseEvent]:
        current_time = frame_id / self.fps
        face_found   = False
        yaw          = 0.0

        if self._fm is not None and driver_crop is not None and driver_crop.size > 0:
            try:
                rgb = cv2.cvtColor(driver_crop, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                result = self._fm.process(rgb)
                if result.multi_face_landmarks:
                    face_found = True
                    yaw = self._yaw_ratio(result.multi_face_landmarks[0].landmark, w)
                    self._yaw_history.append(yaw)
            except Exception:
                pass

        if not face_found:
            self._face_lost_count += 1
        else:
            self._face_lost_count = 0
            self._was_face_visible_recently = True

        if (self._face_lost_count >= self.face_lost_min and
                self._was_face_visible_recently and
                current_time - self._last_event["arkaya_bakma"] >= self.cooldown_sec):
            self._last_event["arkaya_bakma"] = current_time
            self._face_lost_count = 0
            self._was_face_visible_recently = False
            conf   = round(min(0.85, 0.55 + yolo_arkaya_cf * 0.30), 2)
            source = "fusion" if yolo_arkaya_cf > 0.20 else "mediapipe"
            return HeadPoseEvent(frame_id, round(current_time, 2),
                                  "arkaya_bakma", conf, source)

        if (yolo_arkaya_cf > 0.45 and
                current_time - self._last_event["arkaya_bakma"] >= self.cooldown_sec):
            self._last_event["arkaya_bakma"] = current_time
            return HeadPoseEvent(frame_id, round(current_time, 2),
                                  "arkaya_bakma", round(yolo_arkaya_cf * 0.95, 2), "yolo")

        if face_found and abs(yaw) >= self.yaw_side:
            self._side_count += 1
        elif yolo_etrafa_cf > 0.25:
            self._side_count += 1
        else:
            self._side_count = max(0, self._side_count - 2)

        if (self._side_count >= self.min_frames and
                current_time - self._last_event["etrafa_bakinma"] >= self.cooldown_sec):
            self._last_event["etrafa_bakinma"] = current_time
            self._side_count = 0
            avg_yaw     = float(np.mean([abs(y) for y in self._yaw_history] or [self.yaw_side]))
            mp_strength = min(1.0, avg_yaw / max(self.yaw_back, 1e-6))
            conf        = round(min(0.92, 0.55 + mp_strength * 0.25 + yolo_etrafa_cf * 0.10), 2)
            source      = "fusion" if (face_found and yolo_etrafa_cf > 0.20) else (
                          "mediapipe" if face_found else "yolo")
            return HeadPoseEvent(frame_id, round(current_time, 2),
                                  "etrafa_bakinma", conf, source)

        return None

    def reset(self):
        self._side_count                = 0
        self._face_lost_count           = 0
        self._was_face_visible_recently = False
        self._yaw_history.clear()


def make_headpose_json(event: HeadPoseEvent) -> dict:
    return {
        "zaman_saniye"    : event.zaman_saniye,
        "kategori"        : "sofor_eylemi",
        "etiket"          : event.etiket,
        "confidence_score": event.confidence_score,
    }