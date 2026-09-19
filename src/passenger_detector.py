from collections import defaultdict
from typing import Optional, Tuple
import numpy as np

class PassengerDetector:
    """
    Classifies detected persons into passenger zones based on
    their bbox position inside the vehicle ROI.

    Turkey = RIGHT-hand drive → driver is on the RIGHT side (cx >= 0.50)

    Zone map (viewed from front camera):
    ┌─────────────────┬─────────────────┐
    │   on_koltuk     │   SOFOR ZONE    │  cy < FRONT_Y
    │  (front-left)   │  (front-right)  │
    ├─────────────────┼─────────────────┤
    │  arka_koltuk_1  │  arka_koltuk_2  │  cy >= FRONT_Y
    │  (rear-left)    │  (rear-right)   │
    └─────────────────┴─────────────────┘
         cx < 0.50         cx >= 0.50
    """

    FRONT_Y      = 0.52
    DRIVER_X     = 0.50
    MIN_FRAMES   = 2
    EMIT_COOLDOWN= 4.0

    def __init__(self):
        self._zone_count  = defaultdict(int)
        self._zone_conf   = defaultdict(list)
        self._last_logged = {}

    def classify(self, bbox_in_roi, roi_shape) -> Tuple[Optional[str], float]:
        x1, y1, x2, y2 = bbox_in_roi
        rh, rw = roi_shape[:2]
        cx = ((x1+x2)/2.0) / max(rw, 1)
        cy = ((y1+y2)/2.0) / max(rh, 1)
        is_front = cy < self.FRONT_Y
        is_right = cx >= self.DRIVER_X
        if is_front and is_right:
            return None, 0.0
        cx_dist = abs(cx - self.DRIVER_X)
        cy_dist = abs(cy - self.FRONT_Y)
        conf = round(min(0.92, 0.55 + cx_dist*0.5 + cy_dist*0.4), 2)
        if is_front and not is_right:
            return "on_koltuk", conf
        if not is_front and not is_right:
            return "arka_koltuk_1", conf
        return "arka_koltuk_2", conf

    def update(self, bbox_in_roi, roi_shape, base_conf, current_time):
        zone, conf = self.classify(bbox_in_roi, roi_shape)
        if zone is None:
            return None, 0.0
        self._zone_count[zone] += 1
        self._zone_conf[zone].append(base_conf)
        if self._zone_count[zone] >= self.MIN_FRAMES:
            last = self._last_logged.get(zone, -999.0)
            if current_time - last >= self.EMIT_COOLDOWN:
                self._last_logged[zone] = current_time
                emit_conf = round(min(0.92, float(np.mean(self._zone_conf[zone]))*1.1), 2)
                return zone, emit_conf
        return None, 0.0

    def flush(self, current_time):
        events = []
        for zone, cnt in self._zone_count.items():
            if cnt >= self.MIN_FRAMES and zone not in self._last_logged:
                conf = round(min(0.88, float(np.mean(self._zone_conf[zone]))*1.05), 2)
                events.append({"zaman_saniye": round(current_time, 2),
                               "kategori": "yolcular", "etiket": zone,
                               "confidence_score": conf})
                self._last_logged[zone] = current_time
        return events