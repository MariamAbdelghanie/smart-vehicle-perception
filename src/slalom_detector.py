from collections import deque
from typing import Optional

class SlalomDetector:
    def __init__(self, fps=25.0, min_switches=5, window_sec=4.0,
                 min_shift_px=30, min_total_disp=60, cooldown=6.0):
        self.fps         = fps
        self.min_sw      = min_switches
        self.win         = window_sec
        self.min_shift   = min_shift_px
        self.min_disp    = min_total_disp
        self.cooldown    = cooldown
        self._history    = deque()
        self._last_event = -999.0

    def update(self, frame_id: int, center_x: float) -> Optional[dict]:
        t = frame_id / self.fps
        self._history.append((t, center_x))
        cutoff = t - self.win
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()
        if len(self._history) < 6:
            return None
        if t - self._last_event < self.cooldown:
            return None
        pts = list(self._history)
        all_x = [p[1] for p in pts]
        if max(all_x) - min(all_x) < self.min_disp:
            return None
        switches = 0
        prev_dir = None
        for i in range(1, len(pts)):
            dx = pts[i][1] - pts[i-1][1]
            if abs(dx) < self.min_shift:
                continue
            d = "R" if dx > 0 else "L"
            if prev_dir and d != prev_dir:
                switches += 1
            prev_dir = d
        if switches >= self.min_sw:
            self._last_event = t
            conf = round(min(0.93, 0.55 + switches * 0.09), 2)
            return {"zaman_saniye": round(t,2), "kategori": "sofor_eylemi",
                    "etiket": "slalom", "confidence_score": conf}
        return None