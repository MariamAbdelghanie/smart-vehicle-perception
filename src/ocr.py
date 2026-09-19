import re
import cv2
import numpy as np
from collections import Counter
from src.utils import smart_plate_correct, PLATE_REGEX

_reader = None

def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        try:
            _reader = easyocr.Reader(["en"], gpu=True, verbose=False)
        except Exception:
            _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        print("  [OCR] EasyOCR hazir")
    return _reader

def _deskew(crop):
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.dilate(cv2.Canny(gray, 50, 150), np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return crop
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < crop.shape[0] * crop.shape[1] * 0.15:
            return crop
        angle = cv2.minAreaRect(largest)[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) < 3.0:
            return crop
        h, w = crop.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return crop

def _prep(crop, mode):
    h, w = crop.shape[:2]
    sc = max(2, min(8, int(600 // max(h, 1))))
    big = cv2.resize(crop, (w * sc, h * sc), interpolation=cv2.INTER_LANCZOS4)
    g = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    if mode == "otsu":
        bl = cv2.GaussianBlur(g, (3, 3), 0)
        _, out = cv2.threshold(bl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(out) > 200:
            out = cv2.bitwise_not(out)
        return out
    elif mode == "clahe":
        out = cv2.createCLAHE(2.5, (8, 8)).apply(g)
        bl = cv2.GaussianBlur(out, (0, 0), 1.0)
        return cv2.addWeighted(out, 1.6, bl, -0.6, 0)
    return g

def read_plate(crop: np.ndarray):
    if crop is None or crop.size == 0:
        return "", 0.0
    try:
        reader = _get_reader()
        deskewed = _deskew(crop)
        variants = [crop] if np.array_equal(deskewed, crop) else [crop, deskewed]

        raw_results = []
        for variant in variants:
            for mode in ["otsu", "clahe", "raw"]:
                try:
                    img = _prep(variant, mode)
                    res = reader.readtext(img, detail=1, paragraph=False)
                    if not res:
                        continue
                    raw = re.sub(r"[^A-Z0-9]", "", "".join(x[1] for x in res).upper())
                    if not raw:
                        continue
                    conf = float(np.mean([x[2] for x in res]))
                    raw_results.append((raw, conf))
                except Exception:
                    continue

        if not raw_results:
            return "", 0.0

        valid_candidates = []
        for raw, conf in raw_results:
            for cand in smart_plate_correct(raw):
                if PLATE_REGEX.match(cand):
                    valid_candidates.append((cand, conf))
                    break

        if valid_candidates:
            cnt = Counter(c[0] for c in valid_candidates)
            best = cnt.most_common(1)[0][0]
            cf = float(np.mean([c[1] for c in valid_candidates if c[0] == best]))
            return best, cf

        cnt = Counter(r[0] for r in raw_results)
        best = cnt.most_common(1)[0][0]
        cf = float(np.mean([r[1] for r in raw_results if r[0] == best]))

        best = best.replace("0", "8").replace("O", "0")
        best = best.replace("I", "T").replace("L", "1")
        best = best.replace("5", "8").replace("S", "5")
        best = best.replace("B", "8").replace("G", "6")
        best = best.replace("2", "Z").replace("Z", "2")

        return best, cf

    except Exception as e:
        print(f"  [OCR] Hata: {e}")
        return "", 0.0