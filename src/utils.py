import re
import cv2
import numpy as np
from collections import Counter

def enhance_frame(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def classify_vehicle_color(crop: np.ndarray) -> str:
    if crop is None or crop.size == 0:
        return "beyaz"
    h, w = crop.shape[:2]
    if h < 10 or w < 10:
        return "beyaz"
    region = crop[int(0.35*h):int(0.78*h), int(0.12*w):int(0.88*w)]
    if region.size == 0:
        region = crop
    flat = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    H, S, V = flat[:,0], flat[:,1], flat[:,2]
    mask = (V > 15) & (V < 248)
    if mask.sum() < 30:
        mask = np.ones(len(V), dtype=bool)
    H, S, V = H[mask], S[mask], V[mask]
    mv, ms = float(np.median(V)), float(np.median(S))
    if mv < 65:
        return "siyah"
    if ms < 42:
        return "beyaz" if mv > 170 else "gri"
    cm = S > 42
    if cm.sum() < 10:
        cm = np.ones(len(S), dtype=bool)
    mh = float(np.median(H[cm]))
    if mh < 6 or mh >= 170:
        return "kirmizi"
    elif mh < 18:
        return "kahverengi" if mv < 130 else "turuncu"
    elif mh < 33:
        return "sari"
    elif mh < 85:
        return "yesil"
    elif mh < 140:
        return "mavi"
    else:
        return "kirmizi"

PLATE_REGEX = re.compile(
    r"^(0[1-9]|[1-7][0-9]|8[01])"
    r"(([A-Z])(\d{4,5})|([A-Z]{2})(\d{3,4})|([A-Z]{3})(\d{2,3}))$"
)

SHAPE_TYPE = [
    (2.5, 99, "minibus"),
    (1.8, 2.5, "suv"),
    (1.55, 1.8, "sedan"),
    (1.25, 1.55, "hatchback"),
    (0.0, 1.25, "minibus"),
]

def shape_type_from_bbox(bbox):
    x1, y1, x2, y2 = bbox
    r = max(x2-x1, 1) / max(y2-y1, 1)
    for mn, mx, lbl in SHAPE_TYPE:
        if mn <= r < mx:
            return lbl
    return "sedan"

def smart_plate_correct(raw: str):
    raw = re.sub(r"[^A-Z0-9]", "", raw.upper())
    if len(raw) < 5:
        return [raw]
    city = raw[:2]
    rest = raw[2:]
    letter_end = 0
    for i, ch in enumerate(rest):
        if ch.isdigit() and i >= 1:
            letter_end = i
            break
    if letter_end == 0:
        letter_end = min(3, len(rest))
    letters = rest[:letter_end]
    digits = rest[letter_end:]
    lf = {"0":"O","1":"I","5":"S","8":"B","6":"G"}
    d0 = {"O":"0","Q":"0","I":"1","L":"1","B":"8","S":"5","Z":"2","G":"6"}
    d8 = {"O":"8","Q":"0","I":"1","L":"1","B":"8","S":"5","Z":"2","G":"6"}
    fl = "".join(lf.get(ch, ch) for ch in letters)
    dd0 = "".join(d0.get(ch, ch) for ch in digits)
    dd8 = "".join(d8.get(ch, ch) for ch in digits)
    
    candidates = {raw, city+letters+dd0, city+letters+dd8, city+fl+dd0, city+fl+dd8}
    
    for cand in list(candidates):
        if "0" in cand[2:]:
            cand_fixed = cand.replace("0", "8")
            if PLATE_REGEX.match(cand_fixed):
                candidates.add(cand_fixed)
    
    for cand in list(candidates):
        if "I" in cand[2:]:
            cand_fixed = cand.replace("I", "T")
            if PLATE_REGEX.match(cand_fixed):
                candidates.add(cand_fixed)
    
    return list(candidates)

def normalize_plate(raw: str) -> str:
    for cand in smart_plate_correct(raw):
        if PLATE_REGEX.match(cand):
            return cand
    return "tespit_edilemedi"

def best_vehicle_type(type_votes, type_counts, shape_list):
    valid = {k: v for k, v in type_counts.items() if v >= 2}
    if valid:
        avg = {k: type_votes[k]/type_counts[k] for k in valid}
        winner = max(valid, key=lambda k: (valid[k], avg.get(k, 0)))
        if winner in ["minibus", "panelvan"]:
            return "sedan"
        if winner in ["sedan", "hatchback"] and shape_list:
            asp = Counter(shape_list).most_common(1)[0][0]
            if asp in ["sedan", "hatchback"]:
                return asp
        return winner
    if type_counts:
        return max(type_counts, key=type_counts.get)
    return Counter(shape_list).most_common(1)[0][0] if shape_list else "sedan"

def aggregate_vehicle_info(type_votes, type_counts, shape_list,
                           color_votes, plate_list, conf_list):
    tip = best_vehicle_type(type_votes, type_counts, shape_list)
    renk = Counter(color_votes).most_common(1)[0][0] if color_votes else "beyaz"
    plaka = "tespit_edilemedi"
    pc = 0.0
    if plate_list:
        hits = []
        for dc, txt in plate_list:
            p = normalize_plate(txt)
            if p != "tespit_edilemedi":
                hits.append((p, dc))
        if hits:
            cm = Counter(h[0] for h in hits)
            plaka = cm.most_common(1)[0][0]
            pc = float(np.mean([h[1] for h in hits if h[0] == plaka]))
    tc = float(np.mean(conf_list)) if conf_list else 0.0
    cc = 1.0 if color_votes else 0.5
    confs = [c for c in [tc, cc, pc] if c > 0]
    overall = round(float(np.mean(confs)), 3) if confs else 0.0
    return {"tip": tip, "plaka": plaka, "renk": renk,
            "confidence_score": max(0.0, min(1.0, overall))}

def format_output(video_id: str, vehicle_info: dict, detections: list) -> dict:
    return {"video_id": video_id, "arac_bilgisi": vehicle_info, "tespitler": detections}