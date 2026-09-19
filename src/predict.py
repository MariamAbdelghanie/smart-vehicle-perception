import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import torch
import numpy as np
from collections import defaultdict

from src.utils import (enhance_frame, classify_vehicle_color,
                       shape_type_from_bbox, aggregate_vehicle_info, format_output)
from src.ocr import read_plate
from src.esneme_detector import EsnemeDetector, make_esneme_json
from src.slalom_detector import SlalomDetector
from src.passenger_detector import PassengerDetector
from src.teknocan_detector import scan_teknocan

VEHICLE_TYPES = ["sedan","suv","hatchback","pickup","minibus","panelvan","kamyon"]
TYPE_IDS      = set(range(7))
PLAKA_ID      = 16
COLOR_IDS     = set(range(7,16))
COLOR_NAMES   = ["beyaz","siyah","gri","kirmizi","mavi","sari","yesil","turuncu","kahverengi"]

YOLO11S_CLASSES = [
    "arkaya_bakma","esneme","sigara_icme","su_icme","telefonla_konusma",
    "slalom","etrafa_bakinma","emniyet_kemeri_ihlali",
    "teknocan","bilgisayar","arka_koltuk_1","arka_koltuk_2","on_koltuk",
]
SOFOR_IDS = set(range(8))
NESNE_IDS = {8,9}
YOLCU_IDS = {10,11,12}
ALGO_IDS  = {5,10,11}
KATEGORI_MAP = {
    **{i:"sofor_eylemi" for i in SOFOR_IDS},
    **{i:"nesneler"     for i in NESNE_IDS},
    **{i:"yolcular"     for i in YOLCU_IDS},
}

CONF_THRESH = {
    "esneme":0.28,
    "emniyet_kemeri_ihlali":0.28,
    "telefonla_konusma":0.01,
    "sigara_icme":0.30,
    "su_icme":0.30,
    "arkaya_bakma":0.15,
    "etrafa_bakinma":0.15,
    "teknocan":0.08,
    "bilgisayar":0.22,
    "default":0.20,
}

HEAD_TURN_CLASSES = {0: "arkaya_bakma", 6: "etrafa_bakinma"}

_SMOKE_LO = np.array([0, 0, 180], dtype=np.uint8)
_SMOKE_HI = np.array([180, 38, 255], dtype=np.uint8)
_TIP_LO = np.array([5, 150, 80], dtype=np.uint8)
_TIP_HI = np.array([25, 255, 255], dtype=np.uint8)
_WATER_LO = np.array([90, 10, 175], dtype=np.uint8)
_WATER_HI = np.array([130, 80, 255], dtype=np.uint8)
_BRIGHT_LO = np.array([0, 0, 170], dtype=np.uint8)
_BRIGHT_HI = np.array([180, 55, 255], dtype=np.uint8)
_PHONE_LO = np.array([0, 0, 80], dtype=np.uint8)
_PHONE_HI = np.array([180, 80, 200], dtype=np.uint8)

def _disambiguate_sigara_su(roi, bbox, cid, yolo_cf):
    if roi is None or roi.size == 0:
        return cid, yolo_cf
    x1,y1,x2,y2 = [int(v) for v in bbox]
    h,w = roi.shape[:2]
    x1,y1 = max(0,x1-12), max(0,y1-12)
    x2,y2 = min(w,x2+12), min(h,y2+12)
    crop = roi[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
        return cid, yolo_cf
    bh,bw = crop.shape[:2]
    aspect = bh / max(bw, 1)
    n = bh * bw
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    smoke_r = np.count_nonzero(cv2.inRange(hsv, _SMOKE_LO, _SMOKE_HI)) / max(n, 1)
    tip_r = np.count_nonzero(cv2.inRange(hsv, _TIP_LO, _TIP_HI)) / max(n, 1)
    water_r = np.count_nonzero(cv2.inRange(hsv, _WATER_LO, _WATER_HI)) / max(n, 1)
    bright_r = np.count_nonzero(cv2.inRange(hsv, _BRIGHT_LO, _BRIGHT_HI)) / max(n, 1)
    phone_r = np.count_nonzero(cv2.inRange(hsv, _PHONE_LO, _PHONE_HI)) / max(n, 1)
    score_sigara = tip_r * 7.0 + smoke_r * 2.5
    score_su = water_r * 3.5 + bright_r * 1.5 + (0.4 if aspect > 1.3 else 0.0)
    score_phone = phone_r * 3.0
    if score_phone > 0.12 and score_phone > score_sigara:
        return 4, yolo_cf
    if score_sigara > 0.06 and score_sigara > score_su * 1.8 and score_sigara > score_phone * 1.5:
        new_cf = round(min(1.0, yolo_cf + 0.05), 3)
        return 2, new_cf
    if score_su > 0.06 and score_su > score_sigara * 1.5:
        new_cf = round(min(1.0, yolo_cf + 0.04), 3)
        return 3, new_cf
    return cid, yolo_cf

def _head_pose_from_bbox(bbox_in_roi, roi_shape):
    x1,y1,x2,y2 = bbox_in_roi
    rh,rw = roi_shape[:2]
    cx = ((x1+x2)/2.0)/max(rw,1)
    cy = ((y1+y2)/2.0)/max(rh,1)
    if cy > 0.60:
        return "arkaya_bakma"
    if cx < 0.25 or cx > 0.75:
        return "etrafa_bakinma"
    return None

def _load_yolo(path: str):
    from ultralytics import YOLO
    candidates = [path,
                  path.replace("/app/models/","/app/weights/"),
                  path.replace("/app/weights/","/app/models/")]
    for p in candidates:
        if os.path.exists(p):
            print(f"  [YOLO] {p}")
            return YOLO(p)
    raise FileNotFoundError(f"Agirlik bulunamadi: {path}")

class _Buf:
    def __init__(self, cd=2.0):
        self._l = {}; self._cd = cd
    def ok(self, lbl, t, cd=None):
        c = cd or self._cd
        if t - self._l.get(lbl,-999.) >= c:
            self._l[lbl]=t; return True
        return False

def run_inference(video_path: str, weights_detection: str, weights_cabin: str) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [Device] {device.upper()}")

    model_m = _load_yolo(weights_detection)
    model_s = _load_yolo(weights_cabin)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video acilamadi: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = max(1, int(round(fps / 6)))
    max_frames = 600
    print(f"  [Video] FPS={fps:.1f} Toplam={total} Her {frame_skip}. kare")

    esneme_det = EsnemeDetector(fps=fps)
    slalom_det = SlalomDetector(fps=fps)
    pax_det = PassengerDetector()
    buf = _Buf()

    type_votes = defaultdict(float)
    type_counts = defaultdict(int)
    shape_list = []
    color_votes = []
    plate_list = []
    conf_list = []
    detections = []
    last_t = 0.0
    fi = 0
    sc = 0

    while sc < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        fi += 1
        if fi % frame_skip != 0:
            continue
        sc += 1
        t = round(fi / fps, 2)
        last_t = t
        fh, fw = frame.shape[:2]
        eframe = enhance_frame(frame)
        vbox = None

        try:
            res_m = model_m(eframe, imgsz=480, conf=0.20, iou=0.45,
                            verbose=False, device=device)
            for r in res_m:
                for box in r.boxes:
                    cid = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    x1,y1=max(0,x1), max(0,y1)
                    x2,y2=min(fw,x2), min(fh,y2)
                    crop = frame[y1:y2,x1:x2].copy() if x2>x1 and y2>y1 else None

                    if cid in TYPE_IDS and conf >= 0.28:
                        lbl = VEHICLE_TYPES[cid]
                        type_votes[lbl] += conf
                        type_counts[lbl] += 1
                        conf_list.append(conf)
                        if crop is not None:
                            color_votes.append(classify_vehicle_color(crop))
                        vbox = (x1,y1,x2,y2)
                        shape_list.append(shape_type_from_bbox(vbox))
                        ev = slalom_det.update(fi, (x1+x2)/2.0)
                        if ev and buf.ok("slalom", t):
                            detections.append(ev)

                    elif cid in COLOR_IDS and conf >= 0.30:
                        color_votes.append(COLOR_NAMES[cid-7])

                    elif cid == PLAKA_ID and crop is not None and conf >= 0.12:
                        txt, _ = read_plate(crop)
                        if txt:
                            plate_list.append((conf, txt))
        except Exception as e:
            print(f"  [M err] {e}")
            continue

        tkc = scan_teknocan(frame, vbox)
        if tkc > 0.12 and buf.ok("teknocan", t, cd=2.5):
            detections.append({"zaman_saniye":t,"kategori":"nesneler",
                               "etiket":"teknocan","confidence_score":tkc})

        roi = frame[vbox[1]:vbox[3], vbox[0]:vbox[2]] if vbox else frame
        if roi is None or roi.size == 0:
            continue
        eroi = enhance_frame(roi)

        try:
            res_s = model_s(eroi, imgsz=320, conf=0.01, iou=0.45,
                            verbose=False, device=device)
            yolo_esneme_cf = 0.0

            for r in res_s:
                for box in r.boxes:
                    cid = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1,y1,x2,y2 = map(int, box.xyxy[0])

                    if cid in ALGO_IDS:
                        continue

                    lbl = YOLO11S_CLASSES[cid]
                    kat = KATEGORI_MAP.get(cid, "sofor_eylemi")
                    thresh = CONF_THRESH.get(lbl, CONF_THRESH["default"])

                    if cid == 1:
                        yolo_esneme_cf = max(yolo_esneme_cf, conf)
                        continue

                    if conf < thresh:
                        continue

                    if cid == 4:
                        if buf.ok(lbl, t):
                            detections.append({"zaman_saniye":t,"kategori":kat,
                                               "etiket":lbl,"confidence_score":round(conf,3)})
                        continue

                    if cid in {2, 3}:
                        new_cid, cf = _disambiguate_sigara_su(roi, (x1,y1,x2,y2), cid, conf)
                        if new_cid == 4:
                            lbl = "telefonla_konusma"
                            kat = "sofor_eylemi"
                            conf = cf
                        else:
                            lbl = YOLO11S_CLASSES[new_cid]
                            kat = KATEGORI_MAP.get(new_cid, "sofor_eylemi")
                            conf = cf
                        if conf < CONF_THRESH.get(lbl, 0.20):
                            continue
                        if buf.ok(lbl, t):
                            detections.append({"zaman_saniye":t,"kategori":kat,
                                               "etiket":lbl,"confidence_score":round(conf,3)})
                        continue

                    if cid in HEAD_TURN_CLASSES:
                        override = _head_pose_from_bbox((x1,y1,x2,y2), roi.shape)
                        if override:
                            lbl = override

                    if cid in SOFOR_IDS:
                        zone, pcf = pax_det.update((x1,y1,x2,y2), roi.shape, conf, t)
                        if zone and buf.ok(zone, t):
                            detections.append({"zaman_saniye":t,"kategori":"yolcular",
                                               "etiket":zone,"confidence_score":round(pcf,3)})

                    if cid in YOLCU_IDS:
                        conf = min(1.0, conf+0.10)

                    if buf.ok(lbl, t):
                        detections.append({"zaman_saniye":t,"kategori":kat,
                                           "etiket":lbl,"confidence_score":round(conf,3)})

            rw = roi.shape[1]
            driver_crop = roi[:, rw//2:]
            ev = esneme_det.update(fi, driver_crop, yolo_cf=yolo_esneme_cf)
            if ev and buf.ok("esneme", ev.zaman_saniye):
                detections.append(make_esneme_json(ev))

        except Exception as e:
            print(f"  [S err] {e}")

    cap.release()

    for ev in pax_det.flush(last_t):
        if buf.ok(ev["etiket"], last_t):
            detections.append(ev)

    print(f"  [Done] {fi} kare islendi, {len(detections)} tespit")

    vehicle_info = aggregate_vehicle_info(
        type_votes, type_counts, shape_list,
        color_votes, plate_list, conf_list
    )
    detections.sort(key=lambda x: x["zaman_saniye"])
    return format_output(os.path.basename(video_path), vehicle_info, detections)