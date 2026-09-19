import cv2
import numpy as np

TEKNOCAN_COLORS = [
    (np.array([100,120,80],dtype=np.uint8), np.array([130,255,255],dtype=np.uint8)),
    (np.array([20, 120,80],dtype=np.uint8), np.array([35, 255,255],dtype=np.uint8)),
    (np.array([0,  130,80],dtype=np.uint8), np.array([12, 255,255],dtype=np.uint8)),
    (np.array([165,130,80],dtype=np.uint8), np.array([180,255,255],dtype=np.uint8)),
]

def scan_teknocan(frame: np.ndarray, vehicle_bbox) -> float:
    if vehicle_bbox is None: return 0.0
    x1,y1,x2,y2 = vehicle_bbox
    fh,fw = frame.shape[:2]
    x1,y1=max(0,x1),max(0,y1); x2,y2=min(fw,x2),min(fh,y2)
    vh,vw = y2-y1, x2-x1
    if vh<20 or vw<20: return 0.0
    wy1=y1+int(0.15*vh); wy2=y1+int(0.60*vh)
    wx1=x1+int(0.15*vw); wx2=x1+int(0.85*vw)
    zone=frame[wy1:wy2,wx1:wx2]
    if zone.size==0: return 0.0
    hsv=cv2.cvtColor(zone,cv2.COLOR_BGR2HSV)
    mask=np.zeros(zone.shape[:2],dtype=np.uint8)
    for lo,hi in TEKNOCAN_COLORS:
        mask=cv2.bitwise_or(mask,cv2.inRange(hsv,lo,hi))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    ratio=np.count_nonzero(mask)/max(zone.shape[0]*zone.shape[1],1)
    if ratio<0.002: return 0.0
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return 0.0
    best_circ=0.0; best_area=0.0
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area<15: continue
        peri=cv2.arcLength(cnt,True)
        if peri<1: continue
        circ=4*np.pi*area/(peri**2)
        if circ>best_circ: best_circ=circ; best_area=area
    if best_circ>0.35:
        return round(min(0.90,0.40+best_circ*0.40+ratio*0.05),2)
    return 0.0