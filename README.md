# Smart Vehicle Perception — TECHNEXUS Road Safety AI

A real-time computer vision pipeline built for TEKNOFEST. It takes a video recorded from a camera facing a vehicle and produces a structured JSON report with:

- **Vehicle information:** type, color, and license plate.
- **Driver actions:** yawning, smoking, drinking, phone use, looking around, looking back, seat belt violation, and slalom driving.
- **Passengers and objects:** seat occupancy (front and rear seats) and competition-specific objects.

The system combines two custom-trained **YOLO11** models with classical computer vision (HSV color analysis, geometry rules), **MediaPipe FaceMesh**, and **EasyOCR**.

---

## Table of contents

- [Features](#features)
- [How it works](#how-it-works)
- [Detected classes](#detected-classes)
- [Output format](#output-format)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Configuration](#configuration)
- [Notes and limitations](#notes-and-limitations)
- [Team](#team)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Features

- **Two-model design:** a medium YOLO11 model for the whole scene (vehicle, color, plate) and a small YOLO11 model for the vehicle cabin (driver, passengers, objects).
- **Vehicle profiling:** type and color are decided by confidence-weighted voting over many frames, with color-analysis and bounding-box-shape fallbacks.
- **Turkish license plate reading:** EasyOCR with several image preprocessing variants, plate validation with the Turkish plate format (province code 01–81), and correction of commonly confused characters (`O/0`, `I/1`, `S/5`, `B/8`).
- **Driver monitoring:** yawning is detected by fusing MediaPipe mouth-opening ratio with YOLO; smoking, drinking, and phone use are separated with color-based checks on top of YOLO.
- **Slalom detection:** based on the left/right movement pattern of the vehicle across frames.
- **Passenger zone classification:** front seat and two rear seat zones are derived from where detections fall inside the vehicle region.
- **Event de-duplication:** every label has a cooldown so the same event is not reported repeatedly.
- **Graceful fallbacks:** if MediaPipe is not available, the detectors continue with YOLO only. If CUDA is available it is used, otherwise the pipeline runs on CPU.
- **Docker-ready:** runs as a container that reads one input video and writes one JSON file.

---

## How it works

```
video.mp4
   │
   ▼
Frame sampling (~6 FPS) + CLAHE contrast enhancement
   │
   ├─► YOLO11m on the full frame
   │      ├─ vehicle type (7 classes), vehicle color (9 classes), license plate
   │      ├─ plate crop ─► EasyOCR ─► Turkish plate rules
   │      ├─ vehicle bounding box ─► slalom detector (movement pattern)
   │      └─ vehicle bounding box ─► teknocan scan (color and shape analysis)
   │
   └─► YOLO11s on the vehicle region (cabin)
          ├─ driver actions (yawning, smoking, drinking, phone, looking around/back, seat belt)
          ├─ yawning ─► MediaPipe mouth-opening ratio + YOLO fusion
          ├─ smoking / drinking / phone ─► color-ratio disambiguation
          └─ passengers ─► seat zone classification

   ▼
Voting, cooldowns, sorting by time  ─►  results.json
```

Step by step:

1. **Sampling.** The video is read at about 6 frames per second (`frame_skip = round(fps / 6)`), up to 600 sampled frames. Each sampled frame is contrast-enhanced with CLAHE in LAB color space.
2. **Vehicle stage.** `yolo11m.pt` finds the vehicle, its color, and the plate on the enhanced frame. Votes are accumulated across frames. The vehicle bounding box becomes the region of interest for the next stage.
3. **Cabin stage.** `yolo11s.pt` runs on the vehicle region with a very low base confidence. Each class then has its own threshold (for example `esneme` 0.28, `sigara_icme` 0.30, `teknocan` 0.08).
4. **Rule-based detectors.** Yawning, slalom, the `teknocan` object, and the smoking/drinking/phone check are computed with dedicated modules (see [Project structure](#project-structure)).
5. **Aggregation.** Vehicle type, color, and plate are chosen by majority voting. Events are sorted by time and written to JSON.

---

## Detected classes

Label names in the output are in Turkish. English meanings are given below.

### Vehicle model (`yolo11m.pt`) — 17 classes

| IDs | Group | Classes |
|---|---|---|
| 0–6 | Vehicle type | `sedan`, `suv`, `hatchback`, `pickup`, `minibus`, `panelvan`, `kamyon` (truck) |
| 7–15 | Vehicle color | `beyaz` (white), `siyah` (black), `gri` (gray), `kirmizi` (red), `mavi` (blue), `sari` (yellow), `yesil` (green), `turuncu` (orange), `kahverengi` (brown) |
| 16 | License plate | `plaka` |

### Cabin model (`yolo11s.pt`) — 13 classes

| Output category | Label | Meaning |
|---|---|---|
| `sofor_eylemi` (driver action) | `arkaya_bakma` | Looking back |
| | `esneme` | Yawning |
| | `sigara_icme` | Smoking |
| | `su_icme` | Drinking water |
| | `telefonla_konusma` | Talking on the phone |
| | `etrafa_bakinma` | Looking around |
| | `emniyet_kemeri_ihlali` | Seat belt violation |
| | `slalom` | Slalom driving (also detected by the movement-pattern module) |
| `nesneler` (objects) | `teknocan` | Competition object |
| | `bilgisayar` | Computer / laptop |
| `yolcular` (passengers) | `on_koltuk` | Front passenger seat occupied |
| | `arka_koltuk_1` | Rear seat 1 occupied |
| | `arka_koltuk_2` | Rear seat 2 occupied |

---

## Output format

The result is saved to `results.json`. The values below are an example.

```json
{
  "video_id": "video.mp4",
  "arac_bilgisi": {
    "tip": "sedan",
    "plaka": "34ABC123",
    "renk": "beyaz",
    "confidence_score": 0.87
  },
  "tespitler": [
    {
      "zaman_saniye": 3.5,
      "kategori": "sofor_eylemi",
      "etiket": "esneme",
      "confidence_score": 0.82
    },
    {
      "zaman_saniye": 7.2,
      "kategori": "yolcular",
      "etiket": "arka_koltuk_1",
      "confidence_score": 0.74
    }
  ]
}
```

| Field | Description |
|---|---|
| `video_id` | Input file name |
| `arac_bilgisi.tip` | Vehicle type |
| `arac_bilgisi.plaka` | License plate, or `tespit_edilemedi` if no valid plate was read |
| `arac_bilgisi.renk` | Vehicle color |
| `arac_bilgisi.confidence_score` | Combined confidence of type, color, and plate |
| `tespitler[].zaman_saniye` | Time of the event in seconds |
| `tespitler[].kategori` | `sofor_eylemi`, `nesneler`, or `yolcular` |
| `tespitler[].etiket` | Event label |
| `tespitler[].confidence_score` | Confidence between 0 and 1 |

---

## Project structure

```
.
├── main.py                    # Entry point: reads the video, runs inference, writes JSON
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── weights/
│   ├── yolo11m.pt             # Vehicle model (type, color, plate)
│   └── yolo11s.pt             # Cabin model (driver, passengers, objects)
└── src/
    ├── predict.py             # Main inference pipeline
    ├── utils.py               # Frame enhancement, color/shape helpers, plate rules, output format
    ├── ocr.py                 # Plate reading with EasyOCR
    ├── esneme_detector.py     # Yawning detection (MediaPipe + YOLO fusion)
    ├── head_pose_detector.py  # Head pose estimation from FaceMesh landmarks
    ├── passenger_detector.py  # Seat zone classification
    ├── slalom_detector.py     # Slalom detection from movement pattern
    ├── teknocan_detector.py   # Color and shape based object scan
    └── tracker.py             # Simple IoU-based tracker
```

---

## Getting started

### Requirements

- Docker (recommended), or Python 3.11 with the packages in `requirements.txt`
- An NVIDIA GPU is optional. The pipeline uses CUDA when available and falls back to CPU.

### Run with Docker

1. **Build the image:**
   ```bash
   docker build -t technexus-road-safety .
   ```

2. **Prepare the input.** Put your video in `data/input/` and name it `video.mp4`:
   ```
   data/
   ├── input/
   │   └── video.mp4
   └── output/
   ```

3. **Run the container:**
   ```bash
   docker run --rm \
     -v "$(pwd)/data/input:/app/data/input" \
     -v "$(pwd)/data/output:/app/data/output" \
     technexus-road-safety
   ```
   Add `--gpus all` to use an NVIDIA GPU (requires the NVIDIA Container Toolkit).

4. **Read the result** from `data/output/results.json`.

### Run without Docker

The input, output, and weight paths are set at the top of `main.py`:

```python
INPUT_PATH  = "/app/data/input/video.mp4"
OUTPUT_PATH = "/app/data/output/results.json"
WEIGHTS_M   = "/app/models/yolo11m.pt"
WEIGHTS_S   = "/app/models/yolo11s.pt"
```

To run locally, install the dependencies, change these paths to your own (for example `weights/yolo11m.pt`), and run:

```bash
pip install -r requirements.txt
python main.py
```

The loader also looks for the weights in `/app/weights/` when they are not found in `/app/models/`.

---

## Configuration

The main settings live in `src/predict.py` and in the detector classes.

| Setting | Where | Default | Purpose |
|---|---|---|---|
| Sampling rate | `predict.py` | about 6 FPS | Frames analyzed per second of video |
| Maximum frames | `predict.py` | 600 sampled frames | Upper limit of processed frames |
| Vehicle model input size | `predict.py` | 480 | Image size for `yolo11m.pt` |
| Cabin model input size | `predict.py` | 320 | Image size for `yolo11s.pt` |
| Class thresholds | `CONF_THRESH` in `predict.py` | per class | Minimum confidence per label |
| Event cooldown | `_Buf` in `predict.py` | 2.0 s | Minimum time between events with the same label |
| Yawning threshold | `EsnemeDetector` | MAR 0.47, 7 frames, 5 s cooldown | Mouth-opening ratio and duration |
| Slalom rule | `SlalomDetector` | 5 direction switches in 4 s | Movement pattern of the vehicle |

---

## Notes and limitations

- **Heuristic confidence:** confidence scores combine YOLO output and rule-based signals. They are useful for ranking events but are not calibrated probabilities.
- **Tuned thresholds:** thresholds and cooldowns were tuned on a limited set of videos, so they may need adjustment for other cameras and lighting.
- **Camera assumption:** seat zones assume a camera facing the vehicle from the front. In a left-hand-drive car the driver appears on the right side of the image.
- **Plate format:** plate validation follows the Turkish plate format only.
- **Modules not connected to the main pipeline:** `head_pose_detector.py` (FaceMesh-based head pose) and `tracker.py` (IoU tracker) are implemented but are not called from `predict.py`. The current pipeline decides looking-around and looking-back events from YOLO detections and their position inside the vehicle region.
- **Fixed paths:** input and output paths are fixed to the container layout.

---

## Team

Developed by team **TECHNEXUS**.

<!-- Add team members and roles here. -->

---

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [MediaPipe](https://github.com/google-ai-edge/mediapipe)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [OpenCV](https://opencv.org/)
- [TEKNOFEST](https://www.teknofest.org/)

---

