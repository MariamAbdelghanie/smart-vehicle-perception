import os
import json
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.predict import run_inference

INPUT_PATH  = "/app/data/input/video.mp4"
OUTPUT_PATH = "/app/data/output/results.json"
WEIGHTS_M   = "/app/models/yolo11m.pt"
WEIGHTS_S   = "/app/models/yolo11s.pt"

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"[ERROR] Video bulunamadi: {INPUT_PATH}")
        sys.exit(1)

    print("[START] TECHNEXUS Road Safety AI")
    print(f"  Video  : {INPUT_PATH}")
    print(f"  Model M: {WEIGHTS_M}")
    print(f"  Model S: {WEIGHTS_S}")

    try:
        result = run_inference(
            video_path         = INPUT_PATH,
            weights_detection  = WEIGHTS_M,
            weights_cabin      = WEIGHTS_S,
        )
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[DONE] Sonuc kaydedildi: {OUTPUT_PATH}")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()