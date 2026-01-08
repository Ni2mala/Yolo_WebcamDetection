import cv2, json, time
from datetime import datetime
from pathlib import Path
import numpy as np
from ultralytics import YOLO

# ---------------- CONFIG ----------------
WEBCAM_IDX = 0
MODEL_NAME = 'yolov8n.pt'
CONF_THRESHOLD = 0.35
AUTO_SAVE = True
SAVE_FULL_FRAMES = False
OUT_DIR = Path('captures')
OUT_DIR.mkdir(exist_ok=True, parents=True)
# ----------------------------------------

session_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
metadata_path = OUT_DIR / f'session_metadata_{session_ts}.json'
metadata = {'start': datetime.now().isoformat(), 'captures': []}

print(f"Loading model: {MODEL_NAME}")
model = YOLO(MODEL_NAME)

cap = cv2.VideoCapture(WEBCAM_IDX)
if not cap.isOpened():
    raise SystemExit("Cannot open webcam")

w, h = int(cap.get(3)), int(cap.get(4))
fps = cap.get(5) or 20
recording, video_writer, save_mode = False, None, AUTO_SAVE

print("Press 'q' quit | 's' auto-save | 'c' capture ")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
    annotated = cv2.cvtColor(results.plot(), cv2.COLOR_RGB2BGR)

    # Save crops if auto-save
    if save_mode:
        for i, box in enumerate(results.boxes):
            conf = float(box.conf)
            if conf < CONF_THRESHOLD: continue
            cls = model.names[int(box.cls)]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]

            ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            path = OUT_DIR / cls / f"{cls}_{ts}_{i}.jpg"
            path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(path), crop)

            metadata['captures'].append({
                'time': datetime.now().isoformat(),
                'file': str(path),
                'label': cls,
                'conf': conf,
                'bbox': [x1, y1, x2, y2]
            })

        if SAVE_FULL_FRAMES:
            fpath = OUT_DIR / f"frame_{ts}.jpg"
            cv2.imwrite(str(fpath), annotated)

    # if recording and video_writer:
       #  video_writer.write(annotated)

    cv2.imshow("YOLO Webcam", annotated)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        save_mode = not save_mode
        print(f"Auto-save: {save_mode}")
    elif key == ord('c'):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = OUT_DIR / f"manual_{ts}.jpg"
        cv2.imwrite(str(path), annotated)
        print(f"Saved frame: {path}")
    elif key == ord('v'):
        recording = not recording
        if recording:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            vid_path = OUT_DIR / f"record_{ts}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(vid_path), fourcc, fps, (w, h))
            print(f"Recording → {vid_path}")
        else:
            video_writer.release()
            video_writer = None
            print("Stopped recording.")

cap.release()
if video_writer: video_writer.release()
cv2.destroyAllWindows()

metadata['end'] = datetime.now().isoformat()
json.dump(metadata, open(metadata_path, 'w'), indent=2)
print(f"Metadata saved to {metadata_path}")