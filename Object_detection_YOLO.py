import cv2
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO

#config
model = YOLO("yolov8m.pt")
camera = cv2.VideoCapture(0)
MAX_PER_CLASS=3
OUT_DIR =Path('captures')
saved_count = defaultdict(int)

while True:
    ret, frame = camera.read()

    result = model.predict(frame, conf=0.65, verbose=False)[0]

    for i, box in enumerate(result.boxes):
        cls = model.names[int(box.cls)]
        conf = float(box.conf)
        x1,y1,x2,y2=map(int, box.xyxy[0])
       # cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        if saved_count[cls] >=MAX_PER_CLASS:
            continue

        crop=frame[y1:y2, x1:x2]
      

        #create folder and save
        class_folder = OUT_DIR/cls
        class_folder.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

        filepath=class_folder/ f'{cls}_{ts}_{i}.jpg'
        cv2.imwrite(str(filepath), crop)
        saved_count[cls] +=1
    annoted_frame = result.plot()
    cv2.imshow("camera", annoted_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()