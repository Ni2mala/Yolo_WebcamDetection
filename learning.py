import cv2
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import json


camera = cv2.VideoCapture(0) #default camera
model = YOLO("yolov8n.pt")
saved_count = 0 #how many crops have been saved
MAX_SAVES = 3 #stop saving after 3 crops
OUT_DIR = Path('captures')
session_ts = datetime.now().strftime('%Y%m%d_%H%M%S')



#meta data setup
metadata_path = OUT_DIR/f'session_metadata_{session_ts}.json'
metadata = {
    'start': datetime.now().isoformat(),
    'captures':[]
}


#to check if the camera is opened correctly
if not camera.isOpened():
    print("error couldn't open camera")
    exit()
OUT_DIR.mkdir(exist_ok=True)



while True:
    #capture frame by frame
    ret, frame = camera.read()
    print(ret) #returns true if frame is captured
    if not ret:
        print("failed to grab frame")
        break

    #yolo expects image in RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #run yolo detection
    results = model.predict(rgb_frame,conf=0.5, verbose=False)[0] #get first result

    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0]) #bounding box
        cls = model.names[int(box.cls)] #class name
        conf = float(box.conf) #confidence
        crop = frame[y1:y2, x1:x2] #extract the detected object
        #y1:y2 rows
        #x1:x2 columns
        print(f"Trying to save crop {saved_count+1}: size={crop.shape}", flush=True)

        #saving detected object with limit
        if saved_count <MAX_SAVES:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            class_folder = OUT_DIR/cls #creates path: captures/person
            class_folder.mkdir(exist_ok=True) #folder is not exist
            filename = f'{cls}_{ts}_{i}.jpg' #folder name format
            filepath = class_folder / filename #full path  captures/person/person_20250109_143052_0.jpg
            cv2.imwrite(str(filepath), crop) #converts to string for cv2
            print(f'saved {filepath}')


            #add metadata entry
            metadata['captures'].append({
                'time': datetime.now().isoformat(),
                'file': str(filepath),
                'label': cls,
                'conf': conf,
                'bbox': [x1,y1,x2,y2]
            })
            saved_count +=1

        print(f"Detected {cls} with {conf:.2f} at ({x1}, {y1}, {x2},{y2})")

    #Draw results on the frame
    annotated_frame = results.plot() #draws bounding boxes

#frame is captured successfully so display it
    cv2.imshow("laptop camera", annotated_frame)

    #press q to quit / to end the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release/close the camera and close windows
camera.release()
cv2.destroyAllWindows()

#save metadata after cleanup
metadata['end'] = datetime.now().isoformat() #adds programs end time
with open(metadata_path, 'w') as f: #opens file for writing, 
    json.dump(metadata, f, indent=2)
print(f'metadata saved to{metadata_path}')

#with automatically closes file when done






