import cv2
from ultralytics import YOLO

camera = cv2.VideoCapture(0) #default camera
model = YOLO("yolov8n.pt")
#to check if the camera is opened correctly

if not camera.isOpened():
    print("error couldn't open camera")
    exit()


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
    results = model.predict(rgb_frame, verbose=False)

    #Draw results on the frame
    annotated_frame = results[0].plot() #draws bounding boxes

#frame is captured successfully so display it
    cv2.imshow("laptop camera", annotated_frame)

    #press q to quit / to end the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release/close the camera and close windows
camera.release()
cv2.destroyAllWindows()







