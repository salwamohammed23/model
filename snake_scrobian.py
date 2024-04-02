import streamlit as st
from ultralytics import YOLO
import cv2
import math 

# page title
st.title('Object Detection with YOLO on Webcam')

# model
model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
classNames = ["person", "bicycle", "car", ...]  # قم بإكمال القائمة classNames

# function to detect objects
def detect_objects(frame):
    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100

            # class name
            cls = int(box.cls[0])

            # object details
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(frame, classNames[cls], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return frame

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# run the Streamlit app
while True:
    success, img = cap.read()

    # call the detect_objects function
    img_with_objects = detect_objects(img)

    # display the image in Streamlit
    st.image(img_with_objects, channels="BGR")

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
st.write('Object Detection ended')
