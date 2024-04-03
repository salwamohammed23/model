import streamlit as st
import numpy as np
from ultralytics import YOLO
import cv2
import math
from PIL import Image

# Load YOLOv5 model with the specified weights
model = YOLO("best.ptt")

# Empty list to store class names
class_names = []

# Read class names from the coco.names file
with open("coco.names", 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')

# Function for object detection using YOLOv5 model
def find(objects, img):
    for det in objects.pandas().xyxy[0].to_numpy():
        x, y, w, h, conf, cls = det
        cv2.rectangle(img, (int(x), int(y)), (int(w), int(h)), (0, 0, 255), 3)
        cv2.putText(img, f'{class_names[int(cls)]} {int(conf*100)}%', (int(x), int(y-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
    return img

# Initialize YOLOv5 model
model.initialize()

# Main loop for object detection using YOLOv5
while True:
    ret, frame = cap.read()
    
    # Perform object detection with YOLOv5
    with st.spinner('Detecting objects...'):
        results = model(frame)
    
    # Display annotated frame with detections
    frame_annotated = find(results, frame)
    
    # Display the annotated frame in the Streamlit app
    st.image(frame_annotated, channels="BGR")
