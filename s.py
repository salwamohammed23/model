from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np
import openai
import cv2
import os
import requests
from streamlit_lottie import st_lottie


st.title("FireWatch AI")
def predict_with_yolov8(img_bytes):
    # Load the YOLOv8 model
    model = YOLO('def predict_with_yolov8(img_bytes):
    # Load the YOLOv8 model
    model = YOLO('skin_can.pt')

    # Convert the image bytes to PIL image
    pil_image = Image.open(img_bytes)

    # Run YOLOv8 segmentation on the image
    results = model.predict(pil_image, imgsz=600,conf=0.3, iou=0.5)
    # Get the path of the new image saved by YOLOv8
    # Assuming inference[0] is the Results object
    res_plotted = results[0].plot()[:, :, ::-1]
    pred= results[0].boxes.cls
    
    return res_plotted,pred')

    # Convert the image bytes to PIL image
    pil_image = Image.open(img_bytes)

    # Run YOLOv8 segmentation on the image
    results = model.predict(pil_image, imgsz=600,conf=0.3, iou=0.5)
    # Get the path of the new image saved by YOLOv8
    # Assuming inference[0] is the Results object
    res_plotted = results[0].plot()[:, :, ::-1]
    pred= results[0].boxes.cls
    
    return res_plotted,pred
