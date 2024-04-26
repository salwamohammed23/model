from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np
import openai
import cv2
import os
import requests
from streamlit_lottie import st_lottie


st.title("skin_canser")
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

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Predict with YOLOv8 and Openai API
if uploaded_file is not None:
        tab1, tab2,tab3 = st.tabs(["Smoke Detection", "News Article","Awarness"])
        out_img, out_name = predict_with_yolov8(uploaded_file)

        with tab1:
            st.title("FireWatch AI")
           
                
        
       

    
