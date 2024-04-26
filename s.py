from ultralytics import YOLO
import streamlit as st
from PIL import Image

# Function to predict with YOLOv8
def predict_with_yolov8(img_bytes):
    # Load the YOLOv8 model
    model = YOLO('skin_can.pt')

    # Convert the image bytes to PIL image
    pil_image = Image.open(img_bytes)

    # Run YOLOv8 segmentation on the image
    results = model.predict(pil_image, imgsz=600, conf=0.3, iou=0.5)

    return results

# Main code
st.title("Skin Cancer Detection")

# User image upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Perform prediction on uploaded image
    results = predict_with_yolov8(uploaded_file)

    # Get the path of the new image saved by YOLOv8
    # Assuming inference[0] is the Results object
    res_plotted = results.plot()[:, :, ::-1]
    pred = results.boxes.cls
    
    # Display image and results
    st.image(res_plotted, use_column_width=True, caption="Image")

    if pred.numel() == 0:
        st.markdown("**No Skin Cancer Detected.**")
    else:
         st.markdown("**Skin Cancer Detected!**")
