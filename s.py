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

    # Get the path of the new image saved by YOLOv8
    # Assuming inference[0] is the Results object
    res_plotted = results[0].plot()[:, :, ::-1]
    pred = results[0].boxes.cls
    
    return res_plotted, pred

# Main code
st.title("Skin Cancer Detection")

# User image upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Perform prediction on uploaded image
if uploaded_file is not None:
    tab1, tab2 = st.tabs(["Skin Cancer Detection", "Webcam"])

    # Prediction on uploaded image
    out_img, out_name = predict_with_yolov8(uploaded_file)

    with tab1:
        st.image(out_img, use_column_width=True, caption="Image")

        if out_name.numel() == 0:
            st.markdown("**No Skin Cancer Detected.**")
        else:
            st.markdown("**Skin Cancer Detected!**")

    with tab2:    
        st.title("Webcam")
        model = YOLO('skin_can.pt')
        with st.expander("Start Camera"):
            camera_image = st.camera_input("Camera")
        if camera_image:
            img = Image.open(camera_image)
            gray_scale, predictions = model.predict(img)
            st.image(gray_scale, caption="Webcam Image")
            st.markdown("**Predictions:**")
            st.write(predictions)
                
        
       

    
