import os
import cv2
from PIL import Image
import pandas as pd
import streamlit as st
from ultralytics import YOLO
from streamlit_image_comparison import image_comparison

model = YOLO('best.pt')
# Split the page into two tabs
tab1, tab2, tab3= st.tabs(["Detection with YOLOv8", "Detection with YOLO_nas", "Detection with YOLOv9"])

with tab1:
    st.title("Detection with YOLOv8")
    st.subheader("Implementing Detection on snake and scrobian dataset")
    st.write("------")

    def save_uploadedfile(uploadedfile):
        with open(os.path.join("./media-directory/", "uploaded_image.jpg"), "wb") as f:
            f.write(uploadedfile.getbuffer())

    st.divider()

    APPLICATION_MODE = st.sidebar.selectbox("Our Options", ["Take Picture", "Upload Picture"])

    if APPLICATION_MODE == "Take Picture":
        picture = st.camera_input("Take a picture")
        st.markdown('')
        if picture:
            st.write("Showing the picture taken:")
            st.image(picture, caption="Selfie")
            if st.button("Segment!"):
                save_uploadedfile(picture)
                st.sidebar.success("Saved File")
        st.write("Click on **Clear photo** to retake picture")

        if os.path.exists("./media-directory/media-directory/ALL-OF-MY-SCORPIONS-mp4_001173520_png.rf.268a30032134ccc8d2ecc0fc4857ffe6.jpg"):
            img_file = "./media-directory/media-directory/ALL-OF-MY-SCORPIONS-mp4_001173520_png.rf.268a30032134ccc8d2ecc0fc4857ffe6.jpg"
            results = model(img_file)
            img = cv2.imread(img_file)

            names_list = []
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    r = box.xyxy[0].astype(int)
                    rect = cv2.rectangle(img, tuple(r[:2]), tuple(r[2:]), (255, 55, 255), 2)
                st.image(rect)

    elif APPLICATION_MODE == "Upload Picture":
        uploaded_file = st.sidebar.file_uploader("Drop a JPG/PNG file", accept_multiple_files=False, type=['jpg', 'png'])
        if uploaded_file is not None:
            new_file_name = "ALL-OF-MY-SCORPIONS-mp4_001173520_png.rf.268a30032134ccc8d2ecc0fc4857ffe6.jpg"
            with open(os.path.join("./media-directory/", new_file_name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            img_file = os.path.join("./media-directory/", new_file_name)
            st.sidebar.success("File saved successfully")

        if os.path.exists(img_file):
            results = model(img_file)
            img = cv2.imread(img_file)

            names_list = []
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    r = box.xyxy[0].astype(int)
                    rect = cv2.rectangle(img, tuple(r[:2]), tuple(r[2:]), (255, 55, 255), 2)
                st.image(rect)

        else:
            st.sidebar.write("You are using a placeholder image. Upload your Image to explore further.")

    st.sidebar.markdown('')
    st.markdown('##### Slider of Uploaded Image and Segments')
    image_comparison(
        img1=img_file,
        img2=img,
        label1="Actual Image",
        label2="Segmented Image",
        width=700,
        starting_position=50,
        show_labels=True,
        make_responsive=True,
        in_memory=True
    )

    st.sidebar.markdown('#### Distribution of identified items')
    if len(names_list) > 0:
        df_x = pd.DataFrame(names_list)
        summary_table = df_x[0].value_counts().rename_axis('unique_values').reset_index(name='counts')
        st.sidebar.dataframe(summary_table, use_container_width=st.session_state.use_container_width)
    else:
        st.sidebar.warning("Unable to identify distinct items - Please retry with a clearer Image")

with tab2:
    st.title("Detection with YOLO_nas")

with tab3:
    st.title("Detection with YOLOv9")
