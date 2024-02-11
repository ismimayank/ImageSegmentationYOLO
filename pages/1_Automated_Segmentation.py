import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
import supervision as sv
import base64

st.set_page_config(page_title="YOLOv8 Image Segmentation")
st.title('Image Segmentation with YOLOv8')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Process image on button click
    if st.button('Process Image'):
        # Save uploaded image
        temp_image = tempfile.NamedTemporaryFile(delete=False)
        temp_image.write(uploaded_file.read())

        source = cv2.imread(temp_image.name)
        image_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

        # Load YOLOv8 model
        model = YOLO('yolov8n-seg.pt')

        # Perform segmentation
        results = model(image_rgb)
        detections = sv.Detections.from_ultralytics(results[0])

        bounding_box_annotator = sv.BoundingBoxAnnotator()
        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

        annotated_image = bounding_box_annotator.annotate(scene=image_rgb,
                                                        detections=detections)
        annotated_image = mask_annotator.annotate(scene=annotated_image,
                                                detections=detections)
        # Display segmented image
        st.image(annotated_image, caption='Segmented Image', use_column_width=True)

        # Provide option to download segmented image
        with st.expander("Download Segmented Image"):
            temp_file = os.path.join(tempfile.gettempdir(), "segmented_image.jpg")
            cv2.imwrite(temp_file, cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            st.markdown(
                f'<a href="data:file/jpg;base64,{base64.b64encode(open(temp_file, "rb").read()).decode()}\" download="segmented_image.jpg">Download Segmented Image</a>',
                unsafe_allow_html=True
            )

        # Cleanup
        os.remove(temp_image.name)
