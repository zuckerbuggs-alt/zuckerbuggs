import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
import tempfile
import os
import time
import pydeck as pdk
import pandas as pd

st.set_page_config(page_title="Space Debris Detector", layout="wide")

st.title("🚀 Space Debris Detection (Streamlit prototype)")

# Sidebar controls
st.sidebar.header("App Mode")
mode = st.sidebar.radio("Choose mode", ["Image Detection", "Video Detection", "Map View"])

# ==========================
# Load YOLO model
@st.cache_resource(ttl=3600)
def load_model_from_hub():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

if mode in ["Image Detection", "Video Detection"]:
    with st.spinner("Loading detection model..."):
        model = load_model_from_hub()
    model.conf = 0.35
    model.iou = 0.45

# ==========================
# Detection helper
def draw_boxes_on_image(img, results):
    boxes = results.xyxy[0].cpu().numpy()
    for *xyxy, conf, cls in boxes:
        x1, y1, x2, y2 = map(int, xyxy)
        label = f"{model.names[int(cls)]} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        ((w, h), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - 15), (x1 + w, y1), (0,255,0), -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    return img

# ==========================
# IMAGE detection
if mode == "Image Detection":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png","tif"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Input image", use_column_width=True)
        if st.button("Run detection"):
            img_np = np.array(image)
            with st.spinner("Running inference..."):
                results = model(img_np, size=640)
            annotated = draw_boxes_on_image(img_np.copy(), results)
            st.image(annotated, caption="Detections", use_column_width=True)
            st.write("Raw results:")
            st.dataframe(results.pandas().xyxy[0].round(3))

# ==========================
# VIDEO detection
elif mode == "Video Detection":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4","mov","avi"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        st.video(video_path)
        if st.button("Run video detection"):
            cap = cv2.VideoCapture(video_path)
            output_frames = []
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            processed = 0
            progress = st.progress(0)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame, size=640)
                frame_annot = draw_boxes_on_image(frame.copy(), results)
                frame_rgb = cv2.cvtColor(frame_annot, cv2.COLOR_BGR2RGB)
                output_frames.append(frame_rgb)
                processed += 1
                progress.progress(processed / frame_count)
            cap.release()
            st.success(f"Processed {processed} frames")
            if len(output_frames) > 0:
                st.image(output_frames[min(5, len(output_frames)-1)], caption="Sample annotated frame", use_column_width=True)

# ==========================
# MAP VIEW
elif mode == "Map View":
    st.subheader("🌍 Space Debris Locations on Map")

    # Example debris data (lat/lon + altitude in km)
    debris_data = pd.DataFrame({
        "lat": [12.9, -7.1, 45.3, 0.0, 30.5],
        "lon": [77.6, 120.9, -73.9, 10.1, -45.2],
        "alt_km": [500, 700, 1200, 800, 600],
        "object": ["Debris-A", "Debris-B", "Debris-C", "Debris-D", "Debris-E"]
    })

    st.dataframe(debris_data)

    # Create 3D globe-style scatterplot
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/dark-v10",
        initial_view_state=pdk.ViewState(
            latitude=0,
            longitude=0,
            zoom=1,
            pitch=30,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=debris_data,
                get_position='[lon, lat]',
                get_color='[200, 30, 0, 160]',
                get_radius="alt_km", # scale radius by altitude
                pickable=True
            )
        ],
        tooltip={"text": "{object}\nLat: {lat}, Lon: {lon}\nAlt: {alt_km} km"}
    ))
