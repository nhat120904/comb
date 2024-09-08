import os
import cv2
import torch
import datetime
import numpy as np
import logging
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLOv10
import streamlit as st

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model and other configurations
def initialize_model():
    model_path = "./weights/yolov10n.pt"
    if not os.path.exists(model_path):
        logger.error(f"Model weights not found at {model_path}")
        raise FileNotFoundError("Model weights file not found")
    
    model = YOLOv10(model_path)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    model.to(device)
    logger.info(f"Using {device} as processing device")
    return model

def load_class_names():
    classes_path = "./configs/coco.names"
    if not os.path.exists(classes_path):
        logger.error(f"Class names file not found at {classes_path}")
        raise FileNotFoundError("Class names file not found")
    
    with open(classes_path, "r") as f:
        class_names = f.read().strip().split("\n")
    return class_names

def process_frame(frame, model, tracker, class_names, colors, conf_thresh):
    results = model(frame, verbose=False)[0]
    detections = []
    for det in results.boxes:
        label, confidence, bbox = det.cls, det.conf, det.xyxy[0]
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(label)
        
        if confidence < conf_thresh:
            continue
        
        detections.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])
    
    tracks = tracker.update_tracks(detections, frame=frame)
    return tracks

def draw_tracks(frame, tracks, class_names, colors):
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        class_id = track.get_det_class()
        x1, y1, x2, y2 = map(int, ltrb)
        color = colors[class_id]
        B, G, R = map(int, color)
        
        text = f"{track_id} - {class_names[class_id]}"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
        cv2.putText(frame, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
    return frame

# Streamlit app
def main():
    st.title("Human Tracking Demo Using YOLOv10 and DeepSORT")
    
    # File uploader for video
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    
    if uploaded_file is not None:
        # Save uploaded file to disk
        video_path = os.path.join("uploads", uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Initialize components
        model = initialize_model()
        class_names = load_class_names()
        tracker = DeepSort(max_age=20, n_init=3)
        
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(class_names), 3))
        
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_path = os.path.join("output", "tracked_video.mp4")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        frame_count = 0
        
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            start = datetime.datetime.now()
            tracks = process_frame(frame, model, tracker, class_names, colors, conf_thresh=0.4)
            frame = draw_tracks(frame, tracks, class_names, colors)
            end = datetime.datetime.now()
            
            logger.info(f"Time to process frame {frame_count}: {(end - start).total_seconds():.2f} seconds")
            frame_count += 1
            
            fps_text = f"FPS: {1 / (end - start).total_seconds():.2f}"
            cv2.putText(frame, fps_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)
            
            writer.write(frame)
            
            stframe.image(frame, channels="BGR", use_column_width=True)
        
        cap.release()
        writer.release()
        st.success("Video processing complete")
        st.video(output_path)

if __name__ == "__main__":
    main()
