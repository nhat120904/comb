import torch
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLOv10

# Load YOLOv10 model
# Replace with the actual path where your YOLOv10 weights are stored
yolo_model_path = './weights/yolov10n.pt'
model = YOLOv10(yolo_model_path)
    
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")

# model.to(device)

# Ensure model is in evaluation mode
# yolo_model.eval()

def detect_people(frame, model):
    # Convert the frame to a tensor and pass it through the YOLOv10 model
    # (assumed that YOLOv10 uses a similar interface as YOLOv5)
    
    img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()  # Convert to tensor, shape (C, H, W)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

    # Forward pass through YOLOv10
    with torch.no_grad():
        predictions = model(img_tensor)
    
    # Assuming the model returns predictions in a similar format: [x1, y1, x2, y2, conf, class_id]
    detections = predictions[0].cpu().numpy()

    # Filter to only keep 'person' class (assuming class 0 is for 'person')
    person_detections = []
    for detection in detections:
        print("check: ", detection[2])
        if int(detection[5]) == 0:  # Only 'person' class
            x1, y1, x2, y2, conf = detection[:5]
            person_detections.append([x1, y1, x2 - x1, y2 - y1, conf])  # Convert to [x, y, w, h, conf]

    return person_detections

# Initialize DeepSORT Tracker
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.3, nn_budget=100)

# Start processing the video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect people using YOLOv10
    bboxes = detect_people(frame, model)

    # Update the tracker with YOLOv10 detections
    tracks = tracker.update_tracks(bboxes, frame=frame)

    # Loop over the tracks and draw them on the frame
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        ltrb = track.to_ltrb()  # Convert to (left, top, right, bottom) bbox format

        # Draw bounding box
        cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (int(ltrb[0]), int(ltrb[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with tracking info
    cv2.imshow('Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
