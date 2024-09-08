import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch

tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.3, nn_budget=100)

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # For YOLOv5

cap = cv2.VideoCapture(0)  # Replace with your video path

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # YOLOv5 forward pass
    results = model(frame)
    
    # Filter detections to include only 'person' class (class index for person is 0 in YOLOv5)
    detections = results.pred[0].cpu().numpy()  # Extracting the prediction tensor
    bboxes = []
    for detection in detections:
        if int(detection[5]) == 0:  # Only consider 'person' class
            x1, y1, x2, y2, conf = detection[:5]
            bboxes.append([x1, y1, x2 - x1, y2 - y1, conf])  # bbox format: [x, y, w, h, confidence]

    # Update tracker with bounding boxes
    tracks = tracker.update_tracks(bboxes, frame=frame)
    
    # Loop over the tracks
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        # Get track information
        track_id = track.track_id
        ltrb = track.to_ltrb()  # Left, Top, Right, Bottom bounding box
        
        # Draw bounding box
        cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (255, 0, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (int(ltrb[0]), int(ltrb[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Display the frame
    cv2.imshow('Tracking', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
