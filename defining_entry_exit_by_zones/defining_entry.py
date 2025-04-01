import os
import cv2
import numpy as np
import datetime
from ultralytics import YOLO

# Define the model filename
model_filename = "yolov8n.pt"

# Check if the model already exists
if not os.path.exists(model_filename):
    print("Downloading YOLOv8n model...")
    model = YOLO("yolov8n.pt", verbose=False)  #Downloads model if not
    model_path = model.ckpt_path  # Gets the downloaded model path
    os.rename(model_path, model_filename)  # Renames and stores in the current folder
else:
    print("Model already exists, loading...")
    model = YOLO(model_filename, verbose=False)

def define_zones(frame_shape):
    """Define zones for entry/exit detection."""
    height, width, _ = frame_shape
    return {
       "dockking_zone": [(width // 2 - 150, height // 2 ), (width // 2 -30, height // 2 + 90)]  # Dockking zone
    }

def check_zone_entry_exit(centroid, zones):
    """Check if an object enters or exits predefined zones."""
    x, y = centroid
    for zone_name, ((x1, y1), (x2, y2)) in zones.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            return zone_name
    return None

def log_detected_objects(logged_centroids, centroid, label, conf, context):
    """Logs detected objects with timestamp, avoiding duplicate logging, and adding context."""
    if tuple(centroid) not in logged_centroids:  # Only log new objects by centroid
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("defining_entry_exit_by_zones\detection_log.txt", "a") as log_file:
            log_file.write(f"{timestamp} - {label} {conf:.2f} - Context: {context}\n")
        logged_centroids.add(tuple(centroid))

def process_video_activity(model, video_path):
    """Processes video with movement classification and zone detection."""
    cap = cv2.VideoCapture(video_path)
    previous_positions = {}
    zones = None  # Initialize zones
    logged_centroids = set()  # To track logged centroids
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if zones is None:
            zones = define_zones(frame.shape)
        
        results = model(frame)
        detected_objects = []
        
        for result in results:
            for box in result.boxes:
                conf = box.conf[0].item()
                if conf < 0.01:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                detected_objects.append((x1, y1, x2, y2, centroid))
        
        # Processing detected objects
        for x1, y1, x2, y2, centroid in detected_objects:
            # Movement classification
            if tuple(centroid) in previous_positions:
                prev_centroid = previous_positions[tuple(centroid)]
                movement = np.linalg.norm(np.array(centroid) - np.array(prev_centroid))
                if movement < 2:
                    context = "Stationary"
                elif movement < 30:
                    context = "Slow Motion"
                else:
                    context = "Fast Motion"
            else:
                context = "New Detection"
            
            # Zone detection
            zone_status = check_zone_entry_exit(centroid, zones)
            if zone_status:
                context += f" | Entered {zone_status}"
            
            # Draw bounding box for object
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{context}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Log detected objects in docking zone
            if zone_status == "dockking_zone":
                log_detected_objects(logged_centroids, centroid, "Ship", conf, context)
        
        # Update previous positions with the current centroids
        for x1, y1, x2, y2, centroid in detected_objects:
            previous_positions[tuple(centroid)] = centroid
        
        # Draw docking zone rectangle
        (x1, y1), (x2, y2) = zones["dockking_zone"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red rectangle for docking zone
        
        cv2.imshow("Activity & Zone Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Run the function
process_video_activity(model, "defining_entry_exit_by_zones\dockking_of_boat.mp4")
