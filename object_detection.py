import datetime
import numpy as np
import cv2
from ultralytics import YOLO
import sqlite3
from motion_detection import analyze_and_log_context

def load_model(model_path="models/yolov8n-pose.pt"):
    """Loads the YOLOv8-Pose model."""
    return YOLO(model_path, verbose=False)

def process_frame(model, frame):
    """Runs YOLOv8-Pose on a frame and returns the processed frame."""
    results = model(frame, show=True)
    return results[0].plot()

def process_image(model, image_path):
    """Processes a single image."""
    image = cv2.imread(image_path)
    processed_image = process_frame(model, image)
    cv2.imshow("YOLOv8-Pose Detection", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def log_detected_objects(logged_ids, object_id, label, conf, context):
    """Logs detected objects with timestamp and adds context to a SQLite database."""
    
    # Create or connect to the database
    conn = sqlite3.connect("detection_log.db")
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            object_id TEXT,
            label TEXT,
            confidence REAL,
            context TEXT,
            timestamp TEXT
        )
    ''')

    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Insert the detected object data into the table
    cursor.execute('''
        INSERT INTO detections (object_id, label, confidence, context, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (object_id, label, conf, context, timestamp))

    # Commit the transaction and close the connection
    conn.commit()
    conn.close()

        # Update the logged_ids dictionary with the new context
    logged_ids[object_id] = context

    # Log to a .txt file as well
    log_entry = f"{timestamp} | Object ID: {object_id} | Label: {label} | Confidence: {conf} | Context: {context}\n"

    # Append the log entry to the text file
    with open("detection_log.txt", "a") as file:
        file.write(log_entry)

    print(f"Logged: {log_entry}")


def process_video(model, video_path):
    """Processes a video file and logs unique detected objects."""
    cap = cv2.VideoCapture(video_path)
    logged_ids = set()  # Track logged objects
    tracker = {}  # Track object positions
    min_confidence = 0.70  # Confidence threshold
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)  # Run YOLO model
        new_tracker = {}
        detected_objects = set()
        
        for result in results:
            for box in result.boxes:
                conf = box.conf[0].item()
                if conf < min_confidence:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[int(box.cls[0])]
                centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                # Assign or update object ID
                object_id = None
                for existing_id, prev_centroid in tracker.items():
                    if np.linalg.norm(np.array(prev_centroid) - np.array(centroid)) < 50:
                        object_id = existing_id
                        break
                
                if object_id is None:
                    object_id = len(tracker) + 1
                
                new_tracker[object_id] = centroid
                detected_objects.add((object_id, label, conf))
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {object_id} - {label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        logged_ids = {}

        # Log only new objects with context
        context = "Video object detection in progress"
        for obj in detected_objects:
            object_id, label, conf = obj  # Unpack tuple
            log_detected_objects(logged_ids, object_id, label, conf, context)
        
        tracker = new_tracker  # Update tracker with new frame info
        
        cv2.imshow("YOLOv8-Pose Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()


global_tracker = {}  # Ensure tracking persists across frames

def process_webcam(model):
    global global_tracker  # Use persistent tracking dictionary
    cap = cv2.VideoCapture(0)
    logged_ids = set()
    next_id = 0
    min_confidence = 0.70

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, show=True)
        new_tracker = {}

        detected_objects = set()

        for result in results:
            for box in result.boxes:
                conf = box.conf[0].item()
                if conf < min_confidence:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[int(box.cls[0])]
                centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                object_id = None
                for existing_id, prev_centroid in global_tracker.items():
                    if np.linalg.norm(np.array(prev_centroid) - np.array(centroid)) < 80:
                        object_id = existing_id
                        break
                
                if object_id is None:
                    object_id = next_id
                    next_id += 1
                
                new_tracker[object_id] = centroid
                detected_objects.add((object_id, label, conf))

        logged_ids = {}  # Dictionary to track the last logged context for each object

        # Call the function with debug prints
        for obj in detected_objects:
            object_id, label, conf = obj
            analyze_and_log_context(global_tracker, new_tracker, logged_ids, object_id, label, conf, log_detected_objects)
        
        global_tracker = new_tracker  # Update global tracker
        
        cv2.imshow("Webcam Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
