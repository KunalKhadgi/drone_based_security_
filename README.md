# Object Detection and Motion Analysis with YOLOv8

## Overview
This project implements object detection and motion analysis using the YOLOv8 model. It processes video files, images, and webcam streams to identify and track objects. The detections are logged into a SQLite database for further analysis.

## Features
- **YOLOv8-based Object Detection**: Detects and tracks objects in videos, images, and real-time webcam streams.
- **Motion Analysis**: Analyzes object movement across frames.
- **Database Logging**: Logs detected objects with timestamps and context.
- **Video and Webcam Processing**: Supports both file-based and real-time detection.
- **Entry Zone Logging**: Detects and logs objects entering specific zones.

## Project Structure
```
|-- main.py                 # Entry point for running video and webcam processing
|-- object_detection.py      # Handles YOLOv8-based detection and tracking
|-- motion_detection.py      # Motion analysis and logging
|-- log_query.py            # Queries and retrieves logs from SQLite database
|-- entry_zone_logger.py     # Logs objects entering a defined entry zone
|-- yolov8n-pose.pt          # Pre-trained YOLOv8 model files 
|-- detection_log.db        # SQLite database storing detected objects
|-- detection_log.txt       # Log file storing object detections
|-- .env                    # Environment variables configuration
```

## Installation
### Prerequisites
Ensure you have Python 3.8+ installed and set up a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Required Dependencies
Install necessary Python libraries:
```bash
pip install ultralytics numpy opencv-python sqlite3 python-dotenv
```

## Usage
### Processing a Video File

* Open main.py
* set value of mode to either of `image`, `video` or `webcam`
* if image or video provide the file path in 'path' variable 

```bash
python main.py
```

### Processing a Webcam Stream
```bash
python main.py
```

### Setting Up Log Query (.env file)
1. Create a `.env` file in the root directory:
```ini
DATABASE_PATH=detection_log.db
LOG_FILE=detection_log.txt
```
2. Ensure the `.env` file is loaded properly in `log_query.py`:
```python
from dotenv import load_dotenv
load_dotenv()
```
3. Run the query script:
```bash
python log_query.py
```

### Querying Logs
To view logged detections from the database, run:
```bash
python log_query.py
```

## Flow of the Project
1. **Loading the Model**
   - YOLOv8 model is loaded from the `models/` directory.
2. **Processing Video/Webcam**
   - Reads frames from a video file or webcam.
   - Runs YOLOv8 detection on each frame.
3. **Tracking Objects**
   - Assigns unique IDs to detected objects.
   - Tracks object movement across frames.
4. **Motion Analysis**
   - Compares positions between frames.
   - Identifies entry/exit events.
5. **Entry Zone Detection (In different folder/Independent flow but same logic 1-4)**
   - Monitors and logs objects entering defined zones.
   - Stores entry data in `detection_log.db`.
6. **Logging Detections**
   - Stores detected objects in `detection_log.db`.
   - Logs detection events in `detection_log.txt`.

---
This README provides a structured guide for setting up and running the project. 

