from object_detection import load_model, process_image, process_video, process_webcam

# Set mode manually ("image", "video", "webcam")
mode = "video"  # Change this as needed
path = "test_cases/nana Patekar laughing meme #shorts.mp4"  # Set image or video path if required 

# Load YOLOv8m-Pose model
model = load_model("yolov8n-pose.pt")
# pose_model = load_model("yolov8m-pose.pt")

if mode == "image":
    if not path:
        print("Error: Provide image path.")
    else:
        process_image(model, path)
elif mode == "video":
    if not path:
        print("Error: Provide video path.")
    else:
        process_video(model, path)
elif mode == "webcam":
    process_webcam(model)
