import cv2
import torch
from ultralytics import YOLO

# Load trained YOLOv8 model
model_path = "runs/detect/train7/weights/best.pt"  # Update if needed
model = YOLO(model_path)

# Set confidence threshold
CONFIDENCE_THRESHOLD = 0.3  # Adjust if needed

# Open the webcam
cap = cv2.VideoCapture(0)  # Use 1 or 2 if external camera is used

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Perform YOLOv8 inference
    results = model(frame, conf=CONFIDENCE_THRESHOLD)  

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0].item())  # Get class index
            conf = box.conf[0].item()  # Confidence score
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

            # Get class name from model
            label = model.names[cls]
            print(f"Detected: {label}, Confidence: {conf:.2f}")

            # Define colors for different classes
            color_map = {
                "Filling": (255, 0, 0),  # Blue
                "Filled": (0, 255, 0),   # Green
                "Overfilled": (0, 0, 255) # Red
            }
            color = color_map.get(label, (255, 255, 255))

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Display the output
    cv2.imshow("YOLOv8 Sand Level Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
