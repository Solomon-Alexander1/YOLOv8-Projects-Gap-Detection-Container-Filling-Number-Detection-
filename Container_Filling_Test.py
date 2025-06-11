import cv2
from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO("runs/detect/train4/weights/best.pt")  # Update path if needed

# Open the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

# Set frame width and height (Optional)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

while cap.isOpened():
    success, frame = cap.read()  # Read frame from the webcam
    if not success:
        print("⚠️ Failed to grab frame")
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Plot the detections on the frame
    annotated_frame = results[0].plot()

    # Display the output frame
    cv2.imshow("YOLOv8 Sand Level Detection", annotated_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
