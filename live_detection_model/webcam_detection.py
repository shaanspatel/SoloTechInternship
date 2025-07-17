import cv2
from ultralytics import YOLO
import time
import os

def main():
    # Model path - update this to your trained model location
    model_path = 'models/best.pt'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        print("Please download the trained model to the 'models' folder")
        return
    
    # Load the trained model
    print("Loading model...")
    model = YOLO(model_path)
    print("Model loaded successfully!")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)  # Use 0 for default camera
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        print("Try changing camera ID (1, 2, etc.) or check webcam permissions")
        return
    
    # Set webcam properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Class names
    class_names = ['hoodie', 'tshirt']
    
    # Colors for bounding boxes (BGR format)
    colors = {
        'hoodie': (0, 255, 0),    # Green
        'tshirt': (255, 0, 0)     # Blue
    }
    
    # Performance tracking
    fps_counter = 0
    fps_start_time = time.time()
    fps = 0
    
    print("🎥 Real-time detection started!")
    print("Controls:")
    print("- Press 'q' to quit")
    print("- Press 's' to save screenshot")
    print("- Press 'c' to change confidence threshold")
    
    confidence = 0.23
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Run detection
        results = model(frame, conf=confidence, verbose=False)
        
        # Draw detections
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get confidence and class
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    if conf >= confidence:
                        # Get class name
                        class_name = class_names[cls] if cls < len(class_names) else f'class_{cls}'
                        color = colors.get(class_name, (0, 255, 255))
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label background
                        label = f'{class_name} {conf:.2f}'
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), color, -1)
                        
                        # Draw label text
                        cv2.putText(frame, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Calculate and draw FPS
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()
        
        # Draw FPS and confidence
        cv2.putText(frame, f'FPS: {fps}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Conf: {confidence:.2f}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Real-time T-shirt Detection', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save screenshot
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"output/screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Screenshot saved as {filename}")
        elif key == ord('c'):
            # Change confidence threshold
            try:
                new_conf = float(input("Enter new confidence threshold (0.1-1.0): "))
                if 0.1 <= new_conf <= 1.0:
                    confidence = new_conf
                    print(f"Confidence threshold changed to {confidence}")
                else:
                    print("Invalid confidence value. Must be between 0.1 and 1.0")
            except ValueError:
                print("Invalid input. Confidence threshold unchanged.")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped.")

if __name__ == "__main__":
    main()
