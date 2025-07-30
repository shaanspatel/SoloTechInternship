import cv2
from ultralytics import YOLO
import time
import os
import numpy as np

def main():
    # Model path - update this to your trained model location
    model_path = '/Users/shaanpatel/Desktop/Personal/solo/model tests/local_detection_system/live_detection_model/models/best.pt'
    
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
    
    # Class names - single class model
    class_name = 'clothes'  # Single class
    
    # Modern color scheme for bounding boxes (BGR format)
    primary_color = (0, 165, 255)      # Orange (modern, vibrant)
    secondary_color = (255, 255, 255)  # White for text
    accent_color = (0, 0, 0)           # Black for contrast
    
    # Performance tracking
    fps_counter = 0
    fps_start_time = time.time()
    fps = 0
    
    print("🎥 Real-time detection started!")
    print("Controls:")
    print("- Press 'q' to quit")
    print("- Press 's' to save screenshot")
    print("- Press 'c' to change confidence threshold")
    print("- Press 'g' to toggle gray neutralization")
    
    confidence = 0.23
    gray_neutralization = True  # Enable gray neutralization by default
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Convert frame to grayscale and back to 3-channel for model input
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame_3ch = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

        # Create a copy for drawing
        display_frame = frame.copy()
        
        # Run detection on grayscale frame
        results = model(gray_frame_3ch, conf=confidence, verbose=False)
        
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
                        # Apply gray neutralization to detected object
                        if gray_neutralization:
                            # Convert the detected region to grayscale
                            detected_region = display_frame[y1:y2, x1:x2]
                            if detected_region.size > 0:  # Check if region is valid
                                gray_region = cv2.cvtColor(detected_region, cv2.COLOR_BGR2GRAY)
                                # Convert back to BGR for consistency
                                gray_region_bgr = cv2.cvtColor(gray_region, cv2.COLOR_GRAY2BGR)
                                # Apply the gray region back to the display frame
                                display_frame[y1:y2, x1:x2] = gray_region_bgr
                        
                        # Draw modern bounding box with rounded corners effect
                        thickness = 3
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), primary_color, thickness)
                        
                        # Add corner accents for modern look
                        corner_length = 15
                        # Top-left corner
                        cv2.line(display_frame, (x1, y1), (x1 + corner_length, y1), primary_color, thickness)
                        cv2.line(display_frame, (x1, y1), (x1, y1 + corner_length), primary_color, thickness)
                        # Top-right corner
                        cv2.line(display_frame, (x2 - corner_length, y1), (x2, y1), primary_color, thickness)
                        cv2.line(display_frame, (x2, y1), (x2, y1 + corner_length), primary_color, thickness)
                        # Bottom-left corner
                        cv2.line(display_frame, (x1, y2 - corner_length), (x1, y2), primary_color, thickness)
                        cv2.line(display_frame, (x1, y2), (x1 + corner_length, y2), primary_color, thickness)
                        # Bottom-right corner
                        cv2.line(display_frame, (x2 - corner_length, y2), (x2, y2), primary_color, thickness)
                        cv2.line(display_frame, (x2, y2), (x2, y2 - corner_length), primary_color, thickness)
                        
                        # Draw modern label with gradient effect
                        label = f'{class_name.upper()} {conf:.2f}'
                        font_scale = 0.7
                        font_thickness = 2
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                        
                        # Label background with padding
                        padding = 8
                        label_bg_x1 = x1
                        label_bg_y1 = y1 - label_size[1] - padding * 2
                        label_bg_x2 = x1 + label_size[0] + padding * 2
                        label_bg_y2 = y1
                        
                        # Draw semi-transparent background
                        overlay = display_frame.copy()
                        cv2.rectangle(overlay, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), primary_color, -1)
                        cv2.addWeighted(overlay, 0.8, display_frame, 0.2, 0, display_frame)
                        
                        # Draw label border
                        cv2.rectangle(display_frame, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), primary_color, 2)
                        
                        # Draw label text with shadow effect
                        text_x = x1 + padding
                        text_y = y1 - padding
                        
                        # Text shadow
                        cv2.putText(display_frame, label, (text_x + 1, text_y + 1), 
                                  cv2.FONT_HERSHEY_SIMPLEX, font_scale, accent_color, font_thickness)
                        # Main text
                        cv2.putText(display_frame, label, (text_x, text_y), 
                                  cv2.FONT_HERSHEY_SIMPLEX, font_scale, secondary_color, font_thickness)
        
        # Calculate and draw FPS
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()
        
        # Draw modern status overlay
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (5, 5), (200, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
        
        # Draw FPS and confidence with modern styling
        cv2.putText(display_frame, f'FPS: {fps}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, secondary_color, 2)
        cv2.putText(display_frame, f'Conf: {confidence:.2f}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, secondary_color, 2)
        
        # Show gray neutralization status
        status = "ON" if gray_neutralization else "OFF"
        status_color = (0, 255, 0) if gray_neutralization else (0, 0, 255)  # Green if ON, Red if OFF
        cv2.putText(display_frame, f'Gray: {status}', (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Display frame
        cv2.imshow('Real-time Clothing Detection - Modern UI', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save screenshot
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"output/screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, display_frame)
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
        elif key == ord('g'):
            # Toggle gray neutralization
            gray_neutralization = not gray_neutralization
            status = "ON" if gray_neutralization else "OFF"
            print(f"Gray neutralization: {status}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped.")

if __name__ == "__main__":
    main()
