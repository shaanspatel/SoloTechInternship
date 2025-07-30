import sys
import os
import cv2
from ultralytics import YOLO

# Usage: python test_image_detection.py path/to/image.jpg

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_image_detection.py <image_path>")
        return
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    model_path = 'models/best.pt'
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return

    print("Loading model...")
    model = YOLO(model_path)
    print("Model loaded!")
    print("Model classes:", model.names)
    print("Single class model detected - all detections will be treated as 'clothes'")

    print(f"Reading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to read image: {image_path}")
        return

    print("Running detection...")
    results = model(image, conf=0.25)

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls] if cls < len(model.names) else f'class_{cls}'
                print(f"Detected: {class_name} (confidence: {conf:.2f}) at [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
                
                # Apply gray neutralization to detected object
                detected_region = image[int(y1):int(y2), int(x1):int(x2)]
                if detected_region.size > 0:  # Check if region is valid
                    gray_region = cv2.cvtColor(detected_region, cv2.COLOR_BGR2GRAY)
                    # Convert back to BGR for consistency
                    gray_region_bgr = cv2.cvtColor(gray_region, cv2.COLOR_GRAY2BGR)
                    # Apply the gray region back to the image
                    image[int(y1):int(y2), int(x1):int(x2)] = gray_region_bgr
                
                # Modern color scheme
                primary_color = (0, 165, 255)      # Orange
                secondary_color = (255, 255, 255)  # White
                accent_color = (0, 0, 0)           # Black
                
                # Draw modern bounding box with corner accents
                thickness = 3
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), primary_color, thickness)
                
                # Add corner accents
                corner_length = 15
                # Top-left corner
                cv2.line(image, (int(x1), int(y1)), (int(x1) + corner_length, int(y1)), primary_color, thickness)
                cv2.line(image, (int(x1), int(y1)), (int(x1), int(y1) + corner_length), primary_color, thickness)
                # Top-right corner
                cv2.line(image, (int(x2) - corner_length, int(y1)), (int(x2), int(y1)), primary_color, thickness)
                cv2.line(image, (int(x2), int(y1)), (int(x2), int(y1) + corner_length), primary_color, thickness)
                # Bottom-left corner
                cv2.line(image, (int(x1), int(y2) - corner_length), (int(x1), int(y2)), primary_color, thickness)
                cv2.line(image, (int(x1), int(y2)), (int(x1) + corner_length, int(y2)), primary_color, thickness)
                # Bottom-right corner
                cv2.line(image, (int(x2) - corner_length, int(y2)), (int(x2), int(y2)), primary_color, thickness)
                cv2.line(image, (int(x2), int(y2)), (int(x2), int(y2) - corner_length), primary_color, thickness)
                
                # Draw modern label
                label = f'{class_name.upper()} {conf:.2f}'
                font_scale = 0.7
                font_thickness = 2
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                
                # Label background with padding
                padding = 8
                label_bg_x1 = int(x1)
                label_bg_y1 = int(y1) - label_size[1] - padding * 2
                label_bg_x2 = int(x1) + label_size[0] + padding * 2
                label_bg_y2 = int(y1)
                
                # Semi-transparent background
                overlay = image.copy()
                cv2.rectangle(overlay, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), primary_color, -1)
                cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
                
                # Label border
                cv2.rectangle(image, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), primary_color, 2)
                
                # Text with shadow effect
                text_x = int(x1) + padding
                text_y = int(y1) - padding
                
                # Text shadow
                cv2.putText(image, label, (text_x + 1, text_y + 1), 
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, accent_color, font_thickness)
                # Main text
                cv2.putText(image, label, (text_x, text_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, secondary_color, font_thickness)

    # Save output image
    out_path = os.path.join('output', 'detected_output_gray.jpg')
    os.makedirs('output', exist_ok=True)
    cv2.imwrite(out_path, image)
    print(f"Detection complete. Gray neutralized output saved to {out_path}")

if __name__ == "__main__":
    main() 