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
                # Draw bounding box
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                label = f'{class_name} {conf:.2f}'
                cv2.putText(image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Save output image
    out_path = os.path.join('output', 'detected_output.jpg')
    os.makedirs('output', exist_ok=True)
    cv2.imwrite(out_path, image)
    print(f"Detection complete. Output saved to {out_path}")

if __name__ == "__main__":
    main() 