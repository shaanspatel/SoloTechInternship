import cv2
import os
from ultralytics import YOLO

def test_setup():
    """Test if everything is set up correctly"""
    
    print("🧪 Testing Local Setup")
    print("=" * 30)
    
    # Test 1: Check if model exists
    model_path = 'models/best.pt'
    if os.path.exists(model_path):
        print("✅ Model found")
    else:
        print("❌ Model not found")
        print("Please download the model to models/best.pt")
        return False
    
    # Test 2: Try to load model
    try:
        print("Loading model...")
        model = YOLO(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Test 3: Check webcam
    print("Testing webcam...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("✅ Webcam accessible")
        ret, frame = cap.read()
        if ret:
            print(f"✅ Webcam working (frame size: {frame.shape})")
        else:
            print("⚠️  Webcam accessible but can't read frames")
        cap.release()
    else:
        print("❌ Webcam not accessible")
        print("Try different camera IDs or check permissions")
    
    # Test 4: Check OpenCV
    print(f"✅ OpenCV version: {cv2.__version__}")
    
    print("\n🎉 Setup test complete!")
    return True

if __name__ == "__main__":
    test_setup()
