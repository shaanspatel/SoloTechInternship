import cv2
import requests
import time
import sys
import test_tts


# MCP API URL
MCP_URL = "http://localhost:8000/predict"

def process_frame(frame):
    """Process a single frame through the YOLO MCP"""
    try:
        # Save frame to temporary file
        frame_path = "current_frame.jpg"
        cv2.imwrite(frame_path, frame)
        
        # Prepare request payload
        payload = {
            "data_path": frame_path,
            "threshold": 0.5
        }
        
        # Send to YOLO MCP
        response = requests.post(MCP_URL, json=payload, timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            print(f"❌ MCP API error: {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {str(e)}")
        return None
    except Exception as e:
        print(f"❌ Processing error: {str(e)}")

        return None
def sayWords():
    print("🎯 Detected 1 objects:")
    # Call the test_tts_mcp function from the test_tts module
    test_tts.test_tts_mcp()
    sys.exit("Object Detected and audio played")



def run_live_detection():
    """Run live video detection"""
    print("🚀 Starting Live Video Detection")
    print("Controls: 'q' to quit")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    print("✅ Camera opened successfully")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame")
                break

            # Process frame
            result = process_frame(frame)
            
            if result:
                # Extract bounding boxes and confidence
                bounding_boxes = result.get("result", [])
                confidence_scores = result.get("confidence", [])
                
                # Print results
                if bounding_boxes:
                    print(f"🎯 Detected {len(bounding_boxes)} objects:")
                    for i, (bbox, conf) in enumerate(zip(bounding_boxes, confidence_scores)):
                        x1, y1, x2, y2 = bbox
                        print(f"   Object {i+1}: Box [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}], Confidence: {conf:.3f}")
                        if conf > 0.8:  # Only call sayWords for high confidence detections
                            sayWords()
                else:
                    print("🔍 No objects detected in this frame")
            else:
                print("❌ Failed to process frame")
            # Show the frame
            cv2.imshow("Live Detection", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 Quitting...")
                break

            # Small delay to prevent overloading
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Live detection stopped")

if __name__ == "__main__":
    run_live_detection() 