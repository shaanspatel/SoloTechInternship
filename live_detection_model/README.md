# Local Real-time T-shirt Detection

This is your local setup for real-time t-shirt and hoodie detection.

## Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the Trained Model:**
   - Copy the `best.pt` file from your Lightning AI studio to the `models/` folder
   - Or download it from: `/teamspace/studios/this_studio/runs/detect/train2/weights/best.pt`

3. **Run Real-time Detection:**
   ```bash
   python webcam_detection.py
   ```

## Controls

- **q** - Quit detection
- **s** - Save screenshot
- **c** - Change confidence threshold

## Troubleshooting

### Webcam Issues
- Try different camera IDs: Change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` or `cv2.VideoCapture(2)`
- Check if your webcam is being used by another application
- Ensure you have proper permissions

### Model Issues
- Make sure `models/best.pt` exists
- Check file permissions
- Verify the model file is not corrupted

### Performance Issues
- Use GPU if available (install CUDA version of PyTorch)
- Reduce input resolution in the script
- Lower confidence threshold
- Close other applications using GPU

## File Structure

```
local_detection_system/
├── models/
│   └── best.pt              # Your trained model
├── output/                  # Screenshots and results
├── test_images/             # Test images
├── webcam_detection.py      # Main detection script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## System Requirements

- Python 3.8+
- Webcam
- 4GB+ RAM
- GPU recommended (but not required)
- Windows/Mac/Linux

## Performance Tips

1. **GPU Acceleration:** Install CUDA version of PyTorch for better performance
2. **Resolution:** Lower resolution = higher FPS
3. **Confidence:** Higher threshold = fewer false positives
4. **Model Size:** YOLO11n is optimized for speed
# SoloTechInternship
