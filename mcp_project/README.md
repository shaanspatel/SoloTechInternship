# MCP Test Project

This project contains various MCP (Model Context Protocol) implementations and tests developed during the Solo Tech Internship.

## 📁 Project Structure

```
mcp_project/
├── client.py              # MCP client implementation
├── tts_mcp.py             # Text-to-Speech MCP server
├── Yolov11nMCP.py         # YOLO object detection MCP server
├── test_tts.py            # TTS functionality tests
├── test.py                # General MCP tests
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── current_frame.jpg     # Test image for object detection
└── test_output.wav       # Test audio output
```

## 🚀 Features

### Text-to-Speech (TTS) MCP Server
- **File:** `tts_mcp.py`
- **Description:** MCP server implementation for text-to-speech functionality
- **Features:**
  - Converts text to speech using TTS libraries
  - Supports multiple audio formats
  - Real-time audio generation

### YOLO Object Detection MCP Server
- **File:** `Yolov11nMCP.py`
- **Description:** MCP server for YOLO-based object detection
- **Features:**
  - Real-time object detection using YOLOv11n
  - Image processing capabilities
  - Detection result formatting

### MCP Client
- **File:** `client.py`
- **Description:** Client implementation for testing MCP servers
- **Features:**
  - Connects to MCP servers
  - Sends requests and processes responses
  - Error handling and logging

## 🛠️ Setup and Installation

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run TTS Tests:**
   ```bash
   python test_tts.py
   ```

3. **Run General MCP Tests:**
   ```bash
   python test.py
   ```

## 📋 Requirements

See `requirements.txt` for the complete list of Python dependencies.

## 🔧 Usage Examples

### Testing TTS Functionality
```python
from test_tts import play_audio, generate_speech

# Generate speech from text
audio = generate_speech("Hello, this is a test message")

# Play the generated audio
play_audio(audio, sample_rate=22050)
```

### Testing Object Detection
```python
from Yolov11nMCP import YOLOMCPServer

# Initialize YOLO MCP server
server = YOLOMCPServer()

# Process image for object detection
results = server.detect_objects("current_frame.jpg")
```

## 📝 Testing

The project includes comprehensive test files:
- `test_tts.py`: Tests text-to-speech functionality
- `test.py`: General MCP protocol tests

## 🤝 Contributing

This project is part of the Solo Tech Internship program. Please refer to the main repository guidelines for contribution standards.

## 📄 License

This project is part of the Solo Tech Internship program. 