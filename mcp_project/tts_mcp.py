from pydantic import BaseModel
from litserve.mcp import MCP
import litserve as ls
import base64
import soundfile as sf
from io import BytesIO
from kokoro import KPipeline
import torch
import numpy as np

# Define the request schema for TTS
class TtsRequest(BaseModel):
    text: str
    voice: str = "af_heart"  # Default to af_heart as shown in the example
    speed: float = 1.0

# Define the MCP API
class Tts(ls.LitAPI):
    def setup(self, device: str):
        # Initialize Kokoro pipeline
        # Force CPU usage to avoid GPU requirements
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Initialize the pipeline with American English
        self.pipeline = KPipeline(lang_code="a")  # 'a' = American English
        
        # Ensure the pipeline runs on CPU
        if hasattr(self.pipeline, 'to'):
            self.pipeline.to('cpu')
        
        print("TTS MCP initialized with Kokoro TTS")

    def decode_request(self, request: TtsRequest):
        return {
            "text": request.text,
            "voice": request.voice,
            "speed": request.speed
        }

    def predict(self, inputs: dict):
        text = inputs["text"]
        voice = inputs["voice"]
        speed = inputs["speed"]

        try:
            # Run Kokoro TTS pipeline
            # The pipeline returns a generator that yields (gs, ps, audio) tuples
            generator = self.pipeline(text, voice=voice)
            
            # Get the first (and typically only) audio segment
            # gs = grapheme sequence, ps = phoneme sequence, audio = audio data
            gs, ps, audio = next(generator)
            
            # Ensure audio is a numpy array
            if torch.is_tensor(audio):
                audio = audio.cpu().numpy()
            elif isinstance(audio, list):
                audio = np.array(audio)
            
            # Apply speed adjustment if needed
            if speed != 1.0:
                # Simple speed adjustment by resampling
                # This is a basic implementation - for production, consider using librosa
                original_length = len(audio)
                new_length = int(original_length / speed)
                indices = np.linspace(0, original_length - 1, new_length, dtype=int)
                audio = audio[indices]
            
            # Convert audio to list for JSON serialization
            audio_list = audio.tolist()

            return {
                "text": text,
                "voice": voice,
                "speed": speed,
                "grapheme_sequence": gs,
                "phoneme_sequence": ps,
                "audio_data": audio_list,  # Raw audio data as list
                "sample_rate": 24000,
                "audio_format": "float32"
            }
            
        except Exception as e:
            # Return error information if synthesis fails
            return {
                "text": text,
                "voice": voice,
                "speed": speed,
                "error": str(e),
                "audio_data": []
            }

    def encode_response(self, output: dict):
        return {
            "text": output["text"],
            "voice": output["voice"],
            "speed": output["speed"],
            "grapheme_sequence": output.get("grapheme_sequence", ""),
            "phoneme_sequence": output.get("phoneme_sequence", ""),
            "audio_data": output["audio_data"],
            "sample_rate": output.get("sample_rate", 24000),
            "audio_format": output.get("audio_format", "float32"),
            "error": output.get("error", "")
        }

# Package and publish the MCP tool
if __name__ == "__main__":
    mcp = MCP(name="TextToSpeech", description="Convert text to speech using Kokoro TTS")
    api = Tts(mcp=mcp)
    server = ls.LitServer(api)
    server.run(port=8001) 