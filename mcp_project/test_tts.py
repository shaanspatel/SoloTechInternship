#!/usr/bin/env python3
"""
Test script for the Kokoro TTS MCP integration
"""

import requests
import soundfile as sf
from io import BytesIO
import numpy as np
import subprocess
import os
import platform

# Audio playback imports (optional)
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

def test_tts_mcp():
    """Test the TTS MCP with Kokoro integration"""
    
    # MCP server URL (assuming it's running on port 8002)
    url = "http://localhost:8001/predict"
    
    # Test text from the Kokoro example
    test_text = """
    Object Detected! Picking up your clothes now. 
    """
    
    # Test request payload
    payload = {
        "text": test_text,
        "voice": "af_nicole",  # Using the voice from the example
        "speed": 1.0
    }
    
    try:
        print("Sending TTS request to MCP server...")
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            if "error" in result and result["error"]:
                print(f"Error: {result['error']}")
                return
            
            print(f"Success! Generated audio for text: {result['text'][:100]}...")
            print(f"Voice used: {result['voice']}")
            print(f"Speed: {result['speed']}")
            print(f"Sample rate: {result['sample_rate']}")
            
            # Get the audio data directly (no need to decode base64)
            audio_data = result["audio_data"]
            sample_rate = result["sample_rate"]
            
            # Convert list back to numpy array
            audio = np.array(audio_data, dtype=np.float32)
            print(f"Audio loaded: {len(audio)} samples at {sample_rate} Hz")
            
            # Save as WAV file
            sf.write("test_output.wav", audio, sample_rate)
            print("Audio saved as 'test_output.wav'")
            
            # Play the audio
            print("Playing audio...")
            play_audio(audio, sample_rate)
            
        else:
            print(f"Error: HTTP {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to MCP server. Make sure it's running on port 8001.")
    except Exception as e:
        print(f"Error: {e}")

def play_audio(audio, sample_rate):
    """Play audio using available audio libraries or system player"""
    if PYGAME_AVAILABLE:
        try:
            # Use pygame (most reliable)
            pygame.mixer.init(frequency=sample_rate, size=-16, channels=1)
            pygame.mixer.music.load(BytesIO(audio.tobytes()))
            pygame.mixer.music.play()
            
            # Wait for audio to finish
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
            
            pygame.mixer.quit()
            print("Audio playback completed!")
            return
        except Exception as e:
            print(f"Pygame failed: {e}")
    
    # Fallback to system audio player
    try:
        system = platform.system()
        if system == "Darwin":  # macOS
            subprocess.run(["afplay", "test_output.wav"], check=True)
            print("Audio playback completed using afplay!")
        elif system == "Linux":
            # Try different Linux audio players
            for player in ["aplay", "paplay", "mpg123"]:
                try:
                    subprocess.run([player, "test_output.wav"], check=True)
                    print(f"Audio playback completed using {player}!")
                    return
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            print("No Linux audio player found")
        elif system == "Windows":
            subprocess.run(["start", "test_output.wav"], shell=True, check=True)
            print("Audio playback started using Windows default player!")
        else:
            print(f"Unknown system: {system}")
            
    except Exception as e:
        print(f"System audio player failed: {e}")
        print("Audio saved as 'test_output.wav' - you can play it manually.")

if __name__ == "__main__":
    test_tts_mcp() 