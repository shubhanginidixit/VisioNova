import json
import wave
import struct
import math
import os
import sys

# Generate a dummy 1-second 440Hz sine wave (A4)
def create_dummy_wav(filename="test_audio.wav"):
    sample_rate = 16000
    duration = 1.0 # seconds
    frequency = 440.0 # Hz
    
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1) # mono
        wav_file.setsampwidth(2) # 2 bytes per sample (16-bit)
        wav_file.setframerate(sample_rate)
        
        for i in range(int(sample_rate * duration)):
            value = int(32767.0 * math.sin(2.0 * math.pi * frequency * i / sample_rate))
            data = struct.pack('<h', value)
            wav_file.writeframesraw(data)
    print(f"Created {filename}")

if __name__ == "__main__":
    # Add backend to path so imports work
    sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
    
    from audio_detector.audio_detector import AudioDeepfakeDetector
    
    create_dummy_wav()
    
    print("Initializing Audio Detector...")
    detector = AudioDeepfakeDetector(use_gpu=True)
    
    print("Running prediction on dummy audio...")
    result = detector.predict("test_audio.wav")
    
    print("\n[RESULT]")
    print(json.dumps(result, indent=2))
    
    # Cleanup
    if os.path.exists("test_audio.wav"):
        os.remove("test_audio.wav")
