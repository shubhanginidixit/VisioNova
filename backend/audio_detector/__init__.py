"""
VisioNova Audio Detector Module
Detects AI-generated and deepfake audio using pretrained HuggingFace models.

Components:
- AudioDeepfakeDetector: Wav2Vec2-based deepfake audio detection (99.7% accuracy)

Models:
- MelodyMachine/Deepfake-audio-detection-V2: Wav2Vec2ForSequenceClassification (95M params)
  Fine-tuned on deepfake audio datasets, supports 16kHz mono audio input.
"""

from .audio_detector import AudioDeepfakeDetector

__all__ = [
    'AudioDeepfakeDetector',
]
