"""
VisioNova Video Detector Module
Detects deepfake and AI-generated videos using pretrained HuggingFace models.

Components:
- VideoDeepfakeDetector: Frame-level and temporal deepfake detection

Models:
- Naman712/Deep-fake-detection: ResNeXt50 + LSTM (87% accuracy)
  Processes video frames for spatial features + temporal consistency analysis.
"""

from .video_detector import VideoDeepfakeDetector

__all__ = [
    'VideoDeepfakeDetector',
]
