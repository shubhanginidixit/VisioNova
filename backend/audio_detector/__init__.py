"""
VisioNova Audio Detector Module
Detects AI-generated and deepfake audio using NII AntiDeepfake models (ASRU 2025).

Models (loaded by priority, auto-fallback on OOM):
- nii-yamagishilab/xls-r-1b-anti-deepfake: XLS-R 1B post-trained, EER 1.35% (In-the-Wild)
- nii-yamagishilab/wav2vec-large-anti-deepfake: Wav2Vec2 Large post-trained, EER 1.91% (In-the-Wild)

Paper: "Post-training for Deepfake Speech Detection" (arXiv:2506.21090)
Training data: 74,000+ hours across 100+ languages (CC BY-NC-SA 4.0)
"""

from .audio_detector import AudioDeepfakeDetector

__all__ = [
    'AudioDeepfakeDetector',
]
