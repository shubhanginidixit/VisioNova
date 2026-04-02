"""
VisioNova Audio Detector Module
Detects AI-generated and deepfake audio using a 5-model weighted ensemble.

Ensemble Members:
- Gustking/wav2vec2-large-xlsr-deepfake-audio-classification (XLS-R 300M, EER 4.01%)
- DavidCombei/wavLM-base-Deepfake_V2 (WavLM-base, 99.62% accuracy)
- Vansh180/deepfake-audio-wav2vec2 (Wav2Vec2-base, bonafide/spoof)
- MelodyMachine/Deepfake-audio-detection-V2 (Wav2Vec2-base, 99.73% accuracy)
- mo-thecreator/Deepfake-audio-detection (Wav2Vec2-base, dataset diversity)

Supports 16kHz mono audio input up to 60 seconds.
"""

from .audio_detector import AudioDeepfakeDetector, AudioEnsembleDetector

__all__ = [
    'AudioDeepfakeDetector',
    'AudioEnsembleDetector',
]
