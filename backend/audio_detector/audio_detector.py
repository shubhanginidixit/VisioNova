"""
Audio Deepfake Detector using pretrained Wav2Vec2 from HuggingFace.

Model: MelodyMachine/Deepfake-audio-detection-V2
- Architecture: Wav2Vec2ForSequenceClassification (95M params)
- Training: Fine-tuned on deepfake audio datasets
- Accuracy: 99.7% on evaluation set
- Input: 16kHz mono audio (auto-resampled if needed)
- Labels: "Bonafide" (real) / "Spoof" (AI-generated/deepfake)
"""

import os
import logging
import tempfile
import numpy as np
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# HuggingFace model ID
HF_MODEL_ID = "MelodyMachine/Deepfake-audio-detection-V2"
TARGET_SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH_SEC = 30  # Process at most 30 seconds


class AudioDeepfakeDetector:
    """Detect AI-generated/deepfake audio using Wav2Vec2."""

    def __init__(self, use_gpu: bool = False):
        """Initialize the audio detector.
        
        Args:
            use_gpu: Whether to use GPU for inference (if available)
        """
        self.use_gpu = use_gpu
        self.model = None
        self.feature_extractor = None
        self.device = None
        self.model_loaded = False
        self._load_attempted = False
        
    def _load_model(self):
        """Lazy-load the Wav2Vec2 model from HuggingFace."""
        if self._load_attempted:
            return
        self._load_attempted = True
        
        try:
            import torch
            from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
            
            self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
            
            logger.info(f"Loading audio deepfake detector from {HF_MODEL_ID}...")
            print(f"Loading audio deepfake detector: {HF_MODEL_ID}...")
            
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(HF_MODEL_ID)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(HF_MODEL_ID)
            self.model.to(self.device)
            self.model.eval()
            
            self.model_loaded = True
            logger.info(f"Audio detector loaded successfully on {self.device}")
            print(f"Audio detector loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load audio detector: {e}")
            print(f"Failed to load audio detector: {e}")
            self.model_loaded = False

    def _load_audio(self, audio_path: str) -> Tuple[Optional[np.ndarray], int]:
        """Load and preprocess audio file.
        
        Supports: .wav, .mp3, .flac, .ogg, .m4a, .webm
        Returns: (audio_array, sample_rate) or (None, 0) on failure
        """
        try:
            import soundfile as sf
            
            audio, sr = sf.read(audio_path)
            
            # Convert stereo to mono
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample to target rate if needed
            if sr != TARGET_SAMPLE_RATE:
                try:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
                    sr = TARGET_SAMPLE_RATE
                except ImportError:
                    # Manual resampling (basic linear interpolation)
                    duration = len(audio) / sr
                    target_len = int(duration * TARGET_SAMPLE_RATE)
                    audio = np.interp(
                        np.linspace(0, len(audio) - 1, target_len),
                        np.arange(len(audio)),
                        audio
                    )
                    sr = TARGET_SAMPLE_RATE
            
            # Trim to max length
            max_samples = MAX_AUDIO_LENGTH_SEC * sr
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            
            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            return audio.astype(np.float32), sr
            
        except ImportError:
            logger.error("soundfile library not installed. Install with: pip install soundfile")
            return None, 0
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            return None, 0

    def _load_audio_from_bytes(self, audio_bytes: bytes, filename: str = "audio.wav") -> Tuple[Optional[np.ndarray], int]:
        """Load audio from bytes (for API uploads)."""
        try:
            suffix = os.path.splitext(filename)[1] or '.wav'
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            
            result = self._load_audio(tmp_path)
            os.unlink(tmp_path)
            return result
        except Exception as e:
            logger.error(f"Failed to load audio from bytes: {e}")
            return None, 0

    def predict(self, audio_input, filename: str = "audio.wav") -> Dict:
        """Detect if audio is AI-generated/deepfake.
        
        Args:
            audio_input: File path (str), bytes, or numpy array (16kHz mono float32)
            filename: Original filename (used for extension detection when input is bytes)
        
        Returns:
            Dict with detection results:
            - prediction: "real" or "ai_generated"
            - confidence: 0-100 confidence score
            - ai_probability: 0-100 probability of being AI-generated
            - human_probability: 0-100 probability of being real
            - model: Model identifier
            - success: Whether detection succeeded
        """
        # Ensure model is loaded
        self._load_model()
        if not self.model_loaded:
            return {
                'prediction': 'unknown',
                'confidence': 0,
                'ai_probability': 50,
                'human_probability': 50,
                'model': HF_MODEL_ID,
                'success': False,
                'error': 'Model not loaded'
            }
        
        try:
            import torch
            
            # Load audio based on input type
            if isinstance(audio_input, str):
                audio, sr = self._load_audio(audio_input)
            elif isinstance(audio_input, bytes):
                audio, sr = self._load_audio_from_bytes(audio_input, filename)
            elif isinstance(audio_input, np.ndarray):
                audio = audio_input
                sr = TARGET_SAMPLE_RATE
            else:
                return {
                    'prediction': 'unknown',
                    'confidence': 0,
                    'ai_probability': 50,
                    'human_probability': 50,
                    'model': HF_MODEL_ID,
                    'success': False,
                    'error': f'Unsupported input type: {type(audio_input)}'
                }
            
            if audio is None:
                return {
                    'prediction': 'unknown',
                    'confidence': 0,
                    'ai_probability': 50,
                    'human_probability': 50,
                    'model': HF_MODEL_ID,
                    'success': False,
                    'error': 'Failed to load audio file'
                }
            
            # Run inference
            inputs = self.feature_extractor(
                audio,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            prob_values = probs[0].tolist()
            
            # Determine label mapping from model config
            id2label = getattr(self.model.config, 'id2label', None)
            real_prob = 0.5
            fake_prob = 0.5
            
            if id2label:
                for idx, label in id2label.items():
                    label_lower = label.lower()
                    if any(r in label_lower for r in ['bonafide', 'real', 'genuine', 'human']):
                        real_prob = prob_values[int(idx)]
                    elif any(f in label_lower for f in ['spoof', 'fake', 'deepfake', 'ai', 'generated']):
                        fake_prob = prob_values[int(idx)]
            else:
                # Default: index 0 = bonafide/real, index 1 = spoof/fake
                real_prob = prob_values[0]
                fake_prob = prob_values[1]
            
            # Normalize
            total = real_prob + fake_prob
            if total > 0:
                real_prob /= total
                fake_prob /= total
            
            prediction = "ai_generated" if fake_prob > 0.5 else "real"
            confidence = round(max(real_prob, fake_prob) * 100, 2)
            
            # Duration info
            duration_sec = len(audio) / sr if sr > 0 else 0
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'ai_probability': round(fake_prob * 100, 2),
                'human_probability': round(real_prob * 100, 2),
                'model': HF_MODEL_ID,
                'duration_seconds': round(duration_sec, 2),
                'sample_rate': sr,
                'success': True,
                'details': {
                    'raw_probabilities': prob_values,
                    'label_mapping': id2label if id2label else {0: 'bonafide', 1: 'spoof'},
                }
            }
            
        except Exception as e:
            logger.error(f"Audio detection failed: {e}")
            return {
                'prediction': 'unknown',
                'confidence': 0,
                'ai_probability': 50,
                'human_probability': 50,
                'model': HF_MODEL_ID,
                'success': False,
                'error': str(e)
            }
    
    def get_model_info(self) -> Dict:
        """Return model metadata."""
        return {
            'model_id': HF_MODEL_ID,
            'architecture': 'Wav2Vec2ForSequenceClassification',
            'parameters': '~95M',
            'input_format': f'{TARGET_SAMPLE_RATE}Hz mono audio',
            'max_duration': f'{MAX_AUDIO_LENGTH_SEC}s',
            'supported_formats': ['wav', 'mp3', 'flac', 'ogg', 'm4a', 'webm'],
            'loaded': self.model_loaded,
            'device': str(self.device) if self.device else 'not loaded'
        }
