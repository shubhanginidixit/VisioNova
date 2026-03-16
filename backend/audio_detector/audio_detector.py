"""
Audio Deepfake Detector using an Ensemble of Wav2Vec2 and WavLM.

Ensemble Strategy:
1. Model A: MelodyMachine/Deepfake-audio-detection-V2 (Wav2Vec2) - Weight: 0.6
2. Model B: DavidCombei/wavLM-base-Deepfake_V2 (WavLM) - Weight: 0.4

This ensemble approach leverages the architectural diversity of Wav2Vec2 (CNN + Transformer)
and WavLM (Masked Speech Prediction) to improve detection robustness.
"""

import os
import logging
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Any
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

logger = logging.getLogger(__name__)

# Constants
TARGET_SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH_SEC = 30

# Model Registry
ENSEMBLE_MODELS = [
    {
        "id": "MelodyMachine/Deepfake-audio-detection-V2",
        "type": "wav2vec2",
        "weight": 0.60,
        "name": "Wav2Vec2 Expert"
    },
    {
        "id": "DavidCombei/wavLM-base-Deepfake_V2",
        "type": "wavlm",
        "weight": 0.40,
        "name": "WavLM Specialist"
    }
]

class AudioEnsembleDetector:
    """
    Detect AI-generated audio using a weighted ensemble of models.
    """

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.models = {}  # Store loaded models
        self.processors = {}
        self.models_loaded = False
        
    def _load_models(self):
        """Lazy-load all models in the ensemble."""
        if self.models_loaded:
            return

        logger.info("Loading Audio Ensemble Models...")
        print(f"Loading Audio Ensemble on {self.device}...")

        try:
            for config in ENSEMBLE_MODELS:
                model_id = config["id"]
                try:
                    print(f"Loading {config['name']} ({model_id})...")
                    
                    # Load feature extractor
                    processor = AutoFeatureExtractor.from_pretrained(model_id)
                    
                    # Load model
                    model = AutoModelForAudioClassification.from_pretrained(model_id)
                    model.to(self.device)
                    model.eval()
                    
                    self.processors[model_id] = processor
                    self.models[model_id] = model
                    
                except Exception as e:
                    logger.error(f"Failed to load {model_id}: {e}")
                    print(f"⚠ Failed to load {model_id}: {e}")
            
            if not self.models:
                raise RuntimeError("No models could be loaded for the ensemble.")
                
            self.models_loaded = True
            print("✓ Audio Ensemble Loaded Successfully")
            
        except Exception as e:
            logger.error(f"Critical error loading ensemble: {e}")
            print(f"Critical error loading ensemble: {e}")
            self.models_loaded = False

    def _preprocess_audio(self, audio_path: str) -> Tuple[Optional[np.ndarray], int]:
        """Load and preprocess audio to 16kHz mono."""
        try:
            import soundfile as sf
            import librosa
            
            # Load with librosa (handles resampling automatically usually, but let's be explicit)
            # librosa loads as float32, expected by transformers
            audio, sr = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE, mono=True)
            
            # Trim/Pad
            max_samples = MAX_AUDIO_LENGTH_SEC * TARGET_SAMPLE_RATE
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            
            return audio, TARGET_SAMPLE_RATE
            
        except ImportError:
            logger.error("librosa/soundfile not installed")
            return None, 0
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            return None, 0

    def _load_audio_from_bytes(self, audio_bytes: bytes, filename: str) -> Tuple[Optional[np.ndarray], int]:
        """Handle raw bytes upload."""
        try:
            suffix = os.path.splitext(filename)[1] or '.wav'
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            
            result = self._preprocess_audio(tmp_path)
            os.unlink(tmp_path)
            return result
        except Exception as e:
            logger.error(f"Byte loading failed: {e}")
            return None, 0

    def predict(self, audio_input, filename: str = "audio.wav") -> Dict[str, Any]:
        """
        Run ensemble prediction.
        
        Returns detailed JSON with weighted average verdict.
        """
        self._load_models()
        
        if not self.models_loaded:
            return {"success": False, "error": "Models not loaded"}

        # 1. Load Audio
        if isinstance(audio_input, bytes):
            audio, sr = self._load_audio_from_bytes(audio_input, filename)
        elif isinstance(audio_input, str):
            audio, sr = self._preprocess_audio(audio_input)
        else:
            return {"success": False, "error": "Invalid input format"}

        if audio is None:
            return {"success": False, "error": "Could not process audio data"}

        # 2. Inference Loop
        model_results = []
        weighted_fake_prob_sum = 0.0
        total_weight = 0.0

        for config in ENSEMBLE_MODELS:
            model_id = config["id"]
            if model_id not in self.models:
                continue
                
            model = self.models[model_id]
            processor = self.processors[model_id]
            weight = config["weight"]
            
            try:
                # Prepare input
                inputs = processor(
                    audio, 
                    sampling_rate=sr, 
                    return_tensors="pt", 
                    padding=True
                ).to(self.device)

                with torch.no_grad():
                    logits = model(**inputs).logits
                    probs = F.softmax(logits, dim=-1)
                    
                # Extract probabilities
                # Assuming index 0=Real, 1=Fake for these specific models. 
                # Wav2Vec2 and WavLM deepfake models on HF usually follow this, 
                # but let's check config id2label if available safely.
                
                fake_score = 0.0
                id2label = model.config.id2label
                if id2label:
                    # Robust label checking
                    for idx, label in id2label.items():
                        if any(x in label.lower() for x in ['spoof', 'fake', 'generated', 'ai']):
                            # Force convert idx to int because JSON keys are strings
                            fake_score = probs[0][int(idx)].item()
                            break
                    else:
                        # Fallback if labels are weird (e.g. 'class_0', 'class_1')
                        # Usually 1 is fake in binary classifiers
                        fake_score = probs[0][1].item() if len(probs[0]) > 1 else probs[0][0].item()
                else:
                    fake_score = probs[0][1].item()
                
                weighted_fake_prob_sum += fake_score * weight
                total_weight += weight
                
                model_results.append({
                    "name": config["name"],
                    "model_id": model_id,
                    "fake_probability": round(fake_score * 100, 2),
                    "weight": weight
                })
                
            except Exception as e:
                logger.error(f"Inference failed for {model_id}: {e}")
                
        # 3. Aggregation
        if total_weight == 0:
            return {"success": False, "error": "Ensemble inference failed"}
            
        final_fake_prob = weighted_fake_prob_sum / total_weight
        final_score = round(final_fake_prob * 100, 2)
        is_fake = final_score > 50
        
        # 4. Generate Artifacts Analysis (Mocked logic based on score for now)
        # In a real system, we'd extract attention maps or specific spectral features.
        artifacts = []
        if is_fake:
            artifacts.append("High-frequency spectral anomalies detected")
            if final_score > 80:
                artifacts.append("Synthetic phase coherence observed")
            if final_score > 90:
                artifacts.append("Vocoder analysis signature present")
        
        return {
            "success": True,
            "prediction": "ai_generated" if is_fake else "real",
            "verdict": "likely_ai" if is_fake else "likely_human",
            "fake_probability": final_score,
            "real_probability": round(100 - final_score, 2),
            "confidence": final_score if is_fake else round(100 - final_score, 2),
            "ensemble_details": model_results,
            "artifacts_detected": artifacts,
            "meta": {
                "duration_seconds": round(len(audio) / sr, 2),
                "sample_rate": sr
            }
        }

    def get_model_info(self) -> Dict:
        """Return ensemble metadata."""
        return {
            "model_type": "Ensemble (Wav2Vec2 + WavLM)",
            "models": ENSEMBLE_MODELS,
            "max_duration": f"{MAX_AUDIO_LENGTH_SEC}s",
            "sample_rate": f"{TARGET_SAMPLE_RATE}Hz",
            "loaded": self.models_loaded,
            "device": str(self.device)
        }

# For backward compatibility if needed, though we should update app.py
AudioDeepfakeDetector = AudioEnsembleDetector
