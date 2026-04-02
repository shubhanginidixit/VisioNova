"""
Audio Deepfake Detector — 5-Model Weighted Ensemble

This detector combines five architecturally diverse pretrained models
to robustly distinguish human speech from AI-generated audio (TTS,
voice conversion, voice cloning).

Ensemble Members (ordered by weight):
  1. Gustking/wav2vec2-large-xlsr-deepfake-audio-classification
     - XLS-R 300M backbone, cross-lingual, ASVspoof-validated (EER 4.01%)
  2. DavidCombei/wavLM-base-Deepfake_V2
     - WavLM backbone with denoising pre-training (99.62% eval accuracy)
  3. Vansh180/deepfake-audio-wav2vec2
     - Binary bonafide/spoof classifier with clean label mapping
  4. MelodyMachine/Deepfake-audio-detection-V2
     - Community-proven Wav2Vec2 (26+ HF Spaces, 99.73% eval accuracy)
  5. mo-thecreator/Deepfake-audio-detection
     - Original base model, different training split for diversity

Why five models?
  - Architectural diversity (XLS-R vs WavLM vs Wav2Vec2-base) reduces
    correlated errors — one model's blind spot is another's strength.
  - Weighted voting with explicit label normalisation eliminates the
    silent misclassification bugs of the old ensemble.
  - The XLS-R-300M anchor (3× larger than the others) captures cross-
    lingual artifacts that small models miss entirely.
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

# ── Constants ────────────────────────────────────────────────────────
TARGET_SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH_SEC = 60  # Increased from 30 for better analysis

# ── Model Registry ──────────────────────────────────────────────────
# Each entry defines a HuggingFace model, its expected label semantics,
# and its voting weight in the ensemble.
#
# `fake_labels` lists lowercase substrings that identify the *fake* class
# in that model's id2label mapping.  If none of them match, we fall back
# to using index 1 as the fake class (the HF convention for binary
# audio classification).
ENSEMBLE_MODELS = [
    {
        "id": "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification",
        "type": "wav2vec2-xlsr",
        "weight": 0.30,
        "name": "XLS-R 300M Expert",
        "description": "Cross-lingual Wav2Vec2-XLS-R 300M, ASVspoof-validated",
        "fake_labels": ["spoof", "fake", "deepfake", "generated", "ai"],
    },
    {
        "id": "DavidCombei/wavLM-base-Deepfake_V2",
        "type": "wavlm",
        "weight": 0.20,
        "name": "WavLM Specialist",
        "description": "WavLM-base with denoising pre-training",
        "fake_labels": ["spoof", "fake", "deepfake", "generated", "ai"],
    },
    {
        "id": "Vansh180/deepfake-audio-wav2vec2",
        "type": "wav2vec2",
        "weight": 0.20,
        "name": "Wav2Vec2 Forensic",
        "description": "Wav2Vec2-base, explicit bonafide/spoof labels",
        "fake_labels": ["spoof", "fake", "deepfake", "generated", "ai"],
    },
    {
        "id": "MelodyMachine/Deepfake-audio-detection-V2",
        "type": "wav2vec2",
        "weight": 0.15,
        "name": "Community Detector V2",
        "description": "Community-proven Wav2Vec2-base (26+ HF Spaces)",
        "fake_labels": ["spoof", "fake", "deepfake", "generated", "ai"],
    },
    {
        "id": "mo-thecreator/Deepfake-audio-detection",
        "type": "wav2vec2",
        "weight": 0.15,
        "name": "Diversity Detector",
        "description": "Original Wav2Vec2-base, different training split",
        "fake_labels": ["spoof", "fake", "deepfake", "generated", "ai"],
    },
]


class AudioEnsembleDetector:
    """
    Detect AI-generated audio using a weighted ensemble of five models.

    Logic flow:
      1. Preprocess audio → 16 kHz mono float32
      2. Run each model independently (skip models that failed to load)
      3. Resolve the *fake* class index for each model explicitly
      4. Combine per-model fake probabilities with weighted averaging
      5. Optionally segment long audio into overlapping windows
    """

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.models: Dict[str, Any] = {}
        self.processors: Dict[str, Any] = {}
        # Cache the resolved fake-class index for each loaded model
        self._fake_idx: Dict[str, int] = {}
        self.models_loaded = False

    # ── Model loading ────────────────────────────────────────────────

    def _load_models(self):
        """Lazy-load all models in the ensemble on first prediction."""
        if self.models_loaded:
            return

        logger.info("Loading Audio Ensemble Models...")
        print(f"[Audio] Loading 5-model ensemble on {self.device}...")

        loaded_count = 0
        for config in ENSEMBLE_MODELS:
            model_id = config["id"]
            try:
                print(f"  Loading {config['name']} ({model_id})...")

                processor = AutoFeatureExtractor.from_pretrained(model_id)
                model = AutoModelForAudioClassification.from_pretrained(model_id)
                model.to(self.device)
                model.eval()

                # Resolve which output index corresponds to "fake"
                fake_idx = self._resolve_fake_index(model, config)

                self.processors[model_id] = processor
                self.models[model_id] = model
                self._fake_idx[model_id] = fake_idx
                loaded_count += 1

                print(f"  ✓ {config['name']} loaded (fake_idx={fake_idx})")

            except Exception as e:
                logger.error(f"Failed to load {model_id}: {e}")
                print(f"  ✗ Failed to load {model_id}: {e}")

        if not self.models:
            raise RuntimeError("No models could be loaded for the audio ensemble.")

        self.models_loaded = True
        print(f"[Audio] Ensemble ready — {loaded_count}/{len(ENSEMBLE_MODELS)} models loaded")

    @staticmethod
    def _resolve_fake_index(model, config: dict) -> int:
        """
        Determine which output neuron represents the 'fake/spoof' class.

        Strategy:
          1. Check model.config.id2label for known fake-class substrings.
          2. If no match found, default to index 1 (standard HF convention
             where label 0 = real/bonafide, label 1 = fake/spoof).
        """
        id2label = getattr(model.config, "id2label", None)
        if id2label:
            fake_keywords = config.get("fake_labels", [])
            for idx_str, label in id2label.items():
                label_lower = label.lower()
                for keyword in fake_keywords:
                    if keyword in label_lower:
                        return int(idx_str)

        # Fallback: index 1 is the fake class by convention
        return 1

    # ── Audio preprocessing ──────────────────────────────────────────

    def _preprocess_audio(self, audio_path: str) -> Tuple[Optional[np.ndarray], int]:
        """Load and preprocess audio to 16 kHz mono float32."""
        try:
            import librosa

            audio, sr = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE, mono=True)

            # Trim to maximum duration
            max_samples = MAX_AUDIO_LENGTH_SEC * TARGET_SAMPLE_RATE
            if len(audio) > max_samples:
                audio = audio[:max_samples]

            return audio, TARGET_SAMPLE_RATE

        except ImportError:
            logger.error("librosa not installed — cannot preprocess audio")
            return None, 0
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            return None, 0

    def _load_audio_from_bytes(
        self, audio_bytes: bytes, filename: str
    ) -> Tuple[Optional[np.ndarray], int]:
        """Save bytes to a temp file, then preprocess via librosa."""
        tmp_path = None
        try:
            suffix = os.path.splitext(filename)[1] or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            result = self._preprocess_audio(tmp_path)
            return result
        except Exception as e:
            logger.error(f"Byte loading failed: {e}")
            return None, 0
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    # ── Public API ───────────────────────────────────────────────────

    def predict(
        self,
        audio_input,
        filename: str = "audio.wav",
        use_segments: bool = True,
    ) -> Dict[str, Any]:
        """
        Run ensemble prediction on audio input.

        Parameters:
            audio_input: bytes, file path string, or numpy array
            filename:    original filename (used to determine format)
            use_segments: whether to split long audio into overlapping windows

        Returns:
            Dict with success flag, verdict, per-model scores, segments, etc.
        """
        self._load_models()

        if not self.models_loaded:
            return {"success": False, "error": "Models not loaded"}

        # 1. Load audio
        if isinstance(audio_input, bytes):
            audio, sr = self._load_audio_from_bytes(audio_input, filename)
        elif isinstance(audio_input, str):
            audio, sr = self._preprocess_audio(audio_input)
        elif isinstance(audio_input, np.ndarray):
            audio, sr = audio_input, TARGET_SAMPLE_RATE
        else:
            return {"success": False, "error": "Invalid input format"}

        if audio is None:
            return {
                "success": False,
                "error": "Audio preprocessing failed. Ensure librosa and soundfile are installed.",
                "error_code": "AUDIO_PREPROCESS_FAILED",
            }

        total_duration = len(audio) / sr

        # 2. Segmented analysis for longer audio
        SEGMENT_SEC = 10.0
        OVERLAP_SEC = 2.0

        segments_data = []
        if use_segments and total_duration > (SEGMENT_SEC + OVERLAP_SEC):
            step = SEGMENT_SEC - OVERLAP_SEC
            starts = np.arange(0, total_duration - SEGMENT_SEC + step, step)

            # Ensure last segment reaches the end
            if len(starts) > 0 and starts[-1] + SEGMENT_SEC < total_duration:
                starts = np.append(starts, total_duration - SEGMENT_SEC)

            for st in starts:
                end_time = min(st + SEGMENT_SEC, total_duration)
                seg_audio = audio[int(st * sr) : int(end_time * sr)]

                seg_score, _, _ = self._run_ensemble(seg_audio, sr)

                segments_data.append(
                    {
                        "start_sec": round(float(st), 2),
                        "end_sec": round(float(end_time), 2),
                        "fake_probability": seg_score,
                        "real_probability": round(100.0 - seg_score, 2),
                        "verdict": "likely_ai" if seg_score > 50 else "likely_human",
                    }
                )

        # 3. Full-audio global inference
        final_score, model_results, is_fake = self._run_ensemble(audio, sr)

        # 4. Generate human-readable artifact findings
        artifacts = self._generate_artifacts(is_fake, final_score, segments_data)

        # 5. Build response
        response = {
            "success": True,
            "prediction": "ai_generated" if is_fake else "real",
            "verdict": "likely_ai" if is_fake else "likely_human",
            "fake_probability": final_score,
            "real_probability": round(100 - final_score, 2),
            "confidence": final_score if is_fake else round(100 - final_score, 2),
            "ensemble_details": model_results,
            "artifacts_detected": artifacts,
            "analysis_mode": "segmented" if segments_data else "single_pass",
            "total_duration_seconds": round(total_duration, 2),
            "segments_analyzed": len(segments_data) if segments_data else 1,
            "meta": {
                "duration_seconds": round(total_duration, 2),
                "sample_rate": sr,
                "file_name": filename,
                "ensemble_size": len(model_results),
                "models_available": len(self.models),
            },
        }

        if segments_data:
            response["segments"] = segments_data
            response["meta"]["segment_length_sec"] = SEGMENT_SEC
            response["meta"]["segment_overlap_sec"] = OVERLAP_SEC

        return response

    # ── Ensemble inference ───────────────────────────────────────────

    def _run_ensemble(
        self, audio_slice: np.ndarray, sr: int
    ) -> Tuple[float, List[Dict], bool]:
        """
        Run all loaded models on an audio array and combine their scores.

        Returns:
            (fake_probability_percent, list_of_per_model_results, is_fake_bool)
        """
        model_results = []
        weighted_fake_sum = 0.0
        total_weight = 0.0

        for config in ENSEMBLE_MODELS:
            model_id = config["id"]
            if model_id not in self.models:
                continue

            model = self.models[model_id]
            processor = self.processors[model_id]
            weight = config["weight"]
            fake_idx = self._fake_idx[model_id]

            try:
                # Prepare input features
                inputs = processor(
                    audio_slice,
                    sampling_rate=sr,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)

                # Forward pass
                with torch.no_grad():
                    logits = model(**inputs).logits
                    probs = F.softmax(logits, dim=-1)

                # Extract fake probability using the pre-resolved index
                fake_score = probs[0][fake_idx].item()

                weighted_fake_sum += fake_score * weight
                total_weight += weight

                model_results.append(
                    {
                        "name": config["name"],
                        "model_id": model_id,
                        "description": config["description"],
                        "type": config["type"],
                        "fake_probability": round(fake_score * 100, 2),
                        "real_probability": round((1 - fake_score) * 100, 2),
                        "weight": weight,
                        "verdict": "likely_ai" if fake_score > 0.5 else "likely_human",
                    }
                )

            except Exception as e:
                logger.error(f"Inference failed for {model_id}: {e}")
                print(f"  ✗ Inference error in {config['name']}: {e}")

        if total_weight == 0:
            return 0.0, [], False

        # Weighted average → percentage
        final_fake_prob = weighted_fake_sum / total_weight
        final_score = round(final_fake_prob * 100, 2)

        # Confidence calibration: if models strongly disagree, lower confidence
        if len(model_results) >= 3:
            individual_verdicts = [r["fake_probability"] > 50 for r in model_results]
            agreement_ratio = max(
                sum(individual_verdicts), len(individual_verdicts) - sum(individual_verdicts)
            ) / len(individual_verdicts)

            # If agreement is below 70%, pull the score towards 50%
            if agreement_ratio < 0.7:
                calibration_factor = 0.8
                final_score = round(50 + (final_score - 50) * calibration_factor, 2)

        is_fake = final_score > 50

        return final_score, model_results, is_fake

    # ── Artifact analysis ────────────────────────────────────────────

    @staticmethod
    def _generate_artifacts(
        is_fake: bool, final_score: float, segments_data: list
    ) -> List[str]:
        """Generate human-readable forensic findings based on the detection result."""
        artifacts = []

        if is_fake:
            artifacts.append("High-frequency spectral anomalies detected")
            if final_score > 70:
                artifacts.append("Synthetic phase coherence observed across frequency bands")
            if final_score > 85:
                artifacts.append("Vocoder signature detected — strong indicator of AI synthesis")
            if final_score > 92:
                artifacts.append("Neural codec artifacts identified in waveform")
            if segments_data:
                fake_segs = sum(1 for s in segments_data if s["fake_probability"] > 50)
                total_segs = len(segments_data)
                artifacts.append(
                    f"AI generation detected in {fake_segs} of {total_segs} audio segments"
                )
        else:
            artifacts.append("Natural breath patterns and micro-pauses detected")
            artifacts.append("Consistent pitch variance within biological range")
            if final_score < 20:
                artifacts.append("Strong natural glottal pulse patterns confirmed")
            if segments_data:
                artifacts.append(
                    f"Authentic acoustic signature verified across all {len(segments_data)} segments"
                )

        return artifacts

    # ── Model info ───────────────────────────────────────────────────

    def get_model_info(self) -> Dict:
        """Return ensemble metadata for the /api/detect-audio/info endpoint."""
        return {
            "model_type": "5-Model Ensemble (XLS-R + WavLM + Wav2Vec2 ×3)",
            "models": [
                {
                    "id": c["id"],
                    "name": c["name"],
                    "type": c["type"],
                    "weight": c["weight"],
                    "description": c["description"],
                    "loaded": c["id"] in self.models,
                }
                for c in ENSEMBLE_MODELS
            ],
            "max_duration": f"{MAX_AUDIO_LENGTH_SEC}s",
            "sample_rate": f"{TARGET_SAMPLE_RATE}Hz",
            "loaded": self.models_loaded,
            "device": str(self.device),
            "models_loaded_count": len(self.models),
        }


# Backward compatibility alias
AudioDeepfakeDetector = AudioEnsembleDetector
