"""
Video Deepfake Detector using pretrained models from HuggingFace.

Primary Model: Naman712/Deep-fake-detection
- Architecture: ResNeXt50 + LSTM for temporal analysis
- Accuracy: 87% on evaluation set
- Input: Video frames (extracted at regular intervals)

Approach:
1. Extract frames from video at regular intervals
2. Run each frame through the image classifier
3. Aggregate frame-level predictions with temporal weighting
4. Report overall video authenticity score
"""

import os
import logging
import tempfile
import numpy as np
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# HuggingFace model for video deepfake detection
HF_MODEL_ID = "Naman712/Deep-fake-detection"
# Fallback: use a proven image-level detector on extracted frames
HF_FALLBACK_MODEL_ID = "dima806/deepfake_vs_real_faces_detection"

MAX_FRAMES = 20  # Maximum frames to extract
DEFAULT_FPS_SAMPLE = 1  # Sample 1 frame per second by default


class VideoDeepfakeDetector:
    """Detect deepfake videos using frame-level analysis with pretrained models."""

    def __init__(self, use_gpu: bool = False):
        """Initialize the video detector.
        
        Args:
            use_gpu: Whether to use GPU for inference (if available)
        """
        self.use_gpu = use_gpu
        self.model = None
        self.processor = None
        self.device = None
        self.model_loaded = False
        self._load_attempted = False
        self._active_model_id = None

    def _load_model(self):
        """Lazy-load the video deepfake detection model."""
        if self._load_attempted:
            return
        self._load_attempted = True

        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")

            # Try primary model first
            try:
                logger.info(f"Loading video detector from {HF_MODEL_ID}...")
                print(f"Loading video deepfake detector: {HF_MODEL_ID}...")
                self.processor = AutoImageProcessor.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
                self.model = AutoModelForImageClassification.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
                self.model.to(self.device)
                self.model.eval()
                self._active_model_id = HF_MODEL_ID
                self.model_loaded = True
                logger.info(f"Video detector loaded: {HF_MODEL_ID}")
                print(f"Video detector loaded: {HF_MODEL_ID}")
                return
            except Exception as e:
                logger.warning(f"Primary video model failed: {e}")
                print(f"Primary video model failed: {e}, trying fallback...")

            # Fallback: use the face deepfake detector on individual frames
            logger.info(f"Trying fallback model: {HF_FALLBACK_MODEL_ID}...")
            print(f"Loading fallback video detector: {HF_FALLBACK_MODEL_ID}...")
            self.processor = AutoImageProcessor.from_pretrained(HF_FALLBACK_MODEL_ID)
            self.model = AutoModelForImageClassification.from_pretrained(HF_FALLBACK_MODEL_ID)
            self.model.to(self.device)
            self.model.eval()
            self._active_model_id = HF_FALLBACK_MODEL_ID
            self.model_loaded = True
            logger.info(f"Fallback video detector loaded: {HF_FALLBACK_MODEL_ID}")
            print(f"Fallback video detector loaded: {HF_FALLBACK_MODEL_ID}")

        except Exception as e:
            logger.error(f"Failed to load video detector: {e}")
            print(f"Failed to load video detector: {e}")
            self.model_loaded = False

    def _extract_frames(self, video_path: str, max_frames: int = MAX_FRAMES) -> List[np.ndarray]:
        """Extract frames from video file at regular intervals.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of frames as numpy arrays (RGB, HWC format)
        """
        frames = []
        try:
            import cv2
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return frames
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            duration = total_frames / fps if fps > 0 else 0
            
            if total_frames <= 0:
                logger.error("Video has no frames")
                cap.release()
                return frames
            
            # Calculate frame indices to sample
            if total_frames <= max_frames:
                indices = list(range(total_frames))
            else:
                indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()
            
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
            
            cap.release()
            logger.info(f"Extracted {len(frames)} frames from video ({duration:.1f}s, {fps:.0f}fps)")
            
        except ImportError:
            logger.error("opencv-python not installed. Install with: pip install opencv-python")
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
        
        return frames

    def _predict_frame(self, frame: np.ndarray) -> Dict:
        """Run detection on a single frame.
        
        Returns:
            Dict with 'real_prob' and 'fake_prob'
        """
        try:
            import torch
            from PIL import Image
            
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(frame)
            
            inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            prob_values = probs[0].tolist()
            
            # Determine label mapping
            id2label = getattr(self.model.config, 'id2label', None)
            real_prob = 0.5
            fake_prob = 0.5
            
            if id2label:
                for idx, label in id2label.items():
                    label_lower = label.lower()
                    if any(r in label_lower for r in ['real', 'genuine', 'authentic', 'human']):
                        real_prob = prob_values[int(idx)]
                    elif any(f in label_lower for f in ['fake', 'deepfake', 'ai', 'generated', 'synthetic', 'spoof']):
                        fake_prob = prob_values[int(idx)]
            else:
                # Default assumption: index 0 = real, index 1 = fake
                real_prob = prob_values[0]
                fake_prob = prob_values[1] if len(prob_values) > 1 else 1.0 - prob_values[0]
            
            return {'real_prob': real_prob, 'fake_prob': fake_prob}
            
        except Exception as e:
            logger.error(f"Frame prediction failed: {e}")
            return {'real_prob': 0.5, 'fake_prob': 0.5}

    def predict(self, video_input, filename: str = "video.mp4") -> Dict:
        """Detect if a video is a deepfake.
        
        Args:
            video_input: File path (str) or bytes
            filename: Original filename (used for temp file extension)
        
        Returns:
            Dict with detection results:
            - prediction: "real" or "deepfake"
            - confidence: 0-100 confidence score
            - ai_probability: 0-100 probability of being AI-generated/deepfake
            - human_probability: 0-100 probability of being real
            - frame_count: Number of frames analyzed
            - frame_results: Per-frame breakdown
            - temporal_consistency: Score for temporal coherence
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
                'frame_count': 0,
                'model': self._active_model_id or HF_MODEL_ID,
                'success': False,
                'error': 'Model not loaded'
            }

        # Handle input type
        video_path = None
        temp_file = None
        
        try:
            if isinstance(video_input, str):
                video_path = video_input
            elif isinstance(video_input, bytes):
                suffix = os.path.splitext(filename)[1] or '.mp4'
                temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
                temp_file.write(video_input)
                temp_file.close()
                video_path = temp_file.name
            else:
                return {
                    'prediction': 'unknown',
                    'confidence': 0,
                    'ai_probability': 50,
                    'human_probability': 50,
                    'model': self._active_model_id or HF_MODEL_ID,
                    'success': False,
                    'error': f'Unsupported input type: {type(video_input)}'
                }

            # Extract frames
            frames = self._extract_frames(video_path)
            if not frames:
                return {
                    'prediction': 'unknown',
                    'confidence': 0,
                    'ai_probability': 50,
                    'human_probability': 50,
                    'frame_count': 0,
                    'model': self._active_model_id or HF_MODEL_ID,
                    'success': False,
                    'error': 'Failed to extract frames from video'
                }

            # Run detection on each frame
            frame_results = []
            fake_probs = []
            
            for i, frame in enumerate(frames):
                result = self._predict_frame(frame)
                frame_results.append({
                    'frame_index': i,
                    'real_probability': round(result['real_prob'] * 100, 2),
                    'fake_probability': round(result['fake_prob'] * 100, 2),
                })
                fake_probs.append(result['fake_prob'])

            # Aggregate frame predictions
            # Use weighted average with higher weight on more extreme predictions
            fake_probs_arr = np.array(fake_probs)
            
            # Simple average
            avg_fake = float(np.mean(fake_probs_arr))
            
            # Weighted by distance from 0.5 (more confident frames count more)
            weights = np.abs(fake_probs_arr - 0.5) + 0.1  # avoid zero weight
            weighted_fake = float(np.average(fake_probs_arr, weights=weights))
            
            # Final score: blend simple and weighted averages
            final_fake_prob = 0.4 * avg_fake + 0.6 * weighted_fake
            final_real_prob = 1.0 - final_fake_prob
            
            # Temporal consistency analysis
            # High variance across frames may indicate partial manipulation
            temporal_std = float(np.std(fake_probs_arr))
            temporal_consistency = max(0, 1.0 - temporal_std * 2)  # Higher = more consistent
            
            # Count frames classified as fake
            fake_frame_count = int(np.sum(fake_probs_arr > 0.5))
            fake_ratio = fake_frame_count / len(fake_probs)
            
            prediction = "deepfake" if final_fake_prob > 0.5 else "real"
            confidence = round(max(final_fake_prob, final_real_prob) * 100, 2)
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'ai_probability': round(final_fake_prob * 100, 2),
                'human_probability': round(final_real_prob * 100, 2),
                'frame_count': len(frames),
                'fake_frame_count': fake_frame_count,
                'fake_frame_ratio': round(fake_ratio * 100, 2),
                'temporal_consistency': round(temporal_consistency * 100, 2),
                'frame_results': frame_results,
                'model': self._active_model_id or HF_MODEL_ID,
                'success': True,
                'details': {
                    'avg_fake_probability': round(avg_fake * 100, 2),
                    'weighted_fake_probability': round(weighted_fake * 100, 2),
                    'temporal_std': round(temporal_std, 4),
                }
            }

        except Exception as e:
            logger.error(f"Video detection failed: {e}")
            return {
                'prediction': 'unknown',
                'confidence': 0,
                'ai_probability': 50,
                'human_probability': 50,
                'frame_count': 0,
                'model': self._active_model_id or HF_MODEL_ID,
                'success': False,
                'error': str(e)
            }
        finally:
            # Clean up temp file
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except OSError:
                    pass

    def get_model_info(self) -> Dict:
        """Return model metadata."""
        return {
            'model_id': self._active_model_id or HF_MODEL_ID,
            'fallback_model_id': HF_FALLBACK_MODEL_ID,
            'architecture': 'ResNeXt50 + LSTM (frame-level analysis)',
            'max_frames': MAX_FRAMES,
            'supported_formats': ['mp4', 'avi', 'mov', 'mkv', 'webm', 'wmv'],
            'loaded': self.model_loaded,
            'device': str(self.device) if self.device else 'not loaded'
        }
