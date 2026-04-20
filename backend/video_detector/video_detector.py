"""
Video Deepfake and AI-Generation Detector.

Ensemble Models:
1. buildborderless/CommunityForensics-DeepfakeDet-ViT (ViT for pure pixel/diffusion noise per-frame)
2. Vansh180/VideoMae-ffc23-deepfake-detector (VideoMAE for 16-frame temporal sequence analysis)

Approach:
1. Extract contiguous frames (using decord/cv2) for temporal batch analysis.
2. Run frame batches through VideoMAE to test for physics/motion inconsistencies.
3. Run keyframes through the CommunityForensics ViT to test for AI diffusion artifacts.
4. Ensemble the scores.
"""

import os
import logging
import tempfile
import numpy as np
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

HF_SPATIAL_MODEL = "buildborderless/CommunityForensics-DeepfakeDet-ViT"
HF_TEMPORAL_MODEL = "Vansh180/VideoMae-ffc23-deepfake-detector"

MAX_FRAMES_VIT = 10  # Maximum singular keyframes to extract for spatial CNN/ViT
NUM_FRAMES_MAE = 16  # Required frames for VideoMAE


class VideoDeepfakeDetector:
    """Detect deepfake videos and full AI-generated videos using an ensemble of spatial and temporal models."""

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        
        # Spatial Model (ViT)
        self.spatial_model = None
        self.spatial_processor = None
        
        # Temporal Model (VideoMAE)
        self.temporal_model = None
        self.temporal_processor = None
        
        self.device = None
        self.models_loaded = False
        self._load_attempted = False

    def _load_models(self):
        """Lazy-load the video deepfake detection ensemble."""
        if self._load_attempted:
            return
        self._load_attempted = True

        try:
            import torch
            from transformers import (
                AutoImageProcessor, 
                AutoModelForImageClassification,
                VideoMAEImageProcessor, 
                VideoMAEForVideoClassification
            )

            self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")

            # Try to load models inside Try/Except block for fallback safety
            try:
                logger.info(f"Loading spatial Video ViT: {HF_SPATIAL_MODEL}...")
                print(f"Loading spatial Video ViT: {HF_SPATIAL_MODEL}...")
                self.spatial_processor = AutoImageProcessor.from_pretrained(HF_SPATIAL_MODEL)
                self.spatial_model = AutoModelForImageClassification.from_pretrained(HF_SPATIAL_MODEL)
                self.spatial_model.to(self.device)
                self.spatial_model.eval()
            except Exception as e:
                logger.warning(f"Spatial model loading failed: {e}")
                print(f"Spatial model loading failed: {e}")

            try:
                logger.info(f"Loading temporal VideoMAE: {HF_TEMPORAL_MODEL}...")
                print(f"Loading temporal VideoMAE: {HF_TEMPORAL_MODEL}...")
                self.temporal_processor = VideoMAEImageProcessor.from_pretrained(HF_TEMPORAL_MODEL)
                self.temporal_model = VideoMAEForVideoClassification.from_pretrained(HF_TEMPORAL_MODEL)
                self.temporal_model.to(self.device)
                self.temporal_model.eval()
            except Exception as e:
                logger.warning(f"Temporal model loading failed: {e}")
                print(f"Temporal model loading failed: {e}")

            if self.spatial_model is not None or self.temporal_model is not None:
                self.models_loaded = True
            else:
                logger.error("Failed to load any video detection models.")

        except Exception as e:
            logger.error(f"Failed to initialize deep learning backend for video detection: {e}")
            self.models_loaded = False

    def _extract_frames_vit(self, video_path: str, max_frames: int = MAX_FRAMES_VIT) -> List[np.ndarray]:
        """Extract singular keyframes for the spatial ViT analysis."""
        frames = []
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                return frames
            
            indices = np.linspace(0, total_frames - 1, min(total_frames, max_frames), dtype=int).tolist()
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
        except Exception as e:
            logger.error(f"ViT Frame extraction failed: {e}")
        return frames

    def _extract_frames_mae(self, video_path: str, num_frames: int = NUM_FRAMES_MAE) -> List[np.ndarray]:
        """Extract a uniform sequence of frames for VideoMAE."""
        frames = []
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            
            # VideoMAE is usually trained on 16 evenly spaced frames
            indices = np.linspace(0, total_frames - 1, min(total_frames, num_frames)).astype(int)
            batch = vr.get_batch(indices).asnumpy()
            
            # If video is extremely short, pad by replicating last frame
            frame_list = [f for f in batch]
            while len(frame_list) < num_frames and len(frame_list) > 0:
                frame_list.append(frame_list[-1])
            
            frames = frame_list
        except Exception as e:
            # Fallback to OpenCV if decord fails
            logger.error(f"VideoMAE Frame extraction (decord) failed, falling back to cv2: {e}")
            try:
                import cv2
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if cap.isOpened() and total_frames > 0:
                    indices = np.linspace(0, total_frames - 1, min(total_frames, num_frames)).astype(int)
                    for idx in indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        if ret:
                            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    while len(frames) < num_frames and len(frames) > 0:
                        frames.append(frames[-1])
                cap.release()
            except Exception as cv2_err:
                logger.error(f"VideoMAE Fallback cv2 extraction failed: {cv2_err}")
                
        return frames

    def _predict_spatial(self, frames: List[np.ndarray]) -> Tuple[float, float, List[Dict]]:
        """Predict AI/Real probability using the spatial ViT on individual frames."""
        if not frames or self.spatial_model is None:
            return 0.5, 0.5, []

        import torch
        from PIL import Image

        frame_results = []
        fake_probs = []

        for i, frame in enumerate(frames):
            try:
                pil_image = Image.fromarray(frame)
                inputs = self.spatial_processor(images=pil_image, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.spatial_model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # CommunityForensics-DeepfakeDet-ViT logic
                prob_values = probs[0].tolist()
                id2label = getattr(self.spatial_model.config, 'id2label', None)
                
                fake_prob = prob_values[1] if len(prob_values) > 1 else prob_values[0]
                real_prob = prob_values[0]

                if id2label:
                    for idx, label in id2label.items():
                        label_lower = label.lower()
                        if 'fake' in label_lower or 'ai' in label_lower or 'synthetic' in label_lower:
                            fake_prob = prob_values[int(idx)]
                        elif 'real' in label_lower or 'human' in label_lower or 'authentic' in label_lower:
                            real_prob = prob_values[int(idx)]

                fake_probs.append(fake_prob)
                frame_results.append({
                    'frame_index': i,
                    'fake_probability': round(fake_prob * 100, 2),
                    'real_probability': round(real_prob * 100, 2)
                })
            except Exception as e:
                logger.error(f"Spatial inference failed for frame {i}: {e}")

        if not fake_probs:
            return 0.5, 0.5, []

        avg_fake_prob = float(np.mean(fake_probs))
        return avg_fake_prob, 1.0 - avg_fake_prob, frame_results

    def _predict_temporal(self, frames: List[np.ndarray]) -> Tuple[float, float]:
        """Predict using the VideoMAE temporal transformer."""
        if not frames or len(frames) != NUM_FRAMES_MAE or self.temporal_model is None:
            return 0.5, 0.5

        import torch
        from PIL import Image

        try:
            pil_frames = [Image.fromarray(f) for f in frames]
            inputs = self.temporal_processor(list(pil_frames), return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.temporal_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            prob_values = probs[0].tolist()
            
            # Vansh180/VideoMae-ffc23-deepfake-detector
            id2label = getattr(self.temporal_model.config, 'id2label', None)
            fake_prob = prob_values[1] if len(prob_values) > 1 else 0.5
            real_prob = prob_values[0] if len(prob_values) > 1 else 0.5
            
            if id2label:
                for idx, label in id2label.items():
                    label_l = label.lower()
                    if 'fake' in label_l or 'manipulated' in label_l:
                        fake_prob = prob_values[int(idx)]
                    elif 'real' in label_l or 'pristine' in label_l:
                        real_prob = prob_values[int(idx)]
                        
            return fake_prob, real_prob
        except Exception as e:
            logger.error(f"Temporal inference failed: {e}")
            return 0.5, 0.5

    def predict(self, video_input, filename: str = "video.mp4") -> Dict:
        """Detect if a video is an AI-generated deepfake.
        
        Args:
            video_input: File path (str) or bytes
            filename: Original filename
        """
        self._load_models()
        
        if not self.models_loaded:
            return {
                'prediction': 'unknown',
                'confidence': 0,
                'ai_probability': 50,
                'human_probability': 50,
                'model': "ensemble_failed",
                'success': False,
                'error': 'Video detection models not loaded/unavailable.'
            }

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
                raise ValueError("Unsupported input type.")

            # 1. Spatial Pathway
            spatial_frames = self._extract_frames_vit(video_path)
            spatial_fake_prob, spatial_real_prob, frame_results = self._predict_spatial(spatial_frames)
            
            # 2. Temporal Pathway
            temporal_frames = self._extract_frames_mae(video_path)
            temporal_fake_prob, temporal_real_prob = self._predict_temporal(temporal_frames)

            # 3. Ensemble
            if self.spatial_model and self.temporal_model:
                final_fake_prob = (spatial_fake_prob + temporal_fake_prob) / 2.0
            elif self.spatial_model:
                final_fake_prob = spatial_fake_prob
            elif self.temporal_model:
                final_fake_prob = temporal_fake_prob
            else:
                final_fake_prob = 0.5

            final_real_prob = 1.0 - final_fake_prob
            prediction = "deepfake" if final_fake_prob > 0.5 else "real"
            confidence = round(max(final_fake_prob, final_real_prob) * 100, 2)
            
            # Additional metric: Variance/consistency across spatial frames
            fake_probs_arr = [res['fake_probability']/100.0 for res in frame_results]
            temporal_std = float(np.std(fake_probs_arr)) if fake_probs_arr else 0.0
            temporal_consistency = max(0, 1.0 - temporal_std * 2)

            return {
                'prediction': prediction,
                'confidence': confidence,
                'ai_probability': round(final_fake_prob * 100, 2),
                'human_probability': round(final_real_prob * 100, 2),
                'frame_count': len(spatial_frames),
                'temporal_consistency': round(temporal_consistency * 100, 2),
                'frame_results': frame_results,
                'model': 'VideoMAE+CommunityForensics Ensemble',
                'success': True,
                'details': {
                    'spatial_fake_prob': round(spatial_fake_prob * 100, 2),
                    'temporal_fake_prob': round(temporal_fake_prob * 100, 2),
                    'temporal_std': round(temporal_std, 4),
                }
            }

        except Exception as e:
            logger.error(f"Video detection pipeline failed: {e}")
            return {
                'prediction': 'unknown',
                'confidence': 0,
                'ai_probability': 50,
                'success': False,
                'error': str(e)
            }
        finally:
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except OSError:
                    pass

    def get_model_info(self) -> Dict:
        return {
            'models': [HF_SPATIAL_MODEL, HF_TEMPORAL_MODEL],
            'architectures': ['ViT (Spatial)', 'VideoMAE (Temporal)'],
            'max_frames_vit': MAX_FRAMES_VIT,
            'loaded': self.models_loaded,
            'device': str(self.device) if self.device else 'not_loaded'
        }
