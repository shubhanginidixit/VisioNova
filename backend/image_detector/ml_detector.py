"""
VisioNova ML-based Image Detectors
State-of-the-art deep learning models for AI-generated image detection.

Models:
1. Ateeqq ViT Detector - Vision Transformer (99.23% accuracy)
2. UniversalFakeDetect - CLIP-based detector (generalizes across generators)
3. Deepfake Detector - Face manipulation detection
"""

import io
import logging
from typing import Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)






class SDXLDetector:
    """
    SDXL / Modern Diffusion Image Detector
    
    Fine-tuned on Wikimedia-SDXL image pairs. Outperforms older detectors
    on images from SDXL, SD3, Flux, and other modern diffusion models.
    
    Model: Organika/sdxl-detector
    Architecture: Swin Transformer (87M params)
    Accuracy: 98.1% (F1: 0.973, AUC: 0.998)
    Downloads: 36K/month
    """
    
    MODEL_ID = "Organika/sdxl-detector"
    
    def __init__(self, device: str = "auto"):
        """Initialize the SDXL detector."""
        self.model = None
        self.processor = None
        self.device = device
        self.model_loaded = False
        self.load_error = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the SDXL detector from HuggingFace."""
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading SDXL detector on {self.device}...")
            
            self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
            self.model = AutoModelForImageClassification.from_pretrained(self.MODEL_ID)
            self.model.to(self.device)
            self.model.eval()
            
            self.id2label = self.model.config.id2label
            self.model_loaded = True
            logger.info(f"SDXL detector loaded. Labels: {self.id2label}")
            
        except ImportError as e:
            self.load_error = f"Missing dependencies: {e}"
            logger.warning(self.load_error)
        except Exception as e:
            self.load_error = f"Failed to load SDXL detector: {e}"
            logger.warning(self.load_error)
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Predict whether an image is AI-generated (SDXL/modern diffusion)."""
        if not self.model_loaded:
            return {
                'success': False,
                'error': self.load_error or "Model not loaded",
                'ai_probability': 50.0,
                'prediction': 'unknown'
            }
        
        try:
            import torch
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            probs_np = probs[0].cpu().numpy()
            predicted_idx = int(np.argmax(probs_np))
            predicted_label = self.id2label[predicted_idx]
            
            # Organika/sdxl-detector: label 0 = "artificial", label 1 = "human"
            ai_prob = self._get_ai_probability(probs_np, predicted_label)
            
            return {
                'success': True,
                'prediction': 'ai_generated' if ai_prob >= 50 else 'real',
                'confidence': float(probs_np[predicted_idx] * 100),
                'ai_probability': ai_prob,
                'all_probabilities': {
                    self.id2label[i]: float(p * 100) for i, p in enumerate(probs_np)
                },
                'model': 'SDXL-Detector-Swin',
                'specialization': 'SDXL, SD3, modern diffusion models'
            }
            
        except Exception as e:
            logger.error(f"SDXL detector error: {e}")
            return {
                'success': False,
                'error': str(e),
                'ai_probability': 50.0,
                'prediction': 'error'
            }
    
    def _get_ai_probability(self, probs: np.ndarray, predicted_label: str) -> float:
        """Extract AI probability from model output."""
        ai_keywords = ['ai', 'fake', 'generated', 'synthetic', 'artificial']
        real_keywords = ['real', 'authentic', 'human', 'natural', 'genuine']
        
        ai_prob = 50.0
        
        for idx, label in self.id2label.items():
            label_lower = label.lower()
            if any(kw in label_lower for kw in ai_keywords):
                ai_prob = float(probs[idx] * 100)
                break
            elif any(kw in label_lower for kw in real_keywords):
                ai_prob = float((1 - probs[idx]) * 100)
                break
        
        return ai_prob


class DeepfakeDetector:
    """
    Deepfake Face Manipulation Detector
    
    Specialized detector for face manipulation (deepfakes).
    Uses model trained on FaceForensics++ and similar datasets.
    
    Model: dima806/deepfake_vs_real_image_detection (optional)
    """
    
    MODEL_ID = "dima806/deepfake_vs_real_image_detection"
    
    def __init__(self, device: str = "auto"):
        """Initialize deepfake detector."""
        self.model = None
        self.processor = None
        self.device = device
        self.model_loaded = False
        self.load_error = None
        self.face_detector = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the deepfake detection model."""
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading Deepfake detector on {self.device}...")
            
            self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
            self.model = AutoModelForImageClassification.from_pretrained(self.MODEL_ID)
            self.model.to(self.device)
            self.model.eval()
            
            self.id2label = self.model.config.id2label
            self.model_loaded = True
            logger.info("Deepfake detector loaded successfully")
            
        except ImportError as e:
            self.load_error = f"Missing dependencies: {e}"
            logger.warning(self.load_error)
        except Exception as e:
            self.load_error = f"Failed to load deepfake model: {e}"
            logger.warning(self.load_error)
    
    def detect_faces(self, image: Image.Image) -> bool:
        """Check if image contains faces (simple heuristic)."""
        # In production, use a proper face detector like MTCNN or RetinaFace
        # For now, we'll assume all images might contain faces
        return True
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Predict if image contains deepfake manipulation.
        
        Args:
            image: PIL Image object
            
        Returns:
            dict with deepfake detection results
        """
        if not self.model_loaded:
            return {
                'success': False,
                'error': self.load_error or "Model not loaded",
                'deepfake_probability': 50.0,
                'has_face': False
            }
        
        try:
            import torch
            
            # Check for faces
            has_face = self.detect_faces(image)
            if not has_face:
                return {
                    'success': True,
                    'has_face': False,
                    'deepfake_probability': 0.0,
                    'ai_probability': 0.0,
                    'note': 'No face detected, deepfake check skipped'
                }
            
            # Ensure RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Process and predict
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            probs = probs[0].cpu().numpy()
            
            # Extract deepfake probability
            deepfake_prob = self._get_deepfake_probability(probs)
            
            return {
                'success': True,
                'has_face': True,
                'deepfake_probability': deepfake_prob,
                'ai_probability': deepfake_prob,
                'all_probabilities': {
                    self.id2label[i]: float(p * 100) for i, p in enumerate(probs)
                },
                'model': 'DeepfakeDetector'
            }
            
        except Exception as e:
            logger.error(f"Deepfake detection error: {e}")
            return {
                'success': False,
                'error': str(e),
                'deepfake_probability': 50.0
            }
    
    def _get_deepfake_probability(self, probs: np.ndarray) -> float:
        """Extract deepfake probability from model output."""
        fake_keywords = ['fake', 'deepfake', 'manipulated', 'synthetic']
        
        for idx, label in self.id2label.items():
            if any(kw in label.lower() for kw in fake_keywords):
                return float(probs[idx] * 100)
        
        # Default: assume index 1 is fake
        if len(probs) >= 2:
            return float(probs[1] * 100)
        
        return 50.0

class FrequencyAnalyzer:
    """
    Frequency Domain Analysis for AI Image Detection
    
    Analyzes FFT/DCT patterns to detect:
    - GAN fingerprints (periodic patterns from transposed convolutions)
    - Upscaling artifacts
    - Compression anomalies
    """
    
    def __init__(self):
        """Initialize frequency analyzer."""
        pass
    
    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        """
        Perform frequency domain analysis.
        
        Args:
            image: PIL Image object
            
        Returns:
            dict with frequency analysis results
        """
        try:
            from scipy import fft
            from scipy import ndimage
            
            # Convert to grayscale numpy array
            if image.mode != 'L':
                gray = image.convert('L')
            else:
                gray = image
            
            img_array = np.array(gray, dtype=np.float32)
            
            # Compute 2D FFT
            fft_result = fft.fft2(img_array)
            fft_shifted = fft.fftshift(fft_result)
            magnitude = np.log(np.abs(fft_shifted) + 1)
            
            # Analyze frequency patterns
            results = {
                'success': True,
                'periodic_patterns': self._detect_periodic_patterns(magnitude),
                'high_freq_energy': self._analyze_high_frequency(magnitude),
                'grid_artifacts': self._detect_grid_artifacts(magnitude),
                'spectral_flatness': self._spectral_flatness(magnitude),
                'ai_probability_contribution': 0.0
            }
            
            # Calculate AI probability contribution from frequency analysis
            ai_score = 0.0
            
            if results['periodic_patterns']['detected']:
                ai_score += 25
            
            if results['grid_artifacts']['detected']:
                ai_score += 20
            
            if results['high_freq_energy']['anomaly']:
                ai_score += 15
            
            if results['spectral_flatness']['is_suspicious']:
                ai_score += 10
            
            results['ai_probability_contribution'] = min(70, ai_score)
            
            return results
            
        except ImportError as e:
            return {
                'success': False,
                'error': f"Missing scipy: {e}",
                'ai_probability_contribution': 0.0
            }
        except Exception as e:
            logger.error(f"Frequency analysis error: {e}")
            return {
                'success': False,
                'error': str(e),
                'ai_probability_contribution': 0.0
            }
    
    def _detect_periodic_patterns(self, magnitude: np.ndarray) -> Dict[str, Any]:
        """Detect periodic patterns typical of GAN upsampling."""
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        
        # Analyze radial profile for peaks
        radial_profile = []
        max_radius = min(center_h, center_w)
        
        for r in range(1, max_radius):
            mask = self._create_ring_mask(h, w, r, r + 1)
            mean_val = np.mean(magnitude[mask])
            radial_profile.append(mean_val)
        
        radial_profile = np.array(radial_profile)
        
        # Detect peaks (potential periodic artifacts)
        if len(radial_profile) > 10:
            from scipy import signal
            peaks, properties = signal.find_peaks(radial_profile, prominence=0.5)
            
            # GAN artifacts often show peaks at specific frequencies
            has_suspicious_peaks = len(peaks) > 3 and len(peaks) < 20
            
            return {
                'detected': has_suspicious_peaks,
                'num_peaks': len(peaks),
                'peak_positions': peaks.tolist() if len(peaks) < 10 else peaks[:10].tolist()
            }
        
        return {'detected': False, 'num_peaks': 0, 'peak_positions': []}
    
    def _create_ring_mask(self, h: int, w: int, r_inner: int, r_outer: int) -> np.ndarray:
        """Create a ring-shaped mask."""
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        return (dist >= r_inner) & (dist < r_outer)
    
    def _analyze_high_frequency(self, magnitude: np.ndarray) -> Dict[str, Any]:
        """Analyze high frequency energy distribution."""
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        
        # Define high frequency region (outer 30%)
        high_freq_mask = self._create_ring_mask(h, w, int(min(h, w) * 0.35), min(h, w) // 2)
        low_freq_mask = self._create_ring_mask(h, w, 0, int(min(h, w) * 0.15))
        
        high_freq_energy = np.mean(magnitude[high_freq_mask])
        low_freq_energy = np.mean(magnitude[low_freq_mask])
        
        # Ratio analysis
        if low_freq_energy > 0:
            ratio = high_freq_energy / low_freq_energy
        else:
            ratio = 0
        
        # AI images often have unusual high/low frequency ratios
        is_anomaly = ratio > 0.8 or ratio < 0.2
        
        return {
            'high_freq_energy': float(high_freq_energy),
            'low_freq_energy': float(low_freq_energy),
            'ratio': float(ratio),
            'anomaly': is_anomaly
        }
    
    def _detect_grid_artifacts(self, magnitude: np.ndarray) -> Dict[str, Any]:
        """Detect grid-like artifacts from convolution operations."""
        h, w = magnitude.shape
        
        # Look for horizontal and vertical lines in frequency domain
        center_h, center_w = h // 2, w // 2
        
        # Horizontal line energy (transposed conv artifact)
        h_line = magnitude[center_h - 2:center_h + 2, :].mean()
        
        # Vertical line energy
        v_line = magnitude[:, center_w - 2:center_w + 2].mean()
        
        # Overall energy
        overall = magnitude.mean()
        
        # Grid artifacts show as strong lines through center
        h_ratio = h_line / overall if overall > 0 else 0
        v_ratio = v_line / overall if overall > 0 else 0
        
        detected = h_ratio > 1.5 or v_ratio > 1.5
        
        return {
            'detected': detected,
            'horizontal_strength': float(h_ratio),
            'vertical_strength': float(v_ratio)
        }
    
    def _spectral_flatness(self, magnitude: np.ndarray) -> Dict[str, Any]:
        """Calculate spectral flatness (measure of uniformity)."""
        # Avoid log of zero
        magnitude = magnitude + 1e-10
        
        # Geometric mean / Arithmetic mean
        log_mean = np.mean(np.log(magnitude))
        geometric_mean = np.exp(log_mean)
        arithmetic_mean = np.mean(magnitude)
        
        flatness = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0
        
        # AI images sometimes have unusual spectral flatness
        # Very flat (close to 1) or very peaky (close to 0) can be suspicious
        is_suspicious = flatness > 0.9 or flatness < 0.1
        
        return {
            'flatness': float(flatness),
            'is_suspicious': is_suspicious
        }




def create_ml_detectors(device: str = "auto", load_all: bool = False) -> Dict[str, Any]:
    """
    Factory function to create ML detector instances.
    
    Args:
        device: Device to use ("auto", "cpu", "cuda")
        load_all: If True, load all detectors including heavy ones
        
    Returns:
        dict with detector instances
    """
    detectors: Dict[str, Any] = {
        'frequency_analyzer': FrequencyAnalyzer()
    }
    
    # 1. Ateeqq SigLIP2 (99.23%)
    try:
        detectors['ateeqq'] = AteeqqDetector(device=device)
    except Exception as e:
        logger.warning(f"Could not load Ateeqq detector: {e}")
        detectors['ateeqq'] = None
        
    # 2. SigLIP DINOv2 (99.97%)
    try:
        detectors['siglip_dinov2'] = SigLIPDINOv2Detector(device=device)
    except Exception as e:
        logger.warning(f"Could not load SigLIP DINOv2 detector: {e}")
        detectors['siglip_dinov2'] = None
        
    # 3. SDXL Detector (98.1%)
    try:
        detectors['sdxl'] = SDXLDetector(device=device)
    except Exception as e:
        logger.warning(f"Could not load SDXL detector: {e}")
        detectors['sdxl'] = None
        
    # 4. Deepfake ViT (98.25%)
    try:
        detectors['deepfake'] = DeepfakeDetector(device=device)
    except Exception as e:
        logger.warning(f"Could not load Deepfake detector: {e}")
        detectors['deepfake'] = None
        
    # 5. DINOv2 Deepfake
    try:
        detectors['dinov2'] = DINOv2DeepfakeDetector(device=device)
    except Exception as e:
        logger.warning(f"Could not load DINOv2 detector: {e}")
        detectors['dinov2'] = None
        
    return detectors










class DINOv2DeepfakeDetector:
    """
    DINOv2-based Deepfake Detector
    
    Uses Meta's DINOv2 self-supervised Vision Transformer backbone,
    which learns universal visual features without labels.
    
    Key strength: Most resilient to image degradation (JPEG compression,
    blur, resizing, social media processing) because DINOv2's features
    are learned from the structure of visual data itself, not from
    specific artifact patterns that degrade with compression.
    
    Model: WpythonW/dinoV2-deepfake-detector
    Architecture: DINOv2 (Meta's self-supervised ViT)
    Best for: Images from social media, messaging apps, or compressed sources
    """
    
    MODEL_ID = "WpythonW/dinoV2-deepfake-detector"
    
    def __init__(self, device: str = "auto"):
        """Initialize the DINOv2 deepfake detector."""
        self.model = None
        self.processor = None
        self.device = device
        self.model_loaded = False
        self.load_error = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the DINOv2 deepfake model from HuggingFace."""
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading DINOv2 deepfake detector on {self.device}...")
            
            self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
            self.model = AutoModelForImageClassification.from_pretrained(self.MODEL_ID)
            self.model.to(self.device)
            self.model.eval()
            
            self.id2label = self.model.config.id2label
            self.model_loaded = True
            logger.info(f"DINOv2 deepfake detector loaded. Labels: {self.id2label}")
            
        except ImportError as e:
            self.load_error = f"Missing dependencies: {e}"
            logger.warning(self.load_error)
        except Exception as e:
            self.load_error = f"Failed to load DINOv2 deepfake model: {e}"
            logger.warning(self.load_error)
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Predict whether an image is a deepfake using DINOv2 features.
        
        Particularly effective on compressed/degraded images from social media.
        """
        if not self.model_loaded:
            return {
                'success': False,
                'error': self.load_error or "Model not loaded",
                'ai_probability': 50.0,
                'prediction': 'unknown'
            }
        
        try:
            import torch
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            probs = probs[0].cpu().numpy()
            predicted_idx = int(np.argmax(probs))
            predicted_label = self.id2label.get(predicted_idx, f"class_{predicted_idx}")
            
            ai_prob = self._get_ai_probability(probs, predicted_label)
            
            return {
                'success': True,
                'prediction': predicted_label,
                'confidence': float(probs[predicted_idx] * 100),
                'ai_probability': ai_prob,
                'all_probabilities': {
                    self.id2label.get(i, f"class_{i}"): float(p * 100)
                    for i, p in enumerate(probs)
                },
                'model': 'DINOv2-Deepfake',
                'specialization': 'Degradation-resilient deepfake detection (social media, JPEG)'
            }
            
        except Exception as e:
            logger.error(f"DINOv2 deepfake prediction error: {e}")
            return {
                'success': False,
                'error': str(e),
                'ai_probability': 50.0,
                'prediction': 'error'
            }
    
    def _get_ai_probability(self, probs: np.ndarray, predicted_label: str) -> float:
        """Extract AI probability from model output."""
        ai_keywords = ['fake', 'deepfake', 'ai', 'generated', 'synthetic', 'artificial']
        real_keywords = ['real', 'authentic', 'human', 'natural', 'genuine']
        
        for idx, label in self.id2label.items():
            label_lower = label.lower()
            if any(kw in label_lower for kw in ai_keywords):
                return float(probs[idx] * 100)
            elif any(kw in label_lower for kw in real_keywords):
                return float((1 - probs[idx]) * 100)
        
        if len(probs) >= 2:
            return float(probs[1] * 100)
        return 50.0




# ============================================================================
# NEW 2026: Five additional pre-trained detectors for maximum quality ensemble
# ============================================================================


class SigLIPDINOv2Detector:
    """
    SigLIP2 + DINOv2 Ensemble Detector (Best Overall)
    
    Combines Google's SigLIP2 (semantic understanding) with Meta's DINOv2
    (self-supervised visual features) using LoRA adapters for efficient
    fine-tuning. Trained on OpenFake dataset with 25+ generators.
    
    Model: Bombek1/ai-image-detector-siglip-dinov2
    Architecture: SigLIP2-SO400M + DINOv2-Large + LoRA (~740M params, ~8M trainable)
    Accuracy: 97.15% cross-dataset, 99.97% AUC on OpenFake
    Strengths: Quality-agnostic (AUC gap 0.0003), covers Flux/MJ v6/DALL-E 3/GPT-Image-1
    """
    
    REPO_ID = "Bombek1/ai-image-detector-siglip-dinov2"
    
    def __init__(self, device: str = "auto"):
        """Initialize the SigLIP2+DINOv2 ensemble detector."""
        self.model = None
        self.siglip_processor = None
        self.dinov2_transform = None
        self.device = device
        self.model_loaded = False
        self.load_error = None
        
        self._load_model()
    
    def _load_model(self):
        """
        Load the Bombek1 ensemble model from HuggingFace.
        
        This model uses a custom architecture (not standard AutoModel),
        so we download model.py and pytorch_model.pt directly from the repo.
        """
        try:
            import torch
            import torch.nn as nn
            import math
            from torch.amp import autocast
            import timm
            from transformers import AutoProcessor, SiglipVisionModel
            from peft import LoraConfig, get_peft_model
            from torchvision import transforms
            from huggingface_hub import hf_hub_download
            
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading SigLIP2+DINOv2 detector on {self.device}...")
            
            # Download the pre-trained checkpoint
            model_path = hf_hub_download(
                repo_id=self.REPO_ID,
                filename="pytorch_model.pt"
            )
            
            # Load checkpoint to get config
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            config = checkpoint.get('config', {})
            
            siglip_name = config.get('siglip_model', 'google/siglip2-so400m-patch14-384')
            dinov2_name = config.get('dinov2_model', 'vit_large_patch14_dinov2.lvd142m')
            image_size = config.get('image_size', 392)
            lora_rank = config.get('lora_rank', 32)
            lora_alpha = config.get('lora_alpha', 64)
            lora_dropout = config.get('lora_dropout', 0.1)
            
            # Build the dual-encoder model architecture
            # SigLIP2 backbone
            siglip = SiglipVisionModel.from_pretrained(siglip_name, torch_dtype=torch.bfloat16)
            siglip_dim = siglip.config.hidden_size
            
            # DINOv2 backbone via timm
            dinov2 = timm.create_model(dinov2_name, pretrained=True, num_classes=0, img_size=image_size)
            dinov2_dim = dinov2.num_features
            
            # Classification head: LayerNorm → Linear → GELU → Dropout → Linear → GELU → Dropout → Linear → Sigmoid
            classifier = nn.Sequential(
                nn.LayerNorm(siglip_dim + dinov2_dim),
                nn.Linear(siglip_dim + dinov2_dim, 512),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1),
            )
            
            # Build full model as a module for state_dict loading
            class _EnsembleAIDetector(nn.Module):
                def __init__(self, siglip, dinov2, classifier):
                    super().__init__()
                    self.siglip = siglip
                    self.dinov2 = dinov2
                    self.classifier = classifier
                
                def forward(self, siglip_pixels, dinov2_pixels):
                    siglip_features = self.siglip(pixel_values=siglip_pixels).pooler_output
                    dinov2_features = self.dinov2(dinov2_pixels)
                    combined = torch.cat([siglip_features.float(), dinov2_features], dim=-1)
                    logits = self.classifier(combined).squeeze(-1)
                    return logits, siglip_features, dinov2_features
            
            full_model = _EnsembleAIDetector(siglip, dinov2, classifier)
            
            # Apply LoRA to SigLIP
            siglip_lora_config = LoraConfig(
                r=lora_rank, lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=lora_dropout, bias="none"
            )
            full_model.siglip = get_peft_model(full_model.siglip, siglip_lora_config)
            
            # Apply LoRA to DINOv2 QKV layers (custom implementation matches Bombek1's LoRALinear)
            class _LoRALinear(nn.Module):
                def __init__(self, original, rank, alpha, dropout=0.1):
                    super().__init__()
                    self.original = original
                    self.scaling = alpha / rank
                    for p in self.original.parameters():
                        p.requires_grad = False
                    self.lora_A = nn.Linear(original.in_features, rank, bias=False)
                    self.lora_B = nn.Linear(rank, original.out_features, bias=False)
                    self.dropout = nn.Dropout(dropout)
                    nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
                    nn.init.zeros_(self.lora_B.weight)
                
                def forward(self, x):
                    return self.original(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
            
            for name, module in full_model.dinov2.named_modules():
                if hasattr(module, 'qkv') and isinstance(module.qkv, nn.Linear):
                    module.qkv = _LoRALinear(module.qkv, lora_rank, lora_alpha, lora_dropout)
            
            # Load the trained weights
            full_model.load_state_dict(checkpoint['model_state_dict'])
            full_model.to(self.device)
            full_model.eval()
            
            self.model = full_model
            
            # Create preprocessors — SigLIP uses AutoProcessor, DINOv2 uses torchvision transforms
            self.siglip_processor = AutoProcessor.from_pretrained(siglip_name)
            self.dinov2_transform = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            self.model_loaded = True
            logger.info("SigLIP2+DINOv2 detector loaded (97.15% cross-dataset accuracy)")
        
        except ImportError as e:
            self.load_error = f"Missing dependencies: {e}. Install: pip install torch transformers timm peft"
            logger.warning(self.load_error)
        except Exception as e:
            self.load_error = f"Failed to load SigLIP2+DINOv2 model: {e}"
            logger.warning(self.load_error)
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Predict whether an image is AI-generated using dual SigLIP2+DINOv2 features.
        
        Uses sigmoid output (not softmax) — probability > 0.5 means AI-generated.
        """
        if not self.model_loaded:
            return {
                'success': False,
                'error': self.load_error or "Model not loaded",
                'ai_probability': 50.0,
                'prediction': 'unknown'
            }
        
        try:
            import torch
            from torch.amp import autocast
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Preprocess for both encoders
            siglip_inputs = self.siglip_processor(images=image, return_tensors="pt")
            siglip_pixels = siglip_inputs["pixel_values"].to(self.device)
            dinov2_pixels = self.dinov2_transform(image).unsqueeze(0).to(self.device)
            
            # Inference with mixed precision if on GPU
            with torch.no_grad():
                with autocast('cuda', enabled=(self.device == 'cuda')):
                    logits, _, _ = self.model(siglip_pixels, dinov2_pixels)
            
            # Sigmoid output: probability of being AI-generated
            probability = torch.sigmoid(logits).item()
            prediction = "ai-generated" if probability > 0.5 else "real"
            confidence = probability if probability > 0.5 else 1 - probability
            
            return {
                'success': True,
                'prediction': prediction,
                'confidence': float(confidence * 100),
                'ai_probability': float(probability * 100),
                'model': 'SigLIP2-DINOv2-Ensemble',
                'specialization': 'Quality-agnostic, 25+ generators (Flux, MJ v6, DALL-E 3, GPT-Image-1)'
            }
        
        except Exception as e:
            logger.error(f"SigLIP2+DINOv2 prediction error: {e}")
            return {
                'success': False,
                'error': str(e),
                'ai_probability': 50.0,
                'prediction': 'error'
            }

class SigLIPDINOv2Detector:
    """
    SigLIP2 + DINOv2 Deepfake Detector (Top Tier — Jan 2026)
    
    BEST OVERALL: Extremely robust across 25+ different AI generators
    including Flux, Midjourney v6, DALL-E 3, SD3, and GPT-Image-1.
    
    Model: Bombek1/SigLIP2-DINOv2-DeepfakeDetection
    Architecture: SigLIP2 + DINOv2 fusion
    Accuracy: 97.15% internal, 99.97% AUC
    """
    
    MODEL_ID = "Bombek1/SigLIP2-DINOv2-DeepfakeDetection"
    
    def __init__(self, device: str = "auto"):
        self.model = None
        self.processor = None
        self.device = device
        self.model_loaded = False
        self.load_error = None
        self.id2label = {}
        
        self._load_model()
    
    def _load_model(self):
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading SigLIP DINOv2 detector on {self.device}...")
            
            self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
            self.model = AutoModelForImageClassification.from_pretrained(self.MODEL_ID)
            self.model.to(self.device)
            self.model.eval()
            
            self.id2label = self.model.config.id2label
            self.model_loaded = True
            logger.info(f"SigLIP DINOv2 detector loaded. Labels: {self.id2label}")
            
        except ImportError as e:
            self.load_error = f"Missing dependencies: {e}"
            logger.warning(self.load_error)
        except Exception as e:
            self.load_error = f"Failed to load SigLIP DINOv2 model: {e}"
            logger.warning(self.load_error)
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        if not self.model_loaded:
            return {
                'success': False,
                'error': self.load_error or "Model not loaded",
                'ai_probability': 50.0,
                'prediction': 'unknown'
            }
        
        try:
            import torch
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            probs = probs[0].cpu().numpy()
            predicted_idx = int(np.argmax(probs))
            predicted_label = self.id2label.get(predicted_idx, f"class_{predicted_idx}")
            
            # Label heuristic: look for 'fake', 'ai' etc vs 'real'
            ai_keywords = ['fake', 'ai', 'synthetic', 'generated']
            real_keywords = ['real', 'authentic', 'human', 'original']
            ai_prob = 50.0
            
            for idx, label in self.id2label.items():
                lbl = str(label).lower()
                if any(k in lbl for k in ai_keywords):
                    ai_prob = float(probs[idx] * 100)
                    break
                elif any(k in lbl for k in real_keywords):
                    ai_prob = float((1.0 - probs[idx]) * 100)
                    break
            
            return {
                'success': True,
                'prediction': predicted_label,
                'confidence': float(probs[predicted_idx] * 100),
                'ai_probability': ai_prob,
                'all_probabilities': {
                    self.id2label.get(i, f"class_{i}"): float(p * 100)
                    for i, p in enumerate(probs)
                },
                'model': 'SigLIP2-DINOv2-Ensemble',
                'specialization': 'Quality-agnostic, 25+ generators'
            }
            
        except Exception as e:
            logger.error(f"SigLIP2+DINOv2 prediction error: {e}")
            return {
                'success': False,
                'error': str(e),
                'ai_probability': 50.0,
                'prediction': 'error'
            }
class AteeqqDetector:
    """
    Ateeqq AI vs Human Image Detector (Top Tier — Dec 2025)
    
    Fine-tuned SigLIP2 model trained on 120K images (60K AI + 60K human).
    One of the most popular and accurate detectors on HuggingFace.
    
    Model: Ateeqq/ai-vs-human-image-detector
    Architecture: SigLIP2 (google/siglip2-base, 92.9M params)
    Accuracy: 99.23% test accuracy, F1: 0.9923
    Downloads: 46K+/month, 27 Spaces using it
    Labels: {0: 'ai', 1: 'hum'}
    Updated: December 2025
    """
    
    MODEL_ID = "Ateeqq/ai-vs-human-image-detector"
    # Hardcoded label mapping — verified from model card
    LABEL_AI = 0    # label 'ai'
    LABEL_REAL = 1  # label 'hum'
    
    def __init__(self, device: str = "auto"):
        """Initialize the Ateeqq detector."""
        self.model = None
        self.processor = None
        self.device = device
        self.model_loaded = False
        self.load_error = None
        self.id2label = {}
        
        self._load_model()
    
    def _load_model(self):
        """Load the Ateeqq SigLIP2 model from HuggingFace."""
        try:
            import torch
            from transformers import AutoImageProcessor, SiglipForImageClassification
            
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading Ateeqq AI-vs-Human detector on {self.device}...")
            
            self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
            self.model = SiglipForImageClassification.from_pretrained(self.MODEL_ID)
            self.model.to(self.device)
            self.model.eval()
            
            self.id2label = self.model.config.id2label
            self.model_loaded = True
            logger.info(f"Ateeqq detector loaded. Labels: {self.id2label}")
            
        except ImportError as e:
            self.load_error = f"Missing dependencies: {e}. Install: pip install transformers torch"
            logger.warning(self.load_error)
        except Exception as e:
            self.load_error = f"Failed to load Ateeqq model: {e}"
            logger.warning(self.load_error)
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Predict whether an image is AI-generated or human-created."""
        if not self.model_loaded:
            return {
                'success': False,
                'error': self.load_error or "Model not loaded",
                'ai_probability': 50.0,
                'prediction': 'unknown'
            }
        
        try:
            import torch
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            probs = probs[0].cpu().numpy()
            predicted_idx = int(np.argmax(probs))
            predicted_label = self.id2label.get(predicted_idx, f"class_{predicted_idx}")
            
            # Hardcoded: label 0 = 'ai', label 1 = 'hum'
            ai_prob = float(probs[self.LABEL_AI] * 100)
            
            return {
                'success': True,
                'prediction': predicted_label,
                'confidence': float(probs[predicted_idx] * 100),
                'ai_probability': ai_prob,
                'all_probabilities': {
                    self.id2label.get(i, f"class_{i}"): float(p * 100)
                    for i, p in enumerate(probs)
                },
                'model': 'Ateeqq-SigLIP2',
                'specialization': 'AI vs Human (SigLIP2, 99.23% accuracy, Dec 2025)'
            }
            
        except Exception as e:
            logger.error(f"Ateeqq prediction error: {e}")
            return {
                'success': False,
                'error': str(e),
                'ai_probability': 50.0,
                'prediction': 'error'
            }






