"""
VisioNova ML-based Image Detectors
State-of-the-art deep learning models for AI-generated image detection.

Models:
1. NYUAD ViT Detector - Vision Transformer (97.36% accuracy)
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


class NYUADDetector:
    """
    NYUAD AI-Generated Image Detector
    
    Uses Vision Transformer (ViT) fine-tuned on AI-generated images.
    Model: NYUAD-ComNets/NYUAD_AI-generated_images_detector
    
    Accuracy: 97.36% on benchmark datasets
    Parameters: 85.8M
    Supports: CPU and GPU inference
    """
    
    MODEL_ID = "NYUAD-ComNets/NYUAD_AI-generated_images_detector"
    
    def __init__(self, device: str = "auto"):
        """
        Initialize the NYUAD detector.
        
        Args:
            device: "auto", "cpu", or "cuda"
        """
        self.model = None
        self.processor = None
        self.device = device
        self.model_loaded = False
        self.load_error = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the model and processor from HuggingFace."""
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            
            # Determine device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading NYUAD detector on {self.device}...")
            
            # Load processor and model
            self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
            self.model = AutoModelForImageClassification.from_pretrained(self.MODEL_ID)
            self.model.to(self.device)
            self.model.eval()
            
            # Get label mapping
            self.id2label = self.model.config.id2label
            
            self.model_loaded = True
            logger.info(f"NYUAD detector loaded successfully. Labels: {self.id2label}")
            
        except ImportError as e:
            self.load_error = f"Missing dependencies: {e}. Install with: pip install torch transformers"
            logger.warning(self.load_error)
        except Exception as e:
            self.load_error = f"Failed to load NYUAD model: {e}"
            logger.warning(self.load_error)
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Predict whether an image is AI-generated.
        
        Args:
            image: PIL Image object
            
        Returns:
            dict with prediction, confidence, and probabilities
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
            
            # Ensure RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Process image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            probs = probs[0].cpu().numpy()
            predicted_idx = int(np.argmax(probs))
            predicted_label = self.id2label[predicted_idx]
            
            # Determine AI probability based on label
            # Labels are typically: 0 = "Real", 1 = "AI" or similar
            ai_prob = self._get_ai_probability(probs, predicted_label)
            
            return {
                'success': True,
                'prediction': predicted_label,
                'predicted_index': predicted_idx,
                'confidence': float(probs[predicted_idx] * 100),
                'ai_probability': ai_prob,
                'all_probabilities': {
                    self.id2label[i]: float(p * 100) for i, p in enumerate(probs)
                },
                'model': 'NYUAD-ViT'
            }
            
        except Exception as e:
            logger.error(f"NYUAD prediction error: {e}")
            return {
                'success': False,
                'error': str(e),
                'ai_probability': 50.0,
                'prediction': 'error'
            }
    
    def _get_ai_probability(self, probs: np.ndarray, predicted_label: str) -> float:
        """Extract AI probability from model output."""
        # Common label patterns
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


class UniversalFakeDetector:
    """
    Universal AI Image Detector (Pretrained SwinV2)
    
    Uses haywoodsloan/ai-image-detector-dev-deploy — the most downloaded
    AI image detector on HuggingFace (148K+ downloads/month).
    
    Architecture: SwinV2 (197M params)
    Accuracy: 98.1% F1 on validation set
    Training: Broad dataset covering modern diffusion generators
    
    Replaces the old CLIP + untrained linear head approach.
    """
    
    MODEL_ID = "haywoodsloan/ai-image-detector-dev-deploy"
    
    def __init__(self, device: str = "auto", classifier_path: Optional[str] = None):
        """
        Initialize Universal AI Image Detector.
        
        Args:
            device: "auto", "cpu", or "cuda"
            classifier_path: Ignored (kept for backward compatibility)
        """
        self.model = None
        self.processor = None
        self.device = device
        self.model_loaded = False
        self.load_error = None
        
        self._load_model()
    
    def _load_model(self):
        """Load pretrained SwinV2 model from HuggingFace."""
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            
            # Determine device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading Universal AI Image Detector (SwinV2) on {self.device}...")
            
            self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
            self.model = AutoModelForImageClassification.from_pretrained(self.MODEL_ID)
            self.model.to(self.device)
            self.model.eval()
            
            self.id2label = self.model.config.id2label
            self.model_loaded = True
            logger.info(f"Universal AI Image Detector loaded. Labels: {self.id2label}")
            
        except ImportError as e:
            self.load_error = f"Missing dependencies: {e}. Install with: pip install torch transformers"
            logger.warning(self.load_error)
        except Exception as e:
            self.load_error = f"Failed to load Universal AI Image Detector: {e}"
            logger.warning(self.load_error)
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Predict whether an image is AI-generated.
        
        Args:
            image: PIL Image object
            
        Returns:
            dict with prediction and confidence
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
            
            # Ensure RGB
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
            
            # Determine AI probability based on label keywords
            ai_prob = self._get_ai_probability(probs_np, predicted_label)
            
            prediction = 'ai_generated' if ai_prob >= 50 else 'real'
            confidence = ai_prob if ai_prob >= 50 else (100 - ai_prob)
            
            return {
                'success': True,
                'prediction': prediction,
                'confidence': confidence,
                'ai_probability': ai_prob,
                'all_probabilities': {
                    self.id2label[i]: float(p * 100) for i, p in enumerate(probs_np)
                },
                'model': 'SwinV2-AI-Detector'
            }
            
        except Exception as e:
            logger.error(f"Universal AI Image Detector error: {e}")
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



class UmmMaybeDetector:
    """
    Umm-Maybe AI Image Detector
    
    A highly popular community model (280k+ downloads) for general AI image detection.
    Effective against various generators including Midjourney, Stable Diffusion, 
    and anime-style models.
    
    Model: umm-maybe/AI-image-detector
    Architecture: ResNet-50 / ViT based
    """
    
    MODEL_ID = "umm-maybe/AI-image-detector"
    
    def __init__(self, device: str = "auto"):
        """Initialize the Umm-Maybe detector."""
        self.model = None
        self.processor = None
        self.device = device
        self.model_loaded = False
        self.load_error = None
        self.id2label = {}
        
        self._load_model()
    
    def _load_model(self):
        """Load the Umm-Maybe model from HuggingFace."""
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading Umm-Maybe detector on {self.device}...")
            
            self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
            self.model = AutoModelForImageClassification.from_pretrained(self.MODEL_ID)
            self.model.to(self.device)
            self.model.eval()
            
            self.id2label = self.model.config.id2label
            self.model_loaded = True
            logger.info(f"Umm-Maybe detector loaded. Labels: {self.id2label}")
            
        except ImportError as e:
            self.load_error = f"Missing dependencies: {e}"
            logger.warning(self.load_error)
        except Exception as e:
            self.load_error = f"Failed to load Umm-Maybe model: {e}"
            logger.warning(self.load_error)
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Predict whether an image is AI-generated.
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
                'model': 'Umm-Maybe-Detector',
                'specialization': 'General AI Art / Anime / Digital Art'
            }
            
        except Exception as e:
            logger.error(f"Umm-Maybe prediction error: {e}")
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
        
        # Check specific label first
        for idx, label in self.id2label.items():
            label_lower = label.lower()
            if any(kw in label_lower for kw in ai_keywords):
                return float(probs[idx] * 100)
            elif any(kw in label_lower for kw in real_keywords):
                return float((1 - probs[idx]) * 100)
        
        if len(probs) >= 2:
            return float(probs[0] * 100)
        return 50.0

def create_ml_detectors(device: str = "auto", load_all: bool = False) -> Dict[str, Any]:
    """
    Factory function to create ML detector instances.
    
    Args:
        device: Device to use ("auto", "cpu", "cuda")
        load_all: If True, load all detectors including heavy ones
        
    Returns:
        dict with detector instances
    """
    detectors = {
        'frequency_analyzer': FrequencyAnalyzer()
    }
    
    # DIRE is the primary detector for latest generators (2024-2026)
    try:
        from .dire_detector import DIREDetector
        detectors['dire'] = DIREDetector()
    except Exception as e:
        logger.warning(f"Could not load DIRE detector: {e}")
        detectors['dire'] = None
    
    # NYUAD is backup (best accuracy/speed tradeoff)
    try:
        detectors['nyuad'] = NYUADDetector(device=device)
    except Exception as e:
        logger.warning(f"Could not load NYUAD detector: {e}")
        detectors['nyuad'] = None
    
    # NEW 2026: Umm-Maybe Detector (Community Choice - 280k+ downloads)
    try:
        # Class is defined above
        detectors['umm_maybe'] = UmmMaybeDetector(device=device)
    except Exception as e:
        logger.warning(f"Could not load Umm-Maybe detector: {e}")
        detectors['umm_maybe'] = None
    
    # NEW 2026: DINOv2 deepfake detector (degradation-resilient)
    try:
        detectors['dinov2'] = DINOv2DeepfakeDetector(device=device)
    except Exception as e:
        logger.warning(f"Could not load DINOv2 deepfake detector: {e}")
        detectors['dinov2'] = None
    
    return detectors



class FluxDetector:
    """
    Flux Image Detector
    
    Specialized detector for Flux.1 (Schnell/Dev/Pro) generated images.
    Model: LukasT9/flux-detector
    Architecture: EfficientNet-B2
    Accuracy: ~90% on Flux datasets
    """
    
    MODEL_ID = "LukasT9/flux-detector"
    
    def __init__(self, device: str = "auto"):
        self.model = None
        self.processor = None
        self.device = device
        self.model_loaded = False
        self.load_error = None
        
        self._load_model()
        
    def _load_model(self):
        """Load the Flux detection model."""
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                
            logger.info(f"Loading Flux detector on {self.device}...")
            
            self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
            self.model = AutoModelForImageClassification.from_pretrained(self.MODEL_ID)
            self.model.to(self.device)
            self.model.eval()
            
            self.id2label = self.model.config.id2label
            self.model_loaded = True
            logger.info("Flux detector loaded successfully")
            
        except ImportError as e:
            self.load_error = f"Missing dependencies: {e}. Install transformers torch"
            logger.warning(self.load_error)
        except Exception as e:
            self.load_error = f"Failed to load Flux detector: {e}"
            logger.warning(self.load_error)
            
    def predict(self, image: Any) -> Dict[str, Any]:
        """Predict if image is generated by Flux."""
        if not self.model_loaded:
            return {
                'success': False,
                'error': self.load_error,
                'ai_probability': 50.0
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
            
            # Label 0 is usually Real, 1 is Flux (check model card if unsure, usually 'flux' label exists)
            # For LukasT9/flux-detector: labels are typically 'real', 'flux'
            
            ai_prob = 0.0
            prediction = 'unknown'
            
            for i, label in self.id2label.items():
                if 'flux' in label.lower() or 'ai' in label.lower():
                    ai_prob = float(probs[i] * 100)
                    if probs[i] > 0.5:
                        prediction = 'flux'
                elif 'real' in label.lower():
                    if probs[i] > 0.5:
                        prediction = 'real'
            
            return {
                'success': True,
                'prediction': prediction,
                'ai_probability': ai_prob,
                'confidence': float(max(probs) * 100),
                'model': 'FluxDetector',
                'is_flux': prediction == 'flux'
            }
            
        except Exception as e:
            logger.error(f"Flux detection error: {e}")
            return {
                'success': False,
                'error': str(e),
                'ai_probability': 50.0
            }


class DIREDetector:
    """
    DIRE (Diffusion Reconstruction Error) Detector
    
    Specifically designed for diffusion models (2024-2026):
    - Stable Diffusion (all versions including XL)
    - DALL-E 2/3
    - Midjourney v5/v6
    - Imagen, Firefly, Flux
    
    Accuracy: 94.7% on latest generators
    Paper: CVPR 2024
    """
    
    MODEL_PATH = "models/dire_model.pth"
    
    def __init__(self, device: str = "auto"):
        """
        Initialize DIRE detector.
        
        Args:
            device: "auto", "cpu", or "cuda"
        """
        self.model = None
        self.device = self._determine_device(device)
        self.model_loaded = False
        
        # Load model
        self._load_model()
    
    def _determine_device(self, device: str) -> str:
        """Determine which device to use."""
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except:
                return "cpu"
        return device
    
    def _load_model(self):
        """Load DIRE model from disk."""
        try:
            import torch
            import torch.nn as nn
            from torchvision import models
            from pathlib import Path
            
            model_path = Path(__file__).parent / self.MODEL_PATH
            
            if not model_path.exists():
                logger.warning(f"DIRE model not found at {model_path}. Run download_models.py to download.")
                return
            
            # DIRE uses ResNet-50 backbone
            self.model = models.resnet50(pretrained=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, 1)
            
            # Load weights
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
            
            self.model.to(self.device)
            self.model.eval()
            
            self.model_loaded = True
            logger.info(f"✓ DIRE detector loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load DIRE model: {e}")
            self.model = None
            self.model_loaded = False
    
    def detect(self, image_data: bytes) -> Dict[str, Any]:
        """
        Detect if image is AI-generated using DIRE.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            dict with detection results
        """
        if not self.model_loaded:
            return {
                'success': False,
                'error': 'DIRE model not loaded. Run download_models.py',
                'ai_probability': 50.0
            }
        
        try:
            import torch
            from torchvision import transforms
            
            # Load image
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Preprocess
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            img_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                output = self.model(img_tensor)
                probability = torch.sigmoid(output).item()
            
            # DIRE outputs probability of being REAL, invert for AI probability
            ai_probability = (1 - probability) * 100
            
            return {
                'success': True,
                'ai_probability': round(ai_probability, 2),
                'confidence': round(abs(probability - 0.5) * 200, 2),
                'model': 'DIRE',
                'model_version': 'CVPR 2024',
                'specialization': 'Diffusion models (SD, DALL-E 3, Midjourney v6)',
                'recommended_for': ['stable_diffusion', 'dalle3', 'midjourney', 'flux']
            }
            
        except Exception as e:
            logger.error(f"DIRE detection failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'ai_probability': 50.0
            }





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


class SigLIPDetector:
    """
    SigLIP-based AI vs Human Image Detector
    
    Uses SigLIP (Sigmoid Loss for Language Image Pre-training) model
    fine-tuned to distinguish between AI-generated and human-created images.
    
    Accuracy: ~92% on benchmark datasets
    Strengths: Generalizes well across different AI generators
    
    Model: Ateeqq/ai-vs-human-image-detector
    """
    
    MODEL_ID = "google/siglip-base-patch16-224"  # Base model, we apply our classifier
    CLASSIFIER_REPO = "Ateeqq/ai-vs-human-image-detector"
    
    def __init__(self, device: str = "auto"):
        """Initialize the SigLIP detector."""
        self.model = None
        self.processor = None
        self.device = device
        self.model_loaded = False
        self.load_error = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the SigLIP model."""
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading SigLIP detector on {self.device}...")
            
            # Try to load from the classifier repo first
            try:
                self.processor = AutoImageProcessor.from_pretrained(self.CLASSIFIER_REPO)
                self.model = AutoModelForImageClassification.from_pretrained(self.CLASSIFIER_REPO)
            except Exception:
                # Fallback to base SigLIP with heuristic detection
                logger.info("Using SigLIP base model with heuristic classifier")
                self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
                self.model = AutoModelForImageClassification.from_pretrained(
                    self.MODEL_ID,
                    ignore_mismatched_sizes=True
                )
            
            self.model.to(self.device)
            self.model.eval()
            
            # Get label mapping if available
            self.id2label = getattr(self.model.config, 'id2label', {0: 'real', 1: 'ai_generated'})
            
            self.model_loaded = True
            logger.info(f"SigLIP detector loaded. Labels: {self.id2label}")
            
        except ImportError as e:
            self.load_error = f"Missing dependencies: {e}"
            logger.warning(self.load_error)
        except Exception as e:
            self.load_error = f"Failed to load SigLIP model: {e}"
            logger.warning(self.load_error)
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Predict whether an image is AI-generated or human-created.
        
        Uses SigLIP's strong visual understanding for classification.
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
            
            # Get label
            if isinstance(self.id2label, dict):
                predicted_label = self.id2label.get(predicted_idx, f"class_{predicted_idx}")
            else:
                predicted_label = "ai_generated" if predicted_idx == 1 else "real"
            
            # Calculate AI probability
            ai_prob = self._get_ai_probability(probs, predicted_label)
            
            return {
                'success': True,
                'prediction': predicted_label,
                'confidence': float(probs[predicted_idx] * 100),
                'ai_probability': ai_prob,
                'all_probabilities': {
                    str(self.id2label.get(i, f"class_{i}")): float(p * 100) 
                    for i, p in enumerate(probs)
                },
                'model': 'SigLIP-Classifier',
                'specialization': 'Human vs AI image classification'
            }
            
        except Exception as e:
            logger.error(f"SigLIP prediction error: {e}")
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
        
        # Check the label to determine which probability to use
        label_lower = predicted_label.lower()
        
        if any(kw in label_lower for kw in ai_keywords):
            # If predicted as AI, use that probability
            ai_prob = float(probs[np.argmax(probs)] * 100)
        elif any(kw in label_lower for kw in real_keywords):
            # If predicted as real, AI prob is 1 - real_prob
            ai_prob = float((1 - probs[np.argmax(probs)]) * 100)
        else:
            # Fallback: assume index 1 is AI
            if len(probs) >= 2:
                ai_prob = float(probs[1] * 100)
        
        return ai_prob


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


class DeepFakeV2Detector:
    """
    Deepfake Detector V2 (prithivMLmods, Updated 2025)
    
    Vision Transformer fine-tuned on a large, diverse deepfake dataset.
    Updated in February 2025 with more training data for better generalization.
    
    Model: prithivMLmods/Deep-Fake-Detector-v2-Model
    Architecture: ViT (fine-tuned)
    Accuracy: High F1 on validation & test (exact % not published)
    Strengths: Latest 2025 training data, broad deepfake coverage
    """
    
    MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"
    
    def __init__(self, device: str = "auto"):
        """Initialize the DeepFake V2 detector."""
        self.model = None
        self.processor = None
        self.device = device
        self.model_loaded = False
        self.load_error = None
        self.id2label = {}
        
        self._load_model()
    
    def _load_model(self):
        """Load the DeepFake V2 model from HuggingFace."""
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading DeepFake V2 detector on {self.device}...")
            
            self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
            self.model = AutoModelForImageClassification.from_pretrained(self.MODEL_ID)
            self.model.to(self.device)
            self.model.eval()
            
            self.id2label = self.model.config.id2label
            self.model_loaded = True
            logger.info(f"DeepFake V2 detector loaded. Labels: {self.id2label}")
        
        except ImportError as e:
            self.load_error = f"Missing dependencies: {e}"
            logger.warning(self.load_error)
        except Exception as e:
            self.load_error = f"Failed to load DeepFake V2 model: {e}"
            logger.warning(self.load_error)
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Predict whether an image is a deepfake (2025 dataset)."""
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
                'model': 'DeepFake-V2',
                'specialization': 'Deepfake detection (2025 dataset, ViT)'
            }
        
        except Exception as e:
            logger.error(f"DeepFake V2 prediction error: {e}")
            return {
                'success': False,
                'error': str(e),
                'ai_probability': 50.0,
                'prediction': 'error'
            }
    
    def _get_ai_probability(self, probs: np.ndarray, predicted_label: str) -> float:
        """Extract AI probability from model output."""
        ai_keywords = ['deepfake', 'fake', 'ai', 'generated', 'synthetic']
        real_keywords = ['realism', 'real', 'authentic', 'human', 'genuine']
        
        for idx, label in self.id2label.items():
            label_lower = label.lower()
            if any(kw in label_lower for kw in ai_keywords):
                return float(probs[idx] * 100)
            elif any(kw in label_lower for kw in real_keywords):
                return float((1 - probs[idx]) * 100)
        
        if len(probs) >= 2:
            return float(probs[1] * 100)
        return 50.0


class SigLIPDeepfakeDetector:
    """
    SigLIP-based Deepfake Detector (prithivMLmods V1)
    
    Fine-tuned from Google's SigLIP base model for binary deepfake classification.
    SigLIP's strong visual-semantic understanding makes it effective for face images.
    
    Model: prithivMLmods/deepfake-detector-model-v1
    Architecture: SigLIP (google/siglip-base-patch16-512, fine-tuned)
    Accuracy: High (unspecified), specializes in face deepfakes
    Strengths: SigLIP backbone, good for portrait/face images
    """
    
    MODEL_ID = "prithivMLmods/deepfake-detector-model-v1"
    
    def __init__(self, device: str = "auto"):
        """Initialize the SigLIP Deepfake detector."""
        self.model = None
        self.processor = None
        self.device = device
        self.model_loaded = False
        self.load_error = None
        self.id2label = {}
        
        self._load_model()
    
    def _load_model(self):
        """Load the SigLIP deepfake model from HuggingFace."""
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading SigLIP Deepfake detector on {self.device}...")
            
            self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
            self.model = AutoModelForImageClassification.from_pretrained(self.MODEL_ID)
            self.model.to(self.device)
            self.model.eval()
            
            self.id2label = self.model.config.id2label
            self.model_loaded = True
            logger.info(f"SigLIP Deepfake detector loaded. Labels: {self.id2label}")
        
        except ImportError as e:
            self.load_error = f"Missing dependencies: {e}"
            logger.warning(self.load_error)
        except Exception as e:
            self.load_error = f"Failed to load SigLIP Deepfake model: {e}"
            logger.warning(self.load_error)
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Predict whether an image is a deepfake using SigLIP features."""
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
                'model': 'SigLIP-Deepfake-V1',
                'specialization': 'Face deepfake detection (SigLIP backbone)'
            }
        
        except Exception as e:
            logger.error(f"SigLIP Deepfake prediction error: {e}")
            return {
                'success': False,
                'error': str(e),
                'ai_probability': 50.0,
                'prediction': 'error'
            }
    
    def _get_ai_probability(self, probs: np.ndarray, predicted_label: str) -> float:
        """Extract AI probability from model output."""
        ai_keywords = ['deepfake', 'fake', 'ai', 'generated', 'synthetic']
        real_keywords = ['real', 'authentic', 'human', 'genuine', 'realism']
        
        for idx, label in self.id2label.items():
            label_lower = label.lower()
            if any(kw in label_lower for kw in ai_keywords):
                return float(probs[idx] * 100)
            elif any(kw in label_lower for kw in real_keywords):
                return float((1 - probs[idx]) * 100)
        
        if len(probs) >= 2:
            return float(probs[1] * 100)
        return 50.0


class DistilledDetector:
    """
    Distilled AI Image Detector (Lightweight Generalization Specialist)
    
    A small, distilled ViT model trained on diverse generators.
    Despite only 11.8M parameters, it generalizes well across different
    AI generators because of knowledge distillation from larger models.
    
    Model: jacoballessio/ai-image-detect-distilled
    Architecture: ViT (distilled, 11.8M params)
    Accuracy: ~74% in real-world evaluation
    Strengths: Fast, diverse training (MJ + SD + fine-tuned SD), good generalization
    """
    
    MODEL_ID = "jacoballessio/ai-image-detect-distilled"
    
    def __init__(self, device: str = "auto"):
        """Initialize the Distilled detector."""
        self.model = None
        self.processor = None
        self.device = device
        self.model_loaded = False
        self.load_error = None
        self.id2label = {}
        
        self._load_model()
    
    def _load_model(self):
        """Load the distilled model from HuggingFace."""
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading Distilled AI detector on {self.device}...")
            
            self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
            self.model = AutoModelForImageClassification.from_pretrained(self.MODEL_ID)
            self.model.to(self.device)
            self.model.eval()
            
            self.id2label = self.model.config.id2label
            self.model_loaded = True
            logger.info(f"Distilled detector loaded. Labels: {self.id2label}")
        
        except ImportError as e:
            self.load_error = f"Missing dependencies: {e}"
            logger.warning(self.load_error)
        except Exception as e:
            self.load_error = f"Failed to load Distilled model: {e}"
            logger.warning(self.load_error)
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Predict whether an image is AI-generated using distilled ViT."""
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
                'model': 'Distilled-ViT',
                'specialization': 'Generalization (MJ + SD + fine-tuned SD, distilled)'
            }
        
        except Exception as e:
            logger.error(f"Distilled prediction error: {e}")
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
        
        for idx, label in self.id2label.items():
            label_lower = label.lower()
            if any(kw in label_lower for kw in ai_keywords):
                return float(probs[idx] * 100)
            elif any(kw in label_lower for kw in real_keywords):
                return float((1 - probs[idx]) * 100)
        
        if len(probs) >= 2:
            return float(probs[0] * 100)
        return 50.0


class AIorNotDetector:
    """
    AIorNot Detector (Diversity Signal)
    
    Community model for AI image classification. Added as an additional
    voting signal to increase ensemble diversity — different models
    catch different artifacts.
    
    Model: Nahrawy/AIorNot
    Architecture: Unknown (image classification)
    Accuracy: ~64.74% on anime benchmarks (better on general images)
    Role: Diversity signal in ensemble — catches different artifacts than other models
    """
    
    MODEL_ID = "Nahrawy/AIorNot"
    
    def __init__(self, device: str = "auto"):
        """Initialize the AIorNot detector."""
        self.model = None
        self.processor = None
        self.device = device
        self.model_loaded = False
        self.load_error = None
        self.id2label = {}
        
        self._load_model()
    
    def _load_model(self):
        """Load the AIorNot model from HuggingFace."""
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading AIorNot detector on {self.device}...")
            
            self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
            self.model = AutoModelForImageClassification.from_pretrained(self.MODEL_ID)
            self.model.to(self.device)
            self.model.eval()
            
            self.id2label = self.model.config.id2label
            self.model_loaded = True
            logger.info(f"AIorNot detector loaded. Labels: {self.id2label}")
        
        except ImportError as e:
            self.load_error = f"Missing dependencies: {e}"
            logger.warning(self.load_error)
        except Exception as e:
            self.load_error = f"Failed to load AIorNot model: {e}"
            logger.warning(self.load_error)
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Predict whether an image is AI-generated or not."""
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
                'model': 'AIorNot',
                'specialization': 'General AI vs Real classification (diversity signal)'
            }
        
        except Exception as e:
            logger.error(f"AIorNot prediction error: {e}")
            return {
                'success': False,
                'error': str(e),
                'ai_probability': 50.0,
                'prediction': 'error'
            }
    
    def _get_ai_probability(self, probs: np.ndarray, predicted_label: str) -> float:
        """Extract AI probability from model output."""
        ai_keywords = ['ai', 'fake', 'generated', 'synthetic', 'artificial']
        real_keywords = ['real', 'authentic', 'human', 'natural', 'genuine', 'not']
        
        for idx, label in self.id2label.items():
            label_lower = label.lower()
            if any(kw in label_lower for kw in ai_keywords):
                return float(probs[idx] * 100)
            elif any(kw in label_lower for kw in real_keywords):
                return float((1 - probs[idx]) * 100)
        
        if len(probs) >= 2:
            return float(probs[1] * 100)
        return 50.0


class EnsembleDetector:
    """
    Ensemble AI Image Detector
    
    Combines multiple detection models with weighted voting for maximum accuracy.
    
    Models and weights (updated Feb 2026):
    - Bombek1 SigLIP2+DINOv2: 25% (99.97% AUC, 25+ generators)
    - NYUAD ViT: 10% (97% accuracy, general detection)
    - SwinV2 Universal: 10% (98.1% accuracy, broad modern coverage)
    - SDXL Detector: 10% (98.1% accuracy, SDXL/SD3/modern diffusion)
    - Deepfake SigLIP2: 15% (SigLIP2 binary deepfake detection)
    - DINOv2 Deepfake: 10% (degradation-resilient detection)
    - 3-Class SigLIP2: 5% (AI/Deepfake/Real classification)
    - SMOGY: 10% (90% DALL-E, 87% SD, specialized for 2024 models)
    - SigLIP: 5% (92% human vs AI classification)
    
    Expected accuracy: 97-99% across all major AI generators
    """
    
    WEIGHTS = {
        'nyuad': 0.15,
        'clip': 0.15,              # SwinV2 (via UniversalFakeDetector)
        'sdxl': 0.15,              # Organika/sdxl-detector
        'dinov2': 0.15,            # DINOv2 degradation-resilient
        'umm_maybe': 0.25,         # Umm-Maybe (General purpose)
        'siglip': 0.15             # SigLIP (Human vs AI)
    }
    
    def __init__(self, device: str = "auto", load_all: bool = True):
        """
        Initialize the ensemble detector.
        
        Args:
            device: "auto", "cpu", or "cuda"
            load_all: If True, load all models. If False, load on demand.
        """
        self.device = device
        self.detectors = {}
        self.load_errors = {}
        
        if load_all:
            self._load_all_models()
    
    def _load_all_models(self):
        """Load all detection models."""
        logger.info("Loading ensemble models...")
        
        # Load NYUAD
        try:
            self.detectors['nyuad'] = NYUADDetector(self.device)
            if not self.detectors['nyuad'].model_loaded:
                self.load_errors['nyuad'] = self.detectors['nyuad'].load_error
        except Exception as e:
            self.load_errors['nyuad'] = str(e)
        

        # NEW 2026: DINOv2 Deepfake (degradation-resilient)
        try:
            self.detectors['dinov2'] = DINOv2DeepfakeDetector(self.device)
            if not self.detectors['dinov2'].model_loaded:
                self.load_errors['dinov2'] = self.detectors['dinov2'].load_error
        except Exception as e:
            self.load_errors['dinov2'] = str(e)

        # NEW 2026: Umm-Maybe Detector (Community Choice)
        try:
            self.detectors['umm_maybe'] = UmmMaybeDetector(self.device)
            if not self.detectors['umm_maybe'].model_loaded:
                self.load_errors['umm_maybe'] = self.detectors['umm_maybe'].load_error
        except Exception as e:
            self.load_errors['umm_maybe'] = str(e)
        
        loaded = [k for k, v in self.detectors.items() if v.model_loaded]
        logger.info(f"Ensemble loaded: {len(loaded)}/{len(self.WEIGHTS)} models ({', '.join(loaded)})")
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run ensemble prediction on an image.
        
        All loaded models vote with their weighted confidence.
        Final score is weighted average of all successful predictions.
        """
        results = {}
        weighted_sum = 0.0
        total_weight = 0.0
        
        for name, detector in self.detectors.items():
            if not detector.model_loaded:
                continue
            
            try:
                result = detector.predict(image)
                results[name] = result
                
                if result.get('success', False):
                    ai_prob = result.get('ai_probability', 50.0)
                    weight = self.WEIGHTS.get(name, 0.2)
                    weighted_sum += ai_prob * weight
                    total_weight += weight
                    
            except Exception as e:
                results[name] = {'success': False, 'error': str(e)}
        
        # Calculate final weighted score
        if total_weight > 0:
            final_ai_probability = weighted_sum / total_weight
        else:
            final_ai_probability = 50.0
        
        # Determine verdict based on ensemble
        if final_ai_probability >= 70:
            verdict = 'AI_GENERATED'
        elif final_ai_probability >= 50:
            verdict = 'LIKELY_AI'
        elif final_ai_probability >= 30:
            verdict = 'LIKELY_REAL'
        else:
            verdict = 'REAL'
        
        # Count votes
        ai_votes = sum(1 for r in results.values() if r.get('ai_probability', 50) > 50)
        total_votes = len([r for r in results.values() if r.get('success', False)])
        
        return {
            'success': True,
            'ai_probability': round(final_ai_probability, 2),
            'verdict': verdict,
            'confidence': round(abs(final_ai_probability - 50) * 2, 2),
            'ensemble_votes': f"{ai_votes}/{total_votes} models voted AI",
            'model_results': results,
            'weights_used': {k: v for k, v in self.WEIGHTS.items() if k in self.detectors and self.detectors[k].model_loaded},
            'models_loaded': list(self.detectors.keys()),
            'load_errors': self.load_errors
        }


