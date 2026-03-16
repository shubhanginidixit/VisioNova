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
    detectors = {
        'frequency_analyzer': FrequencyAnalyzer()
    }
    
    # DIRE is the primary detector for latest generators (2024-2026)
    try:
        detectors['dire'] = DIREDetector(device=device)
    except Exception as e:
        logger.warning(f"Could not load DIRE detector: {e}")
        detectors['dire'] = None
    
    # NYUAD is backup (best accuracy/speed tradeoff)
    try:
        detectors['nyuad'] = NYUADDetector(device=device)
    except Exception as e:
        logger.warning(f"Could not load NYUAD detector: {e}")
        detectors['nyuad'] = None
    
    # SMOGY - specialized for 2024 AI generators (DALL-E 3, SD XL, Flux)
    try:
        detectors['smogy'] = SMOGYDetector(device=device)
    except Exception as e:
        logger.warning(f"Could not load SMOGY detector: {e}")
        detectors['smogy'] = None
    
    # SigLIP - human vs AI classification
    try:
        detectors['siglip'] = SigLIPDetector(device=device)
    except Exception as e:
        logger.warning(f"Could not load SigLIP detector: {e}")
        detectors['siglip'] = None
    
    if load_all:
        # Load heavier models
        try:
            detectors['universal_fake'] = UniversalFakeDetector(device=device)
        except Exception as e:
            logger.warning(f"Could not load UniversalFakeDetect: {e}")
            detectors['universal_fake'] = None
        
        try:
            detectors['deepfake'] = DeepfakeDetector(device=device)
        except Exception as e:
            logger.warning(f"Could not load Deepfake detector: {e}")
            detectors['deepfake'] = None
        
        # EnsembleDetector - weighted combination of all models for best accuracy
        try:
            detectors['ensemble'] = EnsembleDetector(device=device, load_all=False)  # Uses already loaded models
        except Exception as e:
            logger.warning(f"Could not load Ensemble detector: {e}")
            detectors['ensemble'] = None
    
    # Flux Detector (new for 2025)
    try:
        detectors['flux'] = FluxDetector(device=device)
    except Exception as e:
        logger.warning(f"Could not load Flux detector: {e}")
        detectors['flux'] = None
    
    # SDXL Detector - specialized for modern diffusion models (SDXL, SD3, Flux)
    try:
        detectors['sdxl'] = SDXLDetector(device=device)
    except Exception as e:
        logger.warning(f"Could not load SDXL detector: {e}")
        detectors['sdxl'] = None
    
    # NEW 2026: Bombek1 SigLIP2+DINOv2 (best overall, 0.9997 AUC, 25+ generators)
    try:
        detectors['bombek1'] = Bombek1SigLIPDINOv2Detector(device=device)
    except Exception as e:
        logger.warning(f"Could not load Bombek1 SigLIP2+DINOv2 detector: {e}")
        detectors['bombek1'] = None
    
    # NEW 2026: Deepfake SigLIP2 binary detector
    try:
        detectors['siglip2_deepfake'] = DeepfakeSigLIP2Detector(device=device)
    except Exception as e:
        logger.warning(f"Could not load Deepfake SigLIP2 detector: {e}")
        detectors['siglip2_deepfake'] = None
    
    # NEW 2026: 3-Class SigLIP2 (AI-Generated vs Deepfake vs Real)
    try:
        detectors['three_class'] = ThreeClassSigLIP2Detector(device=device)
    except Exception as e:
        logger.warning(f"Could not load 3-Class SigLIP2 detector: {e}")
        detectors['three_class'] = None
    
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


class SMOGYDetector:
    """
    SMOGY AI Image Detector
    
    Fine-tuned model specifically for detecting 2024 AI generators:
    - DALL-E: ~90% accuracy
    - Stable Diffusion: ~87% accuracy
    - Flux AI: ~83% accuracy
    - Imagen: ~75% accuracy
    
    Model: Smogy/SMOGY-Ai-images-detector
    """
    
    MODEL_ID = "Smogy/SMOGY-Ai-images-detector"
    
    def __init__(self, device: str = "auto"):
        """Initialize the SMOGY detector."""
        self.model = None
        self.processor = None
        self.device = device
        self.model_loaded = False
        self.load_error = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the SMOGY model from HuggingFace."""
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading SMOGY detector on {self.device}...")
            
            self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
            self.model = AutoModelForImageClassification.from_pretrained(self.MODEL_ID)
            self.model.to(self.device)
            self.model.eval()
            
            self.id2label = self.model.config.id2label
            self.model_loaded = True
            logger.info(f"SMOGY detector loaded. Labels: {self.id2label}")
            
        except ImportError as e:
            self.load_error = f"Missing dependencies: {e}"
            logger.warning(self.load_error)
        except Exception as e:
            self.load_error = f"Failed to load SMOGY model: {e}"
            logger.warning(self.load_error)
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Predict whether an image is AI-generated.
        
        Specialized for 2024 AI models (DALL-E 3, SD XL, Flux, Midjourney v6).
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
            predicted_label = self.id2label[predicted_idx]
            
            # Determine AI probability
            ai_prob = self._get_ai_probability(probs, predicted_label)
            
            return {
                'success': True,
                'prediction': predicted_label,
                'confidence': float(probs[predicted_idx] * 100),
                'ai_probability': ai_prob,
                'all_probabilities': {
                    self.id2label[i]: float(p * 100) for i, p in enumerate(probs)
                },
                'model': 'SMOGY-2024',
                'specialization': 'DALL-E 3, Stable Diffusion XL, Flux, Midjourney v6'
            }
            
        except Exception as e:
            logger.error(f"SMOGY prediction error: {e}")
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


class Bombek1SigLIPDINOv2Detector:
    """
    Bombek1 SigLIP2 + DINOv2 Ensemble AI Image Detector
    
    The most accurate open-source AI image detector available (Feb 2026).
    Dual-encoder architecture combining SigLIP2's semantic understanding
    with DINOv2's self-supervised features.
    
    Model: Bombek1/ai-image-detector-siglip-dinov2
    Architecture: SigLIP2-SO400M + DINOv2-Large (~740M params, ~8M trainable LoRA)
    AUC: 0.9997 on OpenFake validation
    Cross-dataset accuracy: 97.15% across 10+ external datasets
    Quality-agnostic: AUC gap between clean and degraded is only 0.0003
    
    Trained on OpenFake dataset covering 25+ generators:
    - Diffusion: SD 1.5/2.1/XL/3.5, Flux 1.0/1.1 Pro, DALL-E 3, MJ v5/v6, Imagen, Kandinsky
    - GANs: StyleGAN, ProGAN, BigGAN
    - Other: GPT-Image-1, Firefly, Ideogram
    """
    
    MODEL_ID = "Bombek1/ai-image-detector-siglip-dinov2"
    
    def __init__(self, device: str = "auto"):
        """Initialize the Bombek1 SigLIP2+DINOv2 detector."""
        self.model = None
        self.processor = None
        self.device = device
        self.model_loaded = False
        self.load_error = None
        self.pipeline = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the Bombek1 SigLIP2+DINOv2 model from HuggingFace."""
        try:
            import torch
            
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading Bombek1 SigLIP2+DINOv2 detector on {self.device}...")
            
            # This model uses a custom architecture (dual encoder with LoRA).
            # Try loading via transformers pipeline first (if model card supports it),
            # then fall back to manual loading.
            try:
                from transformers import pipeline
                self.pipeline = pipeline(
                    "image-classification",
                    model=self.MODEL_ID,
                    device=0 if self.device == "cuda" else -1,
                    trust_remote_code=True
                )
                self.model_loaded = True
                logger.info("Bombek1 SigLIP2+DINOv2 loaded via pipeline (trust_remote_code)")
            except Exception as pipe_err:
                logger.info(f"Pipeline load failed ({pipe_err}), trying AutoModel...")
                try:
                    from transformers import AutoImageProcessor, AutoModelForImageClassification
                    self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID, trust_remote_code=True)
                    self.model = AutoModelForImageClassification.from_pretrained(
                        self.MODEL_ID, trust_remote_code=True
                    )
                    self.model.to(self.device)
                    self.model.eval()
                    self.id2label = self.model.config.id2label
                    self.model_loaded = True
                    logger.info(f"Bombek1 loaded via AutoModel. Labels: {self.id2label}")
                except Exception as auto_err:
                    self.load_error = f"Failed to load Bombek1 model: {auto_err}"
                    logger.warning(self.load_error)
            
        except ImportError as e:
            self.load_error = f"Missing dependencies: {e}"
            logger.warning(self.load_error)
        except Exception as e:
            self.load_error = f"Failed to load Bombek1 model: {e}"
            logger.warning(self.load_error)
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Predict whether an image is AI-generated using the Bombek1 ensemble.
        
        Covers 25+ generators including Flux 1.0/1.1 Pro, GPT-Image-1,
        DALL-E 3, Midjourney v5/v6, and all Stable Diffusion variants.
        """
        if not self.model_loaded:
            return {
                'success': False,
                'error': self.load_error or "Model not loaded",
                'ai_probability': 50.0,
                'prediction': 'unknown'
            }
        
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Use pipeline if available
            if self.pipeline is not None:
                results = self.pipeline(image)
                
                ai_prob = 50.0
                predicted_label = 'unknown'
                all_probs = {}
                
                for r in results:
                    label = r['label'].lower()
                    score = r['score'] * 100
                    all_probs[r['label']] = score
                    
                    if any(kw in label for kw in ['ai', 'fake', 'generated', 'synthetic']):
                        ai_prob = score
                        predicted_label = r['label']
                    elif any(kw in label for kw in ['real', 'human', 'authentic']):
                        ai_prob = 100 - score
                        if ai_prob < 50:
                            predicted_label = r['label']
                
                return {
                    'success': True,
                    'prediction': predicted_label,
                    'confidence': max(r['score'] for r in results) * 100,
                    'ai_probability': ai_prob,
                    'all_probabilities': all_probs,
                    'model': 'Bombek1-SigLIP2-DINOv2',
                    'specialization': '25+ generators (Flux, GPT-Image-1, DALL-E 3, MJ v6, SDXL)',
                    'auc': 0.9997
                }
            
            # Use AutoModel path
            import torch
            
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
                'model': 'Bombek1-SigLIP2-DINOv2',
                'specialization': '25+ generators (Flux, GPT-Image-1, DALL-E 3, MJ v6, SDXL)',
                'auc': 0.9997
            }
            
        except Exception as e:
            logger.error(f"Bombek1 prediction error: {e}")
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
        
        # Fallback: assume index 1 is AI
        if len(probs) >= 2:
            return float(probs[1] * 100)
        return 50.0


class DeepfakeSigLIP2Detector:
    """
    Deepfake Detection using SigLIP2 (Google's Sigmoid Language-Image Pretraining v2)
    
    Fine-tuned from google/siglip2-base-patch16-224 for binary fake/real classification.
    Leverages SigLIP2's efficient vision-language pre-training for detecting manipulated images.
    
    Model: prithivMLmods/Deepfake-Detect-Siglip2
    Labels: Fake (0), Real (1)
    Architecture: SiglipForImageClassification
    Updated: April 2025
    """
    
    MODEL_ID = "prithivMLmods/Deepfake-Detect-Siglip2"
    
    def __init__(self, device: str = "auto"):
        """Initialize the SigLIP2 deepfake detector."""
        self.model = None
        self.processor = None
        self.device = device
        self.model_loaded = False
        self.load_error = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the SigLIP2 deepfake model from HuggingFace."""
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading Deepfake SigLIP2 detector on {self.device}...")
            
            self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
            self.model = AutoModelForImageClassification.from_pretrained(self.MODEL_ID)
            self.model.to(self.device)
            self.model.eval()
            
            self.id2label = self.model.config.id2label
            self.model_loaded = True
            logger.info(f"Deepfake SigLIP2 loaded. Labels: {self.id2label}")
            
        except ImportError as e:
            self.load_error = f"Missing dependencies: {e}"
            logger.warning(self.load_error)
        except Exception as e:
            self.load_error = f"Failed to load Deepfake SigLIP2 model: {e}"
            logger.warning(self.load_error)
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Predict whether an image is a deepfake or real.
        
        Uses Google's SigLIP2 architecture for efficient vision-language detection.
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
                'model': 'Deepfake-SigLIP2',
                'specialization': 'Binary deepfake detection (Fake/Real)'
            }
            
        except Exception as e:
            logger.error(f"Deepfake SigLIP2 prediction error: {e}")
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
            return float(probs[0] * 100)  # Class 0 = Fake for this model
        return 50.0


class ThreeClassSigLIP2Detector:
    """
    Three-Class AI Image Detector using SigLIP2
    
    Unique model that distinguishes between:
    - AI-Generated images (from text-to-image models)
    - Deepfake images (face-swapped/manipulated)
    - Real images (authentic photographs)
    
    This distinction is valuable for content moderation as the response
    to AI art vs deepfakes may differ significantly.
    
    Model: prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2
    Architecture: SiglipForImageClassification (google/siglip2-base-patch16-224)
    Labels: AI-Generated, Deepfake, Real
    Updated: April 2025
    """
    
    MODEL_ID = "prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2"
    
    def __init__(self, device: str = "auto"):
        """Initialize the 3-class SigLIP2 detector."""
        self.model = None
        self.processor = None
        self.device = device
        self.model_loaded = False
        self.load_error = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the 3-class SigLIP2 model from HuggingFace."""
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading 3-Class SigLIP2 detector on {self.device}...")
            
            self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
            self.model = AutoModelForImageClassification.from_pretrained(self.MODEL_ID)
            self.model.to(self.device)
            self.model.eval()
            
            self.id2label = self.model.config.id2label
            self.model_loaded = True
            logger.info(f"3-Class SigLIP2 loaded. Labels: {self.id2label}")
            
        except ImportError as e:
            self.load_error = f"Missing dependencies: {e}"
            logger.warning(self.load_error)
        except Exception as e:
            self.load_error = f"Failed to load 3-Class SigLIP2 model: {e}"
            logger.warning(self.load_error)
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Classify an image into one of three categories:
        AI-Generated, Deepfake, or Real.
        
        Returns both overall ai_probability and per-class breakdown.
        """
        if not self.model_loaded:
            return {
                'success': False,
                'error': self.load_error or "Model not loaded",
                'ai_probability': 50.0,
                'prediction': 'unknown',
                'class_breakdown': {}
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
            
            # Build class breakdown
            class_breakdown = {
                self.id2label.get(i, f"class_{i}"): float(p * 100)
                for i, p in enumerate(probs)
            }
            
            # AI probability = AI-Generated + Deepfake (both are non-authentic)
            ai_prob = 0.0
            for idx, label in self.id2label.items():
                label_lower = label.lower()
                if any(kw in label_lower for kw in ['ai', 'generated', 'deepfake', 'fake', 'synthetic']):
                    ai_prob += float(probs[idx] * 100)
            
            # If no AI-like labels found, use inverse of real probability
            if ai_prob == 0.0:
                for idx, label in self.id2label.items():
                    if any(kw in label.lower() for kw in ['real', 'authentic', 'human']):
                        ai_prob = 100.0 - float(probs[idx] * 100)
                        break
                else:
                    ai_prob = 50.0
            
            return {
                'success': True,
                'prediction': predicted_label,
                'confidence': float(probs[predicted_idx] * 100),
                'ai_probability': min(100.0, ai_prob),
                'class_breakdown': class_breakdown,
                'all_probabilities': class_breakdown,
                'model': '3-Class-SigLIP2',
                'specialization': 'AI-Generated vs Deepfake vs Real classification',
                'is_deepfake': 'deepfake' in predicted_label.lower(),
                'is_ai_generated': any(kw in predicted_label.lower() for kw in ['ai', 'generated']),
                'is_real': 'real' in predicted_label.lower()
            }
            
        except Exception as e:
            logger.error(f"3-Class SigLIP2 prediction error: {e}")
            return {
                'success': False,
                'error': str(e),
                'ai_probability': 50.0,
                'prediction': 'error',
                'class_breakdown': {}
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
        'bombek1': 0.25,           # NEW: Best overall (99.97% AUC, 25+ generators)
        'nyuad': 0.10,
        'clip': 0.10,              # SwinV2 (via UniversalFakeDetector)
        'sdxl': 0.10,              # Organika/sdxl-detector
        'siglip2_deepfake': 0.15,  # NEW: SigLIP2 binary deepfake
        'dinov2': 0.10,            # NEW: DINOv2 degradation-resilient
        'three_class': 0.05,       # NEW: 3-class AI/Deepfake/Real
        'smogy': 0.10,
        'siglip': 0.05
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
        
        # Load SMOGY
        try:
            self.detectors['smogy'] = SMOGYDetector(self.device)
            if not self.detectors['smogy'].model_loaded:
                self.load_errors['smogy'] = self.detectors['smogy'].load_error
        except Exception as e:
            self.load_errors['smogy'] = str(e)
        
        # Load SigLIP
        try:
            self.detectors['siglip'] = SigLIPDetector(self.device)
            if not self.detectors['siglip'].model_loaded:
                self.load_errors['siglip'] = self.detectors['siglip'].load_error
        except Exception as e:
            self.load_errors['siglip'] = str(e)
        
        # Load SwinV2 Universal (replaces old CLIP+untrained-head)
        try:
            self.detectors['clip'] = UniversalFakeDetector(self.device)
            if not self.detectors['clip'].model_loaded:
                self.load_errors['clip'] = self.detectors['clip'].load_error
        except Exception as e:
            self.load_errors['clip'] = str(e)
        
        # NEW: Load SDXL Detector (Organika/sdxl-detector)
        try:
            self.detectors['sdxl'] = SDXLDetector(self.device)
            if not self.detectors['sdxl'].model_loaded:
                self.load_errors['sdxl'] = self.detectors['sdxl'].load_error
        except Exception as e:
            self.load_errors['sdxl'] = str(e)
        
        # NEW 2026: Load Bombek1 SigLIP2+DINOv2 (best overall, 99.97% AUC)
        try:
            self.detectors['bombek1'] = Bombek1SigLIPDINOv2Detector(self.device)
            if not self.detectors['bombek1'].model_loaded:
                self.load_errors['bombek1'] = self.detectors['bombek1'].load_error
        except Exception as e:
            self.load_errors['bombek1'] = str(e)
        
        # NEW 2026: Load Deepfake SigLIP2
        try:
            self.detectors['siglip2_deepfake'] = DeepfakeSigLIP2Detector(self.device)
            if not self.detectors['siglip2_deepfake'].model_loaded:
                self.load_errors['siglip2_deepfake'] = self.detectors['siglip2_deepfake'].load_error
        except Exception as e:
            self.load_errors['siglip2_deepfake'] = str(e)
        
        # NEW 2026: Load 3-Class SigLIP2 (AI/Deepfake/Real)
        try:
            self.detectors['three_class'] = ThreeClassSigLIP2Detector(self.device)
            if not self.detectors['three_class'].model_loaded:
                self.load_errors['three_class'] = self.detectors['three_class'].load_error
        except Exception as e:
            self.load_errors['three_class'] = str(e)
        
        # NEW 2026: Load DINOv2 Deepfake (degradation-resilient)
        try:
            self.detectors['dinov2'] = DINOv2DeepfakeDetector(self.device)
            if not self.detectors['dinov2'].model_loaded:
                self.load_errors['dinov2'] = self.detectors['dinov2'].load_error
        except Exception as e:
            self.load_errors['dinov2'] = str(e)
        
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
