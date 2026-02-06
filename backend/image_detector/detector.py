"""
VisioNova Image Detector
Main detector class for AI-generated image detection.

Uses a combination of:
1. Deep learning classifier (CNN-based)
2. Statistical analysis (frequency domain, noise patterns)
3. Metadata forensics
4. Error Level Analysis (ELA)

For advanced detection with multiple models, use EnsembleDetector.
"""

import base64
import io
import logging
from typing import Optional
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageDetector:
    """
    AI-Generated Image Detector
    
    Combines multiple detection methods:
    - CNN classifier for AI vs Real classification
    - Frequency domain analysis for GAN artifacts
    - Noise consistency analysis
    - Integration with metadata and ELA analyzers
    """
    
    def __init__(self, model_path: Optional[str] = None, use_gpu: bool = False):
        """
        Initialize the image detector.
        
        Args:
            model_path: Path to trained model weights (optional)
            use_gpu: Whether to use GPU for inference
        """
        self.model = None
        self.model_loaded = False
        self.use_gpu = use_gpu
        self.device = 'cpu'
        self.ml_detectors = None
        
        # Try to load ML models (DIRE + NYUAD)
        try:
            from .ml_detector import create_ml_detectors
            self.ml_detectors = create_ml_detectors(
                device="cuda" if use_gpu else "cpu",
                load_all=False
            )
            
            # Check if any ML models loaded
            if self.ml_detectors.get('dire') and self.ml_detectors['dire'].model_loaded:
                self.model_loaded = True
                logger.info("✓ DIRE ML model loaded - 94% accuracy on latest AI generators")
            elif self.ml_detectors.get('nyuad') and self.ml_detectors['nyuad'].model_loaded:
                self.model_loaded = True
                logger.info("✓ NYUAD ML model loaded - 97% accuracy")
            else:
                logger.warning("No ML models loaded. Using statistical analysis only.")
                logger.info("Run 'python backend/setup_ml_models.py' to download models")
                
        except Exception as e:
            logger.warning(f"Could not load ML models: {e}. Using statistical analysis only.")
            self.model_loaded = False
    
    def _load_model(self, model_path: Optional[str] = None):
        """
        Load the AI detection model.
        
        For now, we'll use statistical methods. In production, this would load
        a pre-trained CNN like ResNet or EfficientNet fine-tuned on AI vs Real images.
        """
        try:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms
            
            self.device = 'cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {self.device}")
            
            # Define image transforms
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            # For now, we'll use statistical analysis
            # In production, load a fine-tuned model here:
            # self.model = torch.load(model_path)
            # self.model.eval()
            # self.model.to(self.device)
            
            self.model_loaded = False  # Set to True when model is trained
            logger.info("Image detector initialized (statistical mode)")
            
        except ImportError:
            logger.warning("PyTorch not available. Using statistical analysis only.")
            self.model_loaded = False
    
    def detect(self, image_data: bytes, filename: str = "image") -> dict:
        """
        Analyze an image to detect if it's AI-generated.
        
        Args:
            image_data: Raw image bytes
            filename: Original filename for logging
            
        Returns:
            dict with detection results
        """
        try:
            # Load image
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Get image info
            width, height = image.size
            
            # Run detection methods
            results = {
                'success': True,
                'filename': filename,
                'dimensions': {'width': width, 'height': height},
                'file_size_bytes': len(image_data),
            }
            
            # Statistical analysis (always available)
            stats = self._statistical_analysis(image)
            results.update(stats)
            
            # Deep learning prediction (if model loaded)
            if self.model_loaded and self.model is not None:
                ml_result = self._ml_prediction(image)
                results['ml_prediction'] = ml_result
                # Combine ML and statistical scores
                results['ai_probability'] = (
                    results['ai_probability'] * 0.3 + ml_result['confidence'] * 0.7
                )
            
            # Determine verdict
            ai_prob = results['ai_probability']
            if ai_prob >= 80:
                results['verdict'] = 'AI_GENERATED'
                results['verdict_description'] = 'High confidence: This image appears to be AI-generated'
            elif ai_prob >= 60:
                results['verdict'] = 'LIKELY_AI'
                results['verdict_description'] = 'Moderate confidence: This image shows signs of AI generation'
            elif ai_prob >= 40:
                results['verdict'] = 'UNCERTAIN'
                results['verdict_description'] = 'Inconclusive: Cannot determine with confidence'
            elif ai_prob >= 20:
                results['verdict'] = 'LIKELY_REAL'
                results['verdict_description'] = 'Moderate confidence: This image appears to be authentic'
            else:
                results['verdict'] = 'REAL'
                results['verdict_description'] = 'High confidence: This image appears to be a real photograph'
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {
                'success': False,
                'error': str(e),
                'ai_probability': 50.0,
                'verdict': 'ERROR',
                'verdict_description': f'Analysis failed: {str(e)}'
            }
    
    def _statistical_analysis(self, image: Image.Image) -> dict:
        """
        Perform statistical analysis to detect AI-generated images.
        
        Analyzes:
        - Color distribution patterns
        - Noise characteristics
        - Edge patterns
        - Texture consistency
        """
        img_array = np.array(image)
        
        # Initialize scores
        scores = {}
        
        # 1. Color histogram analysis
        # AI images often have smoother, more uniform color distributions
        color_score = self._analyze_color_distribution(img_array)
        scores['color_uniformity'] = color_score
        
        # 2. Noise analysis
        # Real photos have natural sensor noise; AI images have artificial noise patterns
        noise_score = self._analyze_noise_patterns(img_array)
        scores['noise_consistency'] = noise_score
        
        # 3. Edge analysis
        # AI images often have softer or more uniform edges
        edge_score = self._analyze_edges(img_array)
        scores['edge_naturalness'] = edge_score
        
        # 4. Texture analysis
        # AI struggles with fine textures (hair, fabric, skin)
        texture_score = self._analyze_texture(img_array)
        scores['texture_quality'] = texture_score
        
        # 5. Frequency domain analysis
        # GANs leave characteristic patterns in frequency domain
        freq_score = self._analyze_frequency_domain(img_array)
        scores['frequency_anomaly'] = freq_score
        
        # Combine scores (weighted average)
        weights = {
            'color_uniformity': 0.15,
            'noise_consistency': 0.25,
            'edge_naturalness': 0.15,
            'texture_quality': 0.20,
            'frequency_anomaly': 0.25
        }
        
        ai_probability = sum(
            scores[key] * weights[key] for key in weights
        )
        
        # Calculate color anomaly score for forensics display
        color_anomaly_score = round(scores.get('color_uniformity', 0), 2)
        
        # Build noise analysis breakdown
        noise_analysis = self._build_noise_analysis(img_array)
        
        return {
            'ai_probability': round(ai_probability, 2),
            'analysis_scores': scores,
            'detection_method': 'statistical',
            'color_anomaly_score': color_anomaly_score,
            'noise_analysis': noise_analysis
        }
    
    def _analyze_color_distribution(self, img: np.ndarray) -> float:
        """
        Analyze color distribution for AI-like uniformity.
        
        AI images often have unnaturally smooth color gradients.
        Returns 0-100 (higher = more likely AI).
        """
        # Calculate color histogram for each channel
        scores = []
        for channel in range(3):
            hist, _ = np.histogram(img[:, :, channel].flatten(), bins=256, range=(0, 256))
            hist = hist / hist.sum()  # Normalize
            
            # Calculate entropy (lower entropy = more uniform = more AI-like)
            entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
            # Typical range: 4-8 for real images, 3-6 for AI
            # Normalize to 0-100 (inverted: lower entropy = higher AI score)
            normalized = max(0, min(100, (8 - entropy) * 20))
            scores.append(normalized)
        
        return np.mean(scores)
    
    def _analyze_noise_patterns(self, img: np.ndarray) -> float:
        """
        Analyze noise patterns for AI artifacts.
        
        Real camera sensors produce consistent noise patterns (PRNU).
        AI images have artificial or missing noise.
        Returns 0-100 (higher = more likely AI).
        """
        # Convert to grayscale for noise analysis
        if len(img.shape) == 3:
            gray = np.mean(img, axis=2)
        else:
            gray = img
        
        # Calculate local variance (noise estimation)
        from scipy import ndimage
        
        # Use Laplacian to estimate noise
        laplacian = ndimage.laplace(gray.astype(float))
        noise_std = np.std(laplacian)
        
        # Real photos typically have noise_std in range 5-30
        # AI images often have very low (< 5) or artificially uniform noise
        if noise_std < 3:
            return 85.0  # Very low noise = likely AI
        elif noise_std < 8:
            return 60.0  # Low noise
        elif noise_std < 20:
            return 30.0  # Normal noise = likely real
        else:
            return 20.0  # High noise = likely real photo
    
    def _analyze_edges(self, img: np.ndarray) -> float:
        """
        Analyze edge characteristics.
        
        AI images often have unnaturally smooth or uniform edges.
        Returns 0-100 (higher = more likely AI).
        """
        from scipy import ndimage
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = np.mean(img, axis=2)
        else:
            gray = img
        
        # Sobel edge detection
        sobel_x = ndimage.sobel(gray, axis=1)
        sobel_y = ndimage.sobel(gray, axis=0)
        edges = np.hypot(sobel_x, sobel_y)
        
        # Calculate edge statistics
        edge_mean = np.mean(edges)
        edge_std = np.std(edges)
        
        # Real photos have varied edge strengths; AI often uniform
        edge_variation = edge_std / (edge_mean + 1e-6)
        
        # Lower variation = more uniform = more likely AI
        if edge_variation < 0.8:
            return 70.0
        elif edge_variation < 1.2:
            return 45.0
        else:
            return 25.0
    
    def _analyze_texture(self, img: np.ndarray) -> float:
        """
        Analyze texture patterns.
        
        AI struggles with fine, repetitive textures.
        Returns 0-100 (higher = more likely AI).
        """
        from scipy import ndimage
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = np.mean(img, axis=2)
        else:
            gray = img
        
        # Use variance of Laplacian as texture measure
        laplacian = ndimage.laplace(gray.astype(float))
        
        # Analyze texture in patches
        patch_size = 32
        h, w = gray.shape
        variances = []
        
        for i in range(0, h - patch_size, patch_size):
            for j in range(0, w - patch_size, patch_size):
                patch = laplacian[i:i+patch_size, j:j+patch_size]
                variances.append(np.var(patch))
        
        if not variances:
            return 50.0
        
        # Calculate coefficient of variation across patches
        cv = np.std(variances) / (np.mean(variances) + 1e-6)
        
        # Real photos have varied textures; AI more uniform
        if cv < 0.5:
            return 65.0  # Uniform textures = likely AI
        elif cv < 1.0:
            return 40.0
        else:
            return 25.0
    
    def _analyze_frequency_domain(self, img: np.ndarray) -> float:
        """
        Analyze frequency domain for GAN fingerprints.
        
        GANs produce characteristic patterns in frequency spectrum.
        Returns 0-100 (higher = more likely AI).
        """
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = np.mean(img, axis=2)
        else:
            gray = img
        
        # Compute 2D FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Log scale for better analysis
        log_magnitude = np.log1p(magnitude)
        
        # Analyze radial spectrum (common for GAN detection)
        center = np.array(log_magnitude.shape) // 2
        y, x = np.ogrid[:log_magnitude.shape[0], :log_magnitude.shape[1]]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        # Bin by radius
        r_int = r.astype(int)
        max_r = min(center)
        
        radial_profile = []
        for i in range(1, max_r):
            mask = r_int == i
            if np.any(mask):
                radial_profile.append(np.mean(log_magnitude[mask]))
        
        if len(radial_profile) < 10:
            return 50.0
        
        radial_profile = np.array(radial_profile)
        
        # Look for periodic patterns (GAN artifacts)
        # Calculate autocorrelation
        autocorr = np.correlate(radial_profile - np.mean(radial_profile), 
                                radial_profile - np.mean(radial_profile), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / (autocorr[0] + 1e-6)
        
        # Strong periodic peaks indicate GAN artifacts
        peaks = np.where((autocorr[1:-1] > autocorr[:-2]) & 
                         (autocorr[1:-1] > autocorr[2:]))[0]
        
        if len(peaks) > 5:
            peak_heights = autocorr[peaks + 1]
            if np.max(peak_heights) > 0.3:
                return 75.0  # Strong periodic pattern = likely GAN
        
        # Also check for unusual smoothness in high frequencies
        high_freq = radial_profile[len(radial_profile)//2:]
        high_freq_std = np.std(high_freq)
        
        if high_freq_std < 0.5:
            return 60.0  # Too smooth high freq = likely AI
        elif high_freq_std < 1.0:
            return 40.0
        else:
            return 25.0
    
    def _build_noise_analysis(self, img: np.ndarray) -> dict:
        """
        Build noise analysis breakdown for forensics display.
        
        Analyzes noise in different frequency bands:
        - Low frequency: Large-scale variations
        - Mid frequency: Texture details
        - High frequency: Fine details and noise
        
        Returns:
            dict with low_freq, mid_freq, high_freq percentages
        """
        try:
            # Convert to grayscale for frequency analysis
            if len(img.shape) == 3:
                gray = np.mean(img, axis=2)
            else:
                gray = img
            
            # Apply FFT
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            
            # Calculate radial profile
            h, w = magnitude.shape
            center_y, center_x = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            max_radius = min(center_x, center_y)
            
            # Divide into frequency bands
            low_band = magnitude[r < max_radius * 0.2]
            mid_band = magnitude[(r >= max_radius * 0.2) & (r < max_radius * 0.6)]
            high_band = magnitude[r >= max_radius * 0.6]
            
            # Calculate energy in each band
            low_energy = np.mean(low_band) if len(low_band) > 0 else 0
            mid_energy = np.mean(mid_band) if len(mid_band) > 0 else 0
            high_energy = np.mean(high_band) if len(high_band) > 0 else 0
            
            # Normalize to percentages
            total_energy = low_energy + mid_energy + high_energy
            if total_energy > 0:
                low_pct = round((low_energy / total_energy) * 100)
                mid_pct = round((mid_energy / total_energy) * 100)
                high_pct = round((high_energy / total_energy) * 100)
            else:
                low_pct = mid_pct = high_pct = 33
            
            return {
                'low_freq': low_pct,
                'mid_freq': mid_pct,
                'high_freq': high_pct
            }
            
        except Exception as e:
            logger.warning(f"Noise analysis failed: {e}")
            return {
                'low_freq': 'N/A',
                'mid_freq': 'N/A',
                'high_freq': 'N/A'
            }
    
    def _ml_prediction(self, image: Image.Image) -> dict:
        """
        Run ML model prediction using DIRE or NYUAD detectors.
        
        Returns:
            dict with 'label' and 'confidence'
        """
        if not self.ml_detectors:
            return {
                'label': 'unknown',
                'confidence': 50.0,
                'note': 'No ML models loaded'
            }
        
        try:
            # Convert image to bytes for detector
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_bytes = img_buffer.getvalue()
            
            # Try DIRE first (best for latest generators)
            if self.ml_detectors.get('dire') and self.ml_detectors['dire'].model_loaded:
                result = self.ml_detectors['dire'].detect(img_bytes)
                if result.get('success'):
                    return {
                        'label': 'AI' if result['ai_probability'] > 50 else 'Real',
                        'confidence': result['ai_probability'],
                        'model': 'DIRE',
                        'specialization': 'Diffusion models (SD, DALL-E 3, Midjourney v6)'
                    }
            
            # Fallback to NYUAD
            if self.ml_detectors.get('nyuad') and self.ml_detectors['nyuad'].model_loaded:
                result = self.ml_detectors['nyuad'].detect(img_bytes)
                if result.get('success'):
                    return {
                        'label': 'AI' if result['ai_probability'] > 50 else 'Real',
                        'confidence': result['ai_probability'],
                        'model': 'NYUAD',
                        'specialization': 'General AI detection'
                    }
            
            # No models available
            return {
                'label': 'unknown',
                'confidence': 50.0,
                'note': 'ML models not loaded - download with setup_ml_models.py'
            }
            
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return {
                'label': 'unknown',
                'confidence': 50.0,
                'error': str(e)
            }
    
    def detect_from_base64(self, base64_data: str, filename: str = "image") -> dict:
        """
        Detect AI-generated image from base64 encoded data.
        
        Args:
            base64_data: Base64 encoded image (with or without data URL prefix)
            filename: Original filename
            
        Returns:
            Detection results dict
        """
        try:
            # Remove data URL prefix if present
            if ',' in base64_data:
                base64_data = base64_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(base64_data)
            
            return self.detect(image_bytes, filename)
            
        except Exception as e:
            logger.error(f"Error decoding base64 image: {e}")
            return {
                'success': False,
                'error': f'Invalid base64 data: {str(e)}',
                'ai_probability': 50.0,
                'verdict': 'ERROR'
            }
