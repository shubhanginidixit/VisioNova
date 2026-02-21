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
        self.c2pa_detector = None
        self.semantic_detector = None
        
        # Try to load ML models (DIRE + Ateeqq + Flux)
        try:
            from .ml_detector import create_ml_detectors
            self.ml_detectors = create_ml_detectors(
                device="cuda" if use_gpu else "cpu",
                load_all=False
            )
            
            # Check for active models
            loaded_models = []
            if self.ml_detectors.get('flux') and self.ml_detectors['flux'].model_loaded:
                loaded_models.append("Flux")
            if self.ml_detectors.get('dire') and self.ml_detectors['dire'].model_loaded:
                loaded_models.append("DIRE")
            if self.ml_detectors.get('ateeqq') and self.ml_detectors['ateeqq'].model_loaded:
                loaded_models.append("Ateeqq")
            if self.ml_detectors.get('smogy') and self.ml_detectors['smogy'].model_loaded:
                loaded_models.append("SMOGY")
            if self.ml_detectors.get('siglip') and self.ml_detectors['siglip'].model_loaded:
                loaded_models.append("SigLIP")
            
            if loaded_models:
                self.model_loaded = True
                logger.info(f"✓ ML models loaded: {', '.join(loaded_models)}")
            else:
                logger.warning("No ML models loaded. Using statistical analysis only.")
                logger.info("Run 'python backend/setup_ml_models.py' to download models")

            # Initialize Content Credentials detector
            try:
                from .content_credentials import ContentCredentialsDetector
                self.c2pa_detector = ContentCredentialsDetector()
            except ImportError:
                logger.warning("Content Credentials module not found")
                self.c2pa_detector = None
            
            # Initialize Semantic Plausibility detector (Groq LLaVA)
            try:
                from .semantic_detector import SemanticPlausibilityDetector
                self.semantic_detector = SemanticPlausibilityDetector()
                if self.semantic_detector.available:
                    logger.info("✓ Semantic plausibility detector loaded (Groq LLaVA)")
                else:
                    logger.info("Semantic detector: GROQ_API_KEY not set (optional)")
            except ImportError:
                logger.warning("Semantic detector module not found")
                self.semantic_detector = None
                
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
                original_mode = image.mode
                image = image.convert('RGB')
            else:
                original_mode = 'RGB'
            
            # Get image info
            width, height = image.size
            
            # Extract color space and bit depth info
            color_space = original_mode  # e.g., 'RGB', 'RGBA', 'L', 'CMYK', 'P'
            
            # Map PIL modes to human-readable color spaces
            color_space_map = {
                'RGB': 'RGB',
                'RGBA': 'RGBA',
                'L': 'Grayscale',
                'P': 'Palette',
                'CMYK': 'CMYK',
                '1': '1-bit',
                'LAB': 'LAB'
            }
            color_space_name = color_space_map.get(original_mode, original_mode)
            
            # Calculate bits per pixel
            mode_to_bits = {
                '1': 1,
                'L': 8,
                'P': 8,
                'RGB': 24,
                'RGBA': 32,
                'CMYK': 32,
                'YCbCr': 24,
                'LAB': 24,
                'HSV': 24,
                'I': 32,
                'F': 32
            }
            bit_depth = mode_to_bits.get(original_mode, 24)
            
            # Run detection methods
            results = {
                'success': True,
                'analysis_mode': 'ML Ensemble' if self.model_loaded else 'Statistical Analysis (Fallback)',
                'filename': filename,
                'dimensions': {'width': width, 'height': height},
                'file_size': len(image_data),  # Changed from file_size_bytes for frontend compatibility
                'color_space': color_space_name,
                'bit_depth': bit_depth,
            }
            
            # Statistical analysis (always available)
            stats = self._statistical_analysis(image)
            results.update(stats)
            
            # Content Credentials (C2PA) analysis
            if self.c2pa_detector:
                c2pa_result = self.c2pa_detector.analyze(image_data, filename)
                results['c2pa'] = c2pa_result
                
                # If C2PA says it's AI, override or significantly boost probability
                if c2pa_result.get('is_ai_generated'):
                    results['ai_probability'] = max(results.get('ai_probability', 0), 95.0)
                    results['provenance_source'] = c2pa_result.get('ai_generator', 'Unknown AI Tool')

            # Deep learning prediction (if model loaded)
            if self.model_loaded:
                ml_result = self._ml_prediction(image)
                results['ml_prediction'] = ml_result
                # Combine ML and statistical scores
                results['ai_probability'] = (
                    results['ai_probability'] * 0.3 + ml_result['confidence'] * 0.7
                )
                
                # Generate ML heatmap for visualization
                try:
                    ml_heatmap = self._generate_ml_heatmap(image)
                    if ml_heatmap:
                        results['ml_heatmap'] = ml_heatmap
                except Exception as e:
                    logger.warning(f"ML heatmap generation failed: {e}")
            
            # Semantic Plausibility Analysis (Groq LLaVA - common sense detection)
            if hasattr(self, 'semantic_detector') and self.semantic_detector and self.semantic_detector.available:
                try:
                    semantic_result = self.semantic_detector.analyze(image_data)
                    results['semantic_analysis'] = semantic_result
                    
                    if semantic_result.get('success'):
                        # If semantic analysis finds issues, boost AI probability
                        plausibility = semantic_result.get('plausibility_score', 100)
                        if plausibility < 70:  # Low plausibility = likely AI
                            # Calculate boost: lower plausibility = higher AI probability boost
                            ai_boost = (70 - plausibility) * 0.5
                            results['ai_probability'] = min(100, results.get('ai_probability', 50) + ai_boost)
                            logger.info(f"Semantic analysis: plausibility {plausibility}%, AI boost +{ai_boost:.1f}%")
                except Exception as e:
                    logger.warning(f"Semantic analysis failed: {e}")
            
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
        Run ML model prediction using weighted ensemble voting.
        
        Uses all available models with weighted voting for best accuracy:
        - Ateeqq: 30% (99% accuracy, proven general detector)
        - SMOGY: 25% (specialized for 2024 generators)
        - SigLIP: 20% (human vs AI classification)
        - DIRE: 15% (diffusion model detection)
        - Flux: 10% (specialized Flux detection)
        
        Returns:
            dict with 'label', 'confidence', and ensemble details
        """
        if not self.ml_detectors:
            return {
                'label': 'unknown',
                'confidence': 50.0,
                'note': 'No ML models loaded'
            }
        
        # Model weights for ensemble voting
        WEIGHTS = {
            'ateeqq': 0.30,
            'smogy': 0.25,
            'siglip': 0.20,
            'dire': 0.15,
            'flux': 0.10
        }
        
        results = {}
        weighted_sum = 0.0
        total_weight = 0.0
        
        try:
            # Convert image to bytes for some detectors
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_bytes = img_buffer.getvalue()
            
            # Run all available detectors
            # Ateeqq detector
            if self.ml_detectors.get('ateeqq') and self.ml_detectors['ateeqq'].model_loaded:
                try:
                    result = self.ml_detectors['ateeqq'].predict(image)
                    if result.get('success'):
                        ai_prob = result.get('ai_probability', 50.0)
                        results['ateeqq'] = {'ai_probability': ai_prob, 'model': 'Ateeqq-ViT'}
                        weighted_sum += ai_prob * WEIGHTS['ateeqq']
                        total_weight += WEIGHTS['ateeqq']
                except Exception as e:
                    logger.debug(f"Ateeqq failed: {e}")
            
            # SMOGY detector (2024 models)
            if self.ml_detectors.get('smogy') and self.ml_detectors['smogy'].model_loaded:
                try:
                    result = self.ml_detectors['smogy'].predict(image)
                    if result.get('success'):
                        ai_prob = result.get('ai_probability', 50.0)
                        results['smogy'] = {'ai_probability': ai_prob, 'model': 'SMOGY-2024'}
                        weighted_sum += ai_prob * WEIGHTS['smogy']
                        total_weight += WEIGHTS['smogy']
                except Exception as e:
                    logger.debug(f"SMOGY failed: {e}")
            
            # SigLIP detector (human vs AI)
            if self.ml_detectors.get('siglip') and self.ml_detectors['siglip'].model_loaded:
                try:
                    result = self.ml_detectors['siglip'].predict(image)
                    if result.get('success'):
                        ai_prob = result.get('ai_probability', 50.0)
                        results['siglip'] = {'ai_probability': ai_prob, 'model': 'SigLIP'}
                        weighted_sum += ai_prob * WEIGHTS['siglip']
                        total_weight += WEIGHTS['siglip']
                except Exception as e:
                    logger.debug(f"SigLIP failed: {e}")
            
            # DIRE detector (diffusion models)
            if self.ml_detectors.get('dire') and self.ml_detectors['dire'].model_loaded:
                try:
                    result = self.ml_detectors['dire'].detect(img_bytes)
                    if result.get('success'):
                        ai_prob = result.get('ai_probability', 50.0)
                        results['dire'] = {'ai_probability': ai_prob, 'model': 'DIRE'}
                        weighted_sum += ai_prob * WEIGHTS['dire']
                        total_weight += WEIGHTS['dire']
                except Exception as e:
                    logger.debug(f"DIRE failed: {e}")
            
            # Flux detector (Flux.1 specific)
            if self.ml_detectors.get('flux') and self.ml_detectors['flux'].model_loaded:
                try:
                    result = self.ml_detectors['flux'].predict(image)
                    if result.get('success') and result.get('is_flux'):
                        ai_prob = result.get('confidence', 50.0)
                        results['flux'] = {'ai_probability': ai_prob, 'model': 'Flux-Detector'}
                        weighted_sum += ai_prob * WEIGHTS['flux']
                        total_weight += WEIGHTS['flux']
                except Exception as e:
                    logger.debug(f"Flux failed: {e}")
            
            # Calculate ensemble result
            if total_weight > 0:
                final_ai_probability = weighted_sum / total_weight
            else:
                # Fallback if no models ran
                return {
                    'label': 'unknown',
                    'confidence': 50.0,
                    'note': 'No ML models produced results'
                }
            
            # Determine label based on probability
            if final_ai_probability >= 50:
                label = 'AI'
            else:
                label = 'Real'
            
            # Count votes
            ai_votes = sum(1 for r in results.values() if r.get('ai_probability', 50) > 50)
            total_models = len(results)
            
            return {
                'label': label,
                'confidence': round(final_ai_probability, 2),
                'model': 'Ensemble',
                'models_used': list(results.keys()),
                'model_count': total_models,
                'ensemble_votes': f"{ai_votes}/{total_models} voted AI",
                'individual_results': results,
                'specialization': 'Weighted ensemble (Ateeqq+SMOGY+SigLIP+DIRE+Flux)'
            }
            
        except Exception as e:
            logger.error(f"ML ensemble prediction failed: {e}")
            return {
                'label': 'unknown',
                'confidence': 50.0,
                'error': str(e)
            }
    
    def _generate_ml_heatmap(self, image: Image.Image, patch_size: int = 64, stride: int = 32) -> Optional[str]:
        """
        Generate ML-based probability heatmap showing AI likelihood in different regions.
        
        Divides image into overlapping patches and runs ML inference on each patch.
        Creates a visualization showing which regions are more/less likely AI-generated.
        
        Args:
            image: PIL Image to analyze
            patch_size: Size of each patch (default 64x64)
            stride: Stride between patches (default 32 for 50% overlap)
            
        Returns:
            Base64-encoded PNG heatmap overlay, or None if generation fails
        """
        if not self.ml_detectors:
            return None
        
        try:
            import cv2
            
            # Resize image if too large (for performance)
            width, height = image.size
            max_dim = 512
            if max(width, height) > max_dim:
                scale = max_dim / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image_resized = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
            else:
                image_resized = image
                new_width, new_height = width, height
            
            # Convert to numpy array
            img_array = np.array(image_resized)
            
            # Initialize heatmap
            heatmap = np.zeros((new_height, new_width), dtype=np.float32)
            counts = np.zeros((new_height, new_width), dtype=np.float32)
            
            # Get active ML detector
            active_detector = None
            if self.ml_detectors.get('dire') and self.ml_detectors['dire'].model_loaded:
                active_detector = self.ml_detectors['dire']
            elif self.ml_detectors.get('ateeqq') and self.ml_detectors['ateeqq'].model_loaded:
                active_detector = self.ml_detectors['ateeqq']
            
            if not active_detector:
                return None
            
            # Extract and analyze patches
            for y in range(0, new_height - patch_size + 1, stride):
                for x in range(0, new_width - patch_size + 1, stride):
                    # Extract patch
                    patch = img_array[y:y+patch_size, x:x+patch_size]
                    
                    # Convert patch to PIL Image
                    patch_img = Image.fromarray(patch)
                    
                    # Convert to bytes for detector
                    patch_buffer = io.BytesIO()
                    patch_img.save(patch_buffer, format='PNG')
                    patch_bytes = patch_buffer.getvalue()
                    
                    # Run detection on patch
                    try:
                        result = active_detector.detect(patch_bytes)
                        if result.get('success'):
                            prob = result.get('ai_probability', 50) / 100.0
                            # Add to heatmap
                            heatmap[y:y+patch_size, x:x+patch_size] += prob
                            counts[y:y+patch_size, x:x+patch_size] += 1
                    except:
                        # If patch detection fails, use neutral value
                        heatmap[y:y+patch_size, x:x+patch_size] += 0.5
                        counts[y:y+patch_size, x:x+patch_size] += 1
            
            # Average overlapping predictions
            counts[counts == 0] = 1  # Avoid division by zero
            heatmap = heatmap / counts
            
            # Normalize to 0-255
            heatmap_normalized = (heatmap * 255).astype(np.uint8)
            
            # Apply colormap (red = AI, blue = real)
            heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Resize back to original dimensions if needed
            if (new_width, new_height) != (width, height):
                heatmap_colored = cv2.resize(heatmap_colored, (width, height), interpolation=cv2.INTER_LINEAR)
            
            # Convert to PIL and encode as base64
            heatmap_img = Image.fromarray(heatmap_colored)
            buffer = io.BytesIO()
            heatmap_img.save(buffer, format='PNG')
            heatmap_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return f"data:image/png;base64,{heatmap_base64}"
            
        except Exception as e:
            logger.error(f"ML heatmap generation failed: {e}")
            return None
    
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
