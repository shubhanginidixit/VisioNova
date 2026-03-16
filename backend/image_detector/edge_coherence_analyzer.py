"""
VisioNova Edge Coherence Analyzer
Detects AI-generated images by analyzing edge quality and consistency.

AI-generated images often have:
- Unnatural edge transitions (too smooth or too sharp)
- Inconsistent edge quality across the image
- Artifacts at object boundaries
- Halos around edges from diffusion sampling

References:
- "Edge-based Detection of AI-Generated Images" (arXiv 2023)
"""

import io
import logging
from typing import Dict, Any
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class EdgeCoherenceAnalyzer:
    """
    Edge Coherence Analyzer for AI-generated image detection.
    
    Analyzes:
    - Edge gradient profiles
    - Edge sharpness consistency
    - Boundary artifact detection
    - Halo effects around edges
    """
    
    def __init__(self):
        """Initialize edge coherence analyzer."""
        logger.info("Edge Coherence Analyzer initialized")
    
    def detect(self, image_data: bytes) -> Dict[str, Any]:
        """
        Analyze image edges for AI-generation artifacts.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Detection result
        """
        try:
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return self.analyze(image)
            
        except Exception as e:
            logger.error(f"Edge analysis error: {e}")
            return {
                'success': False,
                'error': str(e),
                'ai_probability': 50.0
            }
    
    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze edge coherence."""
        img_array = np.array(image).astype(np.float32)
        gray = np.mean(img_array, axis=2)
        
        # 1. Compute edge map
        edges, gradient_x, gradient_y = self._compute_edges(gray)
        
        # 2. Analyze edge sharpness distribution
        sharpness_result = self._analyze_sharpness(edges, gradient_x, gradient_y)
        
        # 3. Analyze edge coherence across image
        coherence_result = self._analyze_coherence(edges)
        
        # 4. Detect halo artifacts
        halo_result = self._detect_halos(gray, edges)
        
        # 5. Analyze edge profiles
        profile_result = self._analyze_edge_profiles(gradient_x, gradient_y)
        
        # Combine scores
        weights = {
            'sharpness': 0.25,
            'coherence': 0.30,
            'halo': 0.25,
            'profile': 0.20
        }
        
        anomaly_score = (
            sharpness_result['anomaly'] * weights['sharpness'] +
            coherence_result['anomaly'] * weights['coherence'] +
            halo_result['anomaly'] * weights['halo'] +
            profile_result['anomaly'] * weights['profile']
        )
        
        ai_probability = anomaly_score * 100
        
        return {
            'success': True,
            'ai_probability': round(ai_probability, 2),
            'analysis': {
                'sharpness': sharpness_result,
                'coherence': coherence_result,
                'halo_detection': halo_result,
                'edge_profiles': profile_result
            },
            'method': 'Edge Coherence Analysis',
            'verdict': self._get_verdict(ai_probability)
        }
    
    def _compute_edges(self, gray: np.ndarray):
        """Compute edge map using Sobel-like filters."""
        # Sobel kernels
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        # Apply convolution
        gradient_x = self._convolve2d(gray, sobel_x)
        gradient_y = self._convolve2d(gray, sobel_y)
        
        # Edge magnitude
        edges = np.sqrt(gradient_x**2 + gradient_y**2)
        
        return edges, gradient_x, gradient_y
    
    def _convolve2d(self, img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Fast 2D convolution using scipy or numpy fallback."""
        try:
            from scipy.ndimage import convolve
            return convolve(img, kernel, mode='reflect')
        except ImportError:
            try:
                import cv2
                return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT)
            except ImportError:
                # Final fallback: vectorized via stride tricks
                k_h, k_w = kernel.shape
                pad_h, pad_w = k_h // 2, k_w // 2
                padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
                # Use stride tricks for sliding window
                shape = img.shape + kernel.shape
                strides = padded.strides + padded.strides
                windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
                return np.einsum('ijkl,kl->ij', windows, kernel)
    
    def _analyze_sharpness(self, edges: np.ndarray, 
                           grad_x: np.ndarray, grad_y: np.ndarray) -> Dict[str, Any]:
        """
        Analyze edge sharpness distribution.
        
        AI images often have unnaturally uniform or bimodal sharpness.
        """
        # Get significant edges
        threshold = np.percentile(edges, 80)
        significant_edges = edges[edges > threshold]
        
        if len(significant_edges) < 100:
            return {'anomaly': 0.5, 'message': 'Too few edges for analysis'}
        
        # Analyze sharpness distribution
        mean_sharpness = np.mean(significant_edges)
        std_sharpness = np.std(significant_edges)
        
        # Coefficient of variation
        cv = std_sharpness / (mean_sharpness + 1e-10)
        
        # Natural images: CV typically 0.3-0.8
        # AI images: often < 0.3 (too uniform) or > 1.0 (too varied)
        
        if cv < 0.25:
            anomaly = 0.7
            message = 'Unnaturally uniform edge sharpness'
        elif cv > 1.0:
            anomaly = 0.6
            message = 'Highly variable edge sharpness'
        else:
            anomaly = 0.3
            message = 'Natural edge sharpness distribution'
        
        return {
            'anomaly': anomaly,
            'message': message,
            'sharpness_cv': round(cv, 4),
            'mean_sharpness': round(float(mean_sharpness), 2)
        }
    
    def _analyze_coherence(self, edges: np.ndarray) -> Dict[str, Any]:
        """
        Analyze edge coherence across image regions.
        
        Natural images have consistent edge quality.
        AI images may have varying quality across regions.
        """
        h, w = edges.shape
        block_size = min(h, w) // 4
        
        if block_size < 16:
            return {'anomaly': 0.5, 'message': 'Image too small for coherence analysis'}
        
        # Compute edge density in blocks
        block_densities = []
        block_intensities = []
        
        for i in range(0, h - block_size, block_size // 2):
            for j in range(0, w - block_size, block_size // 2):
                block = edges[i:i+block_size, j:j+block_size]
                
                threshold = np.percentile(edges, 70)
                density = np.mean(block > threshold)
                intensity = np.mean(block)
                
                block_densities.append(density)
                block_intensities.append(intensity)
        
        if len(block_densities) < 4:
            return {'anomaly': 0.5, 'message': 'Not enough blocks'}
        
        # Check consistency
        density_cv = np.std(block_densities) / (np.mean(block_densities) + 1e-10)
        intensity_cv = np.std(block_intensities) / (np.mean(block_intensities) + 1e-10)
        
        # Very uniform = possibly AI (generated coherently)
        # Very inconsistent = possibly AI (quality varies)
        
        if density_cv < 0.15:
            anomaly = 0.6
            message = 'Unusually uniform edge distribution'
        elif density_cv > 1.5:
            anomaly = 0.6
            message = 'Inconsistent edge distribution'
        else:
            anomaly = 0.3
            message = 'Natural edge distribution'
        
        return {
            'anomaly': anomaly,
            'message': message,
            'density_cv': round(density_cv, 4)
        }
    
    def _detect_halos(self, gray: np.ndarray, edges: np.ndarray) -> Dict[str, Any]:
        """
        Detect halo artifacts around edges.
        
        Diffusion models often create subtle halos (bright/dark rings)
        around object boundaries due to the denoising process.
        """
        h, w = gray.shape
        
        # Find strong edge locations
        edge_threshold = np.percentile(edges, 90)
        strong_edges = edges > edge_threshold
        
        if np.sum(strong_edges) < 100:
            return {'anomaly': 0.5, 'message': 'Too few strong edges'}
        
        # Analyze intensity profile around edges
        halo_scores = []
        
        # Sample random edge points
        edge_y, edge_x = np.where(strong_edges)
        
        if len(edge_y) > 500:
            indices = np.random.choice(len(edge_y), 500, replace=False)
            edge_y, edge_x = edge_y[indices], edge_x[indices]
        
        for ey, ex in zip(edge_y, edge_x):
            # Check 5-pixel radius around edge
            y_min, y_max = max(0, ey-5), min(h, ey+6)
            x_min, x_max = max(0, ex-5), min(w, ex+6)
            
            if y_max - y_min < 5 or x_max - x_min < 5:
                continue
            
            region = gray[y_min:y_max, x_min:x_max]
            center = gray[ey, ex]
            
            # Look for halo pattern (overshoot/undershoot)
            ring_values = []
            for r in [2, 3, 4]:
                for dy in range(-r, r+1):
                    for dx in range(-r, r+1):
                        if abs(dy) == r or abs(dx) == r:
                            if 0 <= ey+dy < h and 0 <= ex+dx < w:
                                ring_values.append(gray[ey+dy, ex+dx])
            
            if ring_values:
                ring_mean = np.mean(ring_values)
                # Halo = ring brighter or darker than expected gradient
                deviation = abs(ring_mean - center)
                halo_scores.append(deviation)
        
        if not halo_scores:
            return {'anomaly': 0.5, 'message': 'Could not analyze halos'}
        
        avg_halo = np.mean(halo_scores)
        
        # High average deviation around edges = possible halos
        if avg_halo > 30:
            anomaly = 0.3  # Some contrast is normal
        elif avg_halo < 5:
            anomaly = 0.5  # Very smooth edges
        else:
            # Mid-range - check consistency
            halo_std = np.std(halo_scores)
            if halo_std < 5:
                anomaly = 0.6  # Too uniform = AI pattern
                message = 'Uniform edge transitions (possible AI)'
            else:
                anomaly = 0.3
                message = 'Natural edge transitions'
        
        if avg_halo > 30:
            message = 'Normal edge contrast'
        elif avg_halo < 5:
            message = 'Very smooth edge transitions'
        
        return {
            'anomaly': anomaly,
            'message': message,
            'avg_halo_intensity': round(float(avg_halo), 2)
        }
    
    def _analyze_edge_profiles(self, grad_x: np.ndarray, 
                               grad_y: np.ndarray) -> Dict[str, Any]:
        """
        Analyze edge gradient profiles.
        
        Real images: varied gradient shapes
        AI images: often too smooth or stepped gradients
        """
        # Compute gradient magnitude
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Get strong gradient locations
        threshold = np.percentile(grad_mag, 85)
        strong_grads = grad_mag[grad_mag > threshold]
        
        if len(strong_grads) < 100:
            return {'anomaly': 0.5, 'message': 'Too few gradients'}
        
        # Analyze gradient value distribution
        # Real images: heavy-tailed distribution
        # AI images: often more Gaussian
        
        mean_grad = np.mean(strong_grads)
        std_grad = np.std(strong_grads)
        
        # Compute kurtosis (peakedness)
        kurtosis = np.mean(((strong_grads - mean_grad) / (std_grad + 1e-10))**4)
        
        # Real gradients: kurtosis often > 5
        # AI gradients: often closer to 3 (Gaussian)
        
        if kurtosis < 4:
            anomaly = 0.7
            message = 'Gaussian-like gradient distribution (possible AI)'
        elif kurtosis > 20:
            anomaly = 0.3
            message = 'Heavy-tailed gradients (natural)'
        else:
            anomaly = 0.4
            message = 'Mixed gradient characteristics'
        
        return {
            'anomaly': anomaly,
            'message': message,
            'gradient_kurtosis': round(kurtosis, 2)
        }
    
    def _get_verdict(self, ai_probability: float) -> str:
        """Get verdict based on AI probability."""
        if ai_probability >= 70:
            return "LIKELY_AI"
        elif ai_probability >= 50:
            return "POSSIBLY_AI"
        elif ai_probability >= 30:
            return "UNCERTAIN"
        else:
            return "LIKELY_REAL"


def create_edge_analyzer() -> EdgeCoherenceAnalyzer:
    """Factory function for edge coherence analyzer."""
    return EdgeCoherenceAnalyzer()
