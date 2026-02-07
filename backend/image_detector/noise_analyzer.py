"""
VisioNova Noise Pattern Analyzer
Analyzes noise consistency and frequency patterns to detect AI-generated images.

Natural camera images have characteristic noise patterns from sensor physics.
AI-generated images often lack realistic noise or show uniform artificial patterns.
"""

import numpy as np
from PIL import Image
import io
import logging
from typing import Dict, Any, Tuple
import cv2

logger = logging.getLogger(__name__)


class NoiseAnalyzer:
    """
    Analyzes noise patterns in images to detect AI generation.
    
    AI-generated images typically show:
    - Unnaturally uniform noise across the image
    - Missing high-frequency sensor noise
    - Perfect gradients without noise texture
    - Artificial noise patterns in smooth areas
    """
    
    def __init__(self):
        """Initialize the noise analyzer."""
        pass
    
    def analyze(self, image_data: bytes) -> Dict[str, Any]:
        """
        Analyze noise patterns in an image.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            dict with noise analysis results including:
                - noise_consistency: Overall consistency score (0-100)
                - low_freq, mid_freq, high_freq: Frequency band analysis
                - noise_map: Base64-encoded visualization
                - analysis_scores: Detailed metrics
        """
        result = {
            'success': True,
            'noise_consistency': 0,
            'low_freq': 0,
            'mid_freq': 0,
            'high_freq': 0,
            'noise_map': None,
            'pattern_analysis': {}
        }
        
        try:
            # Load image
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            img_array = np.array(image)
            
            # Analyze noise patterns
            noise_metrics = self._analyze_noise_patterns(img_array)
            
            # Calculate frequency distribution
            freq_analysis = self._analyze_frequency_bands(img_array)
            
            # Calculate overall consistency score
            consistency = self._calculate_consistency_score(noise_metrics, freq_analysis)
            
            # Generate noise visualization map
            noise_map_b64 = self._generate_noise_map(img_array, noise_metrics)
            
            # Assemble results
            result['noise_consistency'] = consistency
            result['low_freq'] = freq_analysis['low_freq']
            result['mid_freq'] = freq_analysis['mid_freq']
            result['high_freq'] = freq_analysis['high_freq']
            result['noise_map'] = noise_map_b64
            result['pattern_analysis'] = {
                'noise_level': noise_metrics['noise_level'],
                'uniformity': noise_metrics['uniformity'],
                'texture_variance': noise_metrics['texture_variance'],
                'sensor_pattern': noise_metrics['sensor_pattern_detected'],
                'artificial_smoothing': noise_metrics['artificial_smoothing']
            }
            
            logger.info(f"Noise analysis: consistency={consistency}%, high_freq={freq_analysis['high_freq']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in noise analysis: {e}")
            return {
                'success': False,
                'noise_consistency': 0,
                'low_freq': 0,
                'mid_freq': 0,
                'high_freq': 0,
                'noise_map': None,
                'pattern_analysis': {},
                'error': str(e)
            }
    
    def _analyze_noise_patterns(self, img_array: np.ndarray) -> Dict[str, Any]:
        """
        Analyze noise characteristics in the image.
        
        Returns metrics about noise level, uniformity, and patterns.
        """
        # Convert to grayscale for noise analysis
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Calculate local noise using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_level = laplacian.var()
        
        # Analyze noise uniformity across image regions
        h, w = gray.shape
        patch_size = 32
        noise_patches = []
        
        for i in range(0, h - patch_size, patch_size):
            for j in range(0, w - patch_size, patch_size):
                patch = gray[i:i+patch_size, j:j+patch_size]
                patch_lap = cv2.Laplacian(patch, cv2.CV_64F)
                noise_patches.append(patch_lap.var())
        
        if len(noise_patches) > 0:
            uniformity = 1.0 - (np.std(noise_patches) / (np.mean(noise_patches) + 1e-6))
            uniformity = min(100, max(0, uniformity * 100))
        else:
            uniformity = 0
        
        # Calculate texture variance (natural images have higher variance)
        texture_variance = np.std(gray) / (np.mean(gray) + 1e-6)
        
        # Detect sensor noise pattern (present in real photos, absent in AI images)
        # Real cameras have row/column noise patterns
        row_variance = np.var(np.mean(gray, axis=1))
        col_variance = np.var(np.mean(gray, axis=0))
        sensor_pattern = (row_variance + col_variance) / 2
        
        # Detect artificial smoothing (common in AI images)
        # AI images often have unnaturally smooth gradients
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.mean(edges) / 255.0
        
        # Low edge density in complex areas suggests artificial smoothing
        gradient_smooth = cv2.GaussianBlur(gray, (5, 5), 0)
        smoothing_score = np.mean(np.abs(gray.astype(float) - gradient_smooth.astype(float)))
        
        return {
            'noise_level': float(noise_level),
            'uniformity': float(uniformity),
            'texture_variance': float(texture_variance),
            'sensor_pattern_detected': float(sensor_pattern) > 10,  # Threshold for sensor noise
            'artificial_smoothing': float(smoothing_score),
            'edge_density': float(edge_density)
        }
    
    def _analyze_frequency_bands(self, img_array: np.ndarray) -> Dict[str, int]:
        """
        Analyze frequency distribution in image.
        
        Natural images have characteristic frequency profiles.
        AI images often lack high-frequency noise from camera sensors.
        """
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Compute FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Split into frequency bands
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        
        # Low frequency: center region (DC and nearby)
        low_radius = min(h, w) // 8
        low_mask = np.zeros((h, w), dtype=bool)
        y, x = np.ogrid[:h, :w]
        low_mask = ((y - center_h)**2 + (x - center_w)**2) <= low_radius**2
        low_freq_energy = np.mean(magnitude[low_mask])
        
        # Mid frequency: annular region
        mid_radius = min(h, w) // 4
        mid_mask = (((y - center_h)**2 + (x - center_w)**2) > low_radius**2) & \
                   (((y - center_h)**2 + (x - center_w)**2) <= mid_radius**2)
        mid_freq_energy = np.mean(magnitude[mid_mask])
        
        # High frequency: outer regions
        high_mask = ((y - center_h)**2 + (x - center_w)**2) > mid_radius**2
        high_freq_energy = np.mean(magnitude[high_mask])
        
        # Normalize to 0-100 scale
        total_energy = low_freq_energy + mid_freq_energy + high_freq_energy + 1e-6
        
        return {
            'low_freq': int((low_freq_energy / total_energy) * 100),
            'mid_freq': int((mid_freq_energy / total_energy) * 100),
            'high_freq': int((high_freq_energy / total_energy) * 100)
        }
    
    def _calculate_consistency_score(self, noise_metrics: Dict, freq_analysis: Dict) -> int:
        """
        Calculate overall noise consistency score (0-100).
        
        Higher score = more consistent with natural camera noise
        Lower score = suspicious patterns suggesting AI generation
        """
        score = 50  # Start neutral
        
        # Factor 1: Noise uniformity (30% weight)
        # Too uniform suggests AI generation
        uniformity = noise_metrics['uniformity']
        if uniformity > 85:
            score -= 15  # Too uniform (AI characteristic)
        elif uniformity > 70:
            score += 10  # Good uniformity
        elif uniformity < 40:
            score += 15  # Natural variation
        
        # Factor 2: High-frequency content (25% weight)
        # Natural cameras have significant high-frequency noise
        high_freq = freq_analysis['high_freq']
        if high_freq > 15:
            score += 15  # Good high-frequency content (camera noise)
        elif high_freq < 5:
            score -= 15  # Missing high-frequency (AI characteristic)
        
        # Factor 3: Texture variance (20% weight)
        texture_var = noise_metrics['texture_variance']
        if 0.3 < texture_var < 0.7:
            score += 10  # Natural range
        elif texture_var < 0.1:
            score -= 10  # Too smooth (AI)
        
        # Factor 4: Sensor pattern detection (15% weight)
        if noise_metrics['sensor_pattern_detected']:
            score += 10  # Sensor noise present (real photo indicator)
        else:
            score -= 5  # No sensor pattern (AI or heavily processed)
        
        # Factor 5: Artificial smoothing (10% weight)
        if noise_metrics['artificial_smoothing'] < 5:
            score -= 10  # Unnaturally smooth
        elif noise_metrics['artificial_smoothing'] > 15:
            score += 5  # Natural texture
        
        # Clamp to 0-100
        score = max(0, min(100, score))
        
        return int(score)
    
    def _generate_noise_map(self, img_array: np.ndarray, noise_metrics: Dict) -> str:
        """
        Generate visualization of noise distribution across the image.
        
        Returns base64-encoded PNG image.
        """
        try:
            import base64
            
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Calculate local noise using high-pass filter
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise_residual = cv2.absdiff(gray, blurred)
            
            # Enhance noise visibility
            noise_enhanced = cv2.normalize(noise_residual, None, 0, 255, cv2.NORM_MINMAX)
            
            # Apply colormap (red = high noise, blue = low noise)
            noise_colored = cv2.applyColorMap(noise_enhanced.astype(np.uint8), cv2.COLORMAP_JET)
            
            # Resize if too large (for performance)
            h, w = noise_colored.shape[:2]
            if max(h, w) > 1024:
                scale = 1024 / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                noise_colored = cv2.resize(noise_colored, (new_w, new_h))
            
            # Encode as PNG
            success, buffer = cv2.imencode('.png', noise_colored)
            if success:
                noise_b64 = base64.b64encode(buffer).decode('utf-8')
                return f'data:image/png;base64,{noise_b64}'
            
            return None
            
        except Exception as e:
            logger.error(f"Noise map generation error: {e}")
            return None


# Convenience function
def analyze_noise(image_data: bytes) -> Dict[str, Any]:
    """Convenience function to analyze noise in an image."""
    analyzer = NoiseAnalyzer()
    return analyzer.analyze(image_data)
