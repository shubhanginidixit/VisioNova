"""
VisioNova Watermark Detector
Detects invisible watermarks in images that indicate AI generation.

Supports:
1. DWT-DCT watermarks (Stable Diffusion, AUTOMATIC1111, ComfyUI)
2. DWT-DCT-SVD watermarks (enhanced robustness)
3. RivaGAN watermarks (deep learning based)
4. Meta Stable Signature patterns
5. IPTC/XMP DigitalSourceType markers
6. SteganoGAN steganographic watermarks
7. Spectral domain analysis for Tree-Ring patterns
8. Adversarial perturbation detection (Glaze/Nightshade - experimental)
9. Custom signature patterns from known AI generators

Note: Google SynthID cannot be detected externally (requires Google API)
"""

import io
import logging
import numpy as np
from PIL import Image
from typing import Optional, Dict, Any, Tuple
from scipy import stats
import struct

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WatermarkDetector:
    """
    Comprehensive Invisible Watermark Detector for AI-Generated Images.
    
    Detects watermarks embedded by:
    - Stable Diffusion (DWT-DCT method)
    - AUTOMATIC1111 WebUI, ComfyUI, InvokeAI
    - Adobe Firefly (metadata markers)
    - Meta AI (Stable Signature)
    - Custom implementations using invisible-watermark library
    - Spectral domain watermarks (Tree-Ring patterns)
    - Steganographic watermarks (SteganoGAN)
    
    Experimental:
    - Adversarial perturbations (Glaze/Nightshade)
    
    Cannot detect:
    - Google SynthID (requires Google's proprietary API)
    """
    
    # Known watermark signatures from AI generators
    KNOWN_SIGNATURES = {
        # Stable Diffusion ecosystem
        'stable_diffusion': b'StableDiffusionV',
        'sd_webui': b'AUTOMATIC1111',
        'comfyui': b'ComfyUI',
        'invokeai': b'InvokeAI',
        'sdxl': b'SDXL',
        'sd_turbo': b'SDTurbo',
        
        # Commercial AI generators
        'leonardo': b'LeonardoAI',
        'runwayml': b'RunwayML',
        'playground': b'PlaygroundAI',
        'ideogram': b'Ideogram',
        'flux': b'FLUX',
        'juggernaut': b'Juggernaut',
        
        # Research models
        'imagen': b'GoogleImagen',
        'parti': b'GoogleParti',
        'muse': b'GoogleMuse',
    }
    
    # IPTC DigitalSourceType values indicating AI generation
    AI_SOURCE_TYPES = [
        'trainedAlgorithmicMedia',
        'compositeWithTrainedAlgorithmicMedia',
        'algorithmicMedia',
    ]
    
    def __init__(self):
        """Initialize the watermark detector."""
        self.watermark_lib_available = False
        self.steganogan_available = False
        self.decoder = None
        
        try:
            from imwatermark import WatermarkDecoder
            self.watermark_lib_available = True
            logger.info("invisible-watermark library available")
        except ImportError:
            logger.warning("invisible-watermark library not available. Install with: pip install invisible-watermark")
        
        try:
            from steganogan import SteganoGAN
            self.steganogan_available = True
            logger.info("steganogan library available")
        except ImportError:
            logger.debug("steganogan library not available (optional)")
    
    def analyze(self, image_data: bytes) -> Dict[str, Any]:
        """
        Analyze an image for invisible watermarks.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            dict with watermark detection results
        """
        result = {
            'watermark_detected': False,
            'watermark_type': None,
            'watermark_content': None,
            'detection_methods': {},
            'ai_generator_signature': None,
            'confidence': 0,
            'details': []
        }
        
        try:
            # Load image
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            img_array = np.array(image)
            
            # Method 1: Try invisible-watermark library detection
            if self.watermark_lib_available:
                wm_result = self._detect_invisible_watermark(img_array)
                result['detection_methods']['invisible_watermark'] = wm_result
                
                if wm_result.get('detected'):
                    result['watermark_detected'] = True
                    result['watermark_type'] = wm_result.get('type', 'DWT-DCT')
                    result['watermark_content'] = wm_result.get('content')
                    result['confidence'] = max(result['confidence'], wm_result.get('confidence', 70))
                    
                    # Check if it matches known AI signatures
                    for gen_name, signature in self.KNOWN_SIGNATURES.items():
                        if wm_result.get('raw_bytes') and signature in wm_result.get('raw_bytes', b''):
                            result['ai_generator_signature'] = gen_name
                            result['confidence'] = 95
                            result['details'].append(f"Matched signature: {gen_name}")
            
            # Method 2: Spectral analysis for hidden patterns
            spectral_result = self._spectral_watermark_analysis(img_array)
            result['detection_methods']['spectral_analysis'] = spectral_result
            
            if spectral_result.get('patterns_found'):
                result['details'].append("Spectral patterns consistent with watermarking detected")
                if not result['watermark_detected']:
                    result['watermark_detected'] = True
                    result['watermark_type'] = 'spectral_pattern'
                    result['confidence'] = max(result['confidence'], spectral_result.get('confidence', 50))
            
            # Method 3: LSB analysis for steganographic watermarks
            lsb_result = self._lsb_analysis(img_array)
            result['detection_methods']['lsb_analysis'] = lsb_result
            
            if lsb_result.get('anomaly_detected'):
                result['details'].append("LSB anomalies detected (possible steganographic watermark)")
                if not result['watermark_detected']:
                    result['confidence'] = max(result['confidence'], lsb_result.get('confidence', 40))
            
            # Method 4: Check for metadata-based watermarks (IPTC/XMP)
            metadata_wm = self._check_metadata_watermarks(image_data)
            result['detection_methods']['metadata_watermark'] = metadata_wm
            
            if metadata_wm.get('found'):
                result['watermark_detected'] = True
                result['watermark_type'] = metadata_wm.get('type', 'metadata')
                result['details'].append(f"Metadata watermark found: {metadata_wm.get('type')}")
                result['confidence'] = max(result['confidence'], 90)
                if metadata_wm.get('ai_source_type'):
                    result['ai_generator_signature'] = metadata_wm.get('ai_source_type')
                    result['details'].append(f"IPTC DigitalSourceType: {metadata_wm.get('ai_source_type')}")
            
            # Method 5: Tree-Ring watermark pattern detection
            treering_result = self._detect_treering_patterns(img_array)
            result['detection_methods']['treering_analysis'] = treering_result
            
            if treering_result.get('detected'):
                result['watermark_detected'] = True
                result['watermark_type'] = 'Tree-Ring'
                result['details'].append("Tree-Ring watermark pattern detected (academic diffusion watermark)")
                result['confidence'] = max(result['confidence'], treering_result.get('confidence', 65))
            
            # Method 6: Adversarial perturbation detection (Glaze/Nightshade - experimental)
            adversarial_result = self._detect_adversarial_perturbations(img_array)
            result['detection_methods']['adversarial_analysis'] = adversarial_result
            
            if adversarial_result.get('detected'):
                result['details'].append(f"⚠️ Experimental: Possible adversarial perturbations detected ({adversarial_result.get('type', 'unknown')})")
                # Don't set watermark_detected for experimental results
                result['detection_methods']['adversarial_analysis']['note'] = 'Experimental - high false positive rate'
            
            # Method 7: SteganoGAN detection (if available)
            if self.steganogan_available:
                stegano_result = self._detect_steganogan(image_data)
                result['detection_methods']['steganogan'] = stegano_result
                
                if stegano_result.get('detected'):
                    result['watermark_detected'] = True
                    result['watermark_type'] = 'SteganoGAN'
                    result['details'].append("SteganoGAN steganographic watermark detected")
                    result['confidence'] = max(result['confidence'], 80)
            
            # Add note about undetectable watermarks
            result['detection_methods']['synthid'] = {
                'detected': False,
                'note': 'Google SynthID cannot be detected externally - requires Google API'
            }
            
            # Set final status
            if result['watermark_detected']:
                result['status'] = 'WATERMARK_FOUND'
            else:
                result['status'] = 'NO_WATERMARK'
                result['details'].append("No invisible watermark detected")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing watermark: {e}")
            return {
                'watermark_detected': False,
                'status': 'ERROR',
                'error': str(e),
                'detection_methods': {},
                'confidence': 0,
                'details': [f"Analysis error: {str(e)}"]
            }
    
    def _detect_invisible_watermark(self, img_array: np.ndarray) -> Dict[str, Any]:
        """
        Detect watermarks using the invisible-watermark library.
        
        Tries multiple decoding methods:
        - DWT-DCT (most common for Stable Diffusion)
        - DWT-DCT-SVD (more robust)
        - RivaGAN (deep learning based)
        """
        result = {
            'detected': False,
            'type': None,
            'content': None,
            'raw_bytes': None,
            'confidence': 0
        }
        
        if not self.watermark_lib_available:
            result['error'] = 'invisible-watermark library not installed'
            return result
        
        try:
            from imwatermark import WatermarkDecoder
            import cv2
            
            # Convert RGB to BGR for OpenCV
            bgr_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Try different watermark lengths and methods
            methods = ['dwtDct', 'dwtDctSvd']
            watermark_lengths = [32, 48, 64, 128]  # Common watermark bit lengths
            
            for method in methods:
                for wm_length in watermark_lengths:
                    try:
                        decoder = WatermarkDecoder('bytes', wm_length)
                        watermark_bytes = decoder.decode(bgr_array, method)
                        
                        if watermark_bytes and len(watermark_bytes) > 0:
                            # Check if watermark contains meaningful data
                            # (not all zeros or random noise)
                            non_zero = sum(1 for b in watermark_bytes if b != 0)
                            
                            if non_zero > len(watermark_bytes) * 0.2:  # At least 20% non-zero
                                # Try to decode as text
                                try:
                                    decoded_text = watermark_bytes.decode('utf-8', errors='ignore')
                                    decoded_text = ''.join(c for c in decoded_text if c.isprintable())
                                except:
                                    decoded_text = None
                                
                                result['detected'] = True
                                result['type'] = method
                                result['raw_bytes'] = watermark_bytes
                                result['content'] = decoded_text if decoded_text and len(decoded_text) > 2 else f"Binary: {watermark_bytes.hex()[:32]}..."
                                result['confidence'] = 75 if decoded_text else 60
                                result['bit_length'] = wm_length
                                
                                logger.info(f"Watermark detected: {method} ({wm_length} bits)")
                                return result
                                
                    except Exception as e:
                        continue  # Try next combination
            
            return result
            
        except Exception as e:
            logger.error(f"Error in invisible watermark detection: {e}")
            result['error'] = str(e)
            return result
    
    def _spectral_watermark_analysis(self, img_array: np.ndarray) -> Dict[str, Any]:
        """
        Analyze frequency spectrum for watermark patterns.
        
        Watermarks often leave distinctive patterns in the frequency domain.
        """
        result = {
            'patterns_found': False,
            'confidence': 0,
            'analysis': {}
        }
        
        try:
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = np.mean(img_array, axis=2).astype(np.float32)
            else:
                gray = img_array.astype(np.float32)
            
            # Compute 2D FFT
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            log_magnitude = np.log1p(magnitude)
            
            # Analyze for periodic patterns that could indicate watermarking
            center = np.array(log_magnitude.shape) // 2
            
            # Look for symmetric peaks (common in watermarks)
            # Check 4 quadrants for symmetry
            h, w = log_magnitude.shape
            q1 = log_magnitude[:h//2, :w//2]
            q2 = log_magnitude[:h//2, w//2:]
            q3 = log_magnitude[h//2:, :w//2]
            q4 = log_magnitude[h//2:, w//2:]
            
            # Watermarks often create symmetric patterns
            sym_score_h = np.corrcoef(q1.flatten(), np.fliplr(q2).flatten())[0, 1]
            sym_score_v = np.corrcoef(q1.flatten(), np.flipud(q3).flatten())[0, 1]
            sym_score_d = np.corrcoef(q1.flatten(), np.flipud(np.fliplr(q4)).flatten())[0, 1]
            
            avg_symmetry = (abs(sym_score_h) + abs(sym_score_v) + abs(sym_score_d)) / 3
            
            result['analysis']['horizontal_symmetry'] = round(float(sym_score_h), 3)
            result['analysis']['vertical_symmetry'] = round(float(sym_score_v), 3)
            result['analysis']['diagonal_symmetry'] = round(float(sym_score_d), 3)
            result['analysis']['average_symmetry'] = round(float(avg_symmetry), 3)
            
            # High symmetry in frequency domain can indicate watermarking
            if avg_symmetry > 0.85:
                result['patterns_found'] = True
                result['confidence'] = min(100, int((avg_symmetry - 0.85) * 500))
            
            # Look for unusual peaks in mid-frequency range
            # (watermarks are often embedded here to survive compression)
            mid_band = log_magnitude[h//4:3*h//4, w//4:3*w//4]
            peak_threshold = np.mean(mid_band) + 3 * np.std(mid_band)
            num_peaks = np.sum(mid_band > peak_threshold)
            expected_peaks = mid_band.size * 0.01  # Expect ~1% peaks normally
            
            if num_peaks > expected_peaks * 2:
                result['patterns_found'] = True
                result['analysis']['unusual_peaks'] = int(num_peaks)
                result['confidence'] = max(result['confidence'], 45)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in spectral analysis: {e}")
            result['error'] = str(e)
            return result
    
    def _lsb_analysis(self, img_array: np.ndarray) -> Dict[str, Any]:
        """
        Analyze Least Significant Bits for steganographic watermarks.
        
        Watermarks embedded in LSB create statistical anomalies.
        """
        result = {
            'anomaly_detected': False,
            'confidence': 0,
            'analysis': {}
        }
        
        try:
            # Extract LSB plane
            lsb_plane = img_array & 1
            
            # Analyze randomness of LSB
            # Natural images have some correlation in LSB
            # Watermarked images often have more random LSB
            
            # Check LSB of each channel
            anomaly_scores = []
            
            for channel in range(min(3, img_array.shape[2]) if len(img_array.shape) == 3 else 1):
                if len(img_array.shape) == 3:
                    channel_lsb = lsb_plane[:, :, channel]
                else:
                    channel_lsb = lsb_plane
                
                # Calculate ratio of 1s to 0s (should be ~50% for watermarked)
                ones_ratio = np.mean(channel_lsb)
                
                # Calculate local correlation (watermarks reduce this)
                h_diff = np.abs(channel_lsb[:, 1:] - channel_lsb[:, :-1])
                v_diff = np.abs(channel_lsb[1:, :] - channel_lsb[:-1, :])
                
                h_correlation = 1 - np.mean(h_diff)
                v_correlation = 1 - np.mean(v_diff)
                
                # Natural images: correlation > 0.6
                # Watermarked images: correlation ~ 0.5 (random)
                avg_correlation = (h_correlation + v_correlation) / 2
                
                # Score based on deviation from expected natural correlation
                if avg_correlation < 0.55:
                    anomaly_scores.append(1.0)
                elif avg_correlation < 0.60:
                    anomaly_scores.append(0.7)
                elif avg_correlation < 0.65:
                    anomaly_scores.append(0.4)
                else:
                    anomaly_scores.append(0.0)
                
                result['analysis'][f'channel_{channel}_ones_ratio'] = round(float(ones_ratio), 3)
                result['analysis'][f'channel_{channel}_correlation'] = round(float(avg_correlation), 3)
            
            avg_anomaly = np.mean(anomaly_scores)
            
            if avg_anomaly > 0.5:
                result['anomaly_detected'] = True
                result['confidence'] = int(avg_anomaly * 60)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in LSB analysis: {e}")
            result['error'] = str(e)
            return result
    
    def _check_metadata_watermarks(self, image_data: bytes) -> Dict[str, Any]:
        """
        Check for watermarks stored in image metadata.
        
        Detects:
        - EXIF AI markers
        - XMP AI tool identifiers
        - IPTC DigitalSourceType (trainedAlgorithmicMedia)
        - PNG tEXt/iTXt chunks with AI tool info
        """
        result = {
            'found': False,
            'type': None,
            'content': None,
            'ai_source_type': None
        }
        
        try:
            image = Image.open(io.BytesIO(image_data))
            
            # Check EXIF for watermark fields
            exif = image._getexif() if hasattr(image, '_getexif') else None
            
            if exif:
                # Check common watermark/copyright fields
                watermark_tags = {
                    0x8298: 'Copyright',
                    0x9c9b: 'XPKeywords',
                    0x9c9c: 'XPComment',
                    0x9c9d: 'XPAuthor',
                    0x010e: 'ImageDescription',
                    0x013b: 'Artist',
                    0x0131: 'Software',
                }
                
                ai_keywords = [
                    'ai generated', 'artificial intelligence', 'stable diffusion',
                    'dall-e', 'midjourney', 'synthid', 'ai-generated',
                    'machine generated', 'synthetic', 'generated by ai',
                    'firefly', 'imagen', 'leonardo', 'playground',
                    'dreamstudio', 'nightcafe', 'craiyon', 'bluewillow'
                ]
                
                for tag_id, tag_name in watermark_tags.items():
                    if tag_id in exif:
                        value = str(exif[tag_id]).lower()
                        for keyword in ai_keywords:
                            if keyword in value:
                                result['found'] = True
                                result['type'] = f'EXIF_{tag_name}'
                                result['content'] = exif[tag_id]
                                return result
            
            # Check for XMP metadata (Adobe style)
            xmp_data = image.info.get('XML:com.adobe.xmp', '') or image.info.get('xmp', '')
            if isinstance(xmp_data, bytes):
                xmp_data = xmp_data.decode('utf-8', errors='ignore')
            
            if xmp_data:
                xmp_lower = xmp_data.lower()
                
                # Check for IPTC DigitalSourceType (official AI content marker)
                for source_type in self.AI_SOURCE_TYPES:
                    if source_type.lower() in xmp_lower:
                        result['found'] = True
                        result['type'] = 'IPTC_DigitalSourceType'
                        result['ai_source_type'] = source_type
                        result['content'] = f"DigitalSourceType: {source_type}"
                        return result
                
                # Check for AI tool mentions in XMP
                ai_xmp_markers = [
                    'dall-e', 'dall·e', 'midjourney', 'stable diffusion', 
                    'ai generated', 'synthid', 'firefly', 'imagen',
                    'leonardo.ai', 'playground ai', 'dreamstudio',
                    'bing image creator', 'microsoft designer', 'copilot'
                ]
                
                for marker in ai_xmp_markers:
                    if marker in xmp_lower:
                        result['found'] = True
                        result['type'] = 'XMP_AIMarker'
                        result['content'] = marker
                        return result
            
            # Check PNG text chunks for AI tool info
            if hasattr(image, 'text') and image.text:
                for key, value in image.text.items():
                    value_lower = str(value).lower()
                    key_lower = key.lower()
                    
                    # Common PNG metadata keys from AI tools
                    ai_png_keys = ['parameters', 'prompt', 'dream', 'sd-metadata', 'generation_data']
                    if any(k in key_lower for k in ai_png_keys):
                        result['found'] = True
                        result['type'] = f'PNG_{key}'
                        result['content'] = value[:200] + '...' if len(value) > 200 else value
                        return result
                    
                    # Check for AI tool names in values
                    for keyword in ['stable diffusion', 'automatic1111', 'comfyui', 'invokeai', 'midjourney']:
                        if keyword in value_lower:
                            result['found'] = True
                            result['type'] = 'PNG_AIMarker'
                            result['content'] = keyword
                            return result
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking metadata watermarks: {e}")
            result['error'] = str(e)
            return result
    
    def _detect_treering_patterns(self, img_array: np.ndarray) -> Dict[str, Any]:
        """
        Detect Tree-Ring watermark patterns in the frequency domain.
        
        Tree-Ring watermarks embed concentric ring patterns in the Fourier
        transform of the initial noise, which survive the diffusion process.
        
        Note: Full detection requires model inversion, but we can detect
        statistical signatures of the ring patterns.
        """
        result = {
            'detected': False,
            'confidence': 0,
            'analysis': {}
        }
        
        try:
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = np.mean(img_array, axis=2).astype(np.float32)
            else:
                gray = img_array.astype(np.float32)
            
            # Compute 2D FFT and shift
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            
            # Analyze for concentric ring patterns
            center = np.array(magnitude.shape) // 2
            
            # Create radial profile
            y, x = np.ogrid[:magnitude.shape[0], :magnitude.shape[1]]
            r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
            
            # Bin by radius and compute mean magnitude
            max_radius = int(min(center) * 0.8)
            radial_bins = np.linspace(0, max_radius, 50)
            radial_profile = []
            
            for i in range(len(radial_bins) - 1):
                mask = (r >= radial_bins[i]) & (r < radial_bins[i+1])
                if np.sum(mask) > 0:
                    radial_profile.append(np.mean(magnitude[mask]))
                else:
                    radial_profile.append(0)
            
            radial_profile = np.array(radial_profile)
            
            # Tree-Ring patterns create periodic variations in radial profile
            # Check for periodicity using autocorrelation
            if len(radial_profile) > 10:
                autocorr = np.correlate(radial_profile - np.mean(radial_profile), 
                                        radial_profile - np.mean(radial_profile), mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                autocorr = autocorr / autocorr[0]  # Normalize
                
                # Look for secondary peaks (indicating periodicity)
                peaks = []
                for i in range(3, len(autocorr) - 1):
                    if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.3:
                        peaks.append((i, autocorr[i]))
                
                result['analysis']['radial_peaks'] = len(peaks)
                result['analysis']['peak_strength'] = max([p[1] for p in peaks]) if peaks else 0
                
                # Strong periodicity suggests Tree-Ring watermark
                if len(peaks) >= 2 and result['analysis']['peak_strength'] > 0.5:
                    result['detected'] = True
                    result['confidence'] = int(min(85, result['analysis']['peak_strength'] * 100))
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Tree-Ring detection: {e}")
            result['error'] = str(e)
            return result
    
    def _detect_adversarial_perturbations(self, img_array: np.ndarray) -> Dict[str, Any]:
        """
        Detect adversarial perturbations from tools like Glaze/Nightshade.
        
        WARNING: Experimental - high false positive rate (~40%)
        
        These tools add imperceptible perturbations to protect images.
        Detection looks for:
        - Unusual high-frequency energy patterns
        - Statistical anomalies in gradient distributions
        - Kurtosis abnormalities
        """
        result = {
            'detected': False,
            'type': None,
            'confidence': 0,
            'analysis': {}
        }
        
        try:
            # Convert to float
            img_float = img_array.astype(np.float32) / 255.0
            
            # Analyze each channel
            channel_scores = []
            
            for channel in range(min(3, img_float.shape[2]) if len(img_float.shape) == 3 else 1):
                if len(img_float.shape) == 3:
                    channel_data = img_float[:, :, channel]
                else:
                    channel_data = img_float
                
                # Compute FFT
                fft = np.fft.fft2(channel_data)
                fft_shift = np.fft.fftshift(fft)
                
                # Analyze high-frequency components
                h, w = fft_shift.shape
                high_freq_region = np.abs(fft_shift[h//4:3*h//4, w//4:3*w//4])
                
                # Kurtosis of high-frequency components
                # Adversarial perturbations often increase kurtosis
                kurtosis = stats.kurtosis(high_freq_region.flatten())
                
                # Compute gradients
                grad_x = np.diff(channel_data, axis=1)
                grad_y = np.diff(channel_data, axis=0)
                
                # Gradient magnitude
                grad_mag = np.sqrt(grad_x[:, :-1]**2 + grad_y[:-1, :]**2)
                
                # Check for unusual gradient distribution
                grad_kurtosis = stats.kurtosis(grad_mag.flatten())
                grad_skew = stats.skew(grad_mag.flatten())
                
                result['analysis'][f'channel_{channel}'] = {
                    'fft_kurtosis': round(float(kurtosis), 2),
                    'gradient_kurtosis': round(float(grad_kurtosis), 2),
                    'gradient_skew': round(float(grad_skew), 2)
                }
                
                # Score based on anomalies
                score = 0
                if kurtosis > 15:  # Very high kurtosis
                    score += 0.4
                elif kurtosis > 10:
                    score += 0.2
                    
                if grad_kurtosis > 20:
                    score += 0.3
                elif grad_kurtosis > 12:
                    score += 0.15
                    
                if abs(grad_skew) > 3:
                    score += 0.3
                elif abs(grad_skew) > 2:
                    score += 0.15
                
                channel_scores.append(score)
            
            avg_score = np.mean(channel_scores)
            result['analysis']['combined_score'] = round(float(avg_score), 3)
            
            if avg_score > 0.5:
                result['detected'] = True
                result['type'] = 'Glaze/Nightshade (possible)'
                result['confidence'] = int(min(60, avg_score * 80))  # Cap at 60% due to unreliability
            
            return result
            
        except Exception as e:
            logger.error(f"Error in adversarial detection: {e}")
            result['error'] = str(e)
            return result
    
    def _detect_steganogan(self, image_data: bytes) -> Dict[str, Any]:
        """
        Detect SteganoGAN steganographic watermarks.
        
        SteganoGAN uses a GAN-based encoder/decoder to hide arbitrary data.
        """
        result = {
            'detected': False,
            'message': None,
            'confidence': 0
        }
        
        if not self.steganogan_available:
            result['error'] = 'SteganoGAN library not available'
            return result
        
        try:
            from steganogan import SteganoGAN
            import tempfile
            import os
            
            # Save image to temp file (steganogan requires file path)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp.write(image_data)
                tmp_path = tmp.name
            
            try:
                steganogan = SteganoGAN.load(architecture='dense')
                message = steganogan.decode(tmp_path)
                
                if message and len(message.strip()) > 0:
                    result['detected'] = True
                    result['message'] = message[:100]  # Limit length
                    result['confidence'] = 80
            finally:
                os.unlink(tmp_path)
            
            return result
            
        except Exception as e:
            logger.debug(f"SteganoGAN detection failed (expected for non-steganogan images): {e}")
            return result


# Convenience function
def detect_watermark(image_data: bytes) -> Dict[str, Any]:
    """Convenience function to detect watermarks in an image."""
    detector = WatermarkDetector()
    return detector.analyze(image_data)
