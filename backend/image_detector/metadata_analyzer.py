"""
VisioNova Metadata Analyzer
Extracts and analyzes image metadata (EXIF) for forensic purposes.

AI-generated images typically lack camera EXIF data, while real photos
contain detailed information about the camera, settings, and location.
"""

import io
import logging
from typing import Optional
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetadataAnalyzer:
    """
    Image Metadata Analyzer
    
    Extracts and analyzes EXIF data to help determine image authenticity.
    AI-generated images typically lack or have suspicious metadata.
    """
    
    # Known AI generation software signatures
    AI_SOFTWARE_SIGNATURES = [
        'stable diffusion', 'midjourney', 'dall-e', 'dalle',
        'novelai', 'automatic1111', 'comfyui', 'invoke',
        'dreamstudio', 'leonardo', 'playground', 'ideogram',
        'firefly', 'bing image creator', 'copilot',
        'flux', 'sd', 'sdxl', 'controlnet'
    ]
    
    # Known image editing software
    EDITING_SOFTWARE = [
        'photoshop', 'gimp', 'lightroom', 'capture one',
        'affinity', 'paint.net', 'pixelmator', 'snapseed',
        'vsco', 'canva', 'figma', 'illustrator'
    ]
    
    # Screenshot software signatures
    SCREENSHOT_SOFTWARE = [
        'snipping tool', 'snagit', 'lightshot', 'sharex',
        'greenshot', 'flameshot', 'spectacle', 'screenshot',
        'grab', 'firefox screenshot', 'chrome', 'edge',
        'windows.graphics.capture', 'screencapture'
    ]
    
    # Common screenshot resolutions (width x height)
    SCREENSHOT_RESOLUTIONS = [
        (1920, 1080), (2560, 1440), (3840, 2160),  # 16:9
        (1366, 768), (1600, 900), (1280, 720),     # 16:9
        (1440, 900), (1680, 1050), (1920, 1200),   # 16:10
        (2560, 1600), (3840, 2400),                # 16:10
        (1280, 800), (1024, 768), (1280, 1024),    # 4:3 / 5:4
        (3440, 1440), (2560, 1080),                # Ultrawide
    ]
    
    def __init__(self):
        """Initialize the metadata analyzer."""
        pass
    
    def analyze(self, image_data: bytes) -> dict:
        """
        Analyze image metadata.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            dict with metadata analysis results
        """
        try:
            image = Image.open(io.BytesIO(image_data))
            
            result = {
                'success': True,
                'has_exif': False,
                'has_camera_info': False,
                'has_gps': False,
                'has_timestamp': False,
                'software_detected': None,
                'ai_software_detected': False,
                'editing_software_detected': False,
                'is_screenshot': False,
                'screenshot_indicators': [],
                'anomalies': [],
                'metadata': {},
                'ai_probability_modifier': 0  # -30 to +30 adjustment
            }
            
            # Get basic image info
            result['format'] = image.format
            result['mode'] = image.mode
            result['size'] = {'width': image.width, 'height': image.height}
            
            # Check if this is a screenshot
            screenshot_check = self._detect_screenshot(image)
            result['is_screenshot'] = screenshot_check['is_screenshot']
            result['screenshot_indicators'] = screenshot_check['indicators']
            
            # Extract JPEG quality if available
            if hasattr(image, 'quantization'):
                result['jpeg_quality'] = self._estimate_jpeg_quality(image)
            else:
                result['jpeg_quality'] = None
            
            # Extract EXIF data
            exif_data = self._extract_exif(image)
            
            if exif_data:
                result['has_exif'] = True
                result['metadata'] = exif_data
                
                # Analyze camera info
                camera_info = self._analyze_camera_info(exif_data)
                result.update(camera_info)
                
                # Analyze software
                software_info = self._analyze_software(exif_data)
                result.update(software_info)
                
                # Check for timestamp
                timestamp_info = self._analyze_timestamp(exif_data)
                result.update(timestamp_info)
                
                # Check for GPS
                gps_info = self._analyze_gps(exif_data)
                result.update(gps_info)
                
                # Look for anomalies
                anomalies = self._detect_anomalies(exif_data, image)
                result['anomalies'] = anomalies
            else:
                # No EXIF data - suspicious for real photos
                result['anomalies'].append('No EXIF metadata found')
                result['ai_probability_modifier'] = 15
            
            # Calculate overall AI probability modifier
            result['ai_probability_modifier'] = self._calculate_modifier(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing metadata: {e}")
            return {
                'success': False,
                'error': str(e),
                'has_exif': False,
                'anomalies': [f'Metadata extraction failed: {str(e)}'],
                'ai_probability_modifier': 0
            }
    
    def _extract_exif(self, image: Image.Image) -> dict:
        """Extract EXIF data from image."""
        exif_data = {}
        
        try:
            # Try PIL's getexif() method
            exif = image.getexif()
            if exif:
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    
                    # Handle bytes values
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8', errors='ignore')
                        except:
                            value = str(value)
                    
                    exif_data[tag] = value
            
            # Also try to get IFD data (more detailed EXIF)
            for ifd_id in exif.get_ifd(0x8769) if hasattr(exif, 'get_ifd') else []:
                try:
                    ifd = exif.get_ifd(ifd_id)
                    for tag_id, value in ifd.items():
                        tag = TAGS.get(tag_id, tag_id)
                        if isinstance(value, bytes):
                            try:
                                value = value.decode('utf-8', errors='ignore')
                            except:
                                value = str(value)
                        exif_data[tag] = value
                except:
                    pass
                    
        except Exception as e:
            logger.debug(f"Could not extract EXIF: {e}")
        
        return exif_data
    
    def _analyze_camera_info(self, exif: dict) -> dict:
        """Analyze camera-related metadata."""
        result = {
            'has_camera_info': False,
            'camera_make': None,
            'camera_model': None,
            'lens_info': None
        }
        
        # Check for camera make/model
        if 'Make' in exif:
            result['camera_make'] = str(exif['Make']).strip()
            result['has_camera_info'] = True
        
        if 'Model' in exif:
            result['camera_model'] = str(exif['Model']).strip()
            result['has_camera_info'] = True
        
        # Check for lens info
        lens_fields = ['LensModel', 'LensMake', 'LensInfo']
        for field in lens_fields:
            if field in exif:
                result['lens_info'] = str(exif[field])
                break
        
        return result
    
    def _analyze_software(self, exif: dict) -> dict:
        """Analyze software-related metadata."""
        result = {
            'software_detected': None,
            'ai_software_detected': False,
            'editing_software_detected': False
        }
        
        # Check Software field
        software = None
        for field in ['Software', 'ProcessingSoftware', 'CreatorTool']:
            if field in exif:
                software = str(exif[field]).lower()
                result['software_detected'] = exif[field]
                break
        
        if software:
            # Check for AI generation software
            for ai_sig in self.AI_SOFTWARE_SIGNATURES:
                if ai_sig in software:
                    result['ai_software_detected'] = True
                    break
            
            # Check for editing software
            for edit_sig in self.EDITING_SOFTWARE:
                if edit_sig in software:
                    result['editing_software_detected'] = True
                    break
        
        return result
    
    def _analyze_timestamp(self, exif: dict) -> dict:
        """Analyze timestamp metadata."""
        result = {
            'has_timestamp': False,
            'datetime_original': None,
            'datetime_digitized': None
        }
        
        # Check for original capture time
        if 'DateTimeOriginal' in exif:
            result['has_timestamp'] = True
            result['datetime_original'] = str(exif['DateTimeOriginal'])
        
        if 'DateTimeDigitized' in exif:
            result['datetime_digitized'] = str(exif['DateTimeDigitized'])
        
        if 'DateTime' in exif and not result['has_timestamp']:
            result['has_timestamp'] = True
            result['datetime_original'] = str(exif['DateTime'])
        
        return result
    
    def _analyze_gps(self, exif: dict) -> dict:
        """Analyze GPS metadata."""
        result = {
            'has_gps': False,
            'gps_coordinates': None
        }
        
        # Check for GPS data
        gps_fields = ['GPSLatitude', 'GPSLongitude', 'GPSInfo']
        for field in gps_fields:
            if field in exif:
                result['has_gps'] = True
                break
        
        return result
    
    def _detect_anomalies(self, exif: dict, image: Image.Image) -> list:
        """Detect suspicious patterns in metadata."""
        anomalies = []
        
        # Check for missing expected fields
        if 'Make' not in exif and 'Model' not in exif:
            anomalies.append('No camera make/model information')
        
        # Check for suspicious combinations
        if 'Software' in exif and 'Make' not in exif:
            anomalies.append('Software present but no camera info')
        
        # Check for AI software signatures
        software = str(exif.get('Software', '')).lower()
        for ai_sig in self.AI_SOFTWARE_SIGNATURES:
            if ai_sig in software:
                anomalies.append(f'AI generation software detected: {ai_sig}')
                break
        
        # Check for stripped EXIF (common in AI images shared online)
        if len(exif) < 5 and image.format == 'JPEG':
            anomalies.append('Minimal EXIF data (possibly stripped)')
        
        # Check for inconsistent resolution
        if 'ExifImageWidth' in exif and 'ExifImageHeight' in exif:
            exif_width = exif['ExifImageWidth']
            exif_height = exif['ExifImageHeight']
            if (exif_width != image.width or exif_height != image.height):
                anomalies.append('Image dimensions mismatch with EXIF')
        
        return anomalies
    
    def _calculate_modifier(self, result: dict) -> int:
        """
        Calculate AI probability modifier based on metadata analysis.
        
        Returns value from -30 to +30:
        - Negative values indicate more likely real
        - Positive values indicate more likely AI
        """
        modifier = 0
        
        # Screenshot detection - screenshots are NOT AI-generated
        if result.get('is_screenshot'):
            modifier -= 20  # Strong indicator of real (screen capture)
            return max(-30, min(30, modifier))  # Early return
        
        # Strong indicators of real photo
        if result.get('has_camera_info'):
            modifier -= 15
        if result.get('has_gps'):
            modifier -= 10
        if result.get('has_timestamp'):
            modifier -= 5
        
        # Strong indicators of AI
        if result.get('ai_software_detected'):
            modifier += 30
        if not result.get('has_exif'):
            modifier += 10
        
        # Anomalies increase AI likelihood
        modifier += len(result.get('anomalies', [])) * 3
        
        # Clamp to range
        return max(-30, min(30, modifier))
    
    def _estimate_jpeg_quality(self, image: Image.Image) -> Optional[int]:
        """
        Estimate JPEG quality from quantization tables.
        
        Returns:
            Estimated quality (0-100) or None if not a JPEG
        """
        try:
            if not hasattr(image, 'quantization') or image.format != 'JPEG':
                return None
            
            # Get quantization tables
            qtables = image.quantization
            if not qtables:
                return None
            
            # Use first luminance table
            if 0 in qtables:
                qtable = qtables[0]
            else:
                qtable = list(qtables.values())[0]
            
            # Estimate quality from table values
            # Quality estimation based on quantization table sum
            # Reference: IJG (Independent JPEG Group) quality mapping
            q_sum = sum(qtable)
            
            if q_sum < 100:
                quality = 99
            elif q_sum < 200:
                quality = 95
            elif q_sum < 400:
                quality = 90
            elif q_sum < 800:
                quality = 85
            elif q_sum < 1600:
                quality = 75
            elif q_sum < 3200:
                quality = 60
            elif q_sum < 6400:
                quality = 40
            else:
                quality = 20
            
            return quality
                
        except Exception as e:
            logger.warning(f"Failed to estimate JPEG quality: {e}")
            return None
    
    def _detect_screenshot(self, image: Image.Image) -> dict:
        """
        Detect if an image is a screenshot.
        
        Screenshots typically have:
        - No camera EXIF data
        - Common monitor resolutions
        - Screenshot software signatures
        - PNG format (common for screenshots)
        - Perfectly aligned pixels
        
        Returns:
            dict with 'is_screenshot' (bool) and 'indicators' (list)
        """
        indicators = []
        
        # Check resolution against common screenshot sizes
        width, height = image.width, image.height
        
        # Exact match
        if (width, height) in self.SCREENSHOT_RESOLUTIONS:
            indicators.append(f'Common screenshot resolution: {width}x{height}')
        
        # Also check reversed (portrait mode)
        if (height, width) in self.SCREENSHOT_RESOLUTIONS:
            indicators.append(f'Common screenshot resolution (portrait): {width}x{height}')
        
        # Check for common aspect ratios at any resolution
        aspect_ratio = width / height if height > 0 else 0
        common_ratios = {
            16/9: '16:9',
            16/10: '16:10',
            4/3: '4:3',
            21/9: '21:9',
            32/9: '32:9'
        }
        
        for ratio, name in common_ratios.items():
            if abs(aspect_ratio - ratio) < 0.01:  # Allow small tolerance
                indicators.append(f'Monitor aspect ratio: {name}')
                break
        
        # Check format (PNG is more common for screenshots)
        if image.format == 'PNG':
            indicators.append('PNG format (common for screenshots)')
        
        # Check EXIF for screenshot software
        exif_data = self._extract_exif(image)
        if exif_data:
            software = str(exif_data.get('Software', '')).lower()
            for screenshot_sig in self.SCREENSHOT_SOFTWARE:
                if screenshot_sig in software:
                    indicators.append(f'Screenshot software detected: {screenshot_sig}')
                    break
        
        # Check if resolution is exactly divisible by common UI scaling factors
        if width % 16 == 0 and height % 9 == 0:  # 16:9 ratio with clean division
            scale_factor_w = width // 16
            scale_factor_h = height // 9
            if scale_factor_w == scale_factor_h and scale_factor_w >= 60:
                indicators.append('Clean 16:9 grid alignment (typical for screenshots)')
        
        # Determine if it's a screenshot based on indicators
        is_screenshot = len(indicators) >= 2  # Need at least 2 indicators
        
        return {
            'is_screenshot': is_screenshot,
            'indicators': indicators
        }
