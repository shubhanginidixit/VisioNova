"""
VisioNova Image Detector Module
Detects AI-generated images using deep learning and forensic analysis.

Components:
- ImageDetector: Main AI vs Real classifier (statistical + ML)
- MetadataAnalyzer: EXIF/metadata forensics
- ELAAnalyzer: Error Level Analysis for manipulation detection
- WatermarkDetector: Invisible watermark detection (Stable Diffusion, etc.)
- ContentCredentialsDetector: C2PA/Content Credentials detection (DALL-E 3, etc.)
- ImageExplainer: Groq Vision API for AI-powered image analysis (Llama 4 Scout)
"""

from .detector import ImageDetector
from .metadata_analyzer import MetadataAnalyzer
from .ela_analyzer import ELAAnalyzer
from .watermark_detector import WatermarkDetector
from .content_credentials import ContentCredentialsDetector
from .image_explainer import ImageExplainer, create_image_explainer

__all__ = [
    'ImageDetector', 
    'MetadataAnalyzer', 
    'ELAAnalyzer',
    'WatermarkDetector',
    'ContentCredentialsDetector',
    'ImageExplainer',
    'create_image_explainer'
]
