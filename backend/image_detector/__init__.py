"""
VisioNova Image Detector Module
Detects AI-generated images using deep learning and forensic analysis.

Components:
- ImageDetector: Main AI vs Real classifier (statistical + ML)
- EnsembleDetector: Multi-model ensemble with weighted fusion (RECOMMENDED)
- MetadataAnalyzer: EXIF/metadata forensics
- ELAAnalyzer: Error Level Analysis for manipulation detection
- WatermarkDetector: Invisible watermark detection (Stable Diffusion, Meta, etc.)
- ContentCredentialsDetector: C2PA/Content Credentials detection (DALL-E 3, etc.)
- ImageExplainer: Groq Vision API for AI-powered image analysis (Llama 4 Scout)

ML Models (loaded on demand):
- NYUADDetector: Vision Transformer (97.36% accuracy) - HuggingFace
- UniversalFakeDetector: CLIP-based detector (generalizes across generators)
- DeepfakeDetector: Face manipulation detection
- FrequencyAnalyzer: FFT/DCT GAN fingerprint analysis
"""

from .detector import ImageDetector
from .metadata_analyzer import MetadataAnalyzer
from .ela_analyzer import ELAAnalyzer
from .watermark_detector import WatermarkDetector
from .content_credentials import ContentCredentialsDetector
from .image_explainer import ImageExplainer, create_image_explainer
from .ensemble_detector import EnsembleDetector, create_ensemble_detector

# ML detectors (may require additional dependencies)
try:
    from .ml_detector import (
        NYUADDetector,
        UniversalFakeDetector,
        DeepfakeDetector,
        FrequencyAnalyzer,
        create_ml_detectors
    )
    ML_DETECTORS_AVAILABLE = True
except ImportError:
    ML_DETECTORS_AVAILABLE = False
    NYUADDetector = None
    UniversalFakeDetector = None
    DeepfakeDetector = None
    FrequencyAnalyzer = None
    create_ml_detectors = None

__all__ = [
    # Core detectors
    'ImageDetector', 
    'EnsembleDetector',
    'create_ensemble_detector',
    
    # Analyzers
    'MetadataAnalyzer', 
    'ELAAnalyzer',
    'WatermarkDetector',
    'ContentCredentialsDetector',
    
    # AI Explainer
    'ImageExplainer',
    'create_image_explainer',
    
    # ML Detectors
    'NYUADDetector',
    'UniversalFakeDetector',
    'DeepfakeDetector',
    'FrequencyAnalyzer',
    'create_ml_detectors',
    'ML_DETECTORS_AVAILABLE'
]
