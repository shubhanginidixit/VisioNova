"""
VisioNova Image Detector Module
Detects AI-generated images using deep learning and forensic analysis.

Components:
- ImageDetector: Main AI vs Real classifier (statistical + ML)
- EnsembleDetector: Multi-model ensemble with weighted fusion (RECOMMENDED)
- FastCascadeDetector: Speed-optimized 3-stage detection (3-5x faster)
- MetadataAnalyzer: EXIF/metadata forensics
- ELAAnalyzer: Error Level Analysis for manipulation detection
- WatermarkDetector: Invisible watermark detection (Stable Diffusion, Meta, etc.)
- ContentCredentialsDetector: C2PA/Content Credentials detection (DALL-E 3, etc.)
- ImageExplainer: Groq Vision API for AI-powered image analysis (Llama 4 Scout)

Top 4 ML Models (loaded on demand, 2025-2026):
1. SigLIPDINOv2Detector: Bombek1 SigLIP2+DINOv2 (99.97% AUC, Jan 2026)
2. AteeqqDetector: SigLIP2 (99.23% accuracy, Dec 2025)
3. DeepfakeDetector: dima806 ViT (98.25% accuracy, Jan 2025)
4. SDXLDetector: Organika/sdxl-detector (98.1% for Flux/SDXL)

Removed (low performance / redundant):
- DINOv2DeepfakeDetector: WpythonW DINOv2 (redundant with SigLIPDINOv2)
- FrequencyAnalyzer: FFT/DCT GAN fingerprint analysis (only detects old GANs)

Heuristic/Legacy (weight=0 by default, NOT loaded):
- SBIDetector, DIREDetector, NPRDetector, FaceConsistencyDetector, EdgeCoherenceAnalyzer
"""


from .metadata_analyzer import MetadataAnalyzer
from .ela_analyzer import ELAAnalyzer
from .watermark_detector import WatermarkDetector
from .content_credentials import ContentCredentialsDetector
from .image_explainer import ImageExplainer, create_image_explainer
from .ensemble_detector import EnsembleDetector, create_ensemble_detector
from .fast_cascade_detector import FastCascadeDetector, create_fast_detector
from .noise_analyzer import NoiseAnalyzer

# Accuracy improvement modules (Phase 1)
from .confidence_calibrator import ConfidenceCalibrator, create_calibrator

# High-value accuracy modules (Phase 2)

# ML detectors (may require additional dependencies)
try:
    from .ml_detector import (
        AteeqqDetector,
        SDXLDetector,
        DeepfakeDetector,
        SigLIPDINOv2Detector,
        create_ml_detectors
    )
    ML_DETECTORS_AVAILABLE = True
except ImportError:
    ML_DETECTORS_AVAILABLE = False
    AteeqqDetector = None
    SDXLDetector = None
    DeepfakeDetector = None
    SigLIPDINOv2Detector = None
    create_ml_detectors = None

__all__ = [
    # Core detectors
    'EnsembleDetector',
    'create_ensemble_detector',
    'FastCascadeDetector',
    'create_fast_detector',
    
    # Accuracy improvement detectors (Phase 1)
    'ConfidenceCalibrator',
    'create_calibrator',
    
    # High-value detectors (Phase 2)
    
    # Analyzers
    'MetadataAnalyzer', 
    'ELAAnalyzer',
    'WatermarkDetector',
    'ContentCredentialsDetector',
    'NoiseAnalyzer',
    
    # AI Explainer
    'ImageExplainer',
    'create_image_explainer',
    
    # Top 4 ML Detectors
    'AteeqqDetector',
    'SDXLDetector',
    'DeepfakeDetector',
    'SigLIPDINOv2Detector',
    'create_ml_detectors',
    'ML_DETECTORS_AVAILABLE'
]


