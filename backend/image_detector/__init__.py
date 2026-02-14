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
- SBIDetector: Synthetic Basis Index for diffusion detection (99.95% AUC)
- CopyMoveForgeryDetector: A-KAZE based manipulation detection (98.98% accuracy)
- ConfidenceCalibrator: Temperature scaling for calibrated predictions
- DIREDetector: Diffusion Reconstruction Error detector (99.7% accuracy)
- NPRDetector: Neighboring Pixel Relationships detector (99.1% accuracy)
- FaceConsistencyDetector: Eye reflections, symmetry, lighting analysis
- EdgeCoherenceAnalyzer: Edge quality and artifact detection

ML Models (loaded on demand):
- NYUADDetector: Vision Transformer (97.36% accuracy) - HuggingFace
- UniversalFakeDetect: CLIP-based detector (generalizes across generators)
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
from .fast_cascade_detector import FastCascadeDetector, create_fast_detector
from .noise_analyzer import NoiseAnalyzer

# Accuracy improvement modules (Phase 1)
from .sbi_detector import SBIDetector, create_sbi_detector
from .forgery_detector import CopyMoveForgeryDetector, create_forgery_detector
from .confidence_calibrator import ConfidenceCalibrator, create_calibrator

# High-value accuracy modules (Phase 2)
from .dire_detector import DIREDetector, create_dire_detector
from .npr_detector import NPRDetector, create_npr_detector
from .face_consistency_detector import FaceConsistencyDetector, create_face_detector
from .edge_coherence_analyzer import EdgeCoherenceAnalyzer, create_edge_analyzer

# ML detectors (may require additional dependencies)
try:
    from .ml_detector import (
        NYUADDetector,
        UniversalFakeDetector,
        SDXLDetector,
        DeepfakeDetector,
        FrequencyAnalyzer,
        create_ml_detectors
    )
    ML_DETECTORS_AVAILABLE = True
except ImportError:
    ML_DETECTORS_AVAILABLE = False
    NYUADDetector = None
    UniversalFakeDetector = None
    SDXLDetector = None
    DeepfakeDetector = None
    FrequencyAnalyzer = None
    create_ml_detectors = None

__all__ = [
    # Core detectors
    'ImageDetector', 
    'EnsembleDetector',
    'create_ensemble_detector',
    'FastCascadeDetector',
    'create_fast_detector',
    
    # Accuracy improvement detectors (Phase 1)
    'SBIDetector',
    'create_sbi_detector',
    'CopyMoveForgeryDetector',
    'create_forgery_detector',
    'ConfidenceCalibrator',
    'create_calibrator',
    
    # High-value detectors (Phase 2)
    'DIREDetector',
    'create_dire_detector',
    'NPRDetector',
    'create_npr_detector',
    'FaceConsistencyDetector',
    'create_face_detector',
    'EdgeCoherenceAnalyzer',
    'create_edge_analyzer',
    
    # Analyzers
    'MetadataAnalyzer', 
    'ELAAnalyzer',
    'WatermarkDetector',
    'ContentCredentialsDetector',
    'NoiseAnalyzer',
    
    # AI Explainer
    'ImageExplainer',
    'create_image_explainer',
    
    # ML Detectors
    'NYUADDetector',
    'UniversalFakeDetector',
    'SDXLDetector',
    'DeepfakeDetector',
    'FrequencyAnalyzer',
    'create_ml_detectors',
    'ML_DETECTORS_AVAILABLE'
]


