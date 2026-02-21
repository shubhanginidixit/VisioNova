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

ML Models (loaded on demand, 2025-2026 models prioritized):
- AteeqqDetector: SigLIP2 (99.23% accuracy, Dec 2025) — NEW TOP MODEL
- SigLIPDINOv2Detector: Bombek1 SigLIP2+DINOv2 (99.97% AUC, Jan 2026)
- DeepfakeDetector: dima806 ViT (98.25% accuracy, Jan 2025)
- SDXLDetector: Organika/sdxl-detector (98.1% for Flux/SDXL)
- DINOv2DeepfakeDetector: WpythonW DINOv2 (degradation-resilient)
- DeepFakeV2Detector: prithivMLmods V2 (Feb 2025 dataset)
- SigLIPDeepfakeDetector: prithivMLmods SigLIP V1
- AteeqqDetector: Vision Transformer (99.23% accuracy)
- FrequencyAnalyzer: FFT/DCT GAN fingerprint analysis

Heuristic/Legacy (weight=0 by default, NOT loaded):
- SBIDetector, DIREDetector, NPRDetector, FaceConsistencyDetector, EdgeCoherenceAnalyzer
- UmmMaybeDetector (outdated Oct 2022), DistilledDetector (74%), AIorNotDetector (64.74%)
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
        AteeqqDetector,
        UniversalFakeDetector,
        SDXLDetector,
        DeepfakeDetector,
        FrequencyAnalyzer,
        UmmMaybeDetector,
        DINOv2DeepfakeDetector,
        AteeqqDetector,
        SigLIPDINOv2Detector,
        DeepFakeV2Detector,
        SigLIPDeepfakeDetector,
        create_ml_detectors
    )
    ML_DETECTORS_AVAILABLE = True
except ImportError:
    ML_DETECTORS_AVAILABLE = False
    AteeqqDetector = None
    UniversalFakeDetector = None
    SDXLDetector = None
    DeepfakeDetector = None
    FrequencyAnalyzer = None
    UmmMaybeDetector = None
    DINOv2DeepfakeDetector = None
    AteeqqDetector = None
    SigLIPDINOv2Detector = None
    DeepFakeV2Detector = None
    SigLIPDeepfakeDetector = None
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
    'AteeqqDetector',
    'UniversalFakeDetector',
    'SDXLDetector',
    'DeepfakeDetector',
    'FrequencyAnalyzer',
    'UmmMaybeDetector',
    'DINOv2DeepfakeDetector',
    'AteeqqDetector',
    'SigLIPDINOv2Detector',
    'DeepFakeV2Detector',
    'SigLIPDeepfakeDetector',
    'create_ml_detectors',
    'ML_DETECTORS_AVAILABLE'
]


