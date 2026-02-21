"""
VisioNova Ensemble Detector
Combines multiple detection methods with weighted score fusion.

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                   Ensemble Detection Pipeline                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Image                                                    │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Parallel Detection Methods                  │   │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌─────────┐  │   │
│  │  │Statistical│ │ dima806   │ │   CLIP    │ │Frequency│  │   │
│  │  │ Analysis  │ │ ViT 98.2% │ │ Universal │ │Analysis │  │   │
│  │  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └────┬────┘  │   │
│  │        │             │             │            │        │   │
│  │        ▼             ▼             ▼            ▼        │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │         Weighted Score Fusion Layer             │    │   │
│  │  │  statistical: 20% | dima806: 30% | clip: 30%   │    │   │
│  │  │  frequency: 10% | watermark: 10%               │    │   │
│  │  └────────────────────┬────────────────────────────┘    │   │
│  └───────────────────────┼─────────────────────────────────┘   │
│                          │                                      │
│                          ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Confidence Calibration                      │   │
│  │    - Agreement bonus (models agree → higher confidence)  │   │
│  │    - Watermark override (if found → 95% confidence)     │   │
│  │    - C2PA override (if AI declared → 100% confidence)   │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                          │                                      │
│                          ▼                                      │
│                   Final Verdict                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
"""

import io
import logging
from typing import Dict, Any, Optional, List
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleDetector:
    """
    Ensemble AI Image Detector
    
    Combines multiple detection methods:
    - Statistical analysis (frequency, noise, texture, edges)
    - dima806 ViT (Vision Transformer - 98.25% accuracy on HuggingFace)
    - UniversalFakeDetect (CLIP-based - generalizes across generators)
    - Frequency domain analysis (GAN fingerprints)
    - Watermark detection
    - Content Credentials (C2PA)
    - Metadata forensics
    - ELA manipulation detection
    
    Uses weighted score fusion with confidence calibration.
    """
    
    # Default weights for each detector (sum to 1.0)
    # Updated weights (Feb 2026) — RECENT MODELS ONLY
    # Only models trained on 2024-2026 generators (Flux, DALL-E 3, MJ v6,
    # GPT-Image-1, SDXL) are given weight. Outdated and broken models zeroed.
    DEFAULT_WEIGHTS: Dict[str, float] = {
        # ═══ Tier 1: Best recent models (2025-2026, verified accuracy) ═══
        'ateeqq': 0.27,             # Ateeqq SigLIP2 (99.23% acc, 46K downloads, Dec 2025)
        'siglip_dinov2': 0.27,      # Bombek1 SigLIP2+DINOv2 (99.97% AUC, 25+ generators, Jan 2026)
        
        # ═══ Tier 2: Strong recent models ═══
        'deepfake': 0.16,           # dima806 ViT (98.25% acc, 50K downloads, Jan 2025)
        'sdxl': 0.16,               # Organika/sdxl-detector (98.1% acc, SDXL/Flux specialist)
        'dinov2': 0.10,             # WpythonW DINOv2 (degradation-resilient, social media)
        
        # ═══ Tier 3: Supporting signals ═══
        'frequency': 0.04,          # FFT/DCT analysis (GAN fingerprints)
    }
    
    # Override weights when certain signals are strong
    WATERMARK_OVERRIDE_THRESHOLD = 80  # If watermark confidence > 80%, use it
    C2PA_AI_CONFIDENCE = 100           # C2PA declares AI → 100% confidence
    
    def __init__(
        self,
        use_gpu: bool = False,
        load_ml_models: bool = True,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the ensemble detector.
        
        Args:
            use_gpu: Whether to use GPU for ML models
            load_ml_models: Whether to load heavy ML models (NYUAD, CLIP)
            weights: Custom weights for score fusion
        """
        self.use_gpu = use_gpu
        self.weights: Dict[str, float] = weights or self.DEFAULT_WEIGHTS.copy()
        self.device = "cuda" if use_gpu else "cpu"
        
        # Component detectors
        self.statistical_detector = None
        self.nyuad_detector = None
        self.clip_detector = None
        self.frequency_analyzer = None
        self.watermark_detector = None
        self.metadata_analyzer = None
        self.ela_analyzer = None
        self.c2pa_detector = None
        self.deepfake_detector = None
        
        # Heuristic detectors — only loaded if their weight > 0
        self.sbi_detector = None
        self.forgery_detector = None
        self.calibrator = None
        self.dire_detector = None
        self.npr_detector = None
        self.face_detector = None
        self.edge_analyzer = None
        
        # Pretrained HuggingFace detectors
        self.sdxl_detector = None         # Organika/sdxl-detector (98.1% for modern diffusion)
        
        # 2025-2026 high-accuracy detectors
        self.ateeqq_detector = None        # Ateeqq SigLIP2 (99.23% acc, Dec 2025) — NEW
        self.dinov2_detector = None         # WpythonW DINOv2 deepfake detector
        self.siglip_dinov2_detector = None  # Bombek1 SigLIP2+DINOv2 (97.15% cross-dataset)
        self.deepfake_v2_detector = None    # prithivMLmods Deep-Fake V2 (2025 dataset)
        self.siglip_deepfake_detector = None  # prithivMLmods SigLIP Deepfake V1
        
        # Outdated/low-accuracy — kept for backward compat but NOT loaded by default
        self.umm_maybe_detector = None     # OUTDATED Oct 2022
        self.distilled_detector = None     # 74% accuracy
        self.ai_or_not_detector = None     # 64.74% accuracy
        
        # Load detectors
        self._initialize_detectors(load_ml_models)
        
        logger.info(f"EnsembleDetector initialized (GPU: {use_gpu}, ML models: {load_ml_models})")
    
    def _initialize_detectors(self, load_ml_models: bool):
        """Initialize all component detectors.
        
        Detectors with weight == 0 are SKIPPED to save memory and prevent
        false-positive bias from heuristic/outdated models.
        """
        
        # Helper: should we load a detector given its weight key?
        def _should_load(key: str) -> bool:
            return self.weights.get(key, 0) > 0
        
        # ── Always load lightweight utility detectors ──
        try:
            from .detector import ImageDetector
            self.statistical_detector = ImageDetector(use_gpu=self.use_gpu)
            logger.info("Statistical detector loaded")
        except Exception as e:
            logger.warning(f"Could not load statistical detector: {e}")
        
        try:
            from .watermark_detector import WatermarkDetector
            self.watermark_detector = WatermarkDetector()
            logger.info("Watermark detector loaded")
        except Exception as e:
            logger.warning(f"Could not load watermark detector: {e}")
        
        try:
            from .metadata_analyzer import MetadataAnalyzer
            self.metadata_analyzer = MetadataAnalyzer()
            logger.info("Metadata analyzer loaded")
        except Exception as e:
            logger.warning(f"Could not load metadata analyzer: {e}")
        
        try:
            from .ela_analyzer import ELAAnalyzer
            self.ela_analyzer = ELAAnalyzer()
            logger.info("ELA analyzer loaded")
        except Exception as e:
            logger.warning(f"Could not load ELA analyzer: {e}")
        
        try:
            from .content_credentials import ContentCredentialsDetector
            self.c2pa_detector = ContentCredentialsDetector()
            logger.info("C2PA detector loaded")
        except Exception as e:
            logger.warning(f"Could not load C2PA detector: {e}")
        
        # ── Heuristic detectors: ONLY load when weight > 0 ──
        # These are hand-coded approximations, NOT real ML models.
        # They cause false AI classifications on real images.
        if _should_load('sbi'):
            try:
                from .sbi_detector import SBIDetector
                self.sbi_detector = SBIDetector()
                logger.info("SBI Diffusion detector loaded")
            except Exception as e:
                logger.warning(f"Could not load SBI detector: {e}")
        else:
            logger.info("SBI detector SKIPPED (weight=0)")
        
        if _should_load('sbi'):  # forgery shares SBI's weight bucket
            try:
                from .forgery_detector import CopyMoveForgeryDetector
                self.forgery_detector = CopyMoveForgeryDetector()
                logger.info("Copy-Move Forgery detector loaded")
            except Exception as e:
                logger.warning(f"Could not load forgery detector: {e}")
        
        # Confidence calibrator (always useful)
        try:
            from .confidence_calibrator import ConfidenceCalibrator
            self.calibrator = ConfidenceCalibrator()
            logger.info("Confidence calibrator loaded")
        except Exception as e:
            logger.warning(f"Could not load calibrator: {e}")
        
        if _should_load('dire'):
            try:
                from .dire_detector import DIREDetector
                self.dire_detector = DIREDetector()
                logger.info("DIRE Diffusion detector loaded")
            except Exception as e:
                logger.warning(f"Could not load DIRE detector: {e}")
        else:
            logger.info("DIRE heuristic detector SKIPPED (weight=0)")
        
        if _should_load('npr'):
            try:
                from .npr_detector import NPRDetector
                self.npr_detector = NPRDetector()
                logger.info("NPR detector loaded")
            except Exception as e:
                logger.warning(f"Could not load NPR detector: {e}")
        else:
            logger.info("NPR detector SKIPPED (weight=0)")
        
        if _should_load('face'):
            try:
                from .face_consistency_detector import FaceConsistencyDetector
                self.face_detector = FaceConsistencyDetector()
                logger.info("Face Consistency detector loaded")
            except Exception as e:
                logger.warning(f"Could not load Face detector: {e}")
        else:
            logger.info("Face detector SKIPPED (weight=0)")
        
        if _should_load('edge'):
            try:
                from .edge_coherence_analyzer import EdgeCoherenceAnalyzer
                self.edge_analyzer = EdgeCoherenceAnalyzer()
                logger.info("Edge Coherence analyzer loaded")
            except Exception as e:
                logger.warning(f"Could not load Edge analyzer: {e}")
        else:
            logger.info("Edge analyzer SKIPPED (weight=0)")
        
        # ── Load ML models if requested ──
        if load_ml_models:
            try:
                from .ml_detector import create_ml_detectors
                ml_models = create_ml_detectors(device=self.device, load_all=True)
                
                # ── Frequency analyzer ──
                if _should_load('frequency'):
                    self.frequency_analyzer = ml_models.get('frequency_analyzer')
                    if self.frequency_analyzer:
                        logger.info("Frequency analyzer loaded via factory")
                        
                # ── Tier 1: Best recent detectors (2025-2026) ──
                if _should_load('ateeqq'):
                    self.ateeqq_detector = ml_models.get('ateeqq')
                    if self.ateeqq_detector and getattr(self.ateeqq_detector, 'model_loaded', False):
                        logger.info("Ateeqq SigLIP2 detector loaded (99.23% acc, Dec 2025)")
                        
                if _should_load('siglip_dinov2'):
                    self.siglip_dinov2_detector = ml_models.get('siglip_dinov2')
                    if self.siglip_dinov2_detector and getattr(self.siglip_dinov2_detector, 'model_loaded', False):
                        logger.info("SigLIP2+DINOv2 detector loaded (97.15% cross-dataset)")
                        
                # ── Tier 2: Strong recent models ──
                if _should_load('deepfake'):
                    self.deepfake_detector = ml_models.get('deepfake')
                    if self.deepfake_detector and getattr(self.deepfake_detector, 'model_loaded', False):
                        logger.info("Deepfake detector loaded (dima806, 98.25%)")
                        
                if _should_load('sdxl'):
                    self.sdxl_detector = ml_models.get('sdxl')
                    if self.sdxl_detector and getattr(self.sdxl_detector, 'model_loaded', False):
                        logger.info("SDXL detector loaded (Organika/sdxl-detector)")
                        
                if _should_load('dinov2'):
                    self.dinov2_detector = ml_models.get('dinov2')
                    if self.dinov2_detector and getattr(self.dinov2_detector, 'model_loaded', False):
                        logger.info("DINOv2 deepfake detector loaded")
                    
            except ImportError as e:
                logger.warning(f"ML detectors not available: {e}")
            except Exception as e:
                logger.warning(f"Error loading ML detectors: {e}")
    
    def detect(self, image_data: bytes, filename: str = "image") -> Dict[str, Any]:
        """
        Perform ensemble detection on an image.
        
        Args:
            image_data: Raw image bytes
            filename: Original filename for logging
            
        Returns:
            Comprehensive detection result with combined verdict
        """
        # Check if any ML models are active for analysis mode reporting
        ml_active = (self.ateeqq_detector and self.ateeqq_detector.model_loaded) or \
                    (self.deepfake_detector and self.deepfake_detector.model_loaded) or \
                    (self.sdxl_detector and self.sdxl_detector.model_loaded) or \
                    (self.siglip_dinov2_detector and self.siglip_dinov2_detector.model_loaded) or \
                    (self.dinov2_detector and self.dinov2_detector.model_loaded)

        result: Dict[str, Any] = {
            'success': True,
            'analysis_mode': 'Multi-Model Ensemble' if ml_active else 'Statistical Analysis (Fallback)',
            'filename': filename,
            'ensemble_verdict': None,
            'ai_probability': 50.0,
            'confidence': 50,
            'verdict_description': '',
            'individual_results': {},
            'score_breakdown': {},
            'detection_agreement': {},
            'overrides_applied': [],
            'recommendations': []
        }
        
        try:
            # Load image
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            img_array = np.array(image)
            result['dimensions'] = {'width': image.width, 'height': image.height}
            result['file_size_bytes'] = len(image_data)
            
            # Run all detectors
            scores = {}
            
            # 1. Statistical analysis
            if self.statistical_detector:
                stat_result = self.statistical_detector.detect(image_data, filename)
                result['individual_results']['statistical'] = stat_result
                scores['statistical'] = stat_result.get('ai_probability', 50)
            
            # 2. dima806 ViT detector (primary ML model - 98.25% accuracy)
            if self.deepfake_detector and self.deepfake_detector.model_loaded:
                df_result = self.deepfake_detector.predict(image)
                result['individual_results']['deepfake'] = df_result
                if df_result.get('success', False):
                    scores['deepfake'] = df_result.get('ai_probability', 50)
            
            # 3. Ateeqq SigLIP2 detector (99.23% accuracy, incredibly low FPR)
            if self.ateeqq_detector and self.ateeqq_detector.model_loaded:
                ateeqq_result = self.ateeqq_detector.predict(image)
                result['individual_results']['ateeqq'] = ateeqq_result
                if ateeqq_result.get('success', False):
                    scores['ateeqq'] = ateeqq_result.get('ai_probability', 50)
            
            # 4. SDXL Detector (Organika/sdxl-detector - modern diffusion specialist)
            if self.sdxl_detector and self.sdxl_detector.model_loaded:
                sdxl_result = self.sdxl_detector.predict(image)
                result['individual_results']['sdxl'] = sdxl_result
                if sdxl_result.get('success', False):
                    scores['sdxl'] = sdxl_result.get('ai_probability', 50)
            
            # 5. Frequency analysis
            if self.frequency_analyzer:
                freq_result = self.frequency_analyzer.analyze(image)
                result['individual_results']['frequency'] = freq_result
                scores['frequency'] = freq_result.get('ai_probability_contribution', 0)
            
            # 6. Watermark detection
            watermark_boost = 0
            if self.watermark_detector:
                wm_result = self.watermark_detector.analyze(image_data)
                result['individual_results']['watermark'] = wm_result
                
                if wm_result.get('watermark_detected'):
                    watermark_boost = wm_result.get('confidence', 70)
                    result['overrides_applied'].append(
                        f"AI watermark detected ({wm_result.get('watermark_type')})"
                    )
                scores['watermark'] = watermark_boost
            
            # 7. Metadata analysis
            if self.metadata_analyzer:
                meta_result = self.metadata_analyzer.analyze(image_data)
                result['individual_results']['metadata'] = meta_result
                
                # Adjust based on metadata
                if meta_result.get('ai_software_detected'):
                    result['overrides_applied'].append(
                        f"AI software detected in metadata: {meta_result.get('software_detected')}"
                    )
            
            # 8. C2PA/Content Credentials
            c2pa_override = False
            if self.c2pa_detector:
                c2pa_result = self.c2pa_detector.analyze(image_data, filename)
                result['individual_results']['c2pa'] = c2pa_result
                
                if c2pa_result.get('is_ai_generated'):
                    c2pa_override = True
                    result['overrides_applied'].append(
                        f"C2PA Content Credentials declare AI generation: {c2pa_result.get('ai_generator')}"
                    )
            
            # 9. ELA analysis (for manipulation detection)
            if self.ela_analyzer:
                ela_result = self.ela_analyzer.analyze(image_data)
                result['individual_results']['ela'] = ela_result
            
            # 11. NPR Neighboring Pixel Relationships (99.1% accuracy)
            if self.npr_detector:
                try:
                    npr_result = self.npr_detector.analyze(image)
                    result['individual_results']['npr'] = npr_result
                    if npr_result.get('success'):
                        scores['npr'] = npr_result.get('ai_probability', 50)
                except Exception as e:
                    logger.warning(f"NPR detector error: {e}")
            
            # 12. Face Consistency Analysis
            if self.face_detector:
                try:
                    face_result = self.face_detector.analyze(image)
                    result['individual_results']['face_consistency'] = face_result
                    if face_result.get('success') and face_result.get('applicable'):
                        scores['face'] = face_result.get('ai_probability', 50)
                except Exception as e:
                    logger.warning(f"Face detector error: {e}")
            
            # 13. Edge Coherence Analysis
            if self.edge_analyzer:
                try:
                    edge_result = self.edge_analyzer.analyze(image)
                    result['individual_results']['edge_coherence'] = edge_result
                    if edge_result.get('success'):
                        scores['edge'] = edge_result.get('ai_probability', 50)
                except Exception as e:
                    logger.warning(f"Edge analyzer error: {e}")
            
            # 14. Bombek1 SigLIP2+DINOv2 (best overall - 99.97% AUC, 25+ generators)
            # 15. DINOv2 Deepfake (degradation-resilient, best for social media)
            if self.dinov2_detector and self.dinov2_detector.model_loaded:
                try:
                    dinov2_result = self.dinov2_detector.predict(image)
                    result['individual_results']['dinov2'] = dinov2_result
                    if dinov2_result.get('success', False):
                        scores['dinov2'] = dinov2_result.get('ai_probability', 50)
                except Exception as e:
                    logger.warning(f"DINOv2 deepfake error: {e}")
            
            # ============================================================
            # NEW 2026 Phase 2: Five additional pre-trained ML detectors
            # ============================================================
            
            # 18. Bombek1 SigLIP2+DINOv2 (BEST OVERALL — 97.15% cross-dataset, 99.97% AUC)
            if self.siglip_dinov2_detector and self.siglip_dinov2_detector.model_loaded:
                try:
                    siglip_dinov2_result = self.siglip_dinov2_detector.predict(image)
                    result['individual_results']['siglip_dinov2'] = siglip_dinov2_result
                    if siglip_dinov2_result.get('success', False):
                        scores['siglip_dinov2'] = siglip_dinov2_result.get('ai_probability', 50)
                except Exception as e:
                    logger.warning(f"SigLIP2+DINOv2 detector error: {e}")
            
            # 19. DeepFake V2 (prithivMLmods, 2025 training data)
            if self.deepfake_v2_detector and self.deepfake_v2_detector.model_loaded:
                try:
                    dfv2_result = self.deepfake_v2_detector.predict(image)
                    result['individual_results']['deepfake_v2'] = dfv2_result
                    if dfv2_result.get('success', False):
                        scores['deepfake_v2'] = dfv2_result.get('ai_probability', 50)
                except Exception as e:
                    logger.warning(f"DeepFake V2 detector error: {e}")
            
            # 20. SigLIP Deepfake V1 (prithivMLmods, SigLIP backbone)
            if self.siglip_deepfake_detector and self.siglip_deepfake_detector.model_loaded:
                try:
                    siglip_df_result = self.siglip_deepfake_detector.predict(image)
                    result['individual_results']['siglip_deepfake'] = siglip_df_result
                    if siglip_df_result.get('success', False):
                        scores['siglip_deepfake'] = siglip_df_result.get('ai_probability', 50)
                except Exception as e:
                    logger.warning(f"SigLIP Deepfake detector error: {e}")
            
            # 21. Distilled ViT (jacoballessio, generalization specialist)
            if self.distilled_detector and self.distilled_detector.model_loaded:
                try:
                    distilled_result = self.distilled_detector.predict(image)
                    result['individual_results']['distilled'] = distilled_result
                    if distilled_result.get('success', False):
                        scores['distilled'] = distilled_result.get('ai_probability', 50)
                except Exception as e:
                    logger.warning(f"Distilled detector error: {e}")
            
            # 22. AIorNot (Nahrawy, diversity signal)
            if self.ai_or_not_detector and self.ai_or_not_detector.model_loaded:
                try:
                    aion_result = self.ai_or_not_detector.predict(image)
                    result['individual_results']['ai_or_not'] = aion_result
                    if aion_result.get('success', False):
                        scores['ai_or_not'] = aion_result.get('ai_probability', 50)
                except Exception as e:
                    logger.warning(f"AIorNot detector error: {e}")
            
            # Calculate weighted ensemble score
            final_score = self._calculate_ensemble_score(scores)
            result['score_breakdown'] = {
                'raw_scores': scores,
                'weights_used': self.weights,
                'weighted_score': final_score
            }
            
            # ── MAJORITY-VOTE SAFEGUARD ──
            # Prevents false AI classification when most real ML models vote "real".
            # Only considers models with non-zero weight to avoid heuristic bias.
            weighted_ml_votes = []
            for det_name, score in scores.items():
                w = self.weights.get(det_name, 0)
                if w > 0 and score is not None:
                    weighted_ml_votes.append(score)
            
            if len(weighted_ml_votes) >= 3:
                real_votes = sum(1 for s in weighted_ml_votes if s < 40)
                ai_votes = sum(1 for s in weighted_ml_votes if s >= 60)
                total = len(weighted_ml_votes)
                
                # If 60%+ of weighted models say REAL (< 40% AI), cap score
                if real_votes / total >= 0.6 and final_score > 55:
                    old_score = final_score
                    final_score = min(final_score, 45.0)
                    result['overrides_applied'].append(
                        f"Majority-vote safeguard: {real_votes}/{total} models vote real "
                        f"(score capped from {old_score:.1f}% to {final_score:.1f}%)"
                    )
                    logger.info(f"Majority-vote safeguard activated: {real_votes}/{total} "
                              f"models vote real, score {old_score:.1f}→{final_score:.1f}")
                
                # If 60%+ of weighted models say AI (>= 60% AI), ensure minimum score  
                elif ai_votes / total >= 0.6 and final_score < 55:
                    old_score = final_score
                    final_score = max(final_score, 60.0)
                    result['overrides_applied'].append(
                        f"Majority-vote boost: {ai_votes}/{total} models vote AI "
                        f"(score raised from {old_score:.1f}% to {final_score:.1f}%)"
                    )
            
            # Apply overrides
            if c2pa_override:
                final_score = 100.0
                result['overrides_applied'].append("C2PA override: AI generation declared")
            
            # FIXED: Only apply watermark override if it's a known AI signature
            # Generic watermarks (DWT-DCT with low confidence) should not override
            if watermark_boost > self.WATERMARK_OVERRIDE_THRESHOLD:
                # Check if it was a specific AI signature match
                watermark_result = result['individual_results'].get('watermark', {})
                if watermark_result.get('ai_generator_signature'):
                    final_score = max(final_score, 99.0)
                    result['overrides_applied'].append(f"AI Signature override: {watermark_result.get('ai_generator_signature')}")
                else:
                    # Generic watermark - boost score but don't fully override unless very high confidence
                    if watermark_boost >= 90:
                        final_score = max(final_score, 90.0)
                        result['overrides_applied'].append(f"High-confidence watermark override ({watermark_boost}%)")
                    else:
                        # Just a slight boost for generic watermarks
                        # Don't let it override a low AI score from other detectors
                        pass
            
            # Calculate detection agreement
            agreement_result = self._calculate_agreement(scores)
            result['detection_agreement'] = agreement_result
            
            # Apply agreement bonus/penalty
            agreement = agreement_result
            if agreement['agreement_level'] == 'STRONG':
                # High agreement → boost confidence
                pass  # Already reflected in weighted score
            elif agreement['agreement_level'] == 'WEAK':
                # Low agreement → reduce confidence slightly
                final_score = final_score * 0.95
            
            # Determine final verdict
            result['ai_probability'] = round(final_score, 2)
            result['confidence'] = self._calculate_confidence(scores, agreement)
            result['ensemble_verdict'], result['verdict_description'] = self._determine_verdict(final_score)
            
            # Generate recommendations
            result['recommendations'] = self._generate_recommendations(result)
            
            return self._sanitize_for_json(result)
            
        except Exception as e:
            logger.error(f"Ensemble detection error: {e}")
            return self._sanitize_for_json({
                'success': False,
                'error': str(e),
                'ai_probability': 50.0,
                'ensemble_verdict': 'ERROR',
                'verdict_description': f'Analysis failed: {str(e)}'
            })
    
    def _sanitize_for_json(self, obj: Any) -> Any:
        """
        Recursively convert numpy types to Python native types for JSON serialization.
        Needed because some detectors return numpy.float32/int64.
        """
        if isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_for_json(i) for i in obj]
        elif isinstance(obj, tuple):
            return tuple(self._sanitize_for_json(i) for i in obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    def _calculate_ensemble_score(self, scores: Dict[str, float]) -> float:
        """
        Calculate weighted ensemble score.
        
        Args:
            scores: Dict of detector name → AI probability
            
        Returns:
            Weighted average AI probability
        """
        total_weight: float = 0.0
        weighted_sum: float = 0.0
        
        for detector, score in scores.items():
            if detector in self.weights and score is not None:
                weight = self.weights[detector]
                # Skip zero-weighted detectors entirely — they should not
                # influence the final score under any circumstance.
                if weight <= 0:
                    continue
                weighted_sum += score * weight
                total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            # Fallback: only average scores from detectors with non-zero weight.
            # This prevents zero-weighted heuristic detectors from biasing the
            # result when they're the only ones that loaded.
            valid_scores = [
                s for det, s in scores.items()
                if s is not None and self.weights.get(det, 0) > 0
            ]
            return sum(valid_scores) / len(valid_scores) if valid_scores else 50.0
    
    def _calculate_agreement(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate agreement level between detectors.
        
        Returns analysis of how much detectors agree.
        """
        # Only consider detectors with non-zero weight for agreement
        valid_scores = [
            s for det, s in scores.items()
            if s is not None and s > 0 and self.weights.get(det, 0) > 0
        ]
        
        if len(valid_scores) < 2:
            return {
                'agreement_level': 'INSUFFICIENT_DATA',
                'std_deviation': 0,
                'detectors_agree_ai': 0,
                'detectors_agree_real': 0
            }
        
        std_dev = np.std(valid_scores)
        mean_score = np.mean(valid_scores)
        
        # Count how many think AI vs real
        ai_votes = sum(1 for s in valid_scores if s >= 50)
        real_votes = sum(1 for s in valid_scores if s < 50)
        
        # Determine agreement level
        if std_dev < 10:
            agreement = 'STRONG'
        elif std_dev < 20:
            agreement = 'MODERATE'
        else:
            agreement = 'WEAK'
        
        return {
            'agreement_level': agreement,
            'std_deviation': round(std_dev, 2),
            'mean_score': round(mean_score, 2),
            'detectors_agree_ai': ai_votes,
            'detectors_agree_real': real_votes,
            'total_detectors': len(valid_scores)
        }
    
    def _calculate_confidence(self, scores: Dict[str, float], agreement: Dict[str, Any]) -> int:
        """Calculate overall confidence in the verdict."""
        base_confidence = 50
        
        # More detectors → higher confidence
        num_detectors = agreement.get('total_detectors', 0)
        base_confidence += num_detectors * 5
        
        # Strong agreement → higher confidence
        if agreement['agreement_level'] == 'STRONG':
            base_confidence += 20
        elif agreement['agreement_level'] == 'MODERATE':
            base_confidence += 10
        
        # Extreme scores → higher confidence
        mean = agreement.get('mean_score', 50)
        if mean > 80 or mean < 20:
            base_confidence += 15
        elif mean > 70 or mean < 30:
            base_confidence += 10
        
        return min(100, max(0, base_confidence))
    
    def _determine_verdict(self, ai_probability: float) -> tuple:
        """Determine verdict based on AI probability."""
        if ai_probability >= 85:
            return 'AI_GENERATED', 'High confidence: This image is very likely AI-generated'
        elif ai_probability >= 70:
            return 'LIKELY_AI', 'Moderate-high confidence: This image shows strong signs of AI generation'
        elif ai_probability >= 55:
            return 'POSSIBLY_AI', 'Moderate confidence: This image may be AI-generated'
        elif ai_probability >= 45:
            return 'UNCERTAIN', 'Low confidence: Cannot determine with certainty'
        elif ai_probability >= 30:
            return 'POSSIBLY_REAL', 'Moderate confidence: This image may be authentic'
        elif ai_probability >= 15:
            return 'LIKELY_REAL', 'Moderate-high confidence: This image appears to be authentic'
        else:
            return 'REAL', 'High confidence: This image appears to be a real photograph'
    
    def _generate_recommendations(self, result: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        ai_prob = result.get('ai_probability', 50)
        agreement = result.get('detection_agreement', {})
        
        # Based on verdict certainty
        if agreement.get('agreement_level') == 'WEAK':
            recommendations.append(
                "Detection results show disagreement between methods. Consider additional verification."
            )
        
        if 40 <= ai_prob <= 60:
            recommendations.append(
                "Results are inconclusive. Try reverse image search or check original source."
            )
        
        # Based on what was detected
        watermark = result.get('individual_results', {}).get('watermark', {})
        if watermark.get('watermark_detected'):
            recommendations.append(
                f"AI watermark detected ({watermark.get('watermark_type')}). This strongly suggests AI generation."
            )
        
        c2pa = result.get('individual_results', {}).get('c2pa', {})
        if c2pa.get('has_content_credentials'):
            recommendations.append(
                "Image has Content Credentials (C2PA). Check provenance chain for editing history."
            )
        
        metadata = result.get('individual_results', {}).get('metadata', {})
        if not metadata.get('has_exif'):
            recommendations.append(
                "No EXIF metadata found. Real photos usually contain camera information."
            )
        
        if ai_prob >= 70:
            recommendations.append(
                "High AI probability detected. Verify with the original source before sharing."
            )
        
        return recommendations


def create_ensemble_detector(
    use_gpu: bool = False,
    load_ml_models: bool = True,
    weights: Optional[Dict[str, float]] = None
) -> EnsembleDetector:
    """
    Factory function to create an EnsembleDetector.
    
    Args:
        use_gpu: Whether to use GPU
        load_ml_models: Whether to load ML models
        weights: Custom weights
        
    Returns:
        EnsembleDetector instance
    """
    return EnsembleDetector(
        use_gpu=use_gpu,
        load_ml_models=load_ml_models,
        weights=weights
    )
