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
│  │  │Statistical│ │ NYUAD ViT │ │   CLIP    │ │Frequency│  │   │
│  │  │ Analysis  │ │ Detector  │ │ Universal │ │Analysis │  │   │
│  │  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └────┬────┘  │   │
│  │        │             │             │            │        │   │
│  │        ▼             ▼             ▼            ▼        │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │         Weighted Score Fusion Layer             │    │   │
│  │  │  statistical: 20% | nyuad: 25% | clip: 35%     │    │   │
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
    - NYUAD ViT (Vision Transformer - HuggingFace)
    - UniversalFakeDetect (CLIP-based)
    - Frequency domain analysis (GAN fingerprints)
    - Watermark detection
    - Content Credentials (C2PA)
    - Metadata forensics
    
    Uses weighted score fusion with confidence calibration.
    """
    
    # Default weights for each detector (sum to 1.0)
    DEFAULT_WEIGHTS = {
        'statistical': 0.20,    # Basic statistical analysis
        'nyuad': 0.25,          # NYUAD ViT detector
        'clip': 0.35,           # UniversalFakeDetect (CLIP)
        'frequency': 0.10,      # FFT/DCT analysis
        'watermark': 0.10,      # Watermark detection contribution
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
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
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
        
        # Load detectors
        self._initialize_detectors(load_ml_models)
        
        logger.info(f"EnsembleDetector initialized (GPU: {use_gpu}, ML models: {load_ml_models})")
    
    def _initialize_detectors(self, load_ml_models: bool):
        """Initialize all component detectors."""
        
        # Always load lightweight detectors
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
        
        # Load ML models if requested
        if load_ml_models:
            try:
                from .ml_detector import NYUADDetector, UniversalFakeDetector, FrequencyAnalyzer, DeepfakeDetector
                
                # Frequency analyzer (lightweight)
                self.frequency_analyzer = FrequencyAnalyzer()
                logger.info("Frequency analyzer loaded")
                
                # NYUAD ViT (primary ML detector)
                self.nyuad_detector = NYUADDetector(device=self.device)
                if self.nyuad_detector.model_loaded:
                    logger.info("NYUAD ViT detector loaded")
                else:
                    logger.warning("NYUAD detector failed to load model")
                
                # UniversalFakeDetect (CLIP-based)
                self.clip_detector = UniversalFakeDetector(device=self.device)
                if self.clip_detector.model_loaded:
                    logger.info("UniversalFakeDetect (CLIP) loaded")
                else:
                    logger.warning("CLIP detector failed to load model")
                
                # Deepfake detector (optional, for face images)
                self.deepfake_detector = DeepfakeDetector(device=self.device)
                if self.deepfake_detector.model_loaded:
                    logger.info("Deepfake detector loaded")
                    
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
        result = {
            'success': True,
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
            
            # 2. NYUAD ViT detector
            if self.nyuad_detector and self.nyuad_detector.model_loaded:
                nyuad_result = self.nyuad_detector.predict(image)
                result['individual_results']['nyuad'] = nyuad_result
                scores['nyuad'] = nyuad_result.get('ai_probability', 50)
            
            # 3. UniversalFakeDetect (CLIP)
            if self.clip_detector and self.clip_detector.model_loaded:
                clip_result = self.clip_detector.predict(image)
                result['individual_results']['clip'] = clip_result
                scores['clip'] = clip_result.get('ai_probability', 50)
            
            # 4. Frequency analysis
            if self.frequency_analyzer:
                freq_result = self.frequency_analyzer.analyze(image)
                result['individual_results']['frequency'] = freq_result
                scores['frequency'] = freq_result.get('ai_probability_contribution', 0)
            
            # 5. Watermark detection
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
            
            # 6. Metadata analysis
            if self.metadata_analyzer:
                meta_result = self.metadata_analyzer.analyze(image_data)
                result['individual_results']['metadata'] = meta_result
                
                # Adjust based on metadata
                if meta_result.get('ai_software_detected'):
                    result['overrides_applied'].append(
                        f"AI software detected in metadata: {meta_result.get('software_detected')}"
                    )
            
            # 7. C2PA/Content Credentials
            c2pa_override = False
            if self.c2pa_detector:
                c2pa_result = self.c2pa_detector.analyze(image_data, filename)
                result['individual_results']['c2pa'] = c2pa_result
                
                if c2pa_result.get('is_ai_generated'):
                    c2pa_override = True
                    result['overrides_applied'].append(
                        f"C2PA Content Credentials declare AI generation: {c2pa_result.get('ai_generator')}"
                    )
            
            # 8. ELA analysis (for manipulation detection)
            if self.ela_analyzer:
                ela_result = self.ela_analyzer.analyze(image_data)
                result['individual_results']['ela'] = ela_result
            
            # 9. Deepfake detection (if faces present)
            if self.deepfake_detector and self.deepfake_detector.model_loaded:
                df_result = self.deepfake_detector.predict(image)
                result['individual_results']['deepfake'] = df_result
                
                if df_result.get('has_face') and df_result.get('deepfake_probability', 0) > 70:
                    result['overrides_applied'].append(
                        f"Deepfake indicators detected (confidence: {df_result.get('deepfake_probability')}%)"
                    )
            
            # Calculate weighted ensemble score
            final_score = self._calculate_ensemble_score(scores)
            result['score_breakdown'] = {
                'raw_scores': scores,
                'weights_used': self.weights,
                'weighted_score': final_score
            }
            
            # Apply overrides
            if c2pa_override:
                final_score = 100.0
                result['overrides_applied'].append("C2PA override: AI generation declared")
            
            if watermark_boost > self.WATERMARK_OVERRIDE_THRESHOLD:
                final_score = max(final_score, 95.0)
                result['overrides_applied'].append(f"Watermark override: confidence {watermark_boost}%")
            
            # Calculate detection agreement
            result['detection_agreement'] = self._calculate_agreement(scores)
            
            # Apply agreement bonus/penalty
            agreement = result['detection_agreement']
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
            
            return result
            
        except Exception as e:
            logger.error(f"Ensemble detection error: {e}")
            return {
                'success': False,
                'error': str(e),
                'ai_probability': 50.0,
                'ensemble_verdict': 'ERROR',
                'verdict_description': f'Analysis failed: {str(e)}'
            }
    
    def _calculate_ensemble_score(self, scores: Dict[str, float]) -> float:
        """
        Calculate weighted ensemble score.
        
        Args:
            scores: Dict of detector name → AI probability
            
        Returns:
            Weighted average AI probability
        """
        total_weight = 0
        weighted_sum = 0
        
        for detector, score in scores.items():
            if detector in self.weights and score is not None:
                weight = self.weights[detector]
                weighted_sum += score * weight
                total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            # Fallback: simple average of available scores
            valid_scores = [s for s in scores.values() if s is not None]
            return sum(valid_scores) / len(valid_scores) if valid_scores else 50.0
    
    def _calculate_agreement(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate agreement level between detectors.
        
        Returns analysis of how much detectors agree.
        """
        valid_scores = [s for s in scores.values() if s is not None and s > 0]
        
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
