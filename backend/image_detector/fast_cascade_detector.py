"""
VisioNova Fast Cascade Detector
Implements progressive detection for 3-5x faster average response time.

Strategy:
1. Quick lightweight checks first (frequency analysis, statistical)
2. If result is clear (>90% or <10%), return immediately
3. Only run full ML ensemble for uncertain cases
4. Use FP16 inference for 2x speedup on supported hardware

Performance Target:
- Clear cases: 50-100ms (vs 400-600ms with full ensemble)
- Uncertain cases: Same as before (all models run)
- Average 3-5x faster across typical workloads
"""

import io
import logging
import time
from typing import Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class FastCascadeDetector:
    """
    Fast cascading AI image detector.
    
    Uses progressive detection:
    1. Stage 1: Quick statistical checks (~20ms)
    2. Stage 2: Single fast ML model (~100ms)
    3. Stage 3: Full ensemble (only if uncertain) (~400ms)
    
    Typical speedup: 3-5x for clear cases
    """
    
    # Confidence thresholds for early exit
    CONFIDENT_AI_THRESHOLD = 90.0      # >90% = definitely AI, skip remaining
    CONFIDENT_REAL_THRESHOLD = 10.0    # <10% = definitely real, skip remaining
    UNCERTAIN_LOW = 30.0               # 30-70% = uncertain, run more models
    UNCERTAIN_HIGH = 70.0
    
    # Enable FP16 inference when available
    USE_FP16 = True
    
    def __init__(self, use_gpu: bool = False, enable_fp16: bool = True):
        """
        Initialize the fast cascade detector.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            enable_fp16: Whether to use FP16 half-precision inference
        """
        self.use_gpu = use_gpu
        self.enable_fp16 = enable_fp16 and use_gpu  # FP16 only effective on GPU
        self.device = "cuda" if use_gpu else "cpu"
        
        # Lazy-loaded detectors (only load when needed)
        self._frequency_analyzer = None
        self._fast_detector = None  # Single fast model for stage 2
        self._full_ensemble = None  # Full ensemble for stage 3
        
        # Performance metrics
        self.stats = {
            'total_detections': 0,
            'early_exits_stage1': 0,
            'early_exits_stage2': 0,
            'full_ensemble_runs': 0,
            'avg_time_ms': 0.0
        }
        
        logger.info(f"FastCascadeDetector initialized (GPU: {use_gpu}, FP16: {self.enable_fp16})")
    
    @property
    def frequency_analyzer(self):
        """Lazy load frequency analyzer."""
        if self._frequency_analyzer is None:
            try:
                from .ml_detector import FrequencyAnalyzer
                self._frequency_analyzer = FrequencyAnalyzer()
            except ImportError:
                logger.warning("FrequencyAnalyzer not available")
        return self._frequency_analyzer
    
    @property
    def fast_detector(self):
        """Lazy load the fast single-model detector (deepfake detector - fastest)."""
        if self._fast_detector is None:
            try:
                from .ml_detector import DeepfakeDetector
                self._fast_detector = DeepfakeDetector(device=self.device)
                
                # Apply FP16 optimization if available
                if self.enable_fp16 and self._fast_detector.model_loaded:
                    self._apply_fp16(self._fast_detector)
                    
            except ImportError:
                logger.warning("DeepfakeDetector not available")
        return self._fast_detector
    
    @property
    def full_ensemble(self):
        """Lazy load full ensemble detector."""
        if self._full_ensemble is None:
            try:
                from .ensemble_detector import EnsembleDetector
                self._full_ensemble = EnsembleDetector(
                    use_gpu=self.use_gpu,
                    load_ml_models=True
                )
            except ImportError:
                logger.warning("EnsembleDetector not available")
        return self._full_ensemble
    
    def _apply_fp16(self, detector) -> bool:
        """
        Apply FP16 half-precision optimization to a detector.
        
        Args:
            detector: ML detector with a model attribute
            
        Returns:
            True if FP16 was applied successfully
        """
        try:
            import torch
            
            if hasattr(detector, 'model') and detector.model is not None:
                if torch.cuda.is_available():
                    detector.model = detector.model.half()
                    logger.info(f"Applied FP16 to {detector.__class__.__name__}")
                    return True
        except Exception as e:
            logger.debug(f"Could not apply FP16: {e}")
        return False
    
    def detect(self, image_data: bytes, filename: str = "image") -> Dict[str, Any]:
        """
        Perform fast cascading detection.
        
        Stages:
        1. Quick statistical/frequency analysis (~20ms)
        2. Single fast ML model (~100ms)
        3. Full ensemble (only if uncertain)
        
        Args:
            image_data: Raw image bytes
            filename: Original filename
            
        Returns:
            Detection result with timing and stage information
        """
        start_time = time.time()
        
        result = {
            'success': True,
            'filename': filename,
            'ai_probability': 50.0,
            'confidence': 50,
            'verdict': 'UNCERTAIN',
            'verdict_description': '',
            'cascade_stage': 0,
            'stages_run': [],
            'timing_ms': {}
        }
        
        try:
            # Load image
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            result['dimensions'] = {'width': image.width, 'height': image.height}
            
            # ========== STAGE 1: Quick Statistical Analysis ==========
            stage1_start = time.time()
            stage1_result = self._run_stage1(image, image_data)
            stage1_time = (time.time() - stage1_start) * 1000
            
            result['stages_run'].append('statistical')
            result['timing_ms']['stage1'] = round(stage1_time, 1)
            result['stage1_result'] = stage1_result
            
            # Check for early exit
            stage1_prob = stage1_result.get('ai_probability', 50.0)
            
            if stage1_prob >= self.CONFIDENT_AI_THRESHOLD:
                # Clear AI detection - early exit
                result['ai_probability'] = stage1_prob
                result['cascade_stage'] = 1
                result['verdict'] = 'AI_GENERATED'
                result['verdict_description'] = 'Quick detection: Clear AI indicators found'
                self.stats['early_exits_stage1'] += 1
                return self._finalize_result(result, start_time)
            
            if stage1_prob <= self.CONFIDENT_REAL_THRESHOLD:
                # Clear real image - early exit
                result['ai_probability'] = stage1_prob
                result['cascade_stage'] = 1
                result['verdict'] = 'REAL'
                result['verdict_description'] = 'Quick detection: No AI indicators found'
                self.stats['early_exits_stage1'] += 1
                return self._finalize_result(result, start_time)
            
            # ========== STAGE 2: Fast ML Model ==========
            stage2_start = time.time()
            stage2_result = self._run_stage2(image)
            stage2_time = (time.time() - stage2_start) * 1000
            
            result['stages_run'].append('fast_ml')
            result['timing_ms']['stage2'] = round(stage2_time, 1)
            result['stage2_result'] = stage2_result
            
            # Combine stage 1 and 2 results (weighted)
            if stage2_result.get('success'):
                stage2_prob = stage2_result.get('ai_probability', 50.0)
                combined_prob = stage1_prob * 0.3 + stage2_prob * 0.7
                
                if combined_prob >= self.CONFIDENT_AI_THRESHOLD:
                    result['ai_probability'] = combined_prob
                    result['cascade_stage'] = 2
                    result['verdict'] = 'AI_GENERATED'
                    result['verdict_description'] = 'Fast detection: AI generation confirmed'
                    self.stats['early_exits_stage2'] += 1
                    return self._finalize_result(result, start_time)
                
                if combined_prob <= self.CONFIDENT_REAL_THRESHOLD:
                    result['ai_probability'] = combined_prob
                    result['cascade_stage'] = 2
                    result['verdict'] = 'REAL'
                    result['verdict_description'] = 'Fast detection: Natural image confirmed'
                    self.stats['early_exits_stage2'] += 1
                    return self._finalize_result(result, start_time)
                
                # Check if we're in the "pretty sure" territory
                if combined_prob >= self.UNCERTAIN_HIGH or combined_prob <= self.UNCERTAIN_LOW:
                    result['ai_probability'] = combined_prob
                    result['cascade_stage'] = 2
                    result['verdict'], result['verdict_description'] = self._determine_verdict(combined_prob)
                    self.stats['early_exits_stage2'] += 1
                    return self._finalize_result(result, start_time)
            
            # ========== STAGE 3: Full Ensemble (Uncertain Cases Only) ==========
            stage3_start = time.time()
            stage3_result = self._run_stage3(image_data, filename)
            stage3_time = (time.time() - stage3_start) * 1000
            
            result['stages_run'].append('full_ensemble')
            result['timing_ms']['stage3'] = round(stage3_time, 1)
            result['cascade_stage'] = 3
            self.stats['full_ensemble_runs'] += 1
            
            # Use full ensemble result
            if stage3_result.get('success'):
                result['ai_probability'] = stage3_result.get('ai_probability', 50.0)
                result['confidence'] = stage3_result.get('confidence', 50)
                result['verdict'] = stage3_result.get('ensemble_verdict', 'UNCERTAIN')
                result['verdict_description'] = stage3_result.get('verdict_description', '')
                result['ensemble_details'] = stage3_result.get('individual_results', {})
            
            return self._finalize_result(result, start_time)
            
        except Exception as e:
            logger.error(f"Cascade detection error: {e}")
            return {
                'success': False,
                'error': str(e),
                'ai_probability': 50.0,
                'verdict': 'ERROR',
                'verdict_description': f'Analysis failed: {str(e)}'
            }
    
    def _run_stage1(self, image: Image.Image, image_data: bytes) -> Dict[str, Any]:
        """
        Stage 1: Quick statistical and frequency analysis.
        
        Checks:
        - Frequency domain patterns (GAN fingerprints)
        - Noise consistency
        - C2PA/watermark quick check
        
        Target time: <30ms
        """
        result = {'success': True, 'ai_probability': 50.0}
        signals = []
        
        try:
            # 1. Frequency analysis (fast)
            if self.frequency_analyzer:
                freq_result = self.frequency_analyzer.analyze(image)
                freq_prob = freq_result.get('ai_probability_contribution', 0)
                if freq_prob > 0:
                    signals.append(('frequency', freq_prob))
            
            # 2. Quick noise check
            noise_prob = self._quick_noise_check(np.array(image))
            signals.append(('noise', noise_prob))
            
            # 3. Check for C2PA/watermark indicators (instant if present)
            try:
                from .content_credentials import ContentCredentialsDetector
                c2pa = ContentCredentialsDetector()
                c2pa_result = c2pa.analyze(image_data, "quick_check")
                if c2pa_result.get('is_ai_generated'):
                    # C2PA says AI - very high confidence
                    result['ai_probability'] = 98.0
                    result['c2pa_detected'] = True
                    return result
            except:
                pass
            
            # Combine signals
            if signals:
                total_weight = sum(1 for _ in signals)
                weighted_sum = sum(s[1] for s in signals)
                result['ai_probability'] = weighted_sum / total_weight
                result['signals'] = signals
            
        except Exception as e:
            logger.debug(f"Stage 1 error: {e}")
        
        return result
    
    def _quick_noise_check(self, img_array: np.ndarray) -> float:
        """
        Quick noise consistency check.
        
        AI images often have unnaturally uniform noise.
        Returns AI probability 0-100.
        """
        try:
            from scipy import ndimage
            
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = np.mean(img_array, axis=2)
            else:
                gray = img_array
            
            # Quick Laplacian check
            laplacian = ndimage.laplace(gray.astype(float))
            noise_std = np.std(laplacian)
            
            # Real photos: noise_std typically 5-30
            # AI images: often <5 (too clean) or artificial patterns
            if noise_std < 3:
                return 80.0  # Very clean = likely AI
            elif noise_std < 8:
                return 55.0
            elif noise_std < 20:
                return 30.0
            else:
                return 20.0
                
        except:
            return 50.0
    
    def _run_stage2(self, image: Image.Image) -> Dict[str, Any]:
        """
        Stage 2: Single fast ML model (dima806 ViT - 98.25% accuracy).
        
        Target time: <150ms
        """
        result = {'success': False, 'ai_probability': 50.0}
        
        try:
            if self.fast_detector and self.fast_detector.model_loaded:
                ml_result = self.fast_detector.predict(image)
                if ml_result.get('success'):
                    result['success'] = True
                    result['ai_probability'] = ml_result.get('ai_probability', 50.0)
                    result['model'] = 'dima806-vit'
                    result['fp16_enabled'] = self.enable_fp16
        except Exception as e:
            logger.debug(f"Stage 2 error: {e}")
        
        return result
    
    def _run_stage3(self, image_data: bytes, filename: str) -> Dict[str, Any]:
        """
        Stage 3: Full ensemble detection.
        
        Only runs for uncertain cases (30-70% confidence from stages 1-2).
        """
        try:
            if self.full_ensemble:
                return self.full_ensemble.detect(image_data, filename)
        except Exception as e:
            logger.error(f"Stage 3 error: {e}")
        
        return {'success': False}
    
    def _determine_verdict(self, ai_probability: float) -> Tuple[str, str]:
        """Determine verdict from AI probability."""
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
    
    def _finalize_result(self, result: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """Finalize result with timing and stats."""
        total_time = (time.time() - start_time) * 1000
        result['timing_ms']['total'] = round(total_time, 1)
        
        # Update stats
        self.stats['total_detections'] += 1
        self.stats['avg_time_ms'] = (
            (self.stats['avg_time_ms'] * (self.stats['total_detections'] - 1) + total_time)
            / self.stats['total_detections']
        )
        
        # Set verdict if not already set
        if result.get('verdict') == 'UNCERTAIN' and 'ai_probability' in result:
            result['verdict'], result['verdict_description'] = self._determine_verdict(
                result['ai_probability']
            )
        
        # Add performance info
        result['performance'] = {
            'cascade_stage': result.get('cascade_stage', 3),
            'early_exit': result.get('cascade_stage', 3) < 3,
            'speedup_factor': self._estimate_speedup(result)
        }
        
        return result
    
    def _estimate_speedup(self, result: Dict[str, Any]) -> float:
        """Estimate speedup compared to full ensemble."""
        stage = result.get('cascade_stage', 3)
        if stage == 1:
            return 10.0  # ~50ms vs 500ms
        elif stage == 2:
            return 3.0   # ~150ms vs 500ms
        else:
            return 1.0   # No speedup for uncertain cases
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        total = self.stats['total_detections']
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'early_exit_rate': round(
                (self.stats['early_exits_stage1'] + self.stats['early_exits_stage2']) / total * 100, 1
            ),
            'avg_speedup': round(
                (self.stats['early_exits_stage1'] * 10 + 
                 self.stats['early_exits_stage2'] * 3 + 
                 self.stats['full_ensemble_runs'] * 1) / total, 2
            )
        }


def create_fast_detector(use_gpu: bool = False, enable_fp16: bool = True) -> FastCascadeDetector:
    """
    Factory function to create a FastCascadeDetector.
    
    Args:
        use_gpu: Whether to use GPU acceleration
        enable_fp16: Whether to use FP16 half-precision
        
    Returns:
        FastCascadeDetector instance
    """
    return FastCascadeDetector(use_gpu=use_gpu, enable_fp16=enable_fp16)
