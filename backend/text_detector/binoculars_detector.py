"""
Binoculars Zero-Shot AI Text Detector

Uses dual Falcon-7B models to detect AI-generated text without training.
Works by comparing perplexity scores between two identical models:
- AI text: Low perplexity (predictable, consistent)
- Human text: High perplexity (creative, variable)

Requirements:
- 14GB+ VRAM (GPU required)
- Works on ANY AI model output (future-proof)

Reference: https://github.com/ahans30/Binoculars
"""

import logging
from typing import Dict, Optional
import torch

logger = logging.getLogger(__name__)


class BinocularsDetector:
    """Zero-shot AI text detector using dual Falcon-7B models."""
    
    def __init__(self):
        """Initialize Binoculars detector with GPU check."""
        self.detector = None
        self.device = None
        self._initialize()
    
    def _initialize(self) -> bool:
        """
        Initialize the Binoculars detector.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Check for GPU availability
            if not torch.cuda.is_available():
                logger.warning("GPU not available. Binoculars requires 14GB+ VRAM.")
                return False
            
            self.device = "cuda"
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
            
            if gpu_memory < 14:
                logger.warning(f"GPU has only {gpu_memory:.1f}GB VRAM. Binoculars needs 14GB+.")
                logger.warning("Detection may fail or be very slow.")
            
            # Import and initialize Binoculars
            from binoculars import Binoculars
            
            logger.info("Loading Binoculars dual Falcon-7B models...")
            self.detector = Binoculars()
            logger.info("Binoculars initialized successfully")
            
            return True
            
        except ImportError:
            logger.error("Binoculars package not installed. Run: pip install binoculars")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Binoculars: {str(e)}")
            return False
    
    def is_available(self) -> bool:
        """
        Check if Binoculars detector is available.
        
        Returns:
            bool: True if detector is loaded and ready
        """
        return self.detector is not None
    
    def detect(self, text: str) -> Optional[Dict]:
        """
        Detect if text is AI-generated using Binoculars.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with detection results or None if detection failed:
                - prediction: "AI" or "Human"
                - confidence: 0.0 to 1.0
                - score: Raw Binoculars score (lower = more AI-like)
                - method: "binoculars_zero_shot"
        """
        if not self.is_available():
            logger.error("Binoculars detector not available")
            return None
        
        if not text or len(text.strip()) < 50:
            logger.warning("Text too short for reliable Binoculars detection (min 50 chars)")
            return None
        
        try:
            # Run Binoculars detection
            result = self.detector.predict(text)
            
            # Extract score and prediction
            # Binoculars returns a dict with 'score' and optionally 'label'
            if isinstance(result, dict):
                score = result.get('score', result.get('binoculars_score', 0.0))
            else:
                # Handle case where result is just a float score
                score = float(result)
            
            # Lower score = more AI-like
            # Threshold: typically around 0.5 (can be tuned)
            threshold = 0.5
            is_ai = score < threshold
            
            # Calculate confidence based on distance from threshold
            # The further from threshold, the higher the confidence
            distance = abs(score - threshold)
            confidence = min(0.5 + distance, 0.99)  # Scale to 0.5-0.99
            
            prediction = "AI" if is_ai else "Human"
            
            logger.info(f"Binoculars detection: {prediction} (score={score:.4f}, confidence={confidence:.2f})")
            
            return {
                "prediction": prediction,
                "confidence": float(confidence),
                "score": float(score),
                "threshold": threshold,
                "method": "binoculars_zero_shot",
                "model": "dual_falcon_7b"
            }
            
        except Exception as e:
            logger.error(f"Binoculars detection failed: {str(e)}")
            return None
    
    def get_info(self) -> Dict:
        """
        Get information about the Binoculars detector.
        
        Returns:
            Dict with detector information
        """
        info = {
            "available": self.is_available(),
            "requires_gpu": True,
            "min_vram_gb": 14,
            "model": "Falcon-7B (dual)",
            "method": "Zero-shot perplexity comparison",
            "advantages": [
                "No training required",
                "Works on any AI model output",
                "Future-proof (detects GPT-5, Claude 4, etc.)",
                "90%+ accuracy at 0.01% FPR"
            ],
            "limitations": [
                "Requires 14GB+ GPU VRAM",
                "Slower than lightweight models",
                "Cannot run on CPU"
            ]
        }
        
        if self.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return info
    
    def cleanup(self):
        """Clean up GPU memory."""
        if self.detector is not None:
            del self.detector
            self.detector = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory cleared")


# Singleton instance
_binoculars_instance: Optional[BinocularsDetector] = None


def get_binoculars_detector() -> BinocularsDetector:
    """
    Get or create singleton Binoculars detector instance.
    
    Returns:
        BinocularsDetector instance
    """
    global _binoculars_instance
    
    if _binoculars_instance is None:
        _binoculars_instance = BinocularsDetector()
    
    return _binoculars_instance


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    detector = get_binoculars_detector()
    
    print("\n=== Binoculars Detector Info ===")
    info = detector.get_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    
    if detector.is_available():
        # Test with sample texts
        ai_text = """
        It's important to note that artificial intelligence has fundamentally transformed 
        the landscape of modern technology. Furthermore, machine learning algorithms 
        leverage vast datasets to optimize performance metrics. In conclusion, the 
        integration of neural networks represents a paradigm shift in computational capabilities.
        """
        
        human_text = """
        I was walking down the street yesterday when I saw the funniest thing - a dog 
        wearing sunglasses! Made my whole day honestly. Sometimes it's the little 
        unexpected moments that matter most, you know?
        """
        
        print("\n=== Testing AI Text ===")
        result = detector.detect(ai_text)
        if result:
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Score: {result['score']:.4f}")
        
        print("\n=== Testing Human Text ===")
        result = detector.detect(human_text)
        if result:
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Score: {result['score']:.4f}")
    else:
        print("\n⚠️  Binoculars not available (GPU required)")
