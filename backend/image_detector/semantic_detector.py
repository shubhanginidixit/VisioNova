"""
VisioNova Semantic Plausibility Detector
Uses Groq's free LLaVA vision API for "common sense" AI detection.

Detects physically impossible or implausible scenarios:
- Flying objects that can't fly (dogs, cars, etc.)
- Anatomical errors (extra fingers, distorted limbs)
- Text anomalies (gibberish, misspellings)
- Perspective/shadow inconsistencies
- Merged or impossible objects
"""

import base64
import io
import json
import logging
import os
from typing import Dict, Any, Optional, List
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticPlausibilityDetector:
    """
    Semantic Plausibility Detector using Groq's LLaVA vision model.
    
    Uses "common sense" reasoning to detect AI-generated images by
    identifying physically impossible or implausible content.
    
    Free tier limits:
    - 30 requests/minute
    - 14,400 requests/day
    - 4MB max image size (base64)
    """
    
    # Groq API endpoint
    API_URL = "https://api.groq.com/openai/v1/chat/completions"
    
    # Available vision models (in preference order)
    VISION_MODELS = [
        "llama-3.2-90b-vision-preview",  # Best quality
        "llama-3.2-11b-vision-preview",  # Faster
        "llava-v1.5-7b-4096-preview",    # Fallback
    ]
    
    # Plausibility analysis prompt
    ANALYSIS_PROMPT = """You are an expert image analyst detecting AI-generated images.

Analyze this image for physical impossibilities and AI artifacts. Look for:

1. **Physical Impossibilities**: Objects defying physics (flying dogs, floating cars, impossible gravity)
2. **Anatomical Errors**: Extra fingers, missing limbs, distorted body parts, wrong proportions
3. **Text Anomalies**: Gibberish text, misspelled signs, nonsensical writing
4. **Perspective Errors**: Wrong shadows, impossible angles, inconsistent lighting
5. **Object Merging**: Fused objects, impossible spatial relationships, blended items
6. **Texture Issues**: Unrealistic skin, plastic-looking materials, over-smooth surfaces

Respond in this exact JSON format:
{
    "plausibility_score": <0-100 where 100 is perfectly realistic>,
    "is_likely_ai": <true/false>,
    "confidence": <0-100>,
    "issues_found": [
        {"category": "<category>", "description": "<specific issue>", "severity": "<low/medium/high>"}
    ],
    "explanation": "<brief explanation of your assessment>"
}

Be strict but fair. Real photos should score 85-100. AI images with errors should score lower."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the semantic plausibility detector.
        
        Args:
            api_key: Groq API key. If not provided, reads from GROQ_IMAGE_API_KEY or GROQ_API_KEY env var.
        """
        # Use same env vars as image_explainer.py for consistency
        self.api_key = api_key or os.getenv("GROQ_IMAGE_API_KEY") or os.getenv("GROQ_API_KEY")
        self.model = self.VISION_MODELS[0]  # Use best model by default
        self.available = self.api_key is not None
        
        if not self.available:
            logger.warning("GROQ_API_KEY not set. Semantic plausibility detection disabled.")
            logger.info("Get a free API key at https://console.groq.com")
    
    def analyze(self, image_data: bytes, max_size_mb: float = 4.0) -> Dict[str, Any]:
        """
        Analyze an image for semantic plausibility.
        
        Args:
            image_data: Raw image bytes
            max_size_mb: Maximum image size in MB (default 4MB for Groq)
            
        Returns:
            dict with plausibility analysis results
        """
        if not self.available:
            return {
                'success': False,
                'error': 'GROQ_API_KEY not configured',
                'plausibility_score': 50,
                'is_likely_ai': None
            }
        
        try:
            # Resize image if too large
            image_base64 = self._prepare_image(image_data, max_size_mb)
            
            # Call Groq API
            result = self._call_groq_api(image_base64)
            
            return result
            
        except Exception as e:
            logger.error(f"Semantic analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'plausibility_score': 50,
                'is_likely_ai': None
            }
    
    def _prepare_image(self, image_data: bytes, max_size_mb: float) -> str:
        """
        Prepare image for API: resize if needed and convert to base64.
        """
        # Check size
        size_mb = len(image_data) / (1024 * 1024)
        
        if size_mb <= max_size_mb:
            return base64.b64encode(image_data).decode('utf-8')
        
        # Resize image to fit within size limit
        logger.info(f"Resizing image from {size_mb:.2f}MB to fit {max_size_mb}MB limit")
        
        image = Image.open(io.BytesIO(image_data))
        
        # Calculate resize factor
        factor = (max_size_mb / size_mb) ** 0.5
        new_size = (int(image.width * factor * 0.9), int(image.height * factor * 0.9))
        
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to JPEG for smaller size
        buffer = io.BytesIO()
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
        image.save(buffer, format='JPEG', quality=85)
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def _call_groq_api(self, image_base64: str) -> Dict[str, Any]:
        """
        Call Groq API with the image for analysis.
        """
        import requests
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Try models in order of preference
        for model in self.VISION_MODELS:
            try:
                payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": self.ANALYSIS_PROMPT
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_base64}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.1  # Low temperature for consistent analysis
                }
                
                response = requests.post(
                    self.API_URL,
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    return self._parse_response(response.json(), model)
                elif response.status_code == 429:
                    logger.warning(f"Rate limited on {model}, trying next model...")
                    continue
                elif response.status_code == 400:
                    # Model might not support vision, try next
                    logger.debug(f"Model {model} failed, trying next...")
                    continue
                else:
                    error = response.json().get('error', {}).get('message', str(response.status_code))
                    logger.warning(f"Groq API error: {error}")
                    continue
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on {model}, trying next...")
                continue
            except Exception as e:
                logger.warning(f"Error with {model}: {e}")
                continue
        
        # All models failed
        return {
            'success': False,
            'error': 'All vision models failed',
            'plausibility_score': 50,
            'is_likely_ai': None
        }
    
    def _parse_response(self, response: dict, model: str) -> Dict[str, Any]:
        """
        Parse Groq API response and extract analysis results.
        """
        try:
            content = response['choices'][0]['message']['content']
            
            # Try to extract JSON from response
            # Sometimes the model wraps JSON in markdown code blocks
            if '```json' in content:
                json_start = content.find('```json') + 7
                json_end = content.find('```', json_start)
                content = content[json_start:json_end].strip()
            elif '```' in content:
                json_start = content.find('```') + 3
                json_end = content.find('```', json_start)
                content = content[json_start:json_end].strip()
            
            # Parse JSON
            analysis = json.loads(content)
            
            return {
                'success': True,
                'plausibility_score': analysis.get('plausibility_score', 50),
                'is_likely_ai': analysis.get('is_likely_ai', None),
                'confidence': analysis.get('confidence', 50),
                'issues_found': analysis.get('issues_found', []),
                'explanation': analysis.get('explanation', ''),
                'model_used': model,
                'raw_response': content
            }
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            # Try to extract key information from text
            return self._parse_text_response(content, model)
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            return {
                'success': False,
                'error': f'Response parsing failed: {e}',
                'plausibility_score': 50,
                'is_likely_ai': None
            }
    
    def _parse_text_response(self, content: str, model: str) -> Dict[str, Any]:
        """
        Fallback parser for non-JSON responses.
        """
        content_lower = content.lower()
        
        # Detect AI likelihood from keywords
        ai_keywords = ['ai-generated', 'artificial', 'fake', 'impossible', 'error', 'anomaly']
        real_keywords = ['realistic', 'authentic', 'genuine', 'natural', 'real photo']
        
        ai_score = sum(1 for kw in ai_keywords if kw in content_lower)
        real_score = sum(1 for kw in real_keywords if kw in content_lower)
        
        if ai_score > real_score:
            is_likely_ai = True
            plausibility = 30 + (real_score * 10)
        elif real_score > ai_score:
            is_likely_ai = False
            plausibility = 70 + (real_score * 5)
        else:
            is_likely_ai = None
            plausibility = 50
        
        return {
            'success': True,
            'plausibility_score': min(100, max(0, plausibility)),
            'is_likely_ai': is_likely_ai,
            'confidence': 40,  # Lower confidence for text parsing
            'issues_found': [],
            'explanation': content[:500],
            'model_used': model,
            'parsing_method': 'text_fallback'
        }


def test_semantic_detector():
    """Quick test of the semantic detector."""
    detector = SemanticPlausibilityDetector()
    
    if not detector.available:
        print("❌ GROQ_API_KEY not set")
        print("   Set it with: $env:GROQ_API_KEY = 'your-key-here'")
        return False
    
    print("✓ Semantic detector initialized")
    print(f"  Model: {detector.model}")
    return True


if __name__ == "__main__":
    test_semantic_detector()
