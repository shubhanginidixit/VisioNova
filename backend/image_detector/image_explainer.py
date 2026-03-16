"""
VisioNova Image Explainer
Uses Groq Vision API (Llama 4 Scout) to analyze images and provide 
AI-powered explanations of detection results.

Features:
- Visual analysis of image characteristics
- AI artifact identification
- Human-readable explanations of detection findings
- Contextual recommendations
"""
import os
import base64
import json
import logging
from typing import Dict, Optional, Any
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageExplainer:
    """
    Generates visual analysis and natural language explanations for image detection results.
    Uses Groq's Llama 4 Scout vision model to analyze images and explain findings.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Groq Vision client.
        
        Args:
            api_key: Optional Groq API key. If not provided, uses environment variable.
        """
        self.api_key = api_key or os.getenv('GROQ_IMAGE_API_KEY') or os.getenv('GROQ_API_KEY')
        
        # Llama 4 Scout is the best vision model available on Groq
        self.vision_model = os.getenv('GROQ_VISION_MODEL', 'meta-llama/llama-4-scout-17b-16e-instruct')
        self.text_model = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
        
        if not self.api_key:
            logger.warning("GROQ_IMAGE_API_KEY not found. Image explanations disabled.")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)
            logger.info(f"ImageExplainer initialized with vision model: {self.vision_model}")
    
    def analyze_image(self, image_data: bytes, detection_result: Dict) -> Dict[str, Any]:
        """
        Perform comprehensive AI analysis of an image.
        
        Combines ML detection results with visual AI analysis to provide
        a complete understanding of the image.
        
        Args:
            image_data: Raw image bytes
            detection_result: Results from the ML detector
            
        Returns:
            dict with visual_analysis, explanation, and combined_verdict
        """
        if not self.client:
            return self._fallback_analysis(detection_result)
        
        try:
            # Convert image to base64 for API
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Determine image type from bytes
            image_type = self._detect_image_type(image_data)
            data_url = f"data:image/{image_type};base64,{image_b64}"
            
            # First, get visual analysis from the vision model
            visual_analysis = self._get_visual_analysis(data_url, detection_result)
            
            # Then, generate human-readable explanation
            explanation = self._generate_explanation(detection_result, visual_analysis)
            
            return {
                'success': True,
                'visual_analysis': visual_analysis,
                'explanation': explanation,
                'combined_verdict': self._combine_verdicts(detection_result, visual_analysis),
                'ai_model_used': self.vision_model
            }
            
        except Exception as e:
            logger.error(f"Error in image analysis: {e}")
            return {
                'success': False,
                'error': str(e),
                'visual_analysis': None,
                'explanation': self._fallback_analysis(detection_result)
            }
    
    def _detect_image_type(self, image_data: bytes) -> str:
        """Detect image type from magic bytes."""
        if image_data[:8] == b'\x89PNG\r\n\x1a\n':
            return 'png'
        elif image_data[:2] == b'\xff\xd8':
            return 'jpeg'
        elif image_data[:6] in (b'GIF87a', b'GIF89a'):
            return 'gif'
        elif image_data[:4] == b'RIFF' and image_data[8:12] == b'WEBP':
            return 'webp'
        else:
            return 'jpeg'  # Default to JPEG
    
    def _get_visual_analysis(self, data_url: str, detection_result: Dict) -> Dict[str, Any]:
        """
        Use Llama 4 Scout vision model to analyze the image visually.
        
        Args:
            data_url: Base64 data URL of the image
            detection_result: ML detection results for context
            
        Returns:
            dict with visual analysis findings
        """
        try:
            prompt = """Analyze this image carefully for signs of AI generation or manipulation. Look for:

1. **Anatomical issues**: Hands with wrong number of fingers, distorted limbs, asymmetric faces
2. **Text/symbols**: Gibberish text, malformed letters, nonsensical signs
3. **Background inconsistencies**: Impossible geometry, repeated patterns, melting/warping
4. **Texture artifacts**: Overly smooth skin, plastic-like surfaces, unnatural hair
5. **Lighting problems**: Inconsistent shadows, impossible reflections, unnatural highlights
6. **Edge artifacts**: Blurry transitions, halo effects, unnatural object boundaries
7. **Repetitive patterns**: Clone-stamp artifacts, repeated elements, tiling issues

Provide your analysis in this JSON format:
{
    "is_likely_ai_generated": true/false,
    "confidence": 0-100,
    "visual_artifacts_found": [
        {"type": "artifact type", "description": "what you see", "severity": "low/medium/high"}
    ],
    "areas_of_concern": ["list of specific areas or elements that look suspicious"],
    "authentic_indicators": ["list of elements that suggest authenticity"],
    "overall_assessment": "2-3 sentence summary of your visual analysis"
}

Be thorough but honest - only flag issues you actually observe."""

            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": data_url
                                }
                            }
                        ]
                    }
                ],
                temperature=0.2,
                max_completion_tokens=1024,
                top_p=1
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                # Handle potential markdown code blocks
                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0]
                elif '```' in content:
                    content = content.split('```')[1].split('```')[0]
                
                return json.loads(content)
            except json.JSONDecodeError:
                logger.warning("Could not parse vision model response as JSON")
                return {
                    'is_likely_ai_generated': None,
                    'confidence': 50,
                    'overall_assessment': content,
                    'parse_error': True
                }
                
        except Exception as e:
            logger.error(f"Vision analysis error: {e}")
            return {
                'is_likely_ai_generated': None,
                'confidence': 0,
                'error': str(e),
                'overall_assessment': "Vision analysis unavailable"
            }
    
    def _generate_explanation(self, detection_result: Dict, visual_analysis: Dict) -> Dict[str, Any]:
        """
        Generate a human-readable explanation combining ML and visual analysis.
        
        Args:
            detection_result: ML detection results
            visual_analysis: Visual analysis from vision model
            
        Returns:
            dict with explanation details
        """
        if not self.client:
            return self._fallback_explanation(detection_result, visual_analysis)
        
        try:
            # Build context from both analyses
            context = f"""
ML DETECTION RESULTS:
- AI Probability: {detection_result.get('ai_probability', 'N/A')}%
- Verdict: {detection_result.get('verdict', 'N/A')}
- Detection Method: {detection_result.get('detection_method', 'statistical')}
- Analysis Scores: {json.dumps(detection_result.get('analysis_scores', {}), indent=2)}

VISUAL ANALYSIS (AI Vision):
- AI Generated Assessment: {visual_analysis.get('is_likely_ai_generated', 'Unknown')}
- Visual Confidence: {visual_analysis.get('confidence', 'N/A')}%
- Artifacts Found: {json.dumps(visual_analysis.get('visual_artifacts_found', []), indent=2)}
- Areas of Concern: {visual_analysis.get('areas_of_concern', [])}
- Authentic Indicators: {visual_analysis.get('authentic_indicators', [])}

ADDITIONAL CONTEXT:
- Metadata AI Indicators: {detection_result.get('metadata', {}).get('ai_indicators', [])}
- Watermark Detected: {detection_result.get('watermark', {}).get('watermark_detected', False)}
- C2PA/Content Credentials: {detection_result.get('content_credentials', {}).get('has_content_credentials', False)}
"""

            prompt = f"""Based on the following image analysis results, provide a clear, helpful explanation for the user.

{context}

Provide your response in this exact JSON format:
{{
    "summary": "<2-3 sentence plain-language summary explaining whether the image appears AI-generated and why>",
    "key_findings": [
        "<finding 1 with explanation>",
        "<finding 2>",
        "<finding 3>"
    ],
    "visual_evidence": "<Describe the visual clues that support the verdict, if any>",
    "technical_notes": "<Brief explanation of what the ML scores mean>",
    "confidence_explanation": "<What the confidence level means and any caveats>",
    "recommendations": [
        "<what the user should consider or do next>",
        "<additional recommendation>"
    ]
}}

Guidelines:
- Be factual and objective
- Explain technical terms simply
- Acknowledge uncertainty where it exists
- Be helpful, not alarmist

Respond ONLY with valid JSON."""

            response = self.client.chat.completions.create(
                model=self.text_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert image forensics analyst who explains AI detection results clearly and objectively. Help users understand whether an image may be AI-generated without being alarmist."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_completion_tokens=1024,
                top_p=1
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0]
                elif '```' in content:
                    content = content.split('```')[1].split('```')[0]
                
                return json.loads(content)
            except json.JSONDecodeError:
                return {
                    'summary': content,
                    'parse_error': True
                }
                
        except Exception as e:
            logger.error(f"Explanation generation error: {e}")
            return self._fallback_explanation(detection_result, visual_analysis)
    
    def _combine_verdicts(self, detection_result: Dict, visual_analysis: Dict) -> Dict[str, Any]:
        """
        Combine ML detection and visual analysis into a final verdict.
        
        Args:
            detection_result: ML detection results
            visual_analysis: Visual analysis from vision model
            
        Returns:
            dict with combined verdict and confidence
        """
        ml_prob = detection_result.get('ai_probability', 50)
        visual_prob = visual_analysis.get('confidence', 50)
        visual_ai = visual_analysis.get('is_likely_ai_generated')
        
        # Weight the scores (visual analysis is valuable but ML is more consistent)
        if visual_ai is True:
            combined_prob = (ml_prob * 0.6) + (visual_prob * 0.4)
        elif visual_ai is False:
            # Visual says not AI, reduce probability
            combined_prob = (ml_prob * 0.6) + ((100 - visual_prob) * 0.4)
        else:
            # Visual analysis inconclusive, rely more on ML
            combined_prob = ml_prob
        
        combined_prob = round(combined_prob, 2)
        
        # Determine final verdict
        if combined_prob >= 80:
            verdict = 'AI_GENERATED'
            description = 'High confidence: This image appears to be AI-generated'
        elif combined_prob >= 60:
            verdict = 'LIKELY_AI'
            description = 'Moderate confidence: This image shows signs of AI generation'
        elif combined_prob >= 40:
            verdict = 'UNCERTAIN'
            description = 'Inconclusive: Cannot determine with confidence'
        elif combined_prob >= 20:
            verdict = 'LIKELY_REAL'
            description = 'Moderate confidence: This image appears to be authentic'
        else:
            verdict = 'REAL'
            description = 'High confidence: This image appears to be a real photograph'
        
        return {
            'combined_probability': combined_prob,
            'ml_probability': ml_prob,
            'visual_probability': visual_prob if visual_ai else (100 - visual_prob if visual_ai is False else 50),
            'verdict': verdict,
            'verdict_description': description,
            'analysis_agreement': self._check_agreement(ml_prob, visual_ai, visual_prob)
        }
    
    def _check_agreement(self, ml_prob: float, visual_ai: Optional[bool], visual_prob: float) -> str:
        """Check if ML and visual analysis agree."""
        ml_thinks_ai = ml_prob >= 50
        
        if visual_ai is None:
            return 'VISUAL_INCONCLUSIVE'
        elif ml_thinks_ai == visual_ai:
            return 'STRONG_AGREEMENT'
        else:
            return 'DISAGREEMENT'
    
    def _fallback_analysis(self, detection_result: Dict) -> Dict[str, Any]:
        """Provide basic analysis when API is unavailable."""
        return {
            'success': True,
            'visual_analysis': {
                'is_likely_ai_generated': None,
                'confidence': 0,
                'overall_assessment': 'Visual AI analysis unavailable (API key not configured)',
                'visual_artifacts_found': [],
                'areas_of_concern': [],
                'authentic_indicators': []
            },
            'explanation': self._fallback_explanation(detection_result, {}),
            'combined_verdict': {
                'combined_probability': detection_result.get('ai_probability', 50),
                'verdict': detection_result.get('verdict', 'UNCERTAIN'),
                'verdict_description': detection_result.get('verdict_description', 'Analysis based on statistical methods only'),
                'analysis_agreement': 'ML_ONLY'
            },
            'ai_model_used': None
        }
    
    def _fallback_explanation(self, detection_result: Dict, visual_analysis: Dict = None) -> Dict[str, Any]:
        """Generate a basic explanation without API."""
        ai_prob = detection_result.get('ai_probability', 50)
        verdict = detection_result.get('verdict', 'UNCERTAIN')
        scores = detection_result.get('analysis_scores', {})
        
        findings = []
        if scores.get('frequency_anomaly', 0) > 60:
            findings.append("Frequency domain analysis detected patterns typical of AI-generated images")
        if scores.get('noise_consistency', 0) > 60:
            findings.append("Noise patterns appear artificially uniform")
        if scores.get('texture_quality', 0) > 60:
            findings.append("Texture analysis shows signs of AI generation")
        if detection_result.get('watermark', {}).get('watermark_detected'):
            findings.append(f"AI watermark detected: {detection_result.get('watermark', {}).get('watermark_type', 'Unknown')}")
        
        if not findings:
            findings.append("Statistical analysis completed with no strong indicators")
        
        return {
            'summary': f"Based on statistical analysis, this image has a {ai_prob}% probability of being AI-generated. Verdict: {verdict}",
            'key_findings': findings,
            'visual_evidence': 'Visual AI analysis not available',
            'technical_notes': 'Analysis based on frequency domain, noise patterns, and texture analysis',
            'confidence_explanation': f"The {ai_prob}% confidence is derived from statistical methods. Results may vary.",
            'recommendations': [
                "For more accurate results, consider using multiple detection tools",
                "Look for visual artifacts like distorted hands, text, or impossible geometry"
            ]
        }


def create_image_explainer(api_key: Optional[str] = None) -> ImageExplainer:
    """
    Factory function to create an ImageExplainer instance.
    
    Args:
        api_key: Optional Groq API key
        
    Returns:
        ImageExplainer instance
    """
    return ImageExplainer(api_key=api_key)
