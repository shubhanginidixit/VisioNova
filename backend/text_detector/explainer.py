"""
VisioNova Text Explainer
Uses Groq LLM to generate human-readable explanations of AI detection results.

Takes ML detection results and translates them into:
- Plain language summary
- Key indicators explanation
- Writing improvement suggestions
"""
import os
import json
from typing import Dict, Optional
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class TextExplainer:
    """
    Generates natural language explanations for AI detection results.
    Uses Groq LLM to make technical results understandable.
    """
    
    def __init__(self):
        """Initialize the Groq client."""
        api_key = os.getenv('GROQ_API_KEY')
        self.model = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
        
        if not api_key:
            print("Warning: GROQ_API_KEY not found. Explanations disabled.")
            self.client = None
        else:
            self.client = Groq(api_key=api_key)
    
    def explain(self, detection_result: Dict, original_text: str = "") -> Dict:
        """
        Generate a human-readable explanation of detection results.
        
        Args:
            detection_result: The result from AIContentDetector.predict()
            original_text: The original text that was analyzed (optional)
            
        Returns:
            dict with summary, key_indicators, suggestions, and detailed_breakdown
        """
        if not self.client:
            return self._fallback_explanation(detection_result)
        
        try:
            # Build context from detection result
            context = self._build_context(detection_result, original_text)
            
            prompt = f"""You are an AI writing analysis expert. Based on the following detection results, provide a clear, helpful explanation for the user.

DETECTION RESULTS:
{context}

Provide your response in this exact JSON format:
{{
    "summary": "<2-3 sentence plain-language summary of what was detected and why>",
    "verdict_explanation": "<1 sentence explaining what the verdict means>",
    "key_indicators": [
        "<indicator 1 that led to this conclusion>",
        "<indicator 2>",
        "<indicator 3>"
    ],
    "pattern_breakdown": "<If AI patterns were detected, explain what they are and why they matter>",
    "suggestions": [
        "<actionable suggestion 1 for the writer>",
        "<suggestion 2>",
        "<suggestion 3>"
    ],
    "confidence_note": "<Explain what the confidence score means and any caveats>"
}}

Guidelines:
- Be helpful and educational, not accusatory
- Explain technical terms in simple language
- Focus on patterns, not judgment
- Provide actionable writing tips
- Keep language friendly and supportive

Respond ONLY with valid JSON."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful writing analyst who explains AI detection results in simple, supportive terms. You help writers understand their text's characteristics without being judgmental."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            content = response.choices[0].message.content.strip()
            
            # Handle markdown code blocks
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()
            
            result = json.loads(content)
            result['ai_explained'] = True
            return result
            
        except json.JSONDecodeError as e:
            print(f"Explainer JSON parsing error: {e}")
            return self._fallback_explanation(detection_result)
        except Exception as e:
            print(f"Explainer error: {e}")
            return self._fallback_explanation(detection_result)
    
    def _build_context(self, result: Dict, text: str) -> str:
        """Build context string from detection result."""
        lines = []
        
        # Main prediction
        pred = result.get('prediction', 'unknown')
        conf = result.get('confidence', 0)
        lines.append(f"Prediction: {pred.upper()} ({conf}% confidence)")
        
        # Scores
        scores = result.get('scores', {})
        lines.append(f"Human probability: {scores.get('human', 0)}%")
        lines.append(f"AI probability: {scores.get('ai_generated', 0)}%")
        
        # Metrics
        metrics = result.get('metrics', {})
        if metrics:
            lines.append(f"\nMetrics:")
            lines.append(f"- Word count: {metrics.get('word_count', 0)}")
            lines.append(f"- Sentence count: {metrics.get('sentence_count', 0)}")
            lines.append(f"- Vocabulary richness: {metrics.get('vocabulary_richness', 0)}%")
            
            rhythm = metrics.get('rhythm', {})
            if rhythm:
                lines.append(f"- Rhythm: {rhythm.get('status', 'Unknown')} - {rhythm.get('description', '')}")
            
            burstiness = metrics.get('burstiness', {})
            if burstiness:
                lines.append(f"- Burstiness score: {burstiness.get('score', 0)} (higher = more natural)")
        
        # Patterns
        patterns = result.get('detected_patterns', {})
        if patterns:
            total = patterns.get('total_count', 0)
            lines.append(f"\nDetected AI Patterns: {total} total")
            
            categories = patterns.get('categories', {})
            for cat, info in categories.items():
                examples = info.get('examples', [])[:2]
                lines.append(f"- {info.get('type', cat)}: {info.get('count', 0)} occurrences")
                if examples:
                    lines.append(f"  Examples: {', '.join(examples)}")
        
        # Flagged sentences
        flagged = result.get('flagged_sentences', [])
        if flagged:
            lines.append(f"\nFlagged sentences: {len(flagged)}")
            for i, s in enumerate(flagged[:3], 1):
                lines.append(f"{i}. \"{s.get('text', '')[:80]}...\" ({s.get('ai_score', 0)}% AI)")
        
        # Text sample
        if text:
            lines.append(f"\nText sample (first 200 chars): \"{text[:200]}...\"")
        
        return "\n".join(lines)
    
    def _fallback_explanation(self, result: Dict) -> Dict:
        """Generate basic explanation without LLM."""
        pred = result.get('prediction', 'unknown')
        conf = result.get('confidence', 0)
        patterns = result.get('detected_patterns', {})
        pattern_count = patterns.get('total_count', 0)
        
        if pred == 'ai_generated':
            summary = f"This text has characteristics commonly associated with AI-generated content ({conf:.0f}% confidence)."
            indicators = []
            
            if pattern_count > 0:
                indicators.append(f"Found {pattern_count} common AI writing patterns")
            
            metrics = result.get('metrics', {})
            rhythm = metrics.get('rhythm', {})
            if rhythm.get('status') == 'Uniform':
                indicators.append("Sentence structure is very uniform")
            
            burstiness = metrics.get('burstiness', {})
            if burstiness.get('score', 0) < 0.3:
                indicators.append("Low variation in sentence length")
            
            if not indicators:
                indicators = ["Overall text patterns match AI-generated content"]
            
            suggestions = [
                "Try varying your sentence lengths more",
                "Use more conversational language",
                "Add personal anecdotes or specific details"
            ]
        else:
            summary = f"This text appears to be human-written ({conf:.0f}% confidence)."
            indicators = [
                "Natural variation in writing style",
                "Diverse vocabulary usage",
                "Organic sentence structure"
            ]
            suggestions = [
                "Your writing appears natural",
                "Continue using varied sentence structures",
                "Personal voice is coming through"
            ]
        
        return {
            "summary": summary,
            "verdict_explanation": f"The text was classified as {pred.replace('_', ' ')}.",
            "key_indicators": indicators[:3],
            "pattern_breakdown": f"Detected {pattern_count} AI writing patterns." if pattern_count else "No significant AI patterns detected.",
            "suggestions": suggestions[:3],
            "confidence_note": f"This result has {conf:.0f}% confidence. " + 
                ("Higher values indicate stronger AI signals." if pred == 'ai_generated' else "Higher values indicate more natural writing patterns."),
            "ai_explained": False
        }


if __name__ == "__main__":
    # Test the explainer
    explainer = TextExplainer()
    
    # Mock detection result
    mock_result = {
        "prediction": "ai_generated",
        "confidence": 78.5,
        "scores": {"human": 21.5, "ai_generated": 78.5},
        "metrics": {
            "word_count": 150,
            "sentence_count": 8,
            "vocabulary_richness": 45.2,
            "rhythm": {"status": "Uniform", "description": "Highly consistent rhythm"}
        },
        "detected_patterns": {
            "total_count": 5,
            "categories": {
                "formal_transitions": {"count": 3, "type": "Formal Transition", "examples": ["furthermore", "in conclusion"]},
                "hedging": {"count": 2, "type": "Hedging Language", "examples": ["it's important to note"]}
            }
        }
    }
    
    explanation = explainer.explain(mock_result)
    print("Explanation:")
    print(json.dumps(explanation, indent=2))
