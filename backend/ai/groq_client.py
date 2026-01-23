"""
AI Analyzer - Uses Groq LLM for intelligent fact-check analysis.
Provides better verdict determination and explanations.
"""
import os
from groq import Groq
from dotenv import load_dotenv
import json

load_dotenv()


class AIAnalyzer:
    """Uses LLM to analyze claims and sources for fact-checking."""
    
    def __init__(self):
        api_key = os.getenv('GROQ_API_KEY')
        self.model = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
        
        if not api_key:
            print("Warning: GROQ_API_KEY not found. AI analysis disabled.")
            self.client = None
        else:
            self.client = Groq(api_key=api_key)
    
    def analyze_claim(self, claim: str, sources: list) -> dict:
        """
        Use AI to analyze a claim against found sources.
        Returns structured content for different view tabs.
        
        Args:
            claim: The claim being fact-checked
            sources: List of source dictionaries with title, snippet, trust_level
            
        Returns:
            dict with verdict, confidence, summary, detailed_analysis, and claims
        """
        if not self.client:
            return self._fallback_analysis(sources)
        
        try:
            # Build context from sources
            source_context = self._build_source_context(sources)
            
            # Create prompt for comprehensive analysis
            prompt = f"""You are an expert fact-checker. Analyze the following claim based on the provided sources.

CLAIM TO VERIFY:
"{claim}"

SOURCES FOUND:
{source_context}

Provide a comprehensive analysis in the following JSON format:
{{
    "verdict": "TRUE" | "FALSE" | "PARTIALLY TRUE" | "MISLEADING" | "UNVERIFIABLE",
    "confidence": <number 0-100>,
    
    "confidence_breakdown": {{
        "source_quality": <0-25 based on trustworthiness of sources>,
        "source_quantity": <0-20 based on number of sources>,
        "factcheck_found": <0-25 if fact-check sites found, else 0>,
        "consensus": <0-30 based on agreement between sources>,
        "explanation": "<brief explanation of confidence calculation>"
    }},
    
    "summary": {{
        "one_liner": "<One sentence verdict summary>",
        "key_points": ["<key point 1>", "<key point 2>", "<key point 3>"]
    }},
    
    "detailed_analysis": {{
        "overview": "<2-3 paragraph detailed explanation of the analysis>",
        "methodology": "<How the claim was verified>",
        "context": "<Important context about the claim>",
        "limitations": "<Any limitations in the verification>"
    }},
    
    "source_analysis": [
        {{
            "source_title": "<source title>",
            "stance": "SUPPORTS" | "REFUTES" | "NEUTRAL",
            "relevance": <0-100>,
            "key_excerpt": "<most relevant quote from this source>"
        }}
    ],
    
    "contradictions_found": <true if sources contradict each other, false otherwise>,
    
    "claims": [
        {{
            "statement": "<specific claim extracted>",
            "status": "VERIFIED" | "FALSE" | "MISLEADING" | "UNVERIFIED",
            "evidence": "<evidence supporting or refuting this claim>",
            "source": "<which source supports this>"
        }}
    ]
}}

Rules:
- If sources directly confirm the claim, verdict is TRUE
- If sources directly contradict the claim, verdict is FALSE  
- If claim has some truth but is exaggerated/missing context, verdict is PARTIALLY TRUE or MISLEADING
- If sources don't provide enough information, verdict is UNVERIFIABLE
- Calculate confidence breakdown components accurately
- Analyze each source's stance (SUPPORTS/REFUTES/NEUTRAL) individually
- Flag contradictions_found as true if sources disagree significantly
- Break down the main claim into individual verifiable statements in the "claims" array
- Provide thorough analysis in detailed_analysis section

Respond ONLY with valid JSON, no other text."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert fact-checker and journalist. You analyze claims thoroughly, break them into verifiable components, and provide comprehensive analysis. Always respond in valid JSON format."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500  # Increased for detailed response
            )
            
            # Parse the response
            content = response.choices[0].message.content.strip()
            
            # Try to parse JSON from response
            # Try to parse JSON from response
            # import json (moved to top-level)
            
            # Handle potential markdown code blocks
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()
            
            result = json.loads(content)
            
            return {
                'verdict': result.get('verdict', 'UNVERIFIABLE'),
                'confidence': min(100, max(0, int(result.get('confidence', 50)))),
                'confidence_breakdown': result.get('confidence_breakdown', {
                    'source_quality': 0,
                    'source_quantity': 0,
                    'factcheck_found': 0,
                    'consensus': 0,
                    'explanation': 'Breakdown unavailable'
                }),
                'summary': result.get('summary', {
                    'one_liner': 'Analysis completed.',
                    'key_points': []
                }),
                'detailed_analysis': result.get('detailed_analysis', {
                    'overview': 'Detailed analysis unavailable.',
                    'methodology': '',
                    'context': '',
                    'limitations': ''
                }),
                'source_analysis': result.get('source_analysis', []),
                'contradictions_found': result.get('contradictions_found', False),
                'claims': result.get('claims', []),
                'explanation': result.get('summary', {}).get('one_liner', 'Analysis completed.'),
                'ai_analyzed': True
            }
            
        except json.JSONDecodeError as e:
            print(f"AI JSON parsing error: {e}")
            print(f"Raw response: {content[:500]}")
            return self._fallback_analysis(sources)
        except Exception as e:
            print(f"AI analysis error: {e}")
            print(f"Error type: {type(e).__name__}")
            return self._fallback_analysis(sources)
    
    def _build_source_context(self, sources: list) -> str:
        """Build a text context from sources for the AI prompt."""
        if not sources:
            return "No sources found."
        
        context_parts = []
        for i, source in enumerate(sources[:8], 1):  # Limit to 8 sources
            trust = source.get('trust_level', 'unknown')
            is_factcheck = source.get('is_factcheck_site', False)
            
            source_type = "[FACT-CHECK SITE]" if is_factcheck else f"[{trust.upper()}]"
            
            context_parts.append(
                f"{i}. {source_type} {source.get('title', 'Untitled')}\n"
                f"   Domain: {source.get('domain', 'unknown')}\n"
                f"   Content: {source.get('snippet', '')[:250]}"
            )
        
        return "\n\n".join(context_parts)
    
    def _fallback_analysis(self, sources: list) -> dict:
        """Fallback when AI is unavailable - use simple heuristics."""
        factcheck_count = sum(1 for s in sources if s.get('is_factcheck_site'))
        high_trust = sum(1 for s in sources if s.get('trust_level') == 'high')
        
        if factcheck_count > 0:
            confidence = min(85, 60 + factcheck_count * 10)
            verdict = 'UNVERIFIABLE'
            explanation = f"Found {factcheck_count} fact-check sources. AI analysis unavailable."
        elif high_trust > 0:
            confidence = min(70, 40 + high_trust * 10)
            verdict = 'UNVERIFIABLE'
            explanation = f"Found {high_trust} trusted sources. AI analysis unavailable."
        else:
            confidence = 30
            verdict = 'UNVERIFIABLE'
            explanation = "Could not find trusted sources. AI analysis unavailable."
        
        return {
            'verdict': verdict,
            'confidence': confidence,
            'explanation': explanation,
            'summary': {
                'one_liner': explanation,
                'key_points': [f"Found {len(sources)} sources", "AI analysis unavailable"]
            },
            'detailed_analysis': {
                'overview': explanation,
                'methodology': 'Automated source search without AI analysis.',
                'context': 'AI-powered analysis is currently unavailable.',
                'limitations': 'Cannot determine verdict without AI analysis.'
            },
            'claims': [],
            'ai_analyzed': False
        }


# Quick test
if __name__ == '__main__':
    analyzer = AIAnalyzer()
    
    test_claim = "The moon landing happened in 1969"
    test_sources = [
        {
            'title': 'Apollo 11 - Wikipedia',
            'domain': 'wikipedia.org',
            'snippet': 'Apollo 11 was the American spaceflight that first landed humans on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969.',
            'trust_level': 'medium-high',
            'is_factcheck_site': False
        },
        {
            'title': 'NASA Apollo 11 Mission Overview',
            'domain': 'nasa.gov',
            'snippet': 'Apollo 11 was the first mission to land humans on the Moon. On July 20, 1969, American astronauts Neil Armstrong and Buzz Aldrin became the first humans to walk on the Moon.',
            'trust_level': 'high',
            'is_factcheck_site': False
        }
    ]
    
    result = analyzer.analyze_claim(test_claim, test_sources)
    
    print(f"Claim: {test_claim}")
    print(f"Verdict: {result['verdict']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"Explanation: {result['explanation']}")
    print(f"AI Analyzed: {result['ai_analyzed']}")
