"""
Fact Checker - Main Pipeline
Orchestrates the fact-checking process with AI-powered analysis.
"""
from .input_classifier import InputClassifier
from .content_extractor import ContentExtractor
from .web_searcher import WebSearcher
from .config import Verdict

# Import AI analyzer from separate ai module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai import AIAnalyzer


class FactChecker:
    """
    Main fact-checking pipeline.
    Takes input (claim, question, or URL), searches multiple sources,
    and returns a verdict with evidence.
    """
    
    def __init__(self):
        self.classifier = InputClassifier()
        self.extractor = ContentExtractor()
        self.searcher = WebSearcher()
        self.ai_analyzer = AIAnalyzer()
    
    def check(self, user_input: str) -> dict:
        """
        Run the full fact-checking pipeline.
        
        Args:
            user_input: Text claim, question, or URL to verify
            
        Returns:
            dict with verdict, confidence, sources, and explanation
        """
        # Step 1: Classify input
        classification = self.classifier.classify(user_input)
        input_type = classification['type']
        
        # Step 2: Extract claim(s) to verify
        if input_type == 'url':
            # Extract content from URL
            extracted = self.extractor.extract_from_url(user_input)
            if not extracted['success']:
                return self._error_response(
                    user_input,
                    f"Could not fetch URL: {extracted['error']}"
                )
            
            # Use title + first claim as the search query
            claims = extracted['claims']
            if claims:
                claim = claims[0]
            else:
                claim = extracted['title']
            
            classification['claim'] = claim
            classification['url_title'] = extracted['title']
            classification['url_content'] = extracted['content']
        else:
            claim = classification['claim']
        
        if not claim:
            return self._error_response(user_input, "Could not extract a claim to verify")
        
        # Step 3: Search for evidence
        search_results = self.searcher.search(claim)
        sources = search_results['sources']
        
        if not sources:
            return self._build_response(
                classification,
                Verdict.UNVERIFIABLE,
                confidence=10,
                sources=[],
                explanation="Could not find any sources to verify this claim."
            )
        
        # Step 4: Analyze sources and determine verdict using AI
        ai_result = self._analyze_sources(claim, sources)
        
        # Step 5: Build response with all AI analysis data
        return self._build_response(
            classification,
            ai_result,
            sources
        )
    
    def _analyze_sources(self, claim: str, sources: list) -> dict:
        """
        Analyze sources using AI.
        
        Returns:
            dict with verdict, confidence, summary, detailed_analysis, claims
        """
        return self.ai_analyzer.analyze_claim(claim, sources)
        
        # Check if we have fact-check sites with specific verdicts
        # (In a full implementation, we'd parse the snippets for verdict keywords)
        
        total_trusted = len(high_trust) + len(medium_trust)
        
        # Simple heuristic-based verdict determination
        if factcheck_sites:
            # We have fact-check sources - give them high weight
            confidence = min(90, 70 + len(factcheck_sites) * 10)
            
            # Check snippets for verdict indicators
            verdict = self._extract_verdict_from_snippets(factcheck_sites)
            
            explanation = (
                f"Found {len(factcheck_sites)} fact-check source(s) addressing this claim. "
                f"Total of {len(sources)} sources analyzed."
            )
        elif total_trusted >= 3:
            # Multiple trusted sources
            confidence = min(85, 50 + total_trusted * 10)
            verdict = Verdict.TRUE  # Assume true if trusted sources discuss it
            explanation = (
                f"Found {total_trusted} trusted sources discussing this topic. "
                f"No explicit fact-checks found, but sources appear credible."
            )
        elif total_trusted >= 1:
            # Some trusted sources
            confidence = 50
            verdict = Verdict.PARTIALLY_TRUE
            explanation = (
                f"Found {total_trusted} trusted source(s). "
                f"Limited evidence available for full verification."
            )
        else:
            # Only unknown sources
            confidence = 30
            verdict = Verdict.UNVERIFIABLE
            explanation = (
                "Could not find trusted sources to verify this claim. "
                "The sources found are of unknown reliability."
            )
        
        return verdict, confidence, explanation
    
    def _extract_verdict_from_snippets(self, sources: list) -> str:
        """Extract verdict hints from source snippets."""
        # Keywords that indicate false/misleading claims
        false_indicators = ['false', 'fake', 'hoax', 'debunked', 'myth', 
                          'misleading', 'incorrect', 'wrong', 'untrue']
        true_indicators = ['true', 'correct', 'verified', 'confirmed', 
                          'accurate', 'real', 'factual']
        
        false_count = 0
        true_count = 0
        
        for source in sources:
            snippet = source.get('snippet', '').lower()
            
            for indicator in false_indicators:
                if indicator in snippet:
                    false_count += 1
                    break
            
            for indicator in true_indicators:
                if indicator in snippet:
                    true_count += 1
                    break
        
        if false_count > true_count:
            return Verdict.FALSE
        elif true_count > false_count:
            return Verdict.TRUE
        else:
            return Verdict.UNVERIFIABLE
    
    def _build_response(self, classification: dict, ai_result: dict, sources: list) -> dict:
        """Build the final response object with all analysis data."""
        # Limit sources to top 10
        top_sources = []
        for source in sources[:10]:
            top_sources.append({
                'title': source.get('title', 'Untitled'),
                'url': source.get('url', ''),
                'snippet': source.get('snippet', '')[:200],
                'domain': source.get('domain', ''),
                'trust_level': source.get('trust_level', 'unknown'),
                'is_factcheck': source.get('is_factcheck_site', False)
            })
        
        return {
            'success': True,
            'input': classification['original'],
            'input_type': classification['type'],
            'claim': classification['claim'],
            'verdict': ai_result.get('verdict', 'UNVERIFIABLE'),
            'confidence': ai_result.get('confidence', 50),
            'sources': top_sources,
            'source_count': len(sources),
            'explanation': ai_result.get('explanation', 'Analysis completed.'),
            # New structured data for tabs
            'summary': ai_result.get('summary', {}),
            'detailed_analysis': ai_result.get('detailed_analysis', {}),
            'claims': ai_result.get('claims', []),
            'ai_analyzed': ai_result.get('ai_analyzed', False)
        }
    
    def _error_response(self, user_input: str, error: str) -> dict:
        """Build an error response."""
        return {
            'success': False,
            'input': user_input,
            'input_type': 'unknown',
            'claim': None,
            'verdict': Verdict.UNVERIFIABLE,
            'confidence': 0,
            'sources': [],
            'source_count': 0,
            'explanation': error
        }


# Quick test
if __name__ == '__main__':
    checker = FactChecker()
    
    test_claims = [
        "The moon landing happened in 1969",
        "Is the Earth flat?",
        "COVID-19 vaccines contain microchips"
    ]
    
    for claim in test_claims:
        print(f"\n{'='*60}")
        print(f"Checking: {claim}")
        print('='*60)
        
        result = checker.check(claim)
        
        print(f"Verdict: {result['verdict']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Sources found: {result['source_count']}")
        print(f"Explanation: {result['explanation']}")
        
        if result['sources']:
            print("\nTop sources:")
            for source in result['sources'][:3]:
                print(f"  - [{source['trust_level']}] {source['title']}")
