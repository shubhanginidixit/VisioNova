"""
Fact Checker - Main Pipeline
Orchestrates the fact-checking process with AI-powered analysis.
"""
import hashlib
from functools import lru_cache
from .input_classifier import InputClassifier
from .content_extractor import ContentExtractor
from .web_searcher import WebSearcher
from .config import Verdict

# Import AI analyzer from separate ai module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai import AIAnalyzer


# Module-level cache for fact-check results
# Using a dictionary with TTL would be better for production
_claim_cache = {}
_CACHE_MAX_SIZE = 50


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
    
    def _get_cache_key(self, user_input: str) -> str:
        """Generate a hash key for caching claim results."""
        return hashlib.md5(user_input.strip().lower().encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> dict:
        """Check if result exists in cache."""
        return _claim_cache.get(cache_key)
    
    def _cache_result(self, cache_key: str, result: dict):
        """Store result in cache with size limit."""
        global _claim_cache
        # Simple LRU-like behavior: if cache is full, remove oldest entry
        if len(_claim_cache) >= _CACHE_MAX_SIZE:
            oldest_key = next(iter(_claim_cache))
            del _claim_cache[oldest_key]
        _claim_cache[cache_key] = result
    
    def clear_cache(self):
        """Clear the claim cache."""
        global _claim_cache
        _claim_cache = {}
    
    def cache_info(self) -> dict:
        """Get cache statistics."""
        return {
            'size': len(_claim_cache),
            'max_size': _CACHE_MAX_SIZE
        }
    
    def check(self, user_input: str) -> dict:
        """
        Run the full fact-checking pipeline.
        
        Args:
            user_input: Text claim, question, or URL to verify
            
        Returns:
            dict with verdict, confidence, sources, and explanation
        """
        # Check cache first for repeated claims
        cache_key = self._get_cache_key(user_input)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            cached_result['cached'] = True
            return cached_result
        
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
        result = self._build_response(
            classification,
            ai_result,
            sources
        )
        
        # Cache the result for future repeated claims
        result['cached'] = False
        self._cache_result(cache_key, result.copy())
        
        return result
    
    def deep_check(self, user_input: str) -> dict:
        """
        Run an enhanced fact-checking pipeline with multiple search queries.
        Searches for more sources to provide comprehensive results.
        
        Args:
            user_input: Text claim, question, or URL to verify
            
        Returns:
            dict with verdict, confidence, sources, and explanation (enhanced)
        """
        # Step 1: Classify input
        classification = self.classifier.classify(user_input)
        input_type = classification['type']
        
        # Step 2: Extract claim(s) to verify
        if input_type == 'url':
            extracted = self.extractor.extract_from_url(user_input)
            if not extracted['success']:
                return self._error_response(
                    user_input,
                    f"Could not fetch URL: {extracted['error']}"
                )
            claims = extracted['claims']
            claim = claims[0] if claims else extracted['title']
            classification['claim'] = claim
            classification['url_title'] = extracted['title']
            classification['url_content'] = extracted['content']
        else:
            claim = classification['claim']
        
        if not claim:
            return self._error_response(user_input, "Could not extract a claim to verify")
        
        # Step 3: DEEP SEARCH - Multiple query variations
        all_sources = []
        search_queries = self._generate_search_queries(claim)
        
        for query in search_queries:
            try:
                search_results = self.searcher.search(query)
                sources = search_results.get('sources', [])
                # Add query origin to each source
                for source in sources:
                    source['search_query'] = query
                all_sources.extend(sources)
            except Exception as e:
                print(f"Search failed for query '{query}': {e}")
                continue
        
        # Deduplicate sources by URL
        seen_urls = set()
        unique_sources = []
        for source in all_sources:
            url = source.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_sources.append(source)
        
        if not unique_sources:
            return self._build_response(
                classification,
                Verdict.UNVERIFIABLE,
                confidence=10,
                sources=[],
                explanation="Could not find any sources to verify this claim."
            )
        
        # Step 4: Enhanced AI analysis with more sources
        ai_result = self._analyze_sources(claim, unique_sources)
        
        # Step 5: Build response with deep scan metadata
        result = self._build_response(
            classification,
            ai_result,
            unique_sources
        )
        
        # Add deep scan metadata
        result['deep_scan'] = True
        result['queries_used'] = len(search_queries)
        result['total_sources_found'] = len(all_sources)
        result['unique_sources'] = len(unique_sources)
        result['cached'] = False
        
        return result
    
    def _generate_search_queries(self, claim: str) -> list:
        """
        Generate multiple search query variations for deep scanning.
        
        Args:
            claim: The main claim to search for
            
        Returns:
            list of search queries
        """
        queries = [claim]  # Original claim
        
        # Add fact-check focused query
        queries.append(f"fact check {claim}")
        
        # Add verification query
        queries.append(f"is it true that {claim}")
        
        # Extract key entities/keywords for additional searches
        words = claim.split()
        if len(words) > 5:
            # Use first 5 significant words
            key_words = [w for w in words if len(w) > 3][:5]
            queries.append(" ".join(key_words) + " fact check")
        
        # Add news-focused query
        queries.append(f"{claim} news verification")
        
        return queries[:5]  # Limit to 5 queries to avoid rate limiting
    
    def _analyze_sources(self, claim: str, sources: list) -> dict:
        """
        Analyze sources using AI with fallback to heuristic analysis.
        
        Returns:
            dict with verdict, confidence, summary, detailed_analysis, claims
        """
        # Try AI-powered analysis first
        try:
            ai_result = self.ai_analyzer.analyze_claim(claim, sources)
            if ai_result and ai_result.get('verdict'):
                ai_result['ai_analyzed'] = True
                return ai_result
        except Exception as e:
            print(f"AI analysis failed, falling back to heuristic: {e}")
        
        # Fallback to heuristic-based analysis
        return self._heuristic_analysis(claim, sources)
    
    def _heuristic_analysis(self, claim: str, sources: list) -> dict:
        """
        Fallback heuristic-based analysis when AI is unavailable.
        Categorizes sources by trust level and determines verdict.
        
        Returns:
            dict with verdict, confidence, explanation, and empty structured fields
        """
        # Categorize sources by trust level
        from .config import TRUSTED_FACTCHECK_DOMAINS, TRUSTED_DOMAINS
        
        factcheck_sites = []
        high_trust = []
        medium_trust = []
        
        for source in sources:
            domain = source.get('domain', '').lower()
            if any(trusted in domain for trusted in TRUSTED_FACTCHECK_DOMAINS):
                factcheck_sites.append(source)
            elif any(trusted in domain for trusted in TRUSTED_DOMAINS):
                high_trust.append(source)
            else:
                medium_trust.append(source)
        
        total_trusted = len(factcheck_sites) + len(high_trust)
        
        # Determine verdict based on source quality
        if factcheck_sites:
            # We have fact-check sources - give them high weight
            confidence = min(90, 70 + len(factcheck_sites) * 10)
            verdict = self._extract_verdict_from_snippets(factcheck_sites)
            explanation = (
                f"Found {len(factcheck_sites)} fact-check source(s) addressing this claim. "
                f"Total of {len(sources)} sources analyzed. (Heuristic analysis)"
            )
        elif total_trusted >= 3:
            # Multiple trusted sources
            confidence = min(85, 50 + total_trusted * 10)
            verdict = Verdict.TRUE
            explanation = (
                f"Found {total_trusted} trusted sources discussing this topic. "
                f"No explicit fact-checks found, but sources appear credible. (Heuristic analysis)"
            )
        elif total_trusted >= 1:
            # Some trusted sources
            confidence = 50
            verdict = Verdict.PARTIALLY_TRUE
            explanation = (
                f"Found {total_trusted} trusted source(s). "
                f"Limited evidence available for full verification. (Heuristic analysis)"
            )
        else:
            # Only unknown sources
            confidence = 30
            verdict = Verdict.UNVERIFIABLE
            explanation = (
                "Could not find trusted sources to verify this claim. "
                "The sources found are of unknown reliability. (Heuristic analysis)"
            )
        
        # Calculate confidence breakdown for transparency
        source_quality_score = min(25, len(factcheck_sites) * 10 + len(high_trust) * 5)
        source_quantity_score = min(20, len(sources) * 2)
        factcheck_bonus = 25 if factcheck_sites else 0
        consensus_score = min(30, confidence - source_quality_score - source_quantity_score - factcheck_bonus)
        
        confidence_breakdown = {
            'source_quality': source_quality_score,
            'source_quantity': source_quantity_score,
            'factcheck_found': factcheck_bonus,
            'consensus': max(0, consensus_score),
            'total': confidence,
            'explanation': (
                f"Quality: {source_quality_score}/25 (trusted sources), "
                f"Quantity: {source_quantity_score}/20 ({len(sources)} sources), "
                f"Fact-check: {factcheck_bonus}/25, "
                f"Consensus: {max(0, consensus_score)}/30"
            )
        }
        
        return {
            'verdict': verdict,
            'confidence': confidence,
            'confidence_breakdown': confidence_breakdown,
            'explanation': explanation,
            'summary': {'one_liner': explanation, 'key_points': []},
            'detailed_analysis': {},
            'claims': [],
            'ai_analyzed': False
        }
    
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
