"""
Fact Checker - Main Pipeline
Orchestrates the fact-checking process with AI-powered analysis.
"""
import hashlib
from datetime import datetime
from functools import lru_cache
from .input_classifier import InputClassifier
from .content_extractor import ContentExtractor
from .web_searcher import WebSearcher
from .temporal_analyzer import TemporalAnalyzer
from .credibility_manager import CredibilityManager
from .config import Verdict

# Import AI analyzer from separate ai module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai import AIAnalyzer


# Module-level cache for fact-check results with TTL support
# Cache structure: {cache_key: {'result': dict, 'timestamp': float, 'access_count': int}}
_claim_cache = {}
_CACHE_MAX_SIZE = 50
_CACHE_TTL_HOURS = 24  # Default TTL: 24 hours


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
        self.temporal_analyzer = TemporalAnalyzer()
        self.credibility_manager = CredibilityManager()
    
    def _get_cache_key(self, user_input: str) -> str:
        """Generate a hash key for caching claim results."""
        return hashlib.md5(user_input.strip().lower().encode()).hexdigest()
    
    def _is_cache_valid(self, cache_entry: dict) -> bool:
        """Check if a cache entry is still valid based on TTL."""
        import time
        current_time = time.time()
        cache_time = cache_entry.get('timestamp', 0)
        age_hours = (current_time - cache_time) / 3600
        return age_hours < _CACHE_TTL_HOURS
    
    def _get_cached_result(self, cache_key: str) -> dict:
        """Check if result exists in cache and is still valid."""
        cache_entry = _claim_cache.get(cache_key)
        if cache_entry and self._is_cache_valid(cache_entry):
            # Update access metadata
            cache_entry['access_count'] = cache_entry.get('access_count', 0) + 1
            cache_entry['last_accessed'] = __import__('time').time()
            return cache_entry['result']
        elif cache_entry:
            # Cache expired, remove it
            del _claim_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: dict):
        """Store result in cache with TTL and metadata."""
        import time
        global _claim_cache
        # Simple LRU-like behavior: if cache is full, remove oldest entry
        if len(_claim_cache) >= _CACHE_MAX_SIZE:
            # Remove oldest by timestamp
            oldest_key = min(_claim_cache.keys(), 
                           key=lambda k: _claim_cache[k].get('timestamp', 0))
            del _claim_cache[oldest_key]
        
        _claim_cache[cache_key] = {
            'result': result,
            'timestamp': time.time(),
            'access_count': 1,
            'last_accessed': time.time()
        }
    
    def clear_cache(self):
        """Clear the claim cache."""
        global _claim_cache
        _claim_cache = {}
    
    def cache_info(self) -> dict:
        """Get cache statistics."""
        import time
        valid_entries = sum(1 for entry in _claim_cache.values() 
                          if self._is_cache_valid(entry))
        return {
            'size': len(_claim_cache),
            'valid_entries': valid_entries,
            'expired_entries': len(_claim_cache) - valid_entries,
            'max_size': _CACHE_MAX_SIZE,
            'ttl_hours': _CACHE_TTL_HOURS
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
        Now includes temporal analysis to search from the relevant year.
        
        Args:
            user_input: Text claim, question, or URL to verify
            
        Returns:
            dict with verdict, confidence, sources, temporal context, and explanation (enhanced)
        """
        # Step 1: Classify input
        classification = self.classifier.classify(user_input)
        input_type = classification['type']
        
        # Step 2: Extract claim(s) to verify
        url_for_temporal = None
        if input_type == 'url':
            url_for_temporal = user_input
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
        
        # Step 2.5: TEMPORAL ANALYSIS - Extract time context
        temporal_context = self.temporal_analyzer.extract_temporal_context(
            claim, 
            url=url_for_temporal
        )
        search_period_description = self.temporal_analyzer.format_search_period_description(
            temporal_context
        )
        
        # Step 3: DEEP SEARCH - Multiple query variations with temporal context
        all_sources = []
        search_queries = self._generate_search_queries(claim, temporal_context)
        
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
        
        # Add deep scan metadata with temporal context
        result['deep_scan'] = True
        result['queries_used'] = len(search_queries)
        result['total_sources_found'] = len(all_sources)
        result['unique_sources'] = len(unique_sources)
        result['temporal_context'] = {
            'search_year_from': temporal_context['search_year_from'],
            'time_period': temporal_context['time_period'],
            'is_historical': temporal_context['is_historical'],
            'is_recent': temporal_context['is_recent'],
            'description': search_period_description,
            'years_mentioned': temporal_context['years_mentioned']
        }
        result['cached'] = False
        
        return result
    
    def _generate_search_queries(self, claim: str, temporal_context: dict = None) -> list:
        """
        Generate multiple search query variations for deep scanning.
        Now includes temporal context for year-specific searches.
        
        Args:
            claim: The main claim to search for
            temporal_context: Optional temporal context with years and time period
            
        Returns:
            list of search queries
        """
        queries = [claim]  # Original claim
        
        # Add fact-check focused query
        queries.append(f"fact check {claim}")
        
        # Add verification query
        queries.append(f"is it true that {claim}")
        
        # Add temporal queries if context is available
        if temporal_context and temporal_context.get('search_year_from'):
            year = temporal_context['search_year_from']
            
            # Add year-specific queries for historical claims
            if temporal_context.get('is_historical'):
                queries.append(f"{claim} {year}")
                queries.append(f"{claim} archive {year}")
            
            # Add "since year" query for context
            if year < datetime.now().year:
                queries.append(f"{claim} since {year}")
        
        # Extract key entities/keywords for additional searches
        words = claim.split()
        if len(words) > 5:
            # Use first 5 significant words
            key_words = [w for w in words if len(w) > 3][:5]
            queries.append(" ".join(key_words) + " fact check")
        
        # Add news-focused query
        queries.append(f"{claim} news verification")
        
        return queries[:7]  # Limit to 7 queries to balance coverage and rate limiting
    
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
        # Categorize sources by trust level using CredibilityManager
        factcheck_sites = []
        high_trust = []
        medium_trust = []
        low_trust = []
        
        total_trust_score = 0
        valid_sources_count = 0
        
        for source in sources:
            domain = source.get('domain', '').lower()
            cred_info = self.credibility_manager.get_credibility(domain)
            trust_score = cred_info.get('trust', 50)
            category = cred_info.get('category', 'unknown')
            
            # Enrich source with credibility data
            source['trust_score'] = trust_score
            source['trust_level'] = self.credibility_manager.get_trust_level(domain)
            source['category'] = category
            
            if category == 'factcheck' or trust_score >= 85:
                factcheck_sites.append(source)
            elif trust_score >= 70:
                high_trust.append(source)
            elif trust_score >= 50:
                medium_trust.append(source)
            else:
                low_trust.append(source)
            
            total_trust_score += trust_score
            valid_sources_count += 1
            
        avg_trust = total_trust_score / max(1, valid_sources_count)
        total_trusted = len(factcheck_sites) + len(high_trust)
        
        # Determine verdict based on source quality
        if factcheck_sites:
            # We have fact-check sources - give them high weight
            confidence = min(95, 75 + len(factcheck_sites) * 10)
            verdict = self._extract_verdict_from_snippets(factcheck_sites)
            explanation = (
                f"Found {len(factcheck_sites)} high-trust/fact-check source(s) addressing this claim. "
                f"Average source trust score: {avg_trust:.1f}/100. (Heuristic analysis)"
            )
        elif total_trusted >= 3:
            # Multiple trusted sources
            confidence = min(85, 50 + total_trusted * 10)
            verdict = Verdict.TRUE
            explanation = (
                f"Found {total_trusted} trusted sources discussing this topic. "
                f"No explicit fact-checks found, but sources appear credible (Trust > 70). (Heuristic analysis)"
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
                "The sources found are of unknown or low reliability. (Heuristic analysis)"
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
