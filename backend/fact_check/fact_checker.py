"""
Fact Checker - Main Pipeline
Orchestrates the fact-checking process with AI-powered analysis.
"""
import hashlib
import re
from datetime import datetime
from functools import lru_cache
from .input_classifier import InputClassifier
from .content_extractor import ContentExtractor
from .web_searcher import WebSearcher
from .temporal_analyzer import TemporalAnalyzer
from .credibility_manager import CredibilityManager
from .config import (
    FULL_TEXT_ENRICH_LIMIT,
    FULL_TEXT_MAX_CHARS,
    HIGH_CONFIDENCE_THRESHOLD,
    MAX_RESPONSE_SOURCES,
    PRIMARY_EVIDENCE_CATEGORIES,
    SOURCE_ADMISSION_POLICY,
    Verdict,
)

# Import AI analyzer from the AI package
import concurrent.futures
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AI import AIAnalyzer


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
            result = cached_result.copy()
            result['cached'] = True
            return result
        
        # Step 1: Classify input
        classification = self.classifier.classify(user_input)
        input_type = classification['type']
        
        # Step 2: Extract claim(s) to verify
        if input_type == 'url':
            # Extract content from URL
            extracted = self.extractor.extract_from_url(user_input)
            
            if not extracted['success']:
                print(f"URL extraction failed ({extracted.get('error')}), falling back to search verification.")
                # Fallback: Use the URL itself as the claim to check via search
                claim = user_input
                classification['claim'] = claim
                classification['url_title'] = "Content Unavailable (Verification via Search)"
                classification['url_content'] = ""
            else:
                # For URLs, the title is usually the most accurate representation of the core claim.
                # If title is missing, fallback to the first extracted structural claim.
                claims = extracted.get('claims', [])
                if extracted.get('title') and extracted['title'] not in ["Untitled", "None"]:
                    claim = extracted['title']
                elif claims:
                    claim = claims[0]
                else:
                    claim = user_input
                
                classification['claim'] = claim
                classification['url_title'] = extracted['title']
                classification['url_content'] = extracted['content']
        else:
            claim = classification['claim']
        
        if not claim:
            return self._error_response(user_input, "Could not extract a claim to verify")
        
        # Step 3: Atomic Search & Evidence Gathering
        # 3a. Decompose complex claims into atomic facts for better search
        print(f"Decomposing claim: {claim}")
        atomic_claims = [claim]
        try:
            atomic_claims = self.ai_analyzer.decompose_claim(claim)
            # Limit to 3 sub-claims to prevent explosion
            atomic_claims = atomic_claims[:3]
            print(f"Atomic decomposition: {atomic_claims}")
        except Exception as e:
            print(f"Decomposition failed, using original claim: {e}")
        
        # 3b. Search for each atomic claim
        all_sources = []
        seen_urls = set()
        
        print("Running atomic searches...")
        for sub_claim in atomic_claims:
            print(f"  Searching for: {sub_claim}")
            search_results = self.searcher.search(sub_claim)
            new_sources = search_results.get('sources', [])
            
            for s in new_sources:
                if s['url'] not in seen_urls:
                    seen_urls.add(s['url'])
                    # Tag source with the specific sub-claim it was found for
                    s['related_to_claim'] = sub_claim
                    all_sources.append(s)
        
        sources = all_sources
        sources = self._apply_source_policy(sources)
        verdict_sources = [s for s in sources if s.get('include_in_verdict', False)]
        
        if not sources:
            return self._build_response(
                classification,
                {
                    'verdict': Verdict.UNVERIFIABLE,
                    'confidence': 10,
                    'confidence_breakdown': {
                        'source_quality': 0,
                        'source_quantity': 0,
                        'factcheck_found': 0,
                        'consensus': 0,
                        'total': 10,
                        'explanation': 'No sources were found for this claim.',
                    },
                    'explanation': 'Could not find any sources to verify this claim.',
                    'summary': {
                        'one_liner': 'No evidence found from searchable sources.',
                        'key_points': [],
                    },
                    'detailed_analysis': {},
                    'source_analysis': [],
                    'contradictions_found': False,
                    'claims': [],
                    'ai_analyzed': False,
                },
                []
            )

        temporal_context = self.temporal_analyzer.extract_temporal_context(claim)
        search_period_description = self.temporal_analyzer.format_search_period_description(
            temporal_context
        )

        if not verdict_sources:
            ai_result = {
                'verdict': Verdict.UNVERIFIABLE,
                'confidence': 25,
                'confidence_breakdown': {
                    'source_quality': 5,
                    'source_quantity': min(20, len(sources) * 2),
                    'factcheck_found': 0,
                    'consensus': 0,
                    'total': 25,
                    'explanation': 'Only contextual or policy-excluded sources were found.',
                },
                'explanation': (
                    'Sources were found, but none met the reliable-source policy required '
                    'to influence the verdict.'
                ),
                'summary': {
                    'one_liner': 'No eligible reliable sources were found for verdict scoring.',
                    'key_points': [],
                },
                'detailed_analysis': {
                    'overview': 'Search returned contextual evidence only.',
                    'methodology': 'Applied source policy filters before verdict synthesis.',
                    'context': '',
                    'limitations': 'Reliable-source threshold not satisfied.',
                },
                'source_analysis': [],
                'contradictions_found': False,
                'claims': [],
                'ai_analyzed': False,
                'temporal_context': {
                    'search_year_from': temporal_context['search_year_from'],
                    'time_period': temporal_context['time_period'],
                    'is_historical': temporal_context['is_historical'],
                    'is_recent': temporal_context['is_recent'],
                    'description': search_period_description,
                    'years_mentioned': temporal_context['years_mentioned'],
                },
            }
            result = self._build_response(classification, ai_result, sources)
            result['cached'] = False
            self._cache_result(cache_key, result.copy())
            return result
        
        # Step 4: Enrich top sources with full text (Deep Read)
        # We pick up to 3 high-trust sources to read fully
        self._enrich_sources_with_full_text(verdict_sources)
        
        # Temporal context for calibration and filtering
        # Temporal context already extracted above.
        
        # Step 5: Analyze sources and determine verdict using AI
        ai_result = self._analyze_sources(claim, verdict_sources, temporal_context)
        ai_result['temporal_context'] = {
            'search_year_from': temporal_context['search_year_from'],
            'time_period': temporal_context['time_period'],
            'is_historical': temporal_context['is_historical'],
            'is_recent': temporal_context['is_recent'],
            'description': search_period_description,
            'years_mentioned': temporal_context['years_mentioned'],
        }
        
        # Step 6: Build response with all AI analysis data
        result = self._build_response(
            classification,
            ai_result,
            sources
        )
        result['reliable_source_count'] = len(verdict_sources)
        result['excluded_source_count'] = max(0, len(sources) - len(verdict_sources))
        
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
                print(f"Deep check URL extraction failed ({extracted.get('error')}), falling back to search.")
                claim = user_input
                classification['claim'] = claim
                classification['url_title'] = "Content Unavailable (Verification via Search)"
                classification['url_content'] = ""
            else:
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
        # Use content if available (for URLs) to find the date, otherwise use claim
        temporal_text = claim
        if classification.get('url_content'):
             # Create a snippet of the start of content where dates usually appear
             temporal_text = claim + "\n" + classification['url_content'][:2000]
             
        temporal_context = self.temporal_analyzer.extract_temporal_context(
            temporal_text, 
            url=url_for_temporal
        )
        search_period_description = self.temporal_analyzer.format_search_period_description(
            temporal_context
        )
        
        # Step 3: DEEP SEARCH - Multiple query variations with temporal context
        # PARALLELIZED for better performance
        all_sources = []
        search_queries = self._generate_search_queries(claim, temporal_context)
        
        print(f"Executing {len(search_queries)} search queries in parallel...")
        
        # Parallel search execution
        def execute_search(query):
            """Execute a single search query."""
            try:
                search_results = self.searcher.search(query)
                sources = search_results.get('sources', [])
                # Add query origin to each source
                for source in sources:
                    source['search_query'] = query
                return sources
            except Exception as e:
                print(f"Search failed for query '{query}': {e}")
                return []
        
        # Execute searches concurrently (max 3 at a time to avoid rate limiting)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_query = {executor.submit(execute_search, query): query for query in search_queries}
            
            for future in concurrent.futures.as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    sources = future.result()
                    all_sources.extend(sources)
                    print(f"[OK] Query '{query[:50]}...' returned {len(sources)} sources")
                except Exception as e:
                    print(f"[ERR] Query '{query}' failed with exception: {e}")
        
        print(f"Total sources collected: {len(all_sources)}")
        
        # Deduplicate sources by URL
        seen_urls = set()
        unique_sources = []
        for source in all_sources:
            url = source.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_sources.append(source)
        
        print(f"Unique sources after deduplication: {len(unique_sources)}")
        
        if not unique_sources:
            return self._build_response(
                classification,
                {
                    'verdict': Verdict.UNVERIFIABLE,
                    'confidence': 10,
                    'confidence_breakdown': {
                        'source_quality': 0,
                        'source_quantity': 0,
                        'factcheck_found': 0,
                        'consensus': 0,
                        'total': 10,
                        'explanation': 'No sources were found for this deep check.',
                    },
                    'explanation': 'Could not find any sources to verify this claim.',
                    'summary': {
                        'one_liner': 'Deep scan found no evidence for this claim.',
                        'key_points': [],
                    },
                    'detailed_analysis': {},
                    'source_analysis': [],
                    'contradictions_found': False,
                    'claims': [],
                    'ai_analyzed': False,
                },
                []
            )

        unique_sources = self._apply_source_policy(unique_sources)
        verdict_sources = [s for s in unique_sources if s.get('include_in_verdict', False)]
        
        # Step 4: Enrich top sources with full text (Round 1)
        self._enrich_sources_with_full_text(verdict_sources)
        
        # Step 4.5: ITERATIVE REASONING LOOP (Agentic Search)
        # Ask AI: "Do we have enough info? What is missing?"
        print("Analyzing knowledge gaps...")
        search_queries_round_2 = []
        try:
            search_queries_round_2 = self.ai_analyzer.identify_knowledge_gaps(claim, unique_sources)
        except Exception as e:
            print(f"Gap analysis failed: {e}")

        if search_queries_round_2:
            print(f"Identified gaps. Starting Round 2 search with {len(search_queries_round_2)} queries...")
            round_2_sources = []
            
            # Prepare queries with temporal context
            prepared_queries = []
            for query in search_queries_round_2:
                # Enforce temporal context in follow-up queries if needed
                if temporal_context.get('search_year_from') and str(temporal_context['search_year_from']) not in query:
                    if temporal_context.get('is_historical'):
                        query = f"{query} {temporal_context['search_year_from']}"
                prepared_queries.append(query)
            
            # Parallel Round 2 search execution
            def execute_round2_search(query):
                """Execute a Round 2 search query."""
                try:
                    search_results = self.searcher.search(query)
                    r2_sources = search_results.get('sources', [])
                    for source in r2_sources:
                        source['search_query'] = query
                        source['round'] = 2
                    return r2_sources
                except Exception as e:
                    print(f"Round 2 search failed for '{query}': {e}")
                    return []
            
            # Execute Round 2 searches concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_to_query = {executor.submit(execute_round2_search, q): q for q in prepared_queries}
                
                for future in concurrent.futures.as_completed(future_to_query):
                    query = future_to_query[future]
                    try:
                        sources = future.result()
                        round_2_sources.extend(sources)
                        print(f"[OK] Round 2 query '{query[:50]}...' returned {len(sources)} sources")
                    except Exception as e:
                        print(f"[ERR] Round 2 query '{query}' failed: {e}")

            # Process Round 2 sources
            new_unique_sources = []
            seen_urls_r2 = {s.get('url') for s in unique_sources if s.get('url')}  # Start with existing URLs
            
            for source in round_2_sources:
                url = source.get('url', '')
                if url and url not in seen_urls_r2:
                    seen_urls_r2.add(url)
                    new_unique_sources.append(source)
            
            if new_unique_sources:
                new_unique_sources = self._apply_source_policy(new_unique_sources)
                print(f"Round 2 found {len(new_unique_sources)} new unique sources.")
                # Enrich new sources
                round2_verdict_sources = [s for s in new_unique_sources if s.get('include_in_verdict', False)]
                self._enrich_sources_with_full_text(round2_verdict_sources)
                # Combine sources (Round 1 + Round 2)
                unique_sources.extend(new_unique_sources)
                verdict_sources.extend(round2_verdict_sources)
            else:
                 print("Round 2 found no new unique sources.")

        if not verdict_sources:
            ai_result = {
                'verdict': Verdict.UNVERIFIABLE,
                'confidence': 25,
                'confidence_breakdown': {
                    'source_quality': 5,
                    'source_quantity': min(20, len(unique_sources) * 2),
                    'factcheck_found': 0,
                    'consensus': 0,
                    'total': 25,
                    'explanation': 'Deep scan found only contextual or policy-excluded sources.',
                },
                'explanation': (
                    'Deep scan completed, but no sources met the reliable-source policy '
                    'required for verdict scoring.'
                ),
                'summary': {
                    'one_liner': 'Deep scan found no reliable verdict-eligible sources.',
                    'key_points': [],
                },
                'detailed_analysis': {
                    'overview': 'Deep scan sources were contextual only.',
                    'methodology': 'Applied source policy filters before verdict synthesis.',
                    'context': '',
                    'limitations': 'Reliable-source threshold not satisfied.',
                },
                'source_analysis': [],
                'contradictions_found': False,
                'claims': [],
                'ai_analyzed': False,
            }
        else:
            # Step 5: Enhanced AI analysis with combined verdict-eligible sources
            ai_result = self._analyze_sources(claim, verdict_sources, temporal_context)

        ai_result['temporal_context'] = {
            'search_year_from': temporal_context['search_year_from'],
            'time_period': temporal_context['time_period'],
            'is_historical': temporal_context['is_historical'],
            'is_recent': temporal_context['is_recent'],
            'description': search_period_description,
            'years_mentioned': temporal_context['years_mentioned'],
        }
        
        # Step 6: Build response with deep scan metadata
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
        result['reliable_source_count'] = len(verdict_sources)
        result['excluded_source_count'] = max(0, len(unique_sources) - len(verdict_sources))
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
        """
        queries = []
        is_url = claim.strip().startswith('http')
        
        # 1. Base query
        queries.append(claim)
        
        # If it's a URL, try to extract a better text query from the slug/title
        text_query = claim
        if is_url:
            from urllib.parse import urlparse
            try:
                # Extract slug from URL path
                path = urlparse(claim).path
                # Replace dashes/underscores with spaces and remove extension
                slug = path.split('/')[-1].replace('-', ' ').replace('_', ' ').split('.')[0]
                if len(slug) > 10:  # If we have a decent slug
                    text_query = slug
                    queries.append(text_query)  # Add the cleaned slug as a query
            except:
                pass
        
        # 2. Fact-check query
        queries.append(f"fact check {text_query}")
        queries.append(f"site:factcheck.org {text_query}")
        queries.append(f"site:reuters.com fact check {text_query}")
        queries.append(f"official data {text_query}")
        
        # 3. Verification query
        if not is_url:
            queries.append(f"is it true that {text_query}")
            # ADVERSARIAL SEARCH: Explicitly look for debunking
            queries.append(f"{text_query} fake")
            queries.append(f"{text_query} hoax")
        else:
            queries.append(f"verify {text_query}")
            
        # 4. Temporal queries
        if temporal_context and temporal_context.get('search_year_from'):
            year = temporal_context['search_year_from']
            if temporal_context.get('is_historical'):
                queries.append(f"{text_query} {year}")
                queries.append(f"{text_query} archive {year}")
            elif year < datetime.now().year:
                queries.append(f"{text_query} since {year}")
        
        # 5. Keywords query
        words = text_query.split()
        if len(words) > 5:
            key_words = [w for w in words if len(w) > 3][:5]
            queries.append(" ".join(key_words) + " fact check")

        # Deduplicate while preserving order and cap query fan-out.
        deduped = []
        seen = set()
        for query in queries:
            normalized = query.lower().strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                deduped.append(query)

        return deduped[:10]

    def _apply_source_policy(self, sources: list) -> list:
        """Normalize source metadata and apply verdict-eligibility policy."""
        if not sources:
            return []

        processed = []
        seen_urls = set()

        for source in sources:
            url = source.get('url', '')
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)

            domain = source.get('domain', '').lower().strip()
            policy = self.credibility_manager.get_source_policy(domain)

            merged = {**source}
            merged['source_category'] = policy.get('category', merged.get('source_category', 'unknown'))
            merged['source_tier'] = policy.get('tier', merged.get('source_tier', 4))
            merged['is_factcheck_site'] = bool(policy.get('is_factcheck_site', merged.get('is_factcheck_site', False)))
            merged['bias'] = policy.get('bias', merged.get('bias', 'unknown'))
            merged['trust_level'] = policy.get('trust_level', merged.get('trust_level', 'unknown'))
            merged['include_in_verdict'] = bool(policy.get('include_in_verdict', merged.get('include_in_verdict', False)))
            merged['raw_trust_score'] = int(policy.get('raw_trust_score', merged.get('raw_trust_score', 50)))

            try:
                merged['trust_score'] = int(round(float(merged.get('trust_score', policy.get('trust_score', 50)))))
            except (TypeError, ValueError):
                merged['trust_score'] = int(policy.get('trust_score', 50))

            if merged.get('source_category') in PRIMARY_EVIDENCE_CATEGORIES:
                merged['evidence_role'] = 'primary'
            elif merged.get('include_in_verdict'):
                merged['evidence_role'] = 'secondary'
            else:
                merged['evidence_role'] = 'context'

            if merged.get('include_in_verdict'):
                merged['source_reason'] = merged.get(
                    'source_reason',
                    'Eligible for verdict scoring under reliable-source policy.',
                )
            elif merged.get('source_category') in {'unreliable', 'satire'}:
                merged['excluded_reason'] = 'Blocked source category (unreliable/satire).'
                merged['source_reason'] = merged.get(
                    'source_reason',
                    'Shown for context only; excluded from verdict scoring.',
                )
            else:
                merged['excluded_reason'] = 'Source tier exceeds verdict-eligible threshold.'
                merged['source_reason'] = merged.get(
                    'source_reason',
                    'Context-only source due to reliability tier policy.',
                )

            processed.append(merged)

        processed.sort(
            key=lambda x: (
                1 if x.get('include_in_verdict') else 0,
                x.get('trust_score', 0),
                1 if x.get('is_factcheck_site') else 0,
            ),
            reverse=True,
        )

        return processed

    def _has_primary_evidence(self, sources: list) -> bool:
        """Return True if at least one verdict-eligible source is primary evidence."""
        return any(
            s.get('include_in_verdict', False)
            and s.get('source_category', 'unknown') in PRIMARY_EVIDENCE_CATEGORIES
            for s in sources
        )
    
    def _analyze_sources(self, claim: str, sources: list, temporal_context: dict = None) -> dict:
        """
        Analyze sources using AI with fallback to heuristic analysis.
        
        Returns:
            dict with verdict, confidence, summary, detailed_analysis, claims
        """
        if not sources:
            return self._heuristic_analysis(claim, [])

        # Tag stance and temporal alignment before AI so both AI and fallback can use it
        sources = self._apply_stance_tags(claim, sources)
        sources = self._apply_temporal_filter(sources, temporal_context)

        if not sources:
            return self._heuristic_analysis(claim, [])

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
        eligible_sources = [s for s in sources if s.get('include_in_verdict', True)]

        if not eligible_sources:
            return {
                'verdict': Verdict.UNVERIFIABLE,
                'confidence': 25,
                'confidence_breakdown': {
                    'source_quality': 5,
                    'source_quantity': 0,
                    'factcheck_found': 0,
                    'consensus': 0,
                    'total': 25,
                    'explanation': 'No verdict-eligible reliable sources were available.',
                },
                'explanation': 'No reliable verdict-eligible sources were available for analysis.',
                'summary': {
                    'one_liner': 'Unable to verify due to insufficient reliable evidence.',
                    'key_points': [],
                },
                'detailed_analysis': {
                    'overview': 'The analysis found no sources allowed to influence verdict scoring.',
                    'methodology': 'Applied source policy gates before heuristic synthesis.',
                    'context': '',
                    'limitations': 'Reliable-source threshold not satisfied.',
                },
                'source_analysis': [],
                'contradictions_found': False,
                'claims': [],
                'ai_analyzed': False,
            }

        # Categorize sources by trust level using policy-normalized values.
        factcheck_sites = []
        high_trust = []
        medium_trust = []
        low_trust = []
        support_sources = []
        refute_sources = []
        
        total_trust_score = 0
        valid_sources_count = 0
        
        for source in eligible_sources:
            domain = source.get('domain', '').lower()
            category = source.get('source_category', self.credibility_manager.get_source_category(domain))
            trust_score = int(source.get('trust_score', self.credibility_manager.get_adjusted_trust_score(domain)))

            # Keep normalized fields available in fallback output.
            source['trust_score'] = trust_score
            source['trust_level'] = source.get('trust_level', self.credibility_manager.get_trust_level(domain))
            source['category'] = category
            
            if category == 'factcheck' or trust_score >= 85:
                factcheck_sites.append(source)
            elif trust_score >= 70:
                high_trust.append(source)
            elif trust_score >= 50:
                medium_trust.append(source)
            else:
                low_trust.append(source)
            stance = source.get('stance', 'neutral')
            if stance == 'supports':
                support_sources.append(source)
            elif stance == 'refutes':
                refute_sources.append(source)
            
            total_trust_score += trust_score
            valid_sources_count += 1
            
        avg_trust = total_trust_score / max(1, valid_sources_count)
        total_trusted = len(factcheck_sites) + len(high_trust)
        primary_evidence_found = self._has_primary_evidence(eligible_sources)
        
        # Determine verdict based on source quality and stance consensus
        support_high = sum(1 for s in support_sources if s.get('trust_score', 0) >= 70)
        refute_high = sum(1 for s in refute_sources if s.get('trust_score', 0) >= 70)
        factcheck_support = sum(1 for s in support_sources if s.get('category') == 'factcheck')
        factcheck_refute = sum(1 for s in refute_sources if s.get('category') == 'factcheck')
        contradictions_found = support_high > 0 and refute_high > 0

        if factcheck_refute >= 1 and (refute_high + factcheck_refute) >= 2:
            verdict = Verdict.FALSE
            confidence = min(95, 70 + 10 * (refute_high + factcheck_refute))
            explanation = (
                f"Found {refute_high + factcheck_refute} high-trust/refuting sources including "
                f"{factcheck_refute} fact-check site(s). (Heuristic analysis)"
            )
        elif factcheck_support >= 1 and (support_high + factcheck_support) >= 2:
            verdict = Verdict.TRUE
            confidence = min(95, 70 + 10 * (support_high + factcheck_support))
            explanation = (
                f"Found {support_high + factcheck_support} high-trust/supporting sources including "
                f"{factcheck_support} fact-check site(s). (Heuristic analysis)"
            )
        elif factcheck_sites:
            confidence = min(90, 70 + len(factcheck_sites) * 5)
            verdict = self._extract_verdict_from_snippets(factcheck_sites)
            explanation = (
                f"Fact-check sources found but stance consensus is weak. "
                f"Average source trust score: {avg_trust:.1f}/100. (Heuristic analysis)"
            )
        elif total_trusted >= 3:
            confidence = min(80, 50 + total_trusted * 8)
            verdict = Verdict.PARTIALLY_TRUE
            explanation = (
                f"Found {total_trusted} trusted sources but stance signals are mixed/insufficient. (Heuristic analysis)"
            )
        elif total_trusted >= 1:
            confidence = 45
            verdict = Verdict.UNVERIFIABLE
            explanation = (
                f"Found {total_trusted} trusted source(s) but not enough agreement to verify. (Heuristic analysis)"
            )
        else:
            confidence = 30
            verdict = Verdict.UNVERIFIABLE
            explanation = (
                "Could not find trusted sources to verify this claim. "
                "The sources found are of unknown or low reliability. (Heuristic analysis)"
            )

        # Reduce certainty on conflicting evidence.
        if contradictions_found and verdict in {Verdict.TRUE, Verdict.FALSE}:
            confidence = min(confidence, 75)
            explanation += " Conflicting source signals reduced confidence."

        # Enforce primary-evidence gate for high confidence.
        if confidence >= HIGH_CONFIDENCE_THRESHOLD and not primary_evidence_found:
            confidence = HIGH_CONFIDENCE_THRESHOLD - 5
            explanation += " High confidence capped: no primary-evidence source found."
        
        # Calculate confidence breakdown for transparency
        source_quality_score = min(25, len(factcheck_sites) * 10 + len(high_trust) * 5)
        source_quantity_score = min(20, len(eligible_sources) * 2)
        factcheck_bonus = 25 if factcheck_sites else 0
        consensus_score = min(30, confidence - source_quality_score - source_quantity_score - factcheck_bonus)

        source_analysis = []
        for source in eligible_sources[:8]:
            source_analysis.append({
                'source_title': source.get('title', 'Unknown Source'),
                'stance': source.get('stance', 'neutral').upper(),
                'relevance': min(100, max(0, int(source.get('trust_score', 50)))),
                'key_excerpt': source.get('original_snippet', source.get('snippet', ''))[:300],
            })
        
        confidence_breakdown = {
            'source_quality': source_quality_score,
            'source_quantity': source_quantity_score,
            'factcheck_found': factcheck_bonus,
            'consensus': max(0, consensus_score),
            'total': confidence,
            'explanation': (
                f"Quality: {source_quality_score}/25 (trusted sources), "
                f"Quantity: {source_quantity_score}/20 ({len(eligible_sources)} sources), "
                f"Primary evidence: {'yes' if primary_evidence_found else 'no'}, "
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
            'source_analysis': source_analysis,
            'contradictions_found': contradictions_found,
            'claims': [],
            'ai_analyzed': False
        }
    

    
    def _enrich_sources_with_full_text(self, sources: list):
        """
        Fetch full text for top trusted sources to improve AI context.
        Modifies sources in-place.
        """
        if not sources:
            return
            
        print("Enriching top sources with full text...")
        
        # Filter for high/medium-high trust or fact-check sites
        candidates = []
        for i, source in enumerate(sources):
            # Prioritize fact-checkers and high trust
            score = int(source.get('trust_score', 50))
            is_fc = source.get('is_factcheck_site', False)
            
            # Skip if it's a PDF (ContentExtractor might struggle/slow down) or youtube
            url = source.get('url', '').lower()
            if '.pdf' in url or 'youtube.com' in url or 'youtu.be' in url:
                continue
                
            priority = 0
            if is_fc: priority = 3
            elif score >= 80: priority = 2
            elif score >= 60: priority = 1
            
            if priority > 0:
                candidates.append((i, source, priority))
        
        # Sort by priority and take top N
        candidates.sort(key=lambda x: x[2], reverse=True)
        top_candidates = candidates[:FULL_TEXT_ENRICH_LIMIT]
        
        if not top_candidates:
            return
            
        # Helper function to fetch single URL
        def fetch_text(idx, source):
            url = source.get('url')
            try:
                # Use extracting logic
                extracted = self.extractor.extract_from_url(url)
                if extracted['success'] and extracted['content']:
                    # Truncate to avoid blowing up context for model prompts.
                    full_text = extracted['content'][:FULL_TEXT_MAX_CHARS]
                    # Update source snippet with full text (prefixed)
                    return idx, f"[FULL TEXT] {full_text}..."
            except Exception as e:
                print(f"Failed to enrich source {url}: {e}")
            return idx, None

        # Fetch in parallel
        max_workers = min(3, len(top_candidates))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(fetch_text, idx, src) for idx, src, _ in top_candidates]

            for future in concurrent.futures.as_completed(futures):
                idx, new_text = future.result()
                if new_text:
                    # Keep UI-friendly snippet while passing richer context to analysis.
                    original_snippet = sources[idx].get('original_snippet')
                    if original_snippet is None:
                        sources[idx]['original_snippet'] = sources[idx].get('snippet', '')
                    sources[idx]['snippet'] = new_text
                    sources[idx]['full_text_available'] = True

    def _apply_stance_tags(self, claim: str, sources: list) -> list:
        """Lightweight stance tagging based on snippet/title keywords."""
        if not sources:
            return sources
        claim_lower = claim.lower()
        positive = ['true', 'correct', 'confirmed', 'verified', 'authentic', 'real', 'accurate']
        negative = ['false', 'fake', 'hoax', 'debunked', 'misleading', 'incorrect', 'untrue']

        for src in sources:
            text = f"{src.get('title','')} {src.get('snippet','')}".lower()
            stance = 'neutral'
            if any(k in text for k in negative):
                stance = 'refutes'
            elif any(k in text for k in positive):
                stance = 'supports'
            # If the claim itself is echoed, lean supports
            if stance == 'neutral' and claim_lower[:40] in text:
                stance = 'supports'
            src['stance'] = stance
        return sources

    def _apply_temporal_filter(self, sources: list, temporal_context: dict = None) -> list:
        """Tag sources with temporal match and drop ones clearly outside the claim window."""
        if not sources or not temporal_context or not temporal_context.get('search_year_from'):
            return sources
        year_floor = temporal_context['search_year_from']
        filtered = []
        year_pattern = re.compile(r"\b(19\d{2}|20\d{2})\b")

        for src in sources:
            text = f"{src.get('title','')} {src.get('snippet','')}"
            years = [int(y) for y in year_pattern.findall(text)]
            if not years:
                src['temporal_match'] = True
                filtered.append(src)
                continue
            latest = max(years)
            src['temporal_year'] = latest
            src['temporal_match'] = latest >= year_floor
            if src['temporal_match']:
                filtered.append(src)
        return filtered if filtered else sources

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
        if not isinstance(ai_result, dict):
            ai_result = {
                'verdict': Verdict.UNVERIFIABLE,
                'confidence': 0,
                'explanation': 'Analysis failed to produce a structured response.',
                'summary': {'one_liner': 'Analysis unavailable.', 'key_points': []},
                'detailed_analysis': {},
                'source_analysis': [],
                'contradictions_found': False,
                'claims': [],
                'ai_analyzed': False,
            }

        try:
            confidence = int(ai_result.get('confidence', 50))
        except (TypeError, ValueError):
            confidence = 50

        confidence = min(100, max(0, confidence))
        reliable_source_count = sum(1 for s in sources if s.get('include_in_verdict', False))
        excluded_source_count = max(0, len(sources) - reliable_source_count)
        primary_evidence_found = self._has_primary_evidence(sources)

        default_breakdown = {
            'source_quality': 0,
            'source_quantity': min(20, len(sources) * 2),
            'factcheck_found': 0,
            'consensus': 0,
            'total': confidence,
            'explanation': 'Confidence breakdown unavailable for this analysis path.',
        }

        # Limit sources to configured response size.
        top_sources = []
        for source in sources[:MAX_RESPONSE_SOURCES]:
            snippet = source.get('original_snippet', source.get('snippet', ''))
            top_sources.append({
                'title': source.get('title', 'Untitled'),
                'url': source.get('url', ''),
                'snippet': snippet[:200],
                'domain': source.get('domain', ''),
                'trust_level': source.get('trust_level', 'unknown'),
                'trust_score': source.get('trust_score', 50),
                'raw_trust_score': source.get('raw_trust_score', source.get('trust_score', 50)),
                'source_tier': source.get('source_tier', 4),
                'source_category': source.get('source_category', 'unknown'),
                'bias': source.get('bias', 'unknown'),
                'is_factcheck': source.get('is_factcheck_site', False),
                'stance': source.get('stance', 'neutral'),
                'temporal_match': source.get('temporal_match', True),
                'temporal_year': source.get('temporal_year'),
                'search_query': source.get('search_query'),
                'include_in_verdict': source.get('include_in_verdict', False),
                'source_reason': source.get('source_reason', ''),
                'excluded_reason': source.get('excluded_reason', ''),
                'evidence_role': source.get('evidence_role', 'context'),
            })
        
        return {
            'success': True,
            'input': classification['original'],
            'input_type': classification['type'],
            'claim': classification['claim'],
            'verdict': ai_result.get('verdict', 'UNVERIFIABLE'),
            'confidence': confidence,
            'sources': top_sources,
            'source_count': len(sources),
            'reliable_source_count': ai_result.get('reliable_source_count', reliable_source_count),
            'excluded_source_count': ai_result.get('excluded_source_count', excluded_source_count),
            'primary_evidence_found': ai_result.get('primary_evidence_found', primary_evidence_found),
            'source_policy': SOURCE_ADMISSION_POLICY,
            'explanation': ai_result.get('explanation', 'Analysis completed.'),
            # New structured data for tabs
            'summary': ai_result.get('summary', {'one_liner': '', 'key_points': []}),
            'detailed_analysis': ai_result.get('detailed_analysis', {}),
            'confidence_breakdown': ai_result.get('confidence_breakdown', default_breakdown),
            'source_analysis': ai_result.get('source_analysis', []),
            'contradictions_found': ai_result.get('contradictions_found', False),
            'claims': ai_result.get('claims', []),
            'temporal_context': ai_result.get('temporal_context'),
            'ai_analyzed': ai_result.get('ai_analyzed', False),
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
            'reliable_source_count': 0,
            'excluded_source_count': 0,
            'primary_evidence_found': False,
            'source_policy': SOURCE_ADMISSION_POLICY,
            'explanation': error,
            'summary': {'one_liner': error, 'key_points': []},
            'detailed_analysis': {},
            'confidence_breakdown': {
                'source_quality': 0,
                'source_quantity': 0,
                'factcheck_found': 0,
                'consensus': 0,
                'total': 0,
                'explanation': 'Request failed before analysis could run.',
            },
            'source_analysis': [],
            'contradictions_found': False,
            'claims': [],
            'temporal_context': None,
            'ai_analyzed': False,
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
