"""
Web Searcher
Searches multiple websites to find evidence for/against claims.
"""
import time
import requests
from urllib.parse import urlparse
from ddgs import DDGS
from .config import (
    USER_AGENT, REQUEST_TIMEOUT, GOOGLE_SEARCH_RESULTS,
    GOOGLE_API_KEY, GOOGLE_CSE_ID, PRIMARY_EVIDENCE_CATEGORIES
)
from .credibility_manager import CredibilityManager


class WebSearcher:
    """Searches multiple sources to verify claims."""
    
    # Rate limiting settings
    MIN_REQUEST_INTERVAL = 1.0  # Increased to 1.0s to prevent DDG blocking during deep scans
    
    def __init__(self):
        self.headers = {
            'User-Agent': USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        self._last_request_time = 0
        self.credibility_manager = CredibilityManager()
        
        # Circuit breaker for Google Search (disable if auth fails)
        self.google_broken = False
    
    def _throttle(self):
        """
        Rate limiter to prevent getting blocked by external APIs.
        Ensures minimum interval between consecutive requests.
        """
        elapsed = time.time() - self._last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()
    
    def search(self, claim: str) -> dict:
        """
        Search multiple sources for evidence about a claim.
        
        Args:
            claim: The claim to verify
            
        Returns:
            dict with 'sources' list and 'summary'
        """
        sources = []
        
        if GOOGLE_API_KEY and GOOGLE_CSE_ID and not self.google_broken:
            google_results = self._search_google_custom(claim)
            if google_results:
                sources.extend(google_results)
            else:
                # Fallback to DuckDuckGo if Google fails (or if circuit breaker tripped)
                if not self.google_broken:
                     print("Google search failed/empty. Falling back to DuckDuckGo.")
                ddg_results = self._search_duckduckgo(claim)
                sources.extend(ddg_results)
        else:
            # Direct DuckDuckGo if no keys
            ddg_results = self._search_duckduckgo(claim)
            sources.extend(ddg_results)
        
        # Search Wikipedia (always useful)
        wiki_result = self._search_wikipedia(claim)
        if wiki_result:
            sources.append(wiki_result)
        
        # Categorize and score sources
        scored_sources = self._score_sources(sources)
        
        return {
            'claim': claim,
            'sources': scored_sources,
            'total_found': len(scored_sources)
        }
    
    def _search_google_custom(self, query: str, num_results: int = None) -> list:
        """Search using Google Custom Search JSON API."""
        if num_results is None:
            num_results = GOOGLE_SEARCH_RESULTS
            
        results = []
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': GOOGLE_API_KEY,
                'cx': GOOGLE_CSE_ID,
                'q': query,
                'num': min(num_results, 10)  # Max 10 per request
            }
            
            # Rate limit
            self._throttle()
            
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            
            data = response.json()
            items = data.get('items', [])
            
            for item in items:
                results.append({
                    'source': 'google',
                    'url': item.get('link', ''),
                    'title': item.get('title', 'No title'),
                    'snippet': item.get('snippet', '')[:300],
                    'domain': urlparse(item.get('link', '')).netloc
                })
                
        except Exception as e:
            # Check for auth errors (403/401) to trip circuit breaker
            if "403" in str(e) or "401" in str(e):
                if not self.google_broken:
                    print(f"Google Search API Auth Error: {e}")
                    print("Disabling Google Search for this session and falling back to DuckDuckGo.")
                    self.google_broken = True
            else:
                print(f"Google search error: {e}")
            return []
            
        return results
    
    def _search_duckduckgo(self, query: str, num_results: int = None) -> list:
        """Search using DuckDuckGo API library."""
        if num_results is None:
            num_results = GOOGLE_SEARCH_RESULTS
        
        results = []
        
        try:
            # Rate limit before making request
            self._throttle()
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=num_results):
                    results.append({
                        'source': 'duckduckgo',
                        'url': r.get('href', ''),
                        'title': r.get('title', 'No title'),
                        'snippet': r.get('body', '')[:300],
                        'domain': urlparse(r.get('href', '')).netloc
                    })
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
        
        return results
    
    def _search_wikipedia(self, query: str) -> dict:
        """Search Wikipedia for relevant article."""
        try:
            # Rate limit before making request
            self._throttle()
            
            # Use Wikipedia API
            api_url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'opensearch',
                'search': query,
                'limit': 1,
                'format': 'json'
            }
            
            response = requests.get(
                api_url,
                params=params,
                headers=self.headers,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            data = response.json()
            
            if len(data) >= 4 and data[1] and data[3]:
                title = data[1][0]
                url = data[3][0]
                
                # Get article extract
                extract = self._get_wikipedia_extract(title)
                
                return {
                    'source': 'wikipedia',
                    'url': url,
                    'title': title,
                    'snippet': extract[:500] if extract else "",
                    'domain': 'wikipedia.org'
                }
            
        except Exception as e:
            print(f"Wikipedia search error: {e}")
        
        return None
    
    def _get_wikipedia_extract(self, title: str) -> str:
        """Get extract/summary from Wikipedia article."""
        try:
            api_url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'titles': title,
                'prop': 'extracts',
                'exintro': True,
                'explaintext': True,
                'format': 'json'
            }
            
            response = requests.get(
                api_url,
                params=params,
                headers=self.headers,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            data = response.json()
            pages = data.get('query', {}).get('pages', {})
            
            for page_id, page_data in pages.items():
                if 'extract' in page_data:
                    return page_data['extract']
            
        except requests.RequestException as e:
            print(f"Wikipedia extract request failed: {e}")
        except (KeyError, ValueError) as e:
            print(f"Wikipedia extract parsing error: {e}")
        
        return ""
    
    def _score_sources(self, sources: list) -> list:
        """Score and categorize sources by reliability using credibility database."""
        scored = []
        
        for source in sources:
            domain = source.get('domain', '').lower()
            policy = self.credibility_manager.get_source_policy(domain)
            category = policy.get('category', 'unknown')

            include_in_verdict = bool(policy.get('include_in_verdict', False))
            if include_in_verdict:
                source_reason = 'Eligible for verdict scoring under source policy.'
            elif category in {'unreliable', 'satire'}:
                source_reason = 'Excluded from verdict scoring due to unreliable/satire category.'
            else:
                source_reason = 'Used as contextual evidence only due to source tier policy.'

            evidence_role = 'primary' if category in PRIMARY_EVIDENCE_CATEGORIES else 'secondary'
            if not include_in_verdict:
                evidence_role = 'context'
            
            scored.append({
                **source,
                'trust_level': policy.get('trust_level', 'unknown'),
                'trust_score': policy.get('trust_score', 50),
                'raw_trust_score': policy.get('raw_trust_score', 50),
                'is_factcheck_site': policy.get('is_factcheck_site', False),
                'bias': policy.get('bias', 'unknown'),
                'source_category': category,
                'source_tier': policy.get('tier', 4),
                'include_in_verdict': include_in_verdict,
                'source_reason': source_reason,
                'evidence_role': evidence_role,
            })
        
        # Sort by policy eligibility first, then trust score and fact-check priority.
        scored.sort(
            key=lambda x: (
                1 if x.get('include_in_verdict') else 0,
                x.get('trust_score', 0),
                1 if x.get('is_factcheck_site') else 0,
            ),
            reverse=True,
        )
        
        return scored


# Quick test
if __name__ == '__main__':
    searcher = WebSearcher()
    
    test_claim = "The moon landing happened in 1969"
    results = searcher.search(test_claim)
    
    print(f"Searching for: {test_claim}")
    print(f"Found {results['total_found']} sources:\n")
    
    for source in results['sources'][:5]:
        print(f"  [{source['trust_level']}] {source['title']}")
        print(f"    URL: {source['url']}")
        print(f"    Snippet: {source['snippet'][:100]}...")
        print()
