"""
Web Searcher
Searches multiple websites to find evidence for/against claims.
"""
import time
import requests
from urllib.parse import urlparse
from ddgs import DDGS
from .config import (
    USER_AGENT, REQUEST_TIMEOUT, GOOGLE_SEARCH_RESULTS
)
from .credibility_manager import CredibilityManager


class WebSearcher:
    """Searches multiple sources to verify claims."""
    
    # Rate limiting settings
    MIN_REQUEST_INTERVAL = 0.5  # Minimum 500ms between requests
    
    def __init__(self):
        self.headers = {
            'User-Agent': USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        self._last_request_time = 0
        self.credibility_manager = CredibilityManager()
    
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
        
        # Search DuckDuckGo
        ddg_results = self._search_duckduckgo(claim)
        sources.extend(ddg_results)
        
        # Search Wikipedia
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
            
            # Get credibility info from database
            cred_info = self.credibility_manager.get_credibility(domain)
            trust_score = cred_info.get('trust', 50) / 100  # Normalize to 0-1
            trust_level = self.credibility_manager.get_trust_level(domain)
            is_factcheck = self.credibility_manager.is_factcheck_site(domain)
            
            scored.append({
                **source,
                'trust_level': trust_level,
                'trust_score': trust_score,
                'is_factcheck_site': is_factcheck,
                'bias': cred_info.get('bias', 'unknown'),
                'source_category': cred_info.get('category', 'unknown')
            })
        
        # Sort by trust score (descending)
        scored.sort(key=lambda x: x['trust_score'], reverse=True)
        
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
