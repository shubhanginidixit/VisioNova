"""
Source Credibility Manager
Manages source trust scores and credibility ratings from JSON database.
"""
import json
import os
from functools import lru_cache


class CredibilityManager:
    """Manages source credibility ratings and trust scores."""
    
    def __init__(self, db_path: str = None):
        """
        Initialize credibility manager.
        
        Args:
            db_path: Path to source_credibility.json (defaults to same directory)
        """
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(__file__), 
                'source_credibility.json'
            )
        
        self.db_path = db_path
        self._load_database()
    
    def _load_database(self):
        """Load credibility database from JSON file."""
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                self.db = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Credibility database not found at {self.db_path}")
            self.db = {}
        except json.JSONDecodeError as e:
            print(f"Error parsing credibility database: {e}")
            self.db = {}
    
    @lru_cache(maxsize=256)
    def get_credibility(self, domain: str) -> dict:
        """
        Get credibility information for a domain.
        
        Args:
            domain: Domain name (e.g., 'snopes.com')
            
        Returns:
            dict with trust, bias, category, description
        """
        domain = domain.lower().strip()
        
        # Check all categories
        for category in self.db.values():
            if isinstance(category, dict) and domain in category:
                return category[domain]
        
        # Check for partial matches (e.g., 'en.wikipedia.org' matches 'wikipedia.org')
        for category in self.db.values():
            if isinstance(category, dict):
                for known_domain, info in category.items():
                    if known_domain in domain or domain in known_domain:
                        return info
        
        # Return default for unknown sources
        return {
            'trust': 50,
            'bias': 'unknown',
            'category': 'unknown',
            'description': 'Unknown source'
        }
    
    def is_factcheck_site(self, domain: str) -> bool:
        """Check if domain is a fact-checking site."""
        info = self.get_credibility(domain)
        return info.get('category') == 'factcheck'
    
    def is_unreliable(self, domain: str) -> bool:
        """Check if domain is marked as unreliable."""
        info = self.get_credibility(domain)
        return info.get('category') == 'unreliable'
    
    def get_trust_score(self, domain: str) -> int:
        """Get numerical trust score (0-100) for a domain."""
        return self.get_credibility(domain).get('trust', 50)
    
    def get_trust_level(self, domain: str) -> str:
        """
        Get categorical trust level for a domain.
        
        Returns:
            'high', 'medium-high', 'medium', 'low', or 'unknown'
        """
        trust = self.get_trust_score(domain)
        
        if trust >= 85:
            return 'high'
        elif trust >= 70:
            return 'medium-high'
        elif trust >= 50:
            return 'medium'
        elif trust >= 30:
            return 'low'
        else:
            return 'unreliable'
    
    def reload_database(self):
        """Reload database from disk (for runtime updates)."""
        self.get_credibility.cache_clear()
        self._load_database()


# Quick test
if __name__ == '__main__':
    manager = CredibilityManager()
    
    test_domains = [
        'snopes.com',
        'reuters.com',
        'wikipedia.org',
        'infowars.com',
        'example.com'
    ]
    
    for domain in test_domains:
        info = manager.get_credibility(domain)
        print(f"{domain}:")
        print(f"  Trust: {info['trust']}/100")
        print(f"  Level: {manager.get_trust_level(domain)}")
        print(f"  Bias: {info['bias']}")
        print(f"  Category: {info['category']}")
        print(f"  Factcheck: {manager.is_factcheck_site(domain)}")
        print()
