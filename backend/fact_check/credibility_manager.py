"""
Source Credibility Manager
Manages source trust scores and credibility ratings from JSON database.
"""
import json
import os
from functools import lru_cache
from .config import (
    BLOCKED_SOURCE_CATEGORIES,
    SOURCE_ADMISSION_POLICY,
    SOURCE_CATEGORY_TIERS,
    UNKNOWN_SOURCE_TRUST_PENALTY,
    VERDICT_MAX_SOURCE_TIER,
)


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

    @staticmethod
    def _domain_matches(known_domain: str, domain: str) -> bool:
        """Match exact domain, subdomain, or known suffix pattern."""
        if not known_domain or not domain:
            return False

        known_domain = known_domain.lower().strip()
        domain = domain.lower().strip()

        if known_domain.startswith('.'):
            return domain.endswith(known_domain)

        return domain == known_domain or domain.endswith(f".{known_domain}")
    
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
        
        # Check for subdomain/suffix matches (e.g., 'en.wikipedia.org' matches 'wikipedia.org')
        for category in self.db.values():
            if isinstance(category, dict):
                for known_domain, info in category.items():
                    if self._domain_matches(known_domain, domain):
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

    def get_source_category(self, domain: str) -> str:
        """Get normalized source category from the credibility database."""
        return self.get_credibility(domain).get('category', 'unknown')

    def get_source_tier(self, domain: str) -> int:
        """Get reliability tier for a domain (1=highest trust)."""
        category = self.get_source_category(domain)
        return SOURCE_CATEGORY_TIERS.get(category, SOURCE_CATEGORY_TIERS['unknown'])

    def get_adjusted_trust_score(self, domain: str) -> int:
        """Get policy-adjusted trust score after unknown-source penalties."""
        trust = int(self.get_trust_score(domain))
        category = self.get_source_category(domain)

        if category == 'unknown' and SOURCE_ADMISSION_POLICY in {'hybrid', 'strict'}:
            trust = max(0, trust - UNKNOWN_SOURCE_TRUST_PENALTY)

        return trust

    def is_allowed_for_verdict(self, domain: str) -> bool:
        """Check whether a source is eligible to influence verdict scoring."""
        category = self.get_source_category(domain)
        tier = self.get_source_tier(domain)

        if category in BLOCKED_SOURCE_CATEGORIES:
            return False

        if SOURCE_ADMISSION_POLICY == 'strict':
            return category != 'unknown' and tier <= VERDICT_MAX_SOURCE_TIER

        if SOURCE_ADMISSION_POLICY == 'open':
            return True

        # Hybrid policy.
        return tier <= VERDICT_MAX_SOURCE_TIER

    def get_source_policy(self, domain: str) -> dict:
        """Return policy metadata used by retrieval and verdict synthesis."""
        info = self.get_credibility(domain)
        category = info.get('category', 'unknown')
        tier = self.get_source_tier(domain)
        adjusted_trust = self.get_adjusted_trust_score(domain)

        return {
            'category': category,
            'tier': tier,
            'raw_trust_score': int(info.get('trust', 50)),
            'trust_score': adjusted_trust,
            'trust_level': self.get_trust_level(domain),
            'bias': info.get('bias', 'unknown'),
            'description': info.get('description', 'Unknown source'),
            'is_factcheck_site': category == 'factcheck',
            'include_in_verdict': self.is_allowed_for_verdict(domain),
        }
    
    def get_trust_level(self, domain: str) -> str:
        """
        Get categorical trust level for a domain.
        
        Returns:
            'high', 'medium-high', 'medium', 'low', or 'unknown'
        """
        trust = self.get_adjusted_trust_score(domain)
        
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
