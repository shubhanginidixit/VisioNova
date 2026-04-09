"""
Configuration for the Fact Check module.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API Settings
GOOGLE_SEARCH_RESULTS = 10  # Number of search results to fetch
REQUEST_TIMEOUT = 10  # Seconds to wait for HTTP requests

# Fact-checking policy settings
# Source admission policy: strict | hybrid | open
SOURCE_ADMISSION_POLICY = os.getenv('FACTCHECK_SOURCE_POLICY', 'hybrid').lower()

# Tiering policy (lower tier number = higher reliability)
SOURCE_CATEGORY_TIERS = {
    'factcheck': 1,
    'news_agency': 2,
    'news': 2,
    'government': 3,
    'health': 3,
    'academic': 3,
    'reference': 4,
    'opinion': 5,
    'satire': 5,
    'unreliable': 5,
    'unknown': 4,
}

# Only Tier 1-3 can influence final verdicts.
VERDICT_MAX_SOURCE_TIER = 3

# Apply trust penalties to unknown/unmapped domains in hybrid mode.
UNKNOWN_SOURCE_TRUST_PENALTY = 20

# Hard exclusions from verdict scoring.
BLOCKED_SOURCE_CATEGORIES = {'unreliable', 'satire'}

# High-confidence outputs require at least one primary-evidence source.
PRIMARY_EVIDENCE_CATEGORIES = {'factcheck', 'government', 'health', 'academic'}
HIGH_CONFIDENCE_THRESHOLD = 80

# Source display and enrichment behavior.
MAX_RESPONSE_SOURCES = 10
FULL_TEXT_ENRICH_LIMIT = 5
FULL_TEXT_MAX_CHARS = 2500

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')

# Trusted fact-check domains (highest weight in scoring)
TRUSTED_FACTCHECK_DOMAINS = [
    # International fact-checkers
    'snopes.com',
    'politifact.com',
    'factcheck.org',
    'fullfact.org',
    'leadstories.com',
    # News wire services
    'apnews.com',
    'reuters.com',
    'afp.com',
    # India-specific fact-checkers
    'altnews.in',
    'boomlive.in',
    'thequint.com',
    'factchecker.in',
    'vishvasnews.com',
    'factly.in',
]

# Trusted news/reference domains
TRUSTED_DOMAINS = [
    # Reference sources
    'wikipedia.org',
    'britannica.com',
    # Major international news
    'nytimes.com',
    'washingtonpost.com',
    'theguardian.com',
    'bbc.com',
    'bbc.co.uk',
    'cnn.com',
    'nbcnews.com',
    'cbsnews.com',
    'economist.com',
    # Government & health sources
    'nasa.gov',
    'who.int',
    'cdc.gov',
    'nih.gov',
    '.gov.in',  # Indian government sites
    # Indian news sources
    'thehindu.com',
    'indianexpress.com',
    'ndtv.com',
    'hindustantimes.com',
]

# User agent for web requests
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

# Verdict types
class Verdict:
    TRUE = "TRUE"
    FALSE = "FALSE"
    PARTIALLY_TRUE = "PARTIALLY TRUE"
    MISLEADING = "MISLEADING"
    UNVERIFIABLE = "UNVERIFIABLE"

