"""
Configuration for the Fact Check module.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API Settings
GOOGLE_SEARCH_RESULTS = 10  # Number of search results to fetch
REQUEST_TIMEOUT = 10  # Seconds to wait for HTTP requests

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

