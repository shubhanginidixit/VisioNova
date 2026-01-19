# Fact Check Module Documentation

## Overview
The **Fact Check Module** in VisioNova is designed to verify the credibility of text content and claims by analyzing their source, extracting key information, and understanding the temporal context. It serves as a backend service that powers the fact-checking capabilities of the application.

## Core Components

### 1. Content Extractor (`content_extractor.py`)
**Purpose:** Responsbile for fetching and processing web pages to extract meaningful text and scannable claims.

**Key Features:**
- **URL Text Extraction:** Uses `BeautifulSoup` to parse HTML and extract the main article body, removing clutter like scripts, styles, and navigation elements.
- **Claim Extraction:** Automatically identifies sentences that look like verifiable claims (statements of fact) based on length and structure, filtering out questions or exclamations.
- **Robust Networking:** Implements `requests` with realistic browser headers to avoid 403 Forbidden errors and includes a retry mechanism with exponential backoff for reliability.

**Usage:**
```python
extractor = ContentExtractor()
result = extractor.extract_from_url("https://example.com/article")
# Returns: { 'success': True, 'content': "...", 'claims': ["Claim 1", ...], ... }
```

### 2. Temporal Analyzer (`temporal_analyzer.py`)
**Purpose:** Analyzes text to distinct when the events described likely took place, which is crucial for distinguishing between historical facts and breaking news.

**Key Features:**
- **Date & Year Extraction:** Uses regular expressions to find specific dates (e.g., "January 15, 2020") and years (e.g., "1969") within the text.
- **Context Categorization:** Categorizes content into time periods:
  - **Historical:** (e.g., > 50 years ago)
  - **Recent/Current:** (e.g., last 0-5 years)
- **Search Optimization:** Determines the "Search Year From" to guide external search APIs. For example, if a claim mentions "1969", it knows search for historical records rather than just recent news.

**Usage:**
```python
analyzer = TemporalAnalyzer()
context = analyzer.extract_temporal_context("The moon landing happened in 1969.")
# Returns: { 'search_year_from': 1969, 'is_historical': True, ... }
```

### 3. Credibility Manager (`credibility_manager.py`)
**Purpose:** Assesses the trustworthiness of the source domain.

**Key Features:**
- **Database Lookup:** Checks domains against a local JSON database (`source_credibility.json`) containing known reliable and unreliable sources.
- **Trust Scoring:** Returns a trust score (0-100), bias rating (e.g., "left", "center", "conspiracy"), and category (e.g., "news", "satire", "factcheck").
- **Classification:** Helper methods to quickly identify if a site is a known fact-checker or unreliable.

**Usage:**
```python
manager = CredibilityManager()
info = manager.get_credibility("snopes.com")
# Returns: { 'trust': 95, 'category': 'factcheck', 'bias': 'center' }
```

## Workflow
1.  **Input:** A user provides a URL or text claim.
2.  **Extraction:** `ContentExtractor` pulls the text.
3.  **Analysis:**
    *   `TemporalAnalyzer` determines the time frame.
    *   `CredibilityManager` checks the source's reputation.
4.  **Verification:** (External APIs would typically be used here, guided by the temporal and content data).
