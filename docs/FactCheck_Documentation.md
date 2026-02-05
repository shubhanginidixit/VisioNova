# Fact Check Module

## Executive Summary

The Fact Check module is VisioNova's claim verification system that determines whether statements are true, false, misleading, or unverifiable. It combines multi-source web search, source credibility scoring, temporal context analysis, and AI-powered reasoning to deliver explainable verdicts with supporting evidence.

---

## How It Works (High-Level)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER INPUT                                     │
│              (Claim, Question, or URL)                                   │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Input Classifier    │
                    │  (URL/Question/Claim) │
                    └───────────┬───────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ▼                   ▼                   ▼
      ┌──────────┐       ┌──────────────┐    ┌──────────────┐
      │   URL    │       │   Question   │    │    Claim     │
      │ Content  │       │    →Claim    │    │  (as-is)     │
      │ Extract  │       │  Conversion  │    │              │
      └────┬─────┘       └──────┬───────┘    └──────┬───────┘
           └───────────────────┬───────────────────┘
                               │
                               ▼
                    ┌───────────────────────┐
                    │  Temporal Analyzer    │
                    │  (Extract dates/years)│
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │     Web Searcher      │
                    │  (Google, DDG, Wiki)  │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Credibility Manager  │
                    │   (Score sources)     │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │    AI Analyzer        │
                    │ (or Heuristic Fallback)│
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │      VERDICT          │
                    │  + Explanation        │
                    │  + Confidence         │
                    │  + Sources            │
                    └───────────────────────┘
```

---

## Components

### 1. Input Classifier

**File:** `input_classifier.py`

Detects the type of user input and normalizes it for searching.

#### Input Types
| Type | Example | Processing |
|------|---------|------------|
| **URL** | `https://news.com/article` | Extract content, then verify claims |
| **Question** | `Is the Earth round?` | Convert to claim: "Earth is round" |
| **Claim** | `The moon landing was faked` | Normalize and search directly |

#### Question Conversion Examples
| Question | Converted Claim |
|----------|-----------------|
| "Is the Earth flat?" | "Earth flat" |
| "Did Einstein invent the lightbulb?" | "Einstein invent lightbulb" |
| "Was the moon landing real?" | "moon landing real was" |

#### Normalization
- Removes article metadata (timestamps, "Updated -", source attributions)
- Strips leading/trailing punctuation
- Truncates long pasted articles to first sentence

```python
from fact_check.input_classifier import InputClassifier

classifier = InputClassifier()
result = classifier.classify("Is the moon landing real?")

print(result['type'])     # "question"
print(result['original']) # "Is the moon landing real?"
print(result['claim'])    # "moon landing real"
```

---

### 2. Content Extractor

**File:** `content_extractor.py`

Fetches and cleans web page content when a URL is submitted.

#### Features
- **Smart extraction**: Targets article body, ignores ads/navigation
- **Retry with backoff**: Automatic retry on failures (3 attempts)
- **User-Agent rotation**: Randomizes browser identity to avoid blocks
- **Claim extraction**: Identifies verifiable statements from content

#### Extraction Priority
1. `<article>` tags
2. `<main>` content
3. Semantic tags (`<section>`, `<div class="content">`)
4. Paragraph fallback

```python
from fact_check.content_extractor import ContentExtractor

extractor = ContentExtractor()
result = extractor.extract_from_url("https://example.com/news-article")

if result['success']:
    print(result['title'])   # "Breaking News: Event Happened"
    print(result['content']) # Full article text
    print(result['claims'])  # ["Event happened on Tuesday", ...]
```

---

### 3. Temporal Analyzer

**File:** `temporal_analyzer.py`

Detects dates, years, and time periods in claims to guide search strategy.

#### Why This Matters
| Claim | Temporal Context | Search Strategy |
|-------|------------------|-----------------|
| "The moon landing happened in 1969" | Historical | Search archives from 1969+ |
| "COVID vaccines contain microchips" | Recent | Search recent news (2020+) |
| "Breaking: Leader resigned today" | Current | Search last 24-48 hours |

#### Time Period Categories
| Period | Definition |
|--------|------------|
| `current` | Within last 2 years |
| `recent` | 2-5 years ago |
| `past_decade` | 5-10 years ago |
| `historical_modern` | 10-50 years ago |
| `historical` | 50+ years ago |

#### Extracted Data
- Years mentioned (e.g., `[1969, 2024]`)
- Full dates (e.g., `["January 15, 2020"]`)
- Temporal keywords (e.g., `recently`, `historical`, `yesterday`)

```python
from fact_check.temporal_analyzer import TemporalAnalyzer

analyzer = TemporalAnalyzer()
result = analyzer.extract_temporal_context("The Berlin Wall fell in 1989")

print(result['search_year_from'])  # 1989
print(result['is_historical'])     # True
print(result['time_period'])       # "historical_modern"
```

---

### 4. Web Searcher

**File:** `web_searcher.py`

Searches multiple sources to find evidence supporting or refuting claims.

#### Search Sources
| Source | Method | Use Case |
|--------|--------|----------|
| **Google Custom Search** | REST API | Primary source (if API key configured) |
| **DuckDuckGo** | `ddgs` library | Fallback, no API key needed |
| **Wikipedia** | MediaWiki API | Reference/encyclopedic content |

#### Search Flow
1. Query Google Custom Search (if configured)
2. Fallback to DuckDuckGo if Google fails/unavailable
3. Supplement with Wikipedia article lookup
4. Score all sources by credibility
5. Return ranked results

#### Rate Limiting
- Minimum 1 second between requests
- Prevents API blocks and 429 errors

```python
from fact_check.web_searcher import WebSearcher

searcher = WebSearcher()
results = searcher.search("moon landing 1969")

print(f"Found {results['total_found']} sources")
for source in results['sources'][:3]:
    print(f"[{source['trust_level']}] {source['title']}")
    print(f"  URL: {source['url']}")
```

---

### 5. Credibility Manager

**File:** `credibility_manager.py`

Rates source trustworthiness using a curated database of 70+ domains.

#### Trust Score Ranges
| Score | Level | Examples |
|-------|-------|----------|
| 85-100 | High | Reuters, AP, Snopes, Politifact |
| 70-84 | Medium-High | BBC, The Guardian, Wikipedia |
| 50-69 | Medium | Unknown/unrated domains |
| 30-49 | Low | Tabloids, opinion sites |
| 0-29 | Unreliable | Known misinformation sources |

#### Source Categories
- **Factcheck Sites**: Snopes, Politifact, Factcheck.org, AltNews.in
- **News Agencies**: Reuters, AP, AFP
- **Reference**: Wikipedia, Britannica, .gov domains
- **Major News**: BBC, NYTimes, The Hindu
- **Unreliable**: Known conspiracy/misinformation sites

#### Credibility Database
Stored in `source_credibility.json` with structure:
```json
{
  "factcheck_sites": {
    "snopes.com": {
      "trust": 95,
      "bias": "center",
      "category": "factcheck",
      "description": "Independent fact-checking organization"
    }
  }
}
```

```python
from fact_check.credibility_manager import CredibilityManager

manager = CredibilityManager()

print(manager.get_trust_score("reuters.com"))  # 95
print(manager.get_trust_level("reuters.com"))  # "high"
print(manager.is_factcheck_site("snopes.com")) # True
print(manager.is_unreliable("infowars.com"))   # True
```

---

### 6. AI Analyzer

**Integration:** `fact_checker.py → AI/groq_client.py`

Uses LLM (Groq/LLaMA) to analyze sources and determine verdicts.

#### AI Analysis Process
1. Collect all source snippets and full text (when available)
2. Build context prompt with claim + evidence
3. Request verdict, confidence, and explanation from LLM
4. Parse structured response

#### Fallback: Heuristic Analysis
When AI is unavailable, the system uses rule-based analysis:

1. **Categorize sources** by trust level
2. **Count support/contradict signals** in snippets
3. **Check for fact-check verdicts** (e.g., "FALSE" from Snopes)
4. **Calculate confidence** based on source agreement

```python
# Heuristic verdict logic
if factcheck_sources_say_false:
    verdict = "FALSE"
elif high_trust_sources_support:
    verdict = "TRUE"
elif sources_conflict:
    verdict = "PARTIALLY TRUE"
else:
    verdict = "UNVERIFIABLE"
```

---

### 7. Fact Checker (Main Pipeline)

**File:** `fact_checker.py`

Orchestrates all components into a unified checking pipeline.

#### Two Modes

| Mode | Method | Description |
|------|--------|-------------|
| **Quick Check** | `check()` | Single search query, fast results |
| **Deep Check** | `deep_check()` | Multiple query variations, temporal search, full content fetch |

#### Quick Check Flow
1. Classify input (URL/question/claim)
2. Search web sources
3. Score source credibility
4. Analyze with AI (or heuristics)
5. Build response

#### Deep Check Flow (Enhanced)
1. Classify input
2. Extract temporal context
3. Generate multiple search queries (variations, with/without dates)
4. Search with year filters (if historical)
5. Fetch full text from top sources
6. Analyze with AI (richer context)
7. Build detailed response

#### Caching
- Results cached for 24 hours (TTL)
- Max 50 cached claims
- LRU eviction when full
- Cache key: MD5 hash of normalized claim

#### Response Structure
```python
{
    "success": True,
    "input_type": "claim",
    "original_input": "The moon landing happened in 1969",
    "claim": "moon landing happened 1969",
    "verdict": "TRUE",
    "confidence": 0.95,
    "explanation": "Multiple authoritative sources confirm...",
    "summary": "The Apollo 11 mission landed on July 20, 1969...",
    "sources": [
        {
            "title": "Moon landing - Wikipedia",
            "url": "https://en.wikipedia.org/wiki/Moon_landing",
            "snippet": "The United States landed astronauts...",
            "trust_score": 90,
            "trust_level": "high"
        }
    ],
    "temporal_context": {
        "search_year_from": 1969,
        "time_period": "historical",
        "is_historical": True
    },
    "ai_analyzed": True,
    "cached": False
}
```

---

## Verdict Types

| Verdict | Meaning |
|---------|---------|
| **TRUE** | Claim is accurate based on authoritative sources |
| **FALSE** | Claim is inaccurate, contradicted by evidence |
| **PARTIALLY TRUE** | Claim contains some truth but is misleading or incomplete |
| **MISLEADING** | Claim may be technically true but presented deceptively |
| **UNVERIFIABLE** | Insufficient evidence to determine truth |

---

## Configuration

**File:** `config.py`

| Setting | Default | Description |
|---------|---------|-------------|
| `GOOGLE_SEARCH_RESULTS` | 10 | Number of search results per query |
| `REQUEST_TIMEOUT` | 10 | HTTP request timeout (seconds) |
| `GOOGLE_API_KEY` | env | Google Custom Search API key |
| `GOOGLE_CSE_ID` | env | Google Custom Search Engine ID |

#### Environment Variables
```bash
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_custom_search_engine_id
GROQ_API_KEY=your_groq_api_key  # For AI analysis
```

---

## Usage Examples

### Basic Claim Check
```python
from fact_check.fact_checker import FactChecker

checker = FactChecker()
result = checker.check("COVID-19 vaccines contain microchips")

print(f"Verdict: {result['verdict']}")
print(f"Confidence: {result['confidence']*100:.1f}%")
print(f"Explanation: {result['explanation']}")
```

### Deep Check with Temporal Context
```python
result = checker.deep_check("The Berlin Wall fell in 1989")

print(f"Verdict: {result['verdict']}")
print(f"Time period: {result['temporal_context']['time_period']}")
print(f"Searched from: {result['temporal_context']['search_year_from']}")
```

### URL Verification
```python
result = checker.check("https://news.com/breaking-story")

print(f"Extracted claims: {result['claims']}")
print(f"Overall verdict: {result['verdict']}")
```

---

## Error Handling

The system handles failures gracefully:

| Failure | Fallback |
|---------|----------|
| Google API unavailable | DuckDuckGo search |
| AI analysis fails | Heuristic rule-based analysis |
| Content extraction fails | Search by URL as query |
| Source fetch timeout | Retry with exponential backoff |
| Unknown domain | Default 50% trust score |

---

## Limitations

1. **Recency**: Cannot verify breaking news (< 1 hour old)
2. **Language**: Optimized for English; other languages less accurate
3. **Paywalled content**: Cannot access subscription-only articles
4. **Social media**: Individual posts/tweets not searchable
5. **Opinions**: Cannot verify subjective statements

---

## Future Improvements

1. **Multi-language support**: Expand to Hindi, Spanish, French
2. **Image fact-checking**: Reverse image search integration
3. **Claim extraction from audio/video**: Transcription + verification
4. **Real-time monitoring**: Track claim spread across platforms
5. **Expert sourcing**: Direct API to academic databases
