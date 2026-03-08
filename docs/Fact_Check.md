# Fact Check

## Executive Summary

The Fact Check module is VisioNova's claim verification system that determines whether statements are true, false, misleading, or unverifiable. It combines multi-source web search, source credibility scoring, temporal context analysis, and AI-powered reasoning to deliver explainable verdicts with supporting evidence.

---

## 1. How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                      USER INPUT                              │
│            (Claim, Question, or URL)                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
                ┌───────────────────────┐
                │   Input Classifier    │
                │  (URL/Question/Claim) │
                └───────────┬───────────┘
                            │
            ┌───────────────┼───────────────────┐
            ▼               ▼                   ▼
      ┌──────────┐   ┌──────────────┐    ┌──────────────┐
      │   URL    │   │   Question   │    │    Claim     │
      │ Content  │   │    →Claim    │    │  (as-is)     │
      │ Extract  │   │  Conversion  │    │              │
      └────┬─────┘   └──────┬───────┘    └──────┬───────┘
           └────────────────┬───────────────────┘
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

## 2. Components

### 2.1 Input Classifier (`input_classifier.py`)

Detects the type of user input and normalizes it for searching.

| Type | Example | Processing |
|------|---------|------------|
| **URL** | `https://news.com/article` | Extract content, then verify claims |
| **Question** | `Is the Earth round?` | Convert to claim: "Earth is round" |
| **Claim** | `The moon landing was faked` | Normalize and search directly |

### 2.2 Content Extractor (`content_extractor.py`)

Fetches and cleans web page content when a URL is submitted.
- **Smart extraction**: Targets article body, ignores ads/navigation
- **Retry with backoff**: Automatic retry on failures (3 attempts)
- **Extraction priority**: `<article>` → `<main>` → `<section>` → paragraph fallback

### 2.3 Temporal Analyzer (`temporal_analyzer.py`)

Detects dates, years, and time periods to guide search strategy.

| Period | Definition |
|--------|------------|
| `current` | Within last 2 years |
| `recent` | 2-5 years ago |
| `past_decade` | 5-10 years ago |
| `historical_modern` | 10-50 years ago |
| `historical` | 50+ years ago |

### 2.4 Web Searcher (`web_searcher.py`)

Searches multiple sources to find evidence:

| Source | Method | Use Case |
|--------|--------|----------|
| **Google Custom Search** | REST API | Primary source (if API key configured) |
| **DuckDuckGo** | `ddgs` library | Fallback, no API key needed |
| **Wikipedia** | MediaWiki API | Reference/encyclopedic content |

### 2.5 Credibility Manager (`credibility_manager.py`)

Rates source trustworthiness using a curated database of 70+ domains.

| Score | Level | Examples |
|-------|-------|----------|
| 85-100 | High | Reuters, AP, Snopes, Politifact |
| 70-84 | Medium-High | BBC, The Guardian, Wikipedia |
| 50-69 | Medium | Unknown/unrated domains |
| 30-49 | Low | Tabloids, opinion sites |
| 0-29 | Unreliable | Known misinformation sources |

### 2.6 AI Analyzer (`fact_checker.py → AI/groq_client.py`)

Uses LLM (Groq/LLaMA) to analyze sources and determine verdicts. Falls back to heuristic rule-based analysis when AI is unavailable.

---

## 3. Two Check Modes

| Mode | Method | Description |
|------|--------|-------------|
| **Quick Check** | `check()` | Single search query, fast results |
| **Deep Check** | `deep_check()` | Multiple query variations, temporal search, full content fetch |

### Deep Check Enhanced Flow
1. Classify input
2. Extract temporal context
3. Generate multiple search queries (variations, with/without dates)
4. Search with year filters (if historical)
5. Fetch full text from top sources
6. Analyze with AI (richer context)
7. Build detailed response

### Caching
- Results cached for 24 hours (TTL)
- Max 50 cached claims
- LRU eviction when full
- Cache key: MD5 hash of normalized claim

---

## 4. Verdict Types

| Verdict | Meaning |
|---------|---------|
| **TRUE** | Claim is accurate based on authoritative sources |
| **FALSE** | Claim is inaccurate, contradicted by evidence |
| **PARTIALLY TRUE** | Claim contains some truth but is misleading or incomplete |
| **MISLEADING** | Claim may be technically true but presented deceptively |
| **UNVERIFIABLE** | Insufficient evidence to determine truth |

---

## 5. API Endpoints

### `POST /api/fact-check` — Quick Check
```json
{
  "claim": "The moon landing happened in 1969"
}
```

### `POST /api/deep-fact-check` — Enhanced Check
Same input, but runs multiple query variations and fetches full source text.

### Response Structure
```json
{
  "success": true,
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
    "is_historical": true
  },
  "ai_analyzed": true,
  "cached": false
}
```

---

## 6. Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `GOOGLE_SEARCH_RESULTS` | 10 | Number of search results per query |
| `REQUEST_TIMEOUT` | 10 | HTTP request timeout (seconds) |
| `GOOGLE_API_KEY` | env | Google Custom Search API key |
| `GOOGLE_CSE_ID` | env | Google Custom Search Engine ID |
| `GROQ_API_KEY` | env | AI analysis API key |

---

## 7. Error Handling

| Failure | Fallback |
|---------|----------|
| Google API unavailable | DuckDuckGo search |
| AI analysis fails | Heuristic rule-based analysis |
| Content extraction fails | Search by URL as query |
| Source fetch timeout | Retry with exponential backoff |
| Unknown domain | Default 50% trust score |

---

## 8. Limitations

1. **Recency**: Cannot verify breaking news (< 1 hour old)
2. **Language**: Optimized for English; other languages less accurate
3. **Paywalled content**: Cannot access subscription-only articles
4. **Social media**: Individual posts/tweets not searchable
5. **Opinions**: Cannot verify subjective statements

---

## 9. Future Improvements

1. **Multi-language support**: Expand to Hindi, Spanish, French
2. **Image fact-checking**: Reverse image search integration
3. **Claim extraction from audio/video**: Transcription + verification
4. **Real-time monitoring**: Track claim spread across platforms
5. **Expert sourcing**: Direct API to academic databases

---

## References

1. Google Custom Search API: [developers.google.com](https://developers.google.com/custom-search)
2. DuckDuckGo Search Library: [pypi.org/project/duckduckgo-search](https://pypi.org/project/duckduckgo-search/)
3. MediaWiki API: [mediawiki.org/wiki/API](https://www.mediawiki.org/wiki/API:Main_page)
