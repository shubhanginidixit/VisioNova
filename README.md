# VisioNova

<div align="center">

**Multi-modal AI Credibility Engine for Enterprise Security and Media Integrity**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-purple.svg)](https://groq.com)

</div>

---

## Overview

VisioNova is an AI-powered platform designed to verify the authenticity of digital media and detect misinformation. It combines multiple analysis techniques to provide comprehensive credibility assessments for images, videos, audio, and text content.

## Key Features

### Image Verification
- AI-generated image detection
- Manipulation and tampering analysis
- ELA (Error Level Analysis) heatmaps
- Metadata forensics

### Video Analysis
- Deepfake detection with frame-by-frame analysis
- Lip-sync mismatch detection
- Facial landmark tracking
- Motion consistency verification

### Audio Forensics
- Voice cloning detection
- AI-generated audio identification
- Spectral analysis
- Pitch stability verification

### Text Analysis
- AI vs human-written text classification
- Perplexity and burstiness analysis
- Source reliability checking

### Fact Checking
- AI-Powered Analysis using Llama 3.3 70B via Groq API
- Multi-Source Verification through DuckDuckGo and Wikipedia
- Smart Claim Extraction supporting URLs, questions, and claims
- **NEW:** Detailed Confidence Breakdown (source quality, quantity, consensus, factcheck presence)
- **NEW:** Per-Source Stance Analysis (SUPPORTS/REFUTES/NEUTRAL)
- **NEW:** Contradiction Detection between sources
- **NEW:** Dynamic Source Credibility Database (70+ rated domains)
- **NEW:** User Feedback System for reporting incorrect verdicts
- **NEW:** TTL-based caching with 24-hour expiration
- **NEW:** Rate limiting (5 req/min) and input validation for security
- **NEW:** Retry logic with exponential backoff for failed requests
- Tabbed Results Interface with Summary, Detailed Analysis, and Claims & Evidence views
- Trust Level Scoring for source credibility assessment
- Clickable Source Links for direct access to verification sources

## Project Structure

```
VisioNova/
├── backend/
│   ├── app.py                      # Flask API server
│   ├── ai/
│   │   ├── __init__.py
│   │   └── groq_client.py          # Groq LLM integration
│   └── fact_check/
│       ├── __init__.py
│       ├── fact_checker.py         # Main fact-checking pipeline
│       ├── input_classifier.py     # URL/claim/question detection
│       ├── content_extractor.py    # Web page content extraction
│       ├── web_searcher.py         # DuckDuckGo + Wikipedia search
│       └── config.py               # Trusted domains & settings
├── frontend/
│   ├── html/
│   │   ├── homepage.html           # Landing page
│   │   ├── AnalysisDashboard.html  # Upload & analysis interface
│   │   ├── ResultPage.html         # Image analysis results
│   │   ├── VideoResultPage.html    # Video analysis results
│   │   ├── AudioResultPage.html    # Audio analysis results
│   │   ├── TextResultPage.html     # Text analysis results
│   │   ├── FactCheckPage.html      # Fact-checking interface
│   │   └── ReportPage.html         # Detailed forensic reports
│   ├── css/
│   │   └── styles.css              # Custom styles
│   └── js/
│       ├── fact-check.js           # Fact-check frontend logic
│       └── *.js                    # Other JavaScript modules
├── .env                            # Environment variables
└── README.md
```

## Quick Start

### Prerequisites
- Python 3.10+
- Groq API key (free at groq.com)

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/DhanushPillay/VisioNova.git
   cd VisioNova
   ```

2. Create virtual environment
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

3. Install dependencies
   ```bash
   pip install flask flask-cors flask-limiter python-dotenv requests beautifulsoup4 groq ddgs
   ```

4. Set up environment variables
   
   Create a `.env` file in the `backend/` folder:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_MODEL=llama-3.3-70b-versatile
   ```

5. Run the backend
   ```bash
   python backend/app.py
   ```

6. Open the frontend
   
   Open `frontend/html/FactCheckPage.html` in your browser

## Tech Stack

### Frontend
- HTML5
- Tailwind CSS (via CDN)
- JavaScript
- Google Material Symbols
- Inter Font

### Backend
- Python 3.10+
- Flask (REST API)
- Groq API (Llama 3.3 70B)
- BeautifulSoup (Web scraping)
- DuckDuckGo Search API

## Credibility Scoring

VisioNova uses a Unified Credibility Score (0-100) that combines:
- AI probability analysis
- Manipulation detection confidence
- Metadata verification
- Source reliability assessment

| Score Range | Status |
|-------------|--------|
| 80-100 | Likely Authentic |
| 50-79 | Review Recommended |
| 0-49 | High Risk / Manipulated |

## Fact-Check Verdicts

| Verdict | Description |
|---------|-------------|
| TRUE | Claim is verified by trusted sources |
| FALSE | Claim is contradicted by evidence |
| PARTIALLY TRUE | Some truth but context missing |
| MISLEADING | Technically true but deceptive |
| UNVERIFIABLE | Insufficient evidence to verify |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/api/health` | Enhanced health check with cache stats |
| POST | `/api/fact-check` | Verify a claim or URL (5 req/min limit) |
| POST | `/api/fact-check/deep` | Deep scan with multiple search queries |
| POST | `/api/fact-check/feedback` | Submit user feedback on verdicts |
| GET | `/api/fact-check?q=` | Quick claim verification |

## Contributors

- Dhanush Pillay
- Shubhangini Dixit

## License

This project is for educational purposes.

---

<div align="center">

**Built for truth and transparency**

</div>
