# VisioNova

<div align="center">

**The world's most advanced multi-modal AI credibility engine for enterprise security and media integrity.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-purple.svg)](https://groq.com)

</div>

---

## 🎯 Overview

VisioNova is an AI-powered platform designed to verify the authenticity of digital media and detect misinformation. It combines multiple analysis techniques to provide comprehensive credibility assessments for images, videos, audio, and text content.

## ✨ Key Features

### 🖼️ Image Verification
- AI-generated image detection
- Manipulation and tampering analysis
- ELA (Error Level Analysis) heatmaps
- Metadata forensics

### 🎥 Video Analysis
- Deepfake detection with frame-by-frame analysis
- Lip-sync mismatch detection
- Facial landmark tracking
- Motion consistency verification

### 🎤 Audio Forensics
- Voice cloning detection
- AI-generated audio identification
- Spectral analysis
- Pitch stability verification

### 📝 Text Analysis
- AI vs human-written text classification
- Perplexity and burstiness analysis
- Source reliability checking

### ✅ Fact Checking (NEW!)
- **AI-Powered Analysis** - Uses Llama 3.3 70B via Groq API
- **Multi-Source Verification** - Searches DuckDuckGo + Wikipedia
- **Smart Claim Extraction** - Handles URLs, questions, and claims
- **Tabbed Results Interface**:
  - 📋 **Summary** - Quick verdict with key points
  - 🔍 **Detailed Analysis** - Methodology, context, limitations
  - ✓ **Claims & Evidence** - Individual claim breakdown with clickable sources
- **Trust Level Scoring** - Sources rated by credibility
- **Clickable Source Links** - Direct access to verification sources

## 📁 Project Structure

```
VisioNova/
├── backend/
│   ├── app.py                      # Flask API server
│   ├── ai/
│   │   ├── __init__.py
│   │   └── groq_client.py          # Groq LLM integration (Llama 3.3 70B)
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
├── .env                            # Environment variables (API keys)
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js (optional, for development)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/DhanushPillay/VisioNova.git
   cd VisioNova
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install flask flask-cors python-dotenv requests beautifulsoup4 groq ddgs
   ```

4. **Set up environment variables**
   Create a `.env` file in the `backend/` folder:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_MODEL=llama-3.3-70b-versatile
   ```

5. **Run the backend**
   ```bash
   python backend/app.py
   ```

6. **Open the frontend**
   Open `frontend/html/FactCheckPage.html` in your browser

## 🎨 Tech Stack

### Frontend
- **HTML5** - Structure and semantics
- **Tailwind CSS** - Utility-first styling (via CDN)
- **JavaScript** - Interactive functionality
- **Google Material Symbols** - Icon library
- **Inter Font** - Typography

### Backend
- **Python 3.10+** - Core language
- **Flask** - REST API framework
- **Groq API** - LLM inference (Llama 3.3 70B)
- **BeautifulSoup** - Web scraping
- **DuckDuckGo Search** - Web search API

## 🌙 Design Features

- **Dark Theme** - Modern charcoal & navy color scheme
- **Glassmorphism** - Frosted glass panel effects
- **Responsive** - Mobile-first design approach
- **Micro-animations** - Smooth transitions and hover effects
- **High Contrast** - Accessibility-focused color choices

## 📊 Credibility Scoring

VisioNova uses a **Unified Credibility Score** (0-100) that combines:
- AI probability analysis
- Manipulation detection confidence
- Metadata verification
- Source reliability assessment

| Score Range | Status |
|-------------|--------|
| 80-100 | ✅ Likely Authentic |
| 50-79 | ⚠️ Review Recommended |
| 0-49 | ❌ High Risk / Manipulated |

## 🔍 Fact-Check Verdicts

The AI fact-checker returns one of these verdicts:

| Verdict | Meaning |
|---------|---------|
| ✅ TRUE | Claim is verified by trusted sources |
| ❌ FALSE | Claim is contradicted by evidence |
| ⚠️ PARTIALLY TRUE | Some truth but context missing |
| ⚠️ MISLEADING | Technically true but deceptive |
| ❓ UNVERIFIABLE | Insufficient evidence to verify |

## 🤝 Contributing

**Contributors:**
- Dhanush Pillay
- Shubhangini Dixit

---

<div align="center">

**Built with ❤️ for truth and transparency**

</div>
