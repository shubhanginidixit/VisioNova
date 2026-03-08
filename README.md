# VisioNova

VisioNova is a multi-modal AI credibility engine designed to verify the authenticity of digital media and detect misinformation. It provides comprehensive forensic analysis for images, videos, audio, text, and fact-checking.

## Problem & Solution

VisioNova provides a secure and comprehensive platform to analyze all digital media (images, videos, audio, text) in one place without trusting sources blindly on the internet. Users can easily upload, analyze, and verify the authenticity of content with detailed forensic reports whenever they need them. It keeps information integrity at the forefront, helps users make informed decisions, and ensures they always have access to transparent explanations during verification, giving users full control over what they believe and share.

**The Challenge:**  
Generative AI tools (ChatGPT, Midjourney, ElevenLabs, Sora) have made it trivially easy to create convincing fake content. Deepfakes spread 6x faster than truth on social media, and traditional verification methods (metadata checks, watermarking) fail as AI learns to cover its tracks.

**The VisioNova Solution:**  
A multi-modal forensic engine that doesn't just detect AI media—it *explains* why content is flagged. By analyzing text, images, audio, and video simultaneously, and cross-referencing factual claims against trusted sources, VisioNova restores the "Chain of Trust" in digital media consumption.

## Key Capabilities

### Image Verification
*   **Top 5 Pretrained ML Models (2025-2026):** Uses the highest-accuracy Vision Transformers from Hugging Face:
    1. **Bombek1 SigLIP2+DINOv2** — 99.97% AUC, covers 25+ generators (Jan 2026)
    2. **Ateeqq SigLIP2** — 99.23% accuracy, 46K downloads (Dec 2025)
    3. **dima806 ViT** — 98.25% accuracy, 50K downloads (Jan 2025)
    4. **Organika SDXL-Detector** — 98.1% accuracy, Flux/SDXL specialist
    5. **WpythonW DINOv2** — Degradation-resilient, social media optimized
*   **Ensemble Fusion System:** Combines all 5 models via weighted scoring with majority-vote safeguards to prevent false positives.
*   **Binary Verdict:** Outputs a definitive 100% AI or 100% Human classification.
*   **Supporting Analysis:** Error Level Analysis (ELA), metadata forensics, C2PA content credentials, AI watermark detection, and Groq Vision AI explanations.
*   See [Image Detection Docs](docs/Image_Detection.md) for technical details.

### Video Analysis
*   **Deepfake Detection:** Analyzes frame-by-frame artifacts, facial landmarks, and lip-sync consistency to identify synthetic videos.
*   **Motion Consistency:** Verifies that physical movements obey natural laws.
*   See [Video & Deepfake Strategy](docs/Video_Deepfake_Strategy.md) for technical details.

### Audio Forensics
*   **Voice Cloning Detection:** Identifies synthetic vocal patterns and indicators of text-to-speech generation.
*   **Spectral Analysis:** Examines frequency distribution for anomalies typical of AI audio.
*   See [Audio Detection Strategy](docs/Audio_Detection.md) for technical details.

### Text Analysis
*   **AI Text Detection:** Distinguishes between human-written and AI-generated text using hybrid analysis (Neural Models + Stylometry).
*   **Pattern Recognition:** Identifies common rhetorical patterns found in Large Language Model outputs.
*   **Document Support:** Upload PDF, DOCX, or TXT files for full text extraction and AI detection with sentence-level analysis.
*   **Sentence Highlighting:** Color-coded sentence-by-sentence AI probability (green=human, yellow=uncertain, red=AI) with hover tooltips.
*   **AI Explanation:** Groq/Llama-powered detailed breakdown of detection results with key indicators, pattern analysis, and improvement suggestions.
*   See [Text Detection Strategy](docs/Text_Detection.md) for technical details.

### Fact Checking
*   **Atomic Claim Verification:** Decomposes complex multi-part rumors into individual atomic facts for precise verification.
*   **Claim Verification:** Cross-references claims against trusted news sources and fact-checking databases.
*   **Temporal Analysis:** Contextualizes claims within their correct time period (historical vs. current).
*   **Source Credibility:** Scores the reliability of sources based on a curated database of domain trust ratings.
*   See [Fact Check Documentation](docs/Fact_Check.md) for technical details.

## Technology Stack

*   **Backend:** Python 3.10+, Flask
*   **AI/ML:** PyTorch, Transformers (SigLIP2, DINOv2, ViT, Swin, DeBERTa-v3, RoBERTa), Groq API (Llama 4 Scout)
*   **Image Models:** Bombek1 SigLIP2+DINOv2, Ateeqq SigLIP2, dima806 ViT, Organika SDXL-Detector, WpythonW DINOv2

## Quick Start

### Prerequisites
*   Python 3.10 or higher
*   Groq API Key (for LLM features)

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/DhanushPillay/VisioNova.git
    cd VisioNova
    ```

2.  **Set up Virtual Environment**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure all dependencies from `backend/requirements.txt` are installed if available, or install manually: `flask flask-cors python-dotenv requests beautifulsoup4 groq torch transformers`)*

4.  **Configuration**
    Create a `.env` file in the `backend/` directory:
    ```env
    GROQ_API_KEY=your_api_key_here
    GROQ_MODEL=llama-3.3-70b-versatile
    ```

5.  **Run the Application**
    ```bash
    python backend/app.py
    ```

## Usage

Access the web interface by opening the corresponding HTML files in the `frontend/html/` directory or accessing the local server if configured to serve static files.

*   **Dashboard:** `frontend/html/AnalysisDashboard.html`
*   **Fact Check:** `frontend/html/FactCheckPage.html`

## Project Structure

```
VisioNova/
├── backend/                    # Flask server and core logic
│   ├── AI/                    # AI/LLM integration modules
│   ├── fact_check/            # Fact-checking engine
│   ├── image_detector/        # Image analysis detectors
│   │   ├── ml_detector.py           # Top 5 pretrained ML models
│   │   ├── ensemble_detector.py     # Weighted ensemble orchestrator
│   │   ├── confidence_calibrator.py # Score calibration
│   │   ├── watermark_detector.py    # AI watermark detection
│   │   ├── content_credentials.py   # C2PA credentials
│   │   ├── metadata_analyzer.py     # EXIF metadata forensics
│   │   ├── ela_analyzer.py          # Error Level Analysis
│   │   ├── noise_analyzer.py        # Noise pattern analysis
│   │   ├── image_explainer.py       # Groq Vision AI explanation
│   │   └── fast_cascade_detector.py # Speed-optimized endpoint
│   ├── text_detector/         # Text AI detection & document parsing
│   │   ├── text_detector_service.py  # Core detection engine (offline/ML/binoculars)
│   │   ├── document_parser.py        # PDF/DOCX/TXT extraction with OCR fallback
│   │   ├── binoculars_detector.py    # Zero-shot Falcon-7B detector
│   │   └── preprocessor.py           # NLP preprocessing utilities
│   ├── audio_detector/        # Audio deepfake detection
│   ├── video_detector/        # Video deepfake detection
│   ├── download_image_models.py # Pre-download script for 5 ML models
│   ├── app.py                 # Main Flask application
│   └── requirements.txt       # Python dependencies
├── docs/                      # Documentation
│   ├── Image_Detection.md
│   ├── Text_Detection.md
│   ├── Audio_Detection.md
│   ├── Video_Detection.md
│   └── Fact_Check.md
├── frontend/                  # Web interface
│   ├── html/                  # HTML pages
│   ├── css/                   # Stylesheets
│   └── js/                    # JavaScript modules
└── README.md                  # This file
```

### ML Models Setup

To pre-download all 5 image detection models (recommended):
```bash
python backend/download_image_models.py
```
Then start the server:
```bash
python backend/app.py
```
Models are automatically loaded on first use via the ensemble endpoint.

## Contributors

*   Dhanush Pillay
*   Shubhangini Dixit

## License

This project is intended for educational and research purposes.
