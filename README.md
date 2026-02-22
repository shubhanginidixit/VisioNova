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
*   **State-of-the-art AI Detection:** Uses top-performing 2025-2026 Vision Transformers (Ateeqq SigLIP2, Bombek1 SigLIP2+DINOv2, Organika SDXL-detector) for highly accurate detection across modern generators (Flux, Midjourney v6, DALL-E 3).
*   **Ensemble Fusion System:** Combines multiple ML models via a weighted scoring system with majority-vote safeguards to prevent false positives.
*   **Error Level Analysis (ELA):** Highlights areas of potential manipulation within an image file.
*   **Metadata Forensics & Content Credentials:** Analyzes Exif data and C2PA digital signatures for inconsistencies or AI generation flags.
*   See [Image Detection Strategy](docs/Image_Detection_Strategy.md) for technical details.

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
*   **AI/ML:** PyTorch, Transformers (DeBERTa-v3, RoBERTa, DistilGPT-2), Groq API (Llama 4 Scout)
*   **Document Parsing:** PyMuPDF (PDF), python-docx (DOCX), pytesseract (OCR)
*   **Frontend:** HTML5, Tailwind CSS, JavaScript
*   **Analysis:** OpenCV, Librosa, Scikit-learn

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
│   ├── text_detector/         # Text AI detection & document parsing
│   │   ├── text_detector_service.py  # Core detection engine (offline/ML/binoculars)
│   │   ├── document_parser.py        # PDF/DOCX/TXT extraction with OCR fallback
│   │   ├── binoculars_detector.py    # Zero-shot Falcon-7B detector
│   │   └── preprocessor.py           # NLP preprocessing utilities
│   ├── audio_detector/        # Audio deepfake detection
│   ├── video_detector/        # Video deepfake detection
│   ├── app.py                 # Main Flask application
│   └── requirements.txt       # Python dependencies
├── docs/                      # Documentation
│   ├── ML_SETUP_COMPLETE.md   # ML models setup guide
│   ├── QUICKSTART_ML.md       # Quick start for ML features
│   ├── Image_Detection.md
│   ├── Text_Detection.md
│   ├── Audio_Detection.md
│   ├── Video_Detection.md
│   ├── Fact_Check.md
│   └── SystemArchitecture.md
├── frontend/                  # Web interface
│   ├── html/                  # HTML pages
│   ├── css/                   # Stylesheets
│   └── js/                    # JavaScript modules
├── notebooks/                 # Jupyter notebooks
│   ├── DeBERTa_Training_Notebook.ipynb
│   └── VisioNova_Colab_Training.ipynb
├── scripts/                   # Utility scripts
│   ├── setup_ml_models.py     # ML model setup (Python 3.10)
│   ├── train_deberta.py       # Text model training
│   └── download_models.py     # Legacy model downloader
├── tests/                     # Test files
│   ├── test_image_api.py
│   └── test_binoculars.py
├── results/                   # Training outputs (gitignored)
├── .venv/                     # Python 3.13 environment
├── .venv310/                  # Python 3.10 for ML models (gitignored)
├── requirements_ml.txt        # ML-specific dependencies (PyTorch + CUDA)
└── README.md                  # This file
```

### Key Folders

- **backend/**: Core application logic and API endpoints
- **docs/**: Comprehensive technical documentation
- **frontend/**: User interface (HTML/CSS/JS)
- **notebooks/**: Training notebooks for model development
- **scripts/**: Standalone utilities and training scripts
- **tests/**: Unit and integration tests

### ML Models Setup

For GPU-accelerated image detection (98%+ accuracy):
1. See [docs/ML_SETUP_COMPLETE.md](docs/ML_SETUP_COMPLETE.md)
2. Quick start: [docs/QUICKSTART_ML.md](docs/QUICKSTART_ML.md)

## Contributors

*   Dhanush Pillay
*   Shubhangini Dixit

## License

This project is intended for educational and research purposes.
