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
*   **Multi-Model AI Detection:** Utilizes an ensemble of state-of-the-art vision models (DIRE, UniversalFakeDetect, SigLIP2\+DINOv2) for generation detection.
*   **Granular System Safeguards:** Employs weighted score fusion with majority-vote safeguards to eliminate false positives on highly-processed human photography.
*   **Error Level Analysis (ELA):** Highlights areas of potential manipulation within an image file.
*   **Metadata Forensics:**  Analyzes Exif data for inconsistencies.
*   See [Image Detection](docs/Image_Detection.md) for technical details.

### Video Analysis
*   **Deepfake Detection:** Analyzes frame-by-frame artifacts, facial landmarks, and lip-sync consistency to identify synthetic videos.
*   **Motion Consistency:** Verifies that physical movements obey natural laws.
*   See [Video & Deepfake Detection](docs/Video_Detection.md) for technical details.

### Audio Forensics
*   **Voice Cloning Detection:** Identifies synthetic vocal patterns and indicators of text-to-speech generation.
*   **Spectral Analysis:** Examines frequency distribution for anomalies typical of AI audio.
*   See [Audio Detection](docs/Audio_Detection.md) for technical details.

### Text Analysis
*   **AI Text Detection:** Distinguishes between human-written and AI-generated text using hybrid analysis (Neural Models + Stylometry).
*   **Pattern Recognition:** Identifies common rhetorical patterns found in Large Language Model outputs.
*   **Document Support:** Upload PDF, DOCX, or TXT files for full text extraction and AI detection with sentence-level analysis.
*   **Sentence Highlighting:** Color-coded sentence-by-sentence AI probability (green=human, yellow=uncertain, red=AI) with hover tooltips.
*   **AI Explanation:** Groq/Llama-powered detailed breakdown of detection results with key indicators, pattern analysis, and improvement suggestions.
*   See [Text Detection](docs/Text_Detection.md) for technical details.

### Fact Checking
*   **Atomic Claim Verification:** Decomposes complex multi-part rumors into individual atomic facts for precise verification.
*   **Claim Verification:** Cross-references claims against trusted news sources and fact-checking databases.
*   **Temporal Analysis:** Contextualizes claims within their correct time period (historical vs. current).
*   **Source Credibility:** Scores the reliability of sources based on a curated database of domain trust ratings.
*   See [Fact Check Documentation](docs/Fact_Check.md) for technical details.

## Technology Stack

*   **Backend:** Python 3.10+, Flask
*   **AI/ML:** PyTorch, Transformers (Binoculars/Falcon-7B, Wav2Vec2, WavLM, ViT), Groq API (Llama 4 Scout)
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
│   ├── audio_detector/        # Audio deepfake detection (5-model ensemble)
│   ├── image_detector/        # Image analysis detectors
│   ├── text_detector/         # Text AI detection & document parsing
│   ├── video_detector/        # Video deepfake detection
│   └── app.py                 # Main Flask application
├── docs/                      # Documentation
│   ├── Audio_Detection.md     # Audio models and capabilities
│   ├── Fact_Check.md          # Fact checking integration
│   ├── Image_Detection.md     # Image forgery tools
│   ├── Project_Vision_and_Security.md # Vision & goals
│   ├── Text_Detection.md      # Text and document processing
│   └── Video_Detection.md     # Video frame inspection
├── frontend/                  # Web interface
│   ├── css/                   # Stylesheets
│   ├── html/                  # HTML pages (Dashboards, Results)
│   └── js/                    # Client-side JavaScript
└── README.md                  # This file
```

### Key Folders

- **backend/**: Core application logic and API endpoints
- **docs/**: Comprehensive technical documentation
- **frontend/**: User interface (HTML/CSS/JS)

## Contributors

*   Dhanush Pillay
*   Shubhangini Dixit

## License

This project is intended for educational and research purposes.

