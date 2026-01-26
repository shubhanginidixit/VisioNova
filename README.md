# VisioNova

VisioNova is a multi-modal AI credibility engine designed to verify the authenticity of digital media and detect misinformation. It provides comprehensive forensic analysis for images, videos, audio, text, and fact-checking.

## Overview

In an era of generative AI and digital manipulation, VisioNova serves as a security tool for verifying media integrity. It combines machine learning models, spectral analysis, and cross-referenced fact-checking to provide users with a confidence score regarding the authenticity of the content they consume.

## Key Capabilities

### Image Verification
*   **AI Generation Detection:** Identifies images created by generative adversarial networks (GANs) or diffusion models.
*   **Error Level Analysis (ELA):** Highlights areas of potential manipulation within an image file.
*   **Metadata Forensics:**  Analyzes Exif data for inconsistencies.

### Video Analysis
*   **Deepfake Detection:** Analyzes frame-by-frame artifacts, facial landmarks, and lip-sync consistency to identify synthetic videos.
*   **Motion Consistency:** Verifies that physical movements obey natural laws.

### Audio Forensics
*   **Voice Cloning Detection:** Identifies synthetic vocal patterns and indicators of text-to-speech generation.
*   **Spectral Analysis:** Examines frequency distribution for anomalies typical of AI audio.

### Text Analysis
*   **AI Text Detection:** Distinguishes between human-written and AI-generated text using hybrid analysis (Neural Models + Stylometry).
*   **Pattern Recognition:** Identifies common rhetorical patterns found in Large Language Model outputs.
*   See [Text Detection Documentation](docs/TextDetection_Documentation.md) for technical details.

### Fact Checking
*   **Atomic Claim Verification:** Decomposes complex multi-part rumors into individual atomic facts for precise verification.
*   **Claim Verification:** Cross-references claims against trusted news sources and fact-checking databases.
*   **Temporal Analysis:** Contextualizes claims within their correct time period (historical vs. current).
*   **Source Credibility:** Scores the reliability of sources based on a curated database of domain trust ratings.
*   See [Fact Check Documentation](docs/FactCheck_Documentation.md) for technical details.

## Technology Stack

*   **Backend:** Python 3.10+, Flask
*   **AI/ML:** PyTorch, Transformers (DistilBERT), Groq API (Llama 3)
*   **Frontend:** HTML5, CSS3, JavaScript
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

## Contributors

*   Dhanush Pillay
*   Shubhangini Dixit

## License

This project is intended for educational and research purposes.
