# VisioNova

**Multi-Modal AI Credibility Engine for Digital Media Verification**

VisioNova is a comprehensive forensic analysis platform that verifies the authenticity of digital media and detects misinformation across text, images, audio, and video. It combines state-of-the-art AI detection with explainable verdicts.

---

## The Problem (2025-2026)

| Threat | Scale |
|--------|-------|
| Deepfake videos | 8M+ (16× increase from 2023) |
| AI face-swap attacks | +300% surge |
| AI-generated online content | Est. 50%+ |
| Consumers who can spot deepfakes | 0.1% |

Generative AI tools (Sora, Midjourney, ElevenLabs, ChatGPT) have made it trivially easy to create convincing fake content. Traditional verification methods fail as AI learns to cover its tracks.

## The Solution

VisioNova provides a **multi-modal forensic engine** that doesn't just detect AI media—it *explains why* content is flagged. By analyzing text, images, audio, and video simultaneously, and cross-referencing claims against trusted sources, VisioNova restores the "Chain of Trust" in digital media.

---

## Features

### Text Detection
| Method | Technology | Accuracy |
|--------|------------|----------|
| Neural Detection | DeBERTa-v3 + Ensemble | 95%+ |
| Zero-Shot | Binoculars (perplexity analysis) | Cross-generator |
| Stylometry | Statistical patterns | Supplementary |

**Detects:** ChatGPT, Claude, DeepSeek R1, Gemini, LLaMA, Mistral  
📄 [Text Detection Strategy](docs/Text_Detection_Strategy.md)

---

### Image Detection
| Method | Technology | Performance |
|--------|------------|-------------|
| ViT Detection | NYUAD Model | 97.36% accuracy |
| Forensic Analysis | ELA, Frequency (FFT/DCT) | Artifact detection |
| Provenance | C2PA, SynthID | Watermark verification |

**Detects:** Midjourney, DALL-E 3, Stable Diffusion, Firefly  
📄 [Image Detection Strategy](docs/Image_Detection_Strategy.md)

---

### Audio Detection
| Method | Technology | Performance |
|--------|------------|-------------|
| Deep Learning | HuBERT, WavLM | 2.89% EER |
| Spectral Analysis | MFCCs, Mel-spectrograms | Voice cloning patterns |
| Watermarks | AudioSeal, SynthID | Origin verification |

**Detects:** ElevenLabs, XTTS, Tortoise TTS, voice cloning  
📄 [Audio Detection Strategy](docs/Audio_Detection_Strategy.md)

---

### Video Detection
| Method | Technology | Performance |
|--------|------------|-------------|
| Frame Analysis | EfficientNet-B4 | 95.59% AUC |
| Temporal | LSTM + Landmark tracking | Motion consistency |
| Audio-Visual | SyncNet, TrueSync | Lip-sync verification |
| Biological | rPPG | Pulse signal detection |

**Detects:** Sora, Veo, Runway, DeepFaceLab, face swaps  
📄 [Video Deepfake Strategy](docs/Video_Deepfake_Strategy.md)

---

### Fact Checking
| Feature | Description |
|---------|-------------|
| Multi-Source Search | Google, DuckDuckGo, Wikipedia |
| Credibility Scoring | 70+ rated sources (Snopes, Reuters, AP) |
| Temporal Analysis | Historical vs. current context |
| AI Analysis | LLM-powered verdict with fallback |

**Verdicts:** TRUE, FALSE, PARTIALLY TRUE, MISLEADING, UNVERIFIABLE  
📄 [Fact Check Documentation](docs/FactCheck_Documentation.md)

---

## Technology Stack

| Layer | Technologies |
|-------|--------------|
| **Backend** | Python 3.10+, Flask, Flask-CORS |
| **AI/ML** | PyTorch, Transformers (DeBERTa, ViT, Wav2Vec2), Groq API (LLaMA 3) |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Analysis** | OpenCV, Librosa, Scikit-learn, torchaudio |
| **Search** | Google Custom Search API, DuckDuckGo, MediaWiki API |

---

## Quick Start

### Prerequisites
- Python 3.10+
- Groq API Key (for LLM features)
- Google API Key (optional, for enhanced search)

### Installation

```bash
# Clone repository
git clone https://github.com/DhanushPillay/VisioNova.git
cd VisioNova

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

### Configuration

Create `backend/.env`:

```env
# Required
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile

# Optional (enhanced fact-checking)
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_custom_search_engine_id
```

### Run

```bash
python backend/app.py
```

Access the dashboard at `http://localhost:5000` or open:
- **Dashboard:** `frontend/html/AnalysisDashboard.html`
- **Fact Check:** `frontend/html/FactCheckPage.html`

---

## Project Structure

```
VisioNova/
├── backend/
│   ├── app.py              # Flask API server
│   ├── AI/                 # Groq/LLM integration
│   ├── text_detector/      # Text AI detection
│   ├── image_detector/     # Image forensics
│   ├── fact_check/         # Fact verification pipeline
│   └── requirements.txt
├── frontend/
│   ├── html/               # Page templates
│   ├── js/                 # Frontend logic
│   └── css/                # Styles
└── docs/                   # Strategy documentation
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Problem Statement](docs/Problem_Statement_and_Solutions.md) | The crisis of digital authenticity |
| [Text Detection](docs/Text_Detection_Strategy.md) | DeBERTa, Binoculars, RAID dataset |
| [Image Detection](docs/Image_Detection_Strategy.md) | ViT, ELA, C2PA, SynthID |
| [Audio Detection](docs/Audio_Detection_Strategy.md) | HuBERT, WavLM, AudioSeal |
| [Video Detection](docs/Video_Deepfake_Strategy.md) | Frame analysis, temporal, rPPG |
| [Fact Checking](docs/FactCheck_Documentation.md) | Full verification pipeline |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analyze/text` | POST | Analyze text for AI generation |
| `/api/analyze/image` | POST | Image forensic analysis |
| `/api/analyze/audio` | POST | Audio deepfake detection |
| `/api/analyze/video` | POST | Video manipulation detection |
| `/api/fact-check` | POST | Verify claims against sources |
| `/api/fact-check/deep` | POST | Enhanced temporal-aware check |

---

## Contributors

- **Dhanush Pillay** - Lead Developer
- **Shubhangini Dixit** - Developer

---

## License

This project is intended for educational and research purposes.

---

## Acknowledgments

- [Hugging Face](https://huggingface.co) - Transformers library
- [Groq](https://groq.com) - LLM inference
- [NYUAD](https://github.com/Yanyirong/NYUAD-ViT) - ViT detection model
- [ASVspoof](https://www.asvspoof.org) - Audio deepfake benchmarks
- [FaceForensics++](https://github.com/ondyari/FaceForensics) - Video deepfake datasets
