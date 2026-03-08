# Image Detection

## Executive Summary

VisioNova's image detection pipeline uses an **ensemble of 5 state-of-the-art pretrained ML models** (2025-2026) from Hugging Face to classify images as AI-generated or human-created. The system outputs a **binary verdict** (100% AI or 100% Human) for maximum clarity, backed by weighted score fusion across all models.

---

## 1. The Challenge: Why Image Detection is Getting Harder

### The Evolution of Image Generation

| Era | Technology | Visual Quality | Detection Difficulty |
|-----|------------|----------------|---------------------|
| 2019-2021 | GANs (StyleGAN, BigGAN) | Artifacts visible (hands, text) | Low-Medium |
| 2022-2023 | Diffusion (SD 1.5, DALL-E 2) | Occasional flaws | Medium |
| 2024-2025 | Advanced Diffusion (SDXL, MJ V6, Flux) | Near-photorealistic | High |
| 2025+ | GPT-Image-1, Video Diffusion (Sora, Veo) | Indistinguishable | Very High |

### Why Traditional Methods Fail

- **Statistical analysis** (noise, frequency, texture) — too many false positives on real photos
- **Heuristic detectors** (hand-coded rules) — cannot generalize across generators
- **Single-model approaches** — no single model covers all generators

**Solution:** Use an ensemble of pretrained Vision Transformers, each trained on millions of real and AI images, fused via weighted scoring.

---

## 2. Detection Architecture

### Ensemble Pipeline

```
                         INPUT IMAGE
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
  │  Tier 1 (54%) │  │  Tier 2 (42%) │  │  Tier 3 (4%)  │
  │  Best Models  │  │  Strong Models│  │  FFT/DCT      │
  ├───────────────┤  ├───────────────┤  ├───────────────┤
  │ • Bombek1     │  │ • dima806 ViT │  │ • Frequency   │
  │   SigLIP2+    │  │ • Organika    │  │   Analysis    │
  │   DINOv2      │  │   SDXL-Det    │  │   (GAN        │
  │ • Ateeqq      │  │ • WpythonW    │  │   fingerprint)│
  │   SigLIP2     │  │   DINOv2      │  │               │
  └───────────────┘  └───────────────┘  └───────────────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
                   ┌─────────────────────┐
                   │  Weighted Fusion    │
                   │  + Binary Snap      │
                   │  (>50% → 100% AI)   │
                   └──────────┬──────────┘
                              ▼
                   ┌─────────────────────┐
                   │  100% AI or         │
                   │  100% Human         │
                   └─────────────────────┘
```

### Supporting Signals (Not Scored, Informational)

- **Watermark Detection** — C2PA, SynthID, DWT-DCT, Meta Stable Signature
- **Metadata Forensics** — EXIF/IPTC analysis for AI software flags
- **ELA** — Error Level Analysis for manipulation detection
- **Groq Vision AI** — Natural language explanation of the detection result

---

## 3. Top 5 Pretrained ML Models

### Ensemble Weights

```python
DEFAULT_WEIGHTS = {
    # ═══ Tier 1: Best recent models (2025-2026, verified accuracy) ═══
    'ateeqq': 0.27,             # Ateeqq SigLIP2 (99.23% acc, Dec 2025)
    'siglip_dinov2': 0.27,      # Bombek1 SigLIP2+DINOv2 (99.97% AUC, Jan 2026)

    # ═══ Tier 2: Strong recent models ═══
    'deepfake': 0.16,           # dima806 ViT (98.25% acc, Jan 2025)
    'sdxl': 0.16,               # Organika/sdxl-detector (98.1% acc)
    'dinov2': 0.10,             # WpythonW DINOv2 (degradation-resilient)

    # ═══ Tier 3: Supporting signals ═══
    'frequency': 0.04,          # FFT/DCT analysis (GAN fingerprints)
}
```

### Model Details

#### 1. Bombek1 SigLIP2+DINOv2 (Weight: 27%)

| Property | Value |
|----------|-------|
| Model | `Bombek1/ai-image-detector-siglip-dinov2` |
| Architecture | Hybrid SigLIP2 + DINOv2 (dual-encoder) |
| AUC | **99.97%** |
| Cross-Dataset Accuracy | 97.15% |
| Generators Covered | 25+ (Flux, MJ v6, DALL-E 3, SDXL, GPT-Image-1) |
| Updated | January 2026 |
| Why #1 | Combines semantic (SigLIP2) + self-supervised (DINOv2) features for maximum robustness |

#### 2. Ateeqq SigLIP2 (Weight: 27%)

| Property | Value |
|----------|-------|
| Model | `Ateeqq/ai-vs-human-image-detector` |
| Architecture | SigLIP2 (Google's Sigmoid Language-Image Pretraining) |
| Accuracy | **99.23%** |
| Downloads | 46,000+ |
| Updated | December 2025 |
| Why #2 | Pure SigLIP2 backbone with very high accuracy and fast inference |

#### 3. dima806 ViT (Weight: 16%)

| Property | Value |
|----------|-------|
| Model | `dima806/deepfake_vs_real_image_detection` |
| Architecture | Vision Transformer (ViT-B/16) |
| Accuracy | **98.25%** |
| Downloads | 50,000+ |
| Updated | January 2025 |
| Why #3 | Massive download count, strong community validation, reliable ViT backbone |

#### 4. Organika SDXL-Detector (Weight: 16%)

| Property | Value |
|----------|-------|
| Model | `Organika/sdxl-detector` |
| Architecture | Swin Transformer |
| Accuracy | **98.1%** |
| AUC | 99.8% |
| Updated | 2024 |
| Why #4 | Specialist for modern diffusion models (SDXL, Flux), ONNX support |

#### 5. WpythonW DINOv2 (Weight: 10%)

| Property | Value |
|----------|-------|
| Model | `WpythonW/dinov2-deepfake-image` |
| Architecture | DINOv2 (Self-Supervised ViT) |
| Strength | Degradation-resilient |
| Best For | Social media images (heavy compression, resizing) |
| Why #5 | DINOv2's self-supervised training makes it highly resilient to image transformations |

---

## 4. Key Architectures Explained

| Architecture | Type | Why It Matters for Detection |
|---|---|---|
| **ViT** | Transformer | Captures global context; backbone for CLIP, DINOv2, SigLIP |
| **DINOv2** | Self-Supervised ViT | Most resilient to image transformations (compression, resizing) |
| **SigLIP2** | Vision-Language (Sigmoid) | Google's improved CLIP; better efficiency and precision |
| **Swin Transformer** | Hierarchical ViT | Multi-scale features; excellent for fine-grained artifacts |

### Why Ensembles Beat Single Models

No single model generalizes to ALL generators. GANs leave different fingerprints than diffusion models. An ensemble combining spatial (Swin), global (ViT), semantic (SigLIP2), and self-supervised (DINOv2) features covers the most ground.

---

## 5. Supporting Analysis

### 5.1 Error Level Analysis (ELA)

ELA detects image manipulation by analyzing JPEG compression artifacts.

1. Re-compress the image at a known quality (e.g., 95%)
2. Calculate the difference between original and recompressed versions
3. Manipulated regions show different error levels than authentic areas

### 5.2 Metadata Forensics

Analyzes EXIF/IPTC/XMP headers for:
- AI software signatures (e.g., "DALL-E 3", "Midjourney")
- Inconsistent camera metadata
- Missing sensor data (AI images lack PRNU noise patterns)

### 5.3 Watermark Detection (10 Methods)

| # | Method | Technology | Accuracy |
|---|--------|------------|----------|
| 1 | **DWT-DCT** | Discrete Wavelet + Cosine Transform | 85% |
| 2 | **Meta Stable Signature** | 48-bit neural watermark decoder | 90% |
| 3 | **Spectral Pattern Analysis** | FFT frequency analysis | 50% |
| 4 | **LSB Analysis** | Statistical steganography detection | 45% |
| 5 | **Tree-Ring Detection** | Radial frequency profile analysis | 70% |
| 6 | **Gaussian Shading** | Variance distribution analysis | 60% |
| 7 | **Metadata Watermarks** | EXIF/IPTC/XMP header analysis | 95% |
| 8 | **SteganoGAN** | GAN-based steganography decoder | 80% |
| 9 | **Adversarial Perturbation** | Gradient kurtosis (Experimental) | 30% |
| 10 | **Google SynthID** | Proprietary (cannot detect externally) | N/A |

### 5.4 C2PA Content Credentials

The Coalition for Content Provenance and Authenticity (C2PA) standard embeds verifiable metadata:
- **Creator information**: Who created the content
- **Creation tool**: Software/model used (e.g., "DALL-E 3")
- **Edit history**: Cryptographically signed modification chain
- **Adopters (2025)**: Adobe, Google, Microsoft, OpenAI

### 5.5 Groq Vision AI Explanation

Uses Llama 4 Scout via the Groq API to generate a natural language explanation of detection results, including:
- Key visual indicators found
- Pattern analysis breakdown
- Confidence reasoning

---

## 6. Binary Verdict System

The final AI probability is **snapped to a binary output**:

| Raw Score | Final Output | Verdict |
|-----------|-------------|---------|
| > 50% | **100% AI** | AI-Generated Content Detected |
| ≤ 50% | **0% AI (100% Human)** | Authentic Content |

This is enforced at both the backend API level and the frontend display level to eliminate ambiguous results.

---

## 7. File Structure

```
image_detector/
├── ml_detector.py             # Top 5 pretrained ML model classes
├── ensemble_detector.py       # Weighted ensemble orchestrator
├── confidence_calibrator.py   # Score calibration utilities
├── watermark_detector.py      # AI watermark detection (10 methods)
├── content_credentials.py     # C2PA content credentials
├── metadata_analyzer.py       # EXIF/IPTC metadata forensics
├── ela_analyzer.py            # Error Level Analysis
├── noise_analyzer.py          # Noise pattern analysis
├── image_explainer.py         # Groq Vision AI explanation
├── fast_cascade_detector.py   # Speed-optimized detection endpoint
└── models/                    # Downloaded model weights
```

---

## 8. API Endpoints

### Primary: `/api/detect-image/ensemble` (POST)

**Request (JSON):**
```json
{
  "image": "data:image/jpeg;base64,...",
  "filename": "photo.jpg",
  "load_ml_models": true
}
```

**Response:**
```json
{
  "success": true,
  "ai_probability": 100.0,
  "verdict": "AI_GENERATED",
  "individual_results": {
    "ateeqq": { "ai_probability": 98.7, "success": true },
    "siglip_dinov2": { "ai_probability": 99.1, "success": true },
    "deepfake": { "ai_probability": 97.2, "success": true },
    "sdxl": { "ai_probability": 96.5, "success": true },
    "dinov2": { "ai_probability": 95.8, "success": true }
  },
  "overrides_applied": [],
  "recommendations": []
}
```

### Supporting Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/api/detect-image/fast` | Speed-optimized cascade detection |
| `/api/image/ela` | Error Level Analysis with heatmap |
| `/api/image/metadata` | EXIF/metadata extraction |
| `/api/image/watermark` | Watermark detection |
| `/api/image/content-credentials` | C2PA verification |

---

## 9. Known Limitations

1. **AI-enhanced photos**: Humans using AI for minor edits may be flagged
2. **Heavy JPEG compression**: Destroys some forensic signals
3. **Social media processing**: Platforms strip metadata and recompress
4. **New generators**: Each new model (GPT-Image-1, MJ v7) may require model updates
5. **First-run latency**: Initial model loading takes 30-60 seconds

---

## 10. Model Download

Pre-download all 5 models to avoid first-run delays:

```bash
python backend/download_image_models.py
```

This downloads approximately 2-3 GB of model weights to the Hugging Face cache.

---

## References

1. SigLIP2: Google Research (2025)
2. DINOv2: Meta AI Research (2023)
3. Swin Transformer: Microsoft Research (2021)
4. Vision Transformer (ViT): Google Research (2020)
5. C2PA Standard: [c2pa.org](https://c2pa.org)
6. Hugging Face Model Hub: [huggingface.co](https://huggingface.co)
