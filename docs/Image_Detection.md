# Image Detection

## Executive Summary

This document outlines VisioNova's comprehensive approach to detecting AI-generated and manipulated images. As generative AI evolves from GANs to sophisticated diffusion models (Midjourney V6, DALL-E 3, Stable Diffusion XL), our detection pipeline combines traditional forensic analysis with state-of-the-art deep learning to maintain high accuracy against increasingly realistic synthetic content.

---

## 1. The Challenge: Why Image Detection is Getting Harder

### The Evolution of Image Generation

| Era | Technology | Visual Quality | Detection Difficulty |
|-----|------------|----------------|---------------------|
| 2019-2021 | GANs (StyleGAN, BigGAN) | Artifacts visible (hands, text) | Low-Medium |
| 2022-2023 | Diffusion (SD 1.5, DALL-E 2) | Occasional flaws | Medium |
| 2024-2025 | Advanced Diffusion (SDXL, MJ V6) | Near-photorealistic | High |
| 2025+ | Video + 3D Diffusion (Sora, Veo) | Indistinguishable | Very High |

### Common Visual Artifacts (Becoming Less Reliable)

Traditional tell-tale signs that are becoming harder to spot:
- **Hands/Fingers**: Extra or missing digits (much improved in latest models)
- **Text/Typography**: Garbled or nonsensical text (SDXL + ControlNet fix this)
- **Jewelry/Accessories**: "Melting" into skin
- **Symmetry**: Unnatural facial/object symmetry
- **Backgrounds**: Inconsistent lighting or perspective

---

## 2. Detection Methods Overview

### Multi-Stage Ensemble Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                    INPUT IMAGE                                    │
└────────────────────────┬─────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│  Statistical  │ │  ML Models    │ │  Provenance   │
│  Analysis     │ │  (ViT, CLIP)  │ │  (Watermarks) │
├───────────────┤ ├───────────────┤ ├───────────────┤
│ • ELA         │ │ • NYUAD ViT   │ │ • C2PA/CAI    │
│ • FFT/DCT     │ │ • CLIP        │ │ • SynthID     │
│ • Noise       │ │ • Deepfake    │ │ • DWT-DCT     │
│ • Texture     │ │ • Attribution │ │ • Metadata    │
└───────────────┘ └───────────────┘ └───────────────┘
        │                │                │
        └────────────────┼────────────────┘
                         ▼
              ┌─────────────────────┐
              │  Weighted Fusion    │
              │  + Agreement Score  │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │   Final Verdict     │
              └─────────────────────┘
```

### Detection Performance (Current)

```
Input Image
    │
    ├─→ Statistical Analysis (20%)
    ├─→ dima806 ViT (30%) ← CUDA GPU
    ├─→ CLIP Universal (30%) ← CUDA GPU
    ├─→ Frequency Analysis (10%)
    └─→ Watermark Detection (10%)
         │
         ▼
    Weighted Score Fusion
         │
         ▼
    Final Verdict (with confidence)
```

---

## 3. Statistical and Forensic Analysis

### 3.1 Error Level Analysis (ELA)

ELA detects image manipulation by analyzing JPEG compression artifacts.

**How It Works:**
1. Re-compress the image at a known quality (e.g., 95%)
2. Calculate the difference between original and recompressed versions
3. Manipulated regions show different error levels than authentic areas

| Scenario | ELA Effectiveness |
|----------|-------------------|
| JPEG photo manipulation | High |
| Spliced/composite images | High |
| AI-generated from scratch | Low |
| Re-saved multiple times | Medium |

### 3.2 Frequency Domain Analysis (FFT/DCT)

Analyzes the image in the frequency spectrum to detect generator-specific fingerprints.

**GAN Fingerprints:**
- **Checkerboard artifacts**: Grid-like patterns in high frequencies
- **Periodic peaks**: Anomalous spikes in the Fourier transform
- **Spectral flatness**: Abnormal uniformity in frequency distribution

**Diffusion Model Detection:**
- **Smoother frequency profile**: Less obvious periodic patterns
- **Noise distribution**: Different from camera sensor noise
- **Snap-back analysis**: Real images degrade abruptly when noise is added; AI images degrade smoothly

### 3.3 Noise and Texture Analysis

Camera sensors leave a unique "fingerprint" called Photo-Response Non-Uniformity (PRNU).

- Each camera has a unique noise pattern from sensor manufacturing
- AI-generated images lack authentic PRNU patterns
- AI often fails at micro-texture patterns (skin pores, fabric weave)

**Noise Analyzer Output:**
```json
{
  "noise_consistency": 60,
  "low_freq": 94,
  "mid_freq": 4,
  "high_freq": 1,
  "noise_map": "data:image/png;base64,...",
  "pattern_analysis": {
    "noise_level": 12.5,
    "uniformity": 0.85,
    "texture_variance": 0.42,
    "sensor_pattern": false,
    "artificial_smoothing": 0.73
  }
}
```

---

## 4. Deep Learning Models

### 4.1 Vision Transformer (ViT) Detectors

Vision Transformers process images as sequences of patches, enabling global pattern recognition.

**NYUAD ViT Detector:**
| Property | Value |
|----------|-------|
| Model | `NYUAD-ComNets/NYUAD_AI-generated_images_detector` |
| Base Architecture | ViT-B/16 |
| Accuracy | 97.36% on benchmark datasets |
| Input Size | 224×224 |

**dima806 Deepfake Detector:**
| Property | Value |
|----------|-------|
| Model | `dima806/deepfake_vs_real_image_detection` |
| Architecture | Vision Transformer (ViT-B/16) |
| Accuracy | **98.25%** on test dataset |
| Role | Primary detector in current ensemble |

**Why ViT Over CNN:**
- **Global attention**: Captures long-range dependencies across the entire image
- **Patch-based processing**: Naturally detects inconsistencies across regions
- **Better generalization**: Performs well on unseen generator types

### 4.2 CLIP-Based Detection

CLIP (Contrastive Language-Image Pre-training) provides powerful generalizable features.

**UniversalFakeDetect Approach:**
- Extract 768-dimensional feature vector from CLIP ViT-L/14
- Train linear classifier on real vs. AI features
- Analyze feature statistics (kurtosis, variance patterns)
- Generalizes across different AI generators
- Improved generalization by +25.9% accuracy on unseen models (CVPR 2023)

### 4.3 Deepfake Face Detection

| Signal | Description |
|--------|-------------|
| Lip-sync inconsistency | Audio doesn't match mouth movements |
| Blending artifacts | Edge boundaries between face and background |
| Eye reflections | Inconsistent reflections in pupils |
| Skin texture | Unnatural smoothness or pore patterns |

---

## 5. Watermark and Provenance Detection

### 5.1 Detection Methods (10 Methods)

| # | Method | Technology | Accuracy | False Positive Rate |
|---|--------|------------|----------|---------------------|
| 1 | **DWT-DCT** | Discrete Wavelet + Cosine Transform | 85% | ~2% |
| 2 | **Meta Stable Signature** | 48-bit neural watermark decoder | 90% | ~0.00000001% |
| 3 | **Spectral Pattern Analysis** | FFT frequency analysis | 50% | ~15% |
| 4 | **LSB Analysis** | Statistical steganography detection | 45% | ~20% |
| 5 | **Tree-Ring Detection** | Radial frequency profile analysis | 70% | ~8% |
| 6 | **Gaussian Shading** | Variance distribution analysis | 60% | ~12% |
| 7 | **Metadata Watermarks** | EXIF/IPTC/XMP header analysis | 95% | <1% |
| 8 | **SteganoGAN** | GAN-based steganography decoder | 80% | ~5% |
| 9 | **Adversarial Perturbation** | Gradient kurtosis (Experimental) | 30% | ~40% |
| 10 | **Google SynthID** | Proprietary (cannot detect externally) | N/A | N/A |

### 5.2 Generator Compatibility Matrix

| AI Generator | DWT-DCT | Stable Sig | Spectral | Metadata | Overall |
|-------------|---------|------------|----------|----------|---------|
| **Stable Diffusion** | ✅ High | ❌ | ⚠️ Medium | ⚠️ Sometimes | 75% |
| **AUTOMATIC1111** | ✅ High | ❌ | ⚠️ Medium | ✅ High | 80% |
| **Meta AI** | ❌ | ✅ Very High | ⚠️ Low | ✅ High | 95% |
| **Adobe Firefly** | ❌ | ❌ | ❌ | ✅ Very High | 90% |
| **Midjourney** | ❌ | ❌ | ❌ | ✅ High | 85% |
| **DALL-E 3** | ❌ | ❌ | ❌ | ✅ Very High | 90% |

### 5.3 C2PA Content Credentials

The Coalition for Content Provenance and Authenticity (C2PA) standard embeds verifiable metadata:
- **Creator information**: Who created the content
- **Creation tool**: Software/model used (e.g., "DALL-E 3")
- **Edit history**: Cryptographically signed modification chain
- **Adopters (2025)**: Adobe, Google, Microsoft, OpenAI

### 5.4 Weighted Confidence Algorithm

```python
METHOD_WEIGHTS = {
    'metadata_watermark': 0.95,
    'stable_signature': 0.90,
    'invisible_watermark': 0.85,
    'treering_analysis': 0.70,
    'gaussian_shading': 0.60,
    'spectral_analysis': 0.50,
    'lsb_analysis': 0.45,
    'steganogan': 0.80,
    'adversarial_analysis': 0.30
}

weighted_confidence = Σ(weight × confidence) / Σ(weight)
```

---

## 6. Hugging Face Pre-Trained Models Research

### How AI Image Detection Works

AI image detection answers: **"Was this image created by a human or generated by AI?"** AI generators (GANs, Diffusion Models, Autoregressive models) leave subtle **fingerprints** that trained models can detect.

| Approach | What It Looks For | Strength | Weakness |
|---|---|---|---|
| **Spatial Domain** | Pixel distribution, color anomalies | Intuitive, works on raw images | Struggles with high-quality generators |
| **Frequency Domain** | FFT/DCT spectrum | Very discriminative | Degrades with JPEG compression |
| **Fingerprint / Noise** | Noise patterns unique to each generator | Highly generalizable | Requires large diverse training data |
| **Patch-Based (NPR)** | Inter-patch dependencies | Generalizes across 28+ generators | Computationally heavier |
| **Reconstruction (DIRE)** | Diffusion reconstruction error | Strong cross-generator generalization | Slow |
| **Semantic / VLM** | Scene plausibility via vision-language models | Catches "impossible" scenes | Fails on single-subject images |

### Key Architectures

| Architecture | Type | Why It Matters for Detection |
|---|---|---|
| **ViT** | Transformer | Captures global context; backbone for CLIP, DINOv2, SigLIP |
| **CLIP** | Vision-Language | Pre-trained on 400M image-text pairs; generalizes without task-specific training |
| **DINOv2** | Self-Supervised ViT | Most resilient to image transformations |
| **SigLIP** | Vision-Language (Sigmoid) | Google's improved CLIP; better efficiency and precision |
| **Swin Transformer** | Hierarchical ViT | Multi-scale features; excellent for fine-grained artifacts |

### Top 7 Pre-Trained Models on Hugging Face

#### 1. `umm-maybe/AI-image-detector`
| Property | Value |
|---|---|
| Architecture | ViT |
| Accuracy | 94.2% |
| F1 Score | 95.8% |
| AUC | 98.0% |
| Training Data | Human vs AI art (pre-SDXL era, Oct 2022) |
| Best For | AI-generated art/illustrations |
| ⚠️ Limitation | Does NOT detect SDXL, MJ v5+, DALL-E 3 images |

#### 2. `Organika/sdxl-detector`
| Property | Value |
|---|---|
| Architecture | Swin Transformer |
| Accuracy | **98.1%** |
| AUC | **99.8%** |
| F1 Score | 97.3% |
| Training Data | Wikimedia (real) + SDXL-generated images |
| Best For | Modern diffusion model outputs |
| 🏆 Top Pick | Best single-model accuracy, ONNX support, 36K+ monthly downloads |

#### 3. `dima806/ai_vs_human_generated_image`
| Property | Value |
|---|---|
| Architecture | CNN (EfficientNet-style) |
| Accuracy | **97.87%** |
| F1 Score | 97.87% (macro) |
| Best For | General AI vs Human classification |
| ⚠️ Note | Training data is 1–2 years old; creator warns about concept drift |

#### 4. `Nahrawy/AIorNot`
| Property | Value |
|---|---|
| Architecture | Swin-Tiny (`swin-tiny-patch4-window7-224`) |
| Input Size | 224×224 |
| Best For | Edge/fast deployment (smallest footprint) |

#### 5. `Ateeqq/ai-vs-human-image-detector`
| Property | Value |
|---|---|
| Architecture | SigLIP (Google's Sigmoid Language-Image Pretraining) |
| Best For | Semantic-level detection with vision-language features |

#### 6. `prithivMLmods/Deep-Fake-Detector-v2-Model`
| Property | Value |
|---|---|
| Architecture | ViT (`google/vit-base-patch16-224-in21k` fine-tuned) |
| Labels | `Real`, `Fake` |
| Best For | Face deepfake detection, media authentication |

#### 7. `Bombek1/ai-image-detector-siglip-dinov2`
| Property | Value |
|---|---|
| Architecture | Hybrid SigLIP + DINOv2 |
| Updated | January 2026 |
| Best For | Maximum robustness (dual-encoder: semantic + self-supervised features) |
| 🔥 Newest | Most recently updated; covers latest generators |

### Model Comparison Table

| # | Model | Architecture | Accuracy | Best For | Recency |
|---|---|---|---|---|---|
| 1 | `umm-maybe/AI-image-detector` | ViT | 94.2% | AI art | ⚠️ 2022 |
| 2 | `Organika/sdxl-detector` | Swin | **98.1%** | Diffusion images | ✅ Recent |
| 3 | `dima806/ai_vs_human` | CNN | 97.9% | General | ⚠️ 1-2yr old |
| 4 | `Nahrawy/AIorNot` | Swin-Tiny | N/A | Fast deployment | ✅ Recent |
| 5 | `Ateeqq/ai-vs-human` | SigLIP | N/A | Semantic detection | ✅ 2025 |
| 6 | `prithivMLmods/Deep-Fake-v2` | ViT | N/A | Deepfake faces | ✅ Recent |
| 7 | `Bombek1/siglip-dinov2` | SigLIP+DINOv2 | N/A | Max robustness | ✅ Jan 2026 |

### Key Research Insights

**Why Ensembles Beat Single Models:** No single model generalizes to ALL generators. GANs leave different fingerprints than diffusion models. An ensemble combining spatial (CNN), global (ViT), and semantic (SigLIP) features covers the most ground.

**The Generalization Problem:**
- **UniversalFakeDetect** (CVPR 2023): +25.9% accuracy using CLIP features without explicit fake training
- **NPR** (CVPR 2024): Captures upsampling artifacts across 28+ generators
- **DIRE**: Measures diffusion reconstruction error for cross-generator detection

**Data Freshness Matters:** AI generators evolve rapidly. A model trained in 2022 will miss 2024+ generators. Always check when a model was last trained and on what data.

---

## 7. Ensemble Scoring

### Current VisioNova Weights

```python
DEFAULT_WEIGHTS = {
    "statistical": 0.20,
    "dima806": 0.30,      # Primary ML model (98.25% accuracy)
    "clip_universal": 0.30, # CLIP-based detection
    "frequency": 0.10,
    "watermark": 0.10,
}
```

### Score Fusion Formula

```
Final Score = Σ (weight_i × detector_score_i) / Σ weights

Confidence = base_score × agreement_factor × detector_count_bonus
```

### Agreement Analysis

| Agreement Level | Confidence Modifier |
|-----------------|---------------------|
| All detectors agree | 1.2× |
| Majority agree (>75%) | 1.0× |
| Split decision | 0.8× |
| Strong disagreement | 0.6× |

---

## 8. Confidence Thresholds

| AI Probability | Verdict | Recommendation |
|----------------|---------|----------------|
| 0-25% | Likely Authentic | Low concern |
| 25-45% | Uncertain | Manual review recommended |
| 45-65% | Possibly AI | Further investigation needed |
| 65-85% | Likely AI | High confidence synthetic |
| 85-100% | AI Generated | Very high confidence |

---

## 9. ML Models Setup

### Models Currently Used

1. **dima806/deepfake_vs_real_image_detection** (343 MB) — ViT-B/16, 98.25% accuracy, primary detector
2. **openai/clip-vit-large-patch14** (1.71 GB) — CLIP ViT-L/14, used in UniversalFakeDetect
3. **Stable Signature Decoder** (~15 MB) — Meta's 48-bit neural watermark decoder

### Quick Setup

```powershell
# Activate ML environment
.venv310\Scripts\Activate.ps1

# Start server (auto-loads models on GPU)
python backend/app.py

# Verify GPU
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

### Performance

- **GPU inference**: ~50-100ms per image (RTX 4060)
- **CPU fallback**: ~500-1000ms per image
- **GPU VRAM**: 2-3GB

---

## 10. Training Datasets

| Dataset | Size | Generators | Focus |
|---------|------|------------|-------|
| **GenImage** | 1.3M+ pairs | 8 models (MJ, SD, DALL-E, BigGAN) | Cross-model generalization |
| **CIFAKE** | 120K images | Stable Diffusion v1.4 | CIFAR-10 style classification |
| **MS COCOAI** (2025) | Large | DALL-E 3, MJ V6 | Latest generators |
| **FaceForensics++** | 1M+ frames | Multiple face swap methods | Deepfake detection |
| **DiffusionDB** | 14M+ | Stable Diffusion prompts+outputs | Prompt-aware detection |

---

## 11. Known Limitations

1. **AI-enhanced photos**: Humans using AI for minor edits
2. **Photobashed compositions**: Human artists combining AI and real elements
3. **Heavy JPEG compression**: Destroys forensic signals
4. **Social media processing**: Platforms strip metadata and recompress
5. **New generators**: Each new model (Flux, MJ v7) may evade existing detectors

---

## 12. API Response Structure

```json
{
  "success": true,
  "ai_probability": 75.5,
  "verdict": "LIKELY_AI",
  "dimensions": { "width": 1024, "height": 768 },
  "file_size": 204800,
  "color_space": "RGB",
  "bit_depth": 24,
  "noise_analysis": { "consistency": 60, "low_freq": 94, "mid_freq": 4, "high_freq": 1 },
  "noise_map": "data:image/png;base64,...",
  "analysis_scores": {
    "color_uniformity": 65.0,
    "noise_consistency": 60,
    "edge_naturalness": 45.0,
    "texture_quality": 55.0,
    "frequency_anomaly": 70.0
  },
  "ml_heatmap": "data:image/png;base64,...",
  "metadata": {},
  "ela": {},
  "watermark": {},
  "content_credentials": {},
  "ai_analysis": {}
}
```

---

## 13. Future Improvements Roadmap

### Short-Term (Q1 2026)
1. Add latest generators (SDXL Turbo, DALL-E 4, Midjourney V7) to training
2. Implement DiffCoR for diffusion-specific detection
3. C2PA verification integration with CAI Verify

### Medium-Term (Q2-Q3 2026)
1. Video frame analysis for AI-generated video detection
2. Source attribution (identify which specific model generated an image)
3. Adversarial robustness training

### Long-Term (Q4 2026+)
1. Real-time detection API for streaming content
2. 3D render detection for AI-generated 3D scenes
3. Multimodal provenance (image + audio + video chain)

---

## References

1. GenImage Benchmark: [OpenReview](https://openreview.net/forum?id=genimage)
2. NYUAD ViT Detector: [Hugging Face](https://huggingface.co/NYUAD-ComNets)
3. C2PA Standard: [c2pa.org](https://c2pa.org)
4. CLIP Paper: "Learning Transferable Visual Models" (OpenAI, 2021)
5. Stable Signature: Meta Research (2023)
6. UniversalFakeDetect: CVPR 2023
7. NPR (Neighboring Pixel Relationships): CVPR 2024
8. DIRE (Diffusion Reconstruction Error): CVPR 2024
9. FaceForensics++: [github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics)
