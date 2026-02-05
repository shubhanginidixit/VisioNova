# Image Detection Strategy

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

---

## 3. Statistical and Forensic Analysis

### 3.1 Error Level Analysis (ELA)

ELA detects image manipulation by analyzing JPEG compression artifacts.

#### How It Works
1. Re-compress the image at a known quality (e.g., 95%)
2. Calculate the difference between original and recompressed versions
3. Manipulated regions show different error levels than authentic areas

#### Strengths
- Low computational cost
- Interpretable results (visual heatmap)
- Effective for detecting splicing, cloning, and retouching

#### Limitations
- **Does not work on PNG** or other lossless formats
- **Less effective against AI-generated images** (no compression history)
- Uniform recompression can mask manipulation
- Subjective interpretation required

#### When to Use
| Scenario | ELA Effectiveness |
|----------|-------------------|
| JPEG photo manipulation | High |
| Spliced/composite images | High |
| AI-generated from scratch | Low |
| Re-saved multiple times | Medium |

---

### 3.2 Frequency Domain Analysis (FFT/DCT)

Analyzes the image in the frequency spectrum to detect generator-specific fingerprints.

#### GAN Fingerprints
GANs leave distinctive patterns due to transposed convolution operations:
- **Checkerboard artifacts**: Grid-like patterns in high frequencies
- **Periodic peaks**: Anomalous spikes in the Fourier transform
- **Spectral flatness**: Abnormal uniformity in frequency distribution

#### Diffusion Model Detection
Diffusion models have different signatures:
- **Smoother frequency profile**: Less obvious periodic patterns
- **Noise distribution**: Different from camera sensor noise
- **Snap-back analysis**: Real images degrade abruptly when noise is added; AI images degrade smoothly

#### Detection Metrics
- Spectral flatness measure
- Grid artifact detection score
- High-frequency energy ratio
- Radial frequency distribution

---

### 3.3 Noise and Texture Analysis

Camera sensors leave a unique "fingerprint" called Photo-Response Non-Uniformity (PRNU).

#### Camera Noise Fingerprint
- Each camera has a unique noise pattern from sensor manufacturing
- AI-generated images lack authentic PRNU patterns
- Can verify if an image came from a claimed camera

#### Texture Consistency
AI often fails at:
- Micro-texture patterns (skin pores, fabric weave)
- Natural randomness in organic materials
- Consistent noise distribution across the image

---

## 4. Deep Learning Models

### 4.1 Vision Transformer (ViT) Detectors

Vision Transformers process images as sequences of patches, enabling global pattern recognition.

#### NYUAD ViT Detector
| Property | Value |
|----------|-------|
| Model | `NYUAD-ComNets/NYUAD_AI-generated_images_detector` |
| Base Architecture | ViT-B/16 |
| Accuracy | 97.36% on benchmark datasets |
| Input Size | 224×224 |
| Labels | `artificial`, `real` |

#### Why ViT Over CNN?
- **Global attention**: Captures long-range dependencies across the entire image
- **Patch-based processing**: Naturally detects inconsistencies across regions
- **Better generalization**: Performs well on unseen generator types
- **Semantic understanding**: Detects lighting inconsistencies, impossible geometry

#### Fine-Tuning Configuration

```python
from transformers import ViTForImageClassification, ViTImageProcessor

# Load pre-trained model
model = ViTForImageClassification.from_pretrained(
    "NYUAD-ComNets/NYUAD_AI-generated_images_detector"
)

# For custom fine-tuning
training_args = {
    "learning_rate": 2e-5,
    "batch_size": 32,
    "epochs": 10,
    "image_size": 224,
    "warmup_ratio": 0.1,
}
```

---

### 4.2 CLIP-Based Detection

CLIP (Contrastive Language-Image Pre-training) provides powerful generalizable features.

#### UniversalFakeDetect Approach
Uses CLIP ViT-L/14 with a linear classifier:
1. Extract 768-dimensional feature vector from CLIP
2. Train linear classifier on real vs. AI features
3. Analyze feature statistics (kurtosis, variance patterns)

#### Why CLIP Works for Detection
- **Multimodal understanding**: Trained on 400M image-text pairs
- **Generalization**: Detects unseen forgery techniques
- **Feature space differences**: Real vs. AI images cluster differently

#### Heuristic Detection Without Classifier
Even without training, CLIP features show patterns:
- AI images have more uniform feature activations
- Lower variance in activation patterns
- Abnormal kurtosis values

---

### 4.3 Deepfake Face Detection

Specialized detectors for face manipulation.

#### Key Detection Signals
| Signal | Description |
|--------|-------------|
| Lip-sync inconsistency | Audio doesn't match mouth movements |
| Blending artifacts | Edge boundaries between face and background |
| Temporal flickering | Frame-to-frame inconsistencies in video |
| Eye reflections | Inconsistent reflections in pupils |
| Skin texture | Unnatural smoothness or pore patterns |

#### Models for Face Forensics
- **FaceForensics++** trained models
- **Xception** architecture (high accuracy on faces)
- **EfficientNet-B7** with attention

---

## 5. Watermark and Provenance Detection

### 5.1 C2PA Content Credentials

The Coalition for Content Provenance and Authenticity (C2PA) standard embeds verifiable metadata.

#### What C2PA Contains
- **Creator information**: Who created the content
- **Creation tool**: Software/model used (e.g., "DALL-E 3")
- **Edit history**: Cryptographically signed modification chain
- **AI disclosure**: Whether AI was involved in generation

#### Detection Method
1. Parse image metadata for C2PA manifest
2. Verify cryptographic signature chain
3. Check for "trainedAlgorithmicMedia" tag in IPTC

#### Adopters (2025)
- Adobe (Firefly, Photoshop)
- Google (Imagen, Photos)
- Microsoft (Bing Image Creator)
- OpenAI (DALL-E 3)

---

### 5.2 Invisible Watermarks

AI providers embed imperceptible watermarks that survive most transformations.

#### Detection Methods Implemented

| Method | Technology | Typical Use |
|--------|------------|-------------|
| **DWT-DCT** | Discrete Wavelet + Cosine Transform | Stable Diffusion, ComfyUI |
| **DWT-DCT-SVD** | Enhanced with Singular Value Decomposition | Higher robustness |
| **Stable Signature** | Meta's 48-bit neural watermark | Meta AI generators |
| **SynthID** | Google's spectral watermarking | Imagen, Gemini |
| **Tree-Ring** | Fourier domain patterns | Research methods |

#### Stable Signature Details
- Embeds 48-bit watermark using neural encoder
- False positive rate: ~1 in 10 billion
- Survives cropping, compression, color changes
- Requires decoder model for detection

#### Spectral Analysis
- Analyze frequency domain for embedded patterns
- Look for bimodal variance distributions
- Detect unusual energy concentrations

---

### 5.3 Metadata Analysis

#### EXIF Forensics
Check for contradictions:
- Camera model vs. image resolution mismatch
- GPS coordinates vs. timezone inconsistency
- Software tags indicating AI tools

#### AI Tool Indicators
| Metadata Field | AI Indicator |
|----------------|--------------|
| `Software` | "Midjourney", "DALL-E", "Stable Diffusion" |
| `DigitalSourceType` | "trainedAlgorithmicMedia" |
| `XMP Creator Tool` | AI generation software |
| PNG `tEXt` chunks | Generation parameters, seeds |

---

## 6. Training Datasets

### 6.1 Recommended Datasets

| Dataset | Size | Generators | Focus |
|---------|------|------------|-------|
| **GenImage** | 1.3M+ pairs | 8 models (MJ, SD, DALL-E, BigGAN) | Cross-model generalization |
| **CIFAKE** | 120K images | Stable Diffusion v1.4 | CIFAR-10 style classification |
| **MS COCOAI** (2025) | Large | DALL-E 3, MJ V6 | Latest generators |
| **FaceForensics++** | 1M+ frames | Multiple face swap methods | Deepfake detection |
| **DiffusionDB** | 14M+ | Stable Diffusion prompts+outputs | Prompt-aware detection |

### 6.2 GenImage Benchmark Details

GenImage is the gold standard for evaluating AI image detectors:
- **8 generators**: Midjourney V5, SD 1.4/1.5, DALL-E, BigGAN, ADM, GLIDE, VQDM, Wukong
- **1000 ImageNet-aligned classes**
- **Multiple compression levels**
- **Cross-generator evaluation protocols**

### 6.3 Building Custom Datasets

```python
from datasets import Dataset
from PIL import Image

def create_detection_dataset():
    data = {
        "image": [],
        "label": [],  # 0 = real, 1 = AI
        "generator": []
    }
    
    # Real images from photography datasets
    for img_path in real_photo_paths:
        data["image"].append(Image.open(img_path))
        data["label"].append(0)
        data["generator"].append("real")
    
    # AI images from multiple generators
    for generator in ["midjourney", "dalle3", "sdxl", "imagen"]:
        for img_path in get_ai_images(generator):
            data["image"].append(Image.open(img_path))
            data["label"].append(1)
            data["generator"].append(generator)
    
    return Dataset.from_dict(data)
```

---

## 7. Model Training Guide

### 7.1 Fine-Tuning ViT for Detection

```python
from transformers import ViTForImageClassification, Trainer, TrainingArguments
from torchvision import transforms

# Data augmentation (preserving forensic artifacts)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    # Avoid heavy compression/color jittering that destroys forensic signals
])

# Model setup
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=2,
    id2label={0: "real", 1: "ai_generated"},
    label2id={"real": 0, "ai_generated": 1}
)

training_args = TrainingArguments(
    output_dir="./vit-ai-detector",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    num_train_epochs=10,
    warmup_ratio=0.1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
```

### 7.2 Cross-Generator Generalization

Key training strategies:
1. **Include multiple generators** in training data
2. **Leave-one-out validation**: Train on N-1 generators, test on held-out
3. **Curriculum learning**: Start with obvious fakes, progress to subtle ones
4. **Augmentation limits**: Avoid destroying forensic artifacts

---

## 8. Ensemble Scoring

### Current VisioNova Implementation

```python
DEFAULT_WEIGHTS = {
    "statistical": 0.20,    # ELA, frequency, noise analysis
    "nyuad_vit": 0.35,      # Vision Transformer (primary)
    "clip_universal": 0.25, # CLIP-based detection
    "watermark": 0.20,      # Invisible watermark detection
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

## 9. Confidence Thresholds

| AI Probability | Verdict | Recommendation |
|----------------|---------|----------------|
| 0-25% | Likely Authentic | Low concern |
| 25-45% | Uncertain | Manual review recommended |
| 45-65% | Possibly AI | Further investigation needed |
| 65-85% | Likely AI | High confidence synthetic |
| 85-100% | AI Generated | Very high confidence |

---

## 10. Known Limitations

### Detection Challenges

1. **AI-enhanced photos**: Humans using AI for minor edits
2. **Photobashed compositions**: Human artists combining AI and real elements
3. **Upscaling artifacts**: AI upscalers can trigger false positives
4. **Heavy JPEG compression**: Destroys forensic signals
5. **Social media processing**: Platforms strip metadata and recompress

### False Positive Sources

| Source | Mitigation |
|--------|------------|
| Professional photography | Check for C2PA from camera |
| Digital art (non-AI) | Pattern analysis, metadata |
| Stock photos | Reverse image search |
| Heavily filtered images | Lower confidence scores |

---

## 11. Future Improvements Roadmap

### Short-Term (Q1 2026)
1. **Add latest generators** (SDXL Turbo, DALL-E 4, Midjourney V7) to training
2. **Implement DiffCoR** for diffusion-specific detection
3. **C2PA verification** integration with CAI Verify

### Medium-Term (Q2-Q3 2026)
1. **Video frame analysis** for AI-generated video detection
2. **Source attribution** (identify which specific model generated an image)
3. **Adversarial robustness** training against detection evasion

### Long-Term (Q4 2026+)
1. **Real-time detection API** for streaming content
2. **3D render detection** for AI-generated 3D scenes
3. **Multimodal provenance** (image + audio + video chain)

---

## 12. Best Practices

### For Developers
- Always use ensemble methods (never rely on a single detector)
- Preserve original image quality when possible
- Log detector disagreements for analysis
- Regularly update models with new generator outputs

### For Users
- Check multiple detection tools for important decisions
- Verify metadata and provenance when available
- Consider context and source credibility
- Remember: No detector is 100% accurate

---

## References

1. GenImage Benchmark: [OpenReview](https://openreview.net/forum?id=genimage)
2. NYUAD ViT Detector: [Hugging Face](https://huggingface.co/NYUAD-ComNets)
3. C2PA Standard: [c2pa.org](https://c2pa.org)
4. Content Authenticity Initiative: [contentauthenticity.org](https://contentauthenticity.org)
5. CLIP Paper: "Learning Transferable Visual Models" (OpenAI, 2021)
6. Stable Signature: Meta Research (2023)
7. DiffCoR: IEEE 2024
8. FaceForensics++: [faceforensics.org](https://github.com/ondyari/FaceForensics)
