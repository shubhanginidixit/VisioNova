# Audio Detection

## Executive Summary

VisioNova employs a state-of-the-art deep learning approach to detect synthetic audio. By leveraging the **Wav2Vec2** architecture finetuned on large-scale deepfake datasets (ASVspoof, WaveFake), we can effectively distinguish between bonafide human speech and AI-generated audio (TTS, Voice Conversion).

---

## 1. Scope: What We Are Detecting

We are specifically targeting **AI-Synthesized Speech**, often referred to as "Audio Deepfakes". This includes:

| Type | Examples | Detection Focus |
|------|----------|-----------------|
| **Text-to-Speech (TTS)** | ElevenLabs, XTTS, Bark, Tortoise TTS | Synthetic vocal patterns, missing biological signals |
| **Voice Conversion (VC)** | So-VITS-SVC, RVC, VALL-E | Unnatural formant transitions, spectral anomalies |
| **Voice Cloning** | ElevenLabs cloning, Resemble.AI | Pitch stability, breath pattern absence |

### Out of Scope
- Music generation detection (Suno, Udio)
- Audio editing/splicing detection
- Environmental sound synthesis

---

## 2. Detection Architecture

### Wav2Vec2-Based Pipeline

```
Audio Input (.wav/.mp3/.flac)
    │
    ▼
┌─────────────────────────────┐
│  Preprocessing              │
│  • Resample to 16kHz        │
│  • Normalize amplitude      │
│  • Trim silence             │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Wav2Vec2 Feature Extractor │
│  • 768-dim hidden states    │
│  • Self-supervised features │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Classification Head        │
│  • Linear layers            │
│  • Softmax: bonafide/spoof  │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Spectral Analysis          │
│  • Spectrogram generation   │
│  • Frequency anomalies      │
│  • Biological signal check  │
└────────────┬────────────────┘
             │
             ▼
       Final Verdict
```

### Why Wav2Vec2?

| Feature | Benefit |
|---------|---------|
| Self-supervised pre-training | Learns speech representations without labels |
| 768-dimensional features | Rich representation captures subtle artifacts |
| Temporal modeling | Detects time-domain inconsistencies |
| Transfer learning | Fine-tune on small deepfake datasets |

---

## 3. Key Detection Signals

### 3.1 Spectral Anomalies

| Signal | Human Audio | AI Audio |
|--------|-------------|----------|
| Frequency range | Full spectrum (20Hz-20kHz) | Often truncated at 8-12kHz |
| Harmonics | Natural harmonic series | Missing or artificial harmonics |
| Background noise | Varied, natural | Uniform or absent |
| Formant transitions | Smooth, continuous | Abrupt or mechanical |

### 3.2 Biological Signals

Human speech contains involuntary biological markers that AI struggles to replicate:
- **Breath sounds**: Natural pauses between phrases
- **Micro-tremors**: Slight vocal cord vibrations
- **Pitch variance**: Natural fluctuation (±2-5 Hz)
- **Glottal pulses**: Irregular spacing in human voice

### 3.3 Temporal Patterns

| Pattern | Human | AI |
|---------|-------|----|
| Pause distribution | Varied (50-500ms) | Regular intervals |
| Speaking rate | Dynamic | Often monotonic |
| Emphasis patterns | Context-dependent | Formulaic |

---

## 4. Models and Performance

### Primary: Weighted Ensemble (Wav2Vec2 + WavLM)

VisioNova uses a 3-model weighted ensemble combining state-of-the-art 2025-2026 detectors designed to catch cutting-edge synthetic voices across multiple languages and architectures:

| Model | Architecture | Role & Strength | Weight |
|-------|-------------|-----------------|--------|
| **NII Yamagishi Anti-Deepfake** (`nii-yamagishilab/wav2vec-large-anti-deepfake`) | Wav2Vec2 Large | Zero-shot detection across unseen datasets, robust payload handling. | 40% |
| **WavLM Deepfake V2** (`DavidCombei/wavLM-base-Deepfake_V2`) | WavLM Base | Masked Speech Prediction base; excels against TTS artifacts in noisy environments. | 40% |
| **Deepfake Pattern Fallback** (`mo-thecreator/Deepfake-audio-detection`) | Wav2Vec2 | Standard baseline checking to catch simpler VC anomalies. | 20% |

> **Note:** Models are dynamically loaded (lazy-loading) during inference to manage GPU VRAM. If a model fails to load, the ensemble automatically adjusts remaining weights to ensure completion.

---

## 5. Training Datasets

| Dataset | Size | Focus |
|---------|------|-------|
| **ASVspoof 2021** | 200K+ clips | TTS and VC attacks |
| **ASVspoof 5** | Latest challenge | Modern generators |
| **WaveFake** | 100K+ clips | Diverse TTS systems |
| **In-the-Wild** | 20K+ clips | Real-world deepfakes |

---

## 6. API Endpoint

### `POST /api/detect-audio`

**Input:** `multipart/form-data` with `audio` field  
**Supported formats:** WAV, MP3, FLAC, OGG, M4A  
**Max file size:** 25MB

**Response:**
```json
{
  "success": true,
  "authenticity_score": 72.5,
  "verdict": "LIKELY_HUMAN",
  "spectrogram": "data:image/png;base64,...",
  "anomalies": [
    "Slight frequency truncation above 14kHz",
    "Normal pitch variance detected"
  ],
  "details": {
    "duration": 5.2,
    "sample_rate": 16000,
    "model_confidence": 0.725
  }
}
```

---

## 7. Limitations

1. **Short clips (< 2 seconds)**: Insufficient signal for reliable detection
2. **Music/singing**: Models trained on speech, not musical audio
3. **Poor recording quality**: Low bitrate obscures forensic signals
4. **Real-time cloning**: Newest models (VALL-E 2) approaching human-level quality
5. **Non-speech audio**: Environmental sounds, sound effects not covered

---

## 8. Future Improvements

1. **AudioSeal integration**: Meta's sample-level watermark detection
2. **Ensemble approach**: Combine Wav2Vec2 + WavLM + HuBERT for 2-3% EER
3. **Speaker verification**: Cross-reference with known voice samples
4. **Multilingual support**: Extend beyond English audio
5. **Real-time streaming**: Socket-based detection for live audio feeds

---

## References

1. ASVspoof Challenge: [asvspoof.org](https://www.asvspoof.org)
2. Wav2Vec 2.0: "Self-Supervised Speech Representations" (Facebook AI, 2020)
3. WavLM: "Large-Scale Self-Supervised Pre-Training" (Microsoft, 2022)
4. HuBERT: "Self-Supervised Speech Representation via Masked Prediction" (2021)
5. AudioSeal: Meta's Audio Watermarking (2024)
