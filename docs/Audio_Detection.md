# Audio Detection

## Executive Summary

VisioNova employs a **5-model weighted ensemble** of state-of-the-art self-supervised learning (SSL) architectures to detect synthetic audio. By combining XLS-R, WavLM, and Wav2Vec2 models fine-tuned on deepfake datasets (ASVspoof, WaveFake), we achieve robust detection across diverse AI audio generation methods including TTS, Voice Conversion, and Voice Cloning.

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

### 5-Model Ensemble Pipeline

```
Audio Input (.wav/.mp3/.flac)
    │
    ▼
┌─────────────────────────────┐
│  Preprocessing              │
│  • Resample to 16kHz        │
│  • Convert to mono          │
│  • Normalize amplitude      │
│  • Clip to 60s max          │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│  5-Model Ensemble (parallel inference)                  │
│                                                         │
│  ┌──────────────────┐  ┌──────────────────┐             │
│  │ XLS-R 300M       │  │ WavLM Specialist │             │
│  │ Weight: 0.30     │  │ Weight: 0.20     │             │
│  │ Cross-lingual    │  │ Denoising focus  │             │
│  └────────┬─────────┘  └────────┬─────────┘             │
│           │                     │                       │
│  ┌──────────────────┐  ┌──────────────────┐             │
│  │ Wav2Vec2 Forensic│  │ Community V2     │             │
│  │ Weight: 0.20     │  │ Weight: 0.15     │             │
│  │ bonafide/spoof   │  │ 26+ HF Spaces    │             │
│  └────────┬─────────┘  └────────┬─────────┘             │
│           │                     │                       │
│  ┌──────────────────┐           │                       │
│  │ Diversity Det.   │           │                       │
│  │ Weight: 0.15     │           │                       │
│  │ Different split  │           │                       │
│  └────────┬─────────┘           │                       │
│           │                     │                       │
└───────────┴────────┬────────────┘                       │
                     │                                     
                     ▼
┌─────────────────────────────┐
│  Weighted Vote + Calibration│
│  • Explicit label resolution│
│  • Confidence calibration   │
│  • Agreement check          │
└────────────┬────────────────┘
             │
             ▼
       Final Verdict
```

### Why a 5-Model Ensemble?

| Feature | Benefit |
|---------|---------|
| Architectural diversity | XLS-R + WavLM + Wav2Vec2 reduces correlated blind spots |
| Weighted voting | Larger, more reliable models get higher influence |
| Explicit label resolution | Each model's fake-class is resolved deterministically, eliminating silent misclassification |
| Confidence calibration | When models disagree, the score is pulled toward 50% (uncertainty) instead of a confident wrong answer |
| Cross-lingual features | XLS-R-300M trained on 128 languages catches artifacts across accents |

---

## 3. Models and Performance

### Ensemble Members

| # | Model | Architecture | Params | Weight | Strength |
|---|-------|-------------|--------|--------|----------|
| 1 | `Gustking/wav2vec2-large-xlsr-deepfake-audio-classification` | XLS-R 300M | 300M | 30% | Cross-lingual, EER 4.01%, F1 0.95 on ASVspoof2019 |
| 2 | `DavidCombei/wavLM-base-Deepfake_V2` | WavLM-base | 94M | 20% | Denoising pre-training, 99.62% eval accuracy |
| 3 | `Vansh180/deepfake-audio-wav2vec2` | Wav2Vec2-base | 95M | 20% | Clean bonafide/spoof labels |
| 4 | `MelodyMachine/Deepfake-audio-detection-V2` | Wav2Vec2-base | 95M | 15% | 26+ HF Spaces, 99.73% eval accuracy |
| 5 | `mo-thecreator/Deepfake-audio-detection` | Wav2Vec2-base | 95M | 15% | Different training split → diversity |

### Aggregate Performance

| Metric | Value |
|--------|-------|
| Total Parameters | ~679M across 5 models |
| Disk Space Required | ~2.8 GB |
| Memory (inference) | ~3-4 GB RAM |
| Inference Time | ~5-15s per 10s clip |
| Max Audio Length | 60 seconds |

---

## 4. Key Detection Signals

### 4.1 Spectral Anomalies

| Signal | Human Audio | AI Audio |
|--------|-------------|----------|
| Frequency range | Full spectrum (20Hz-20kHz) | Often truncated at 8-12kHz |
| Harmonics | Natural harmonic series | Missing or artificial harmonics |
| Background noise | Varied, natural | Uniform or absent |
| Formant transitions | Smooth, continuous | Abrupt or mechanical |

### 4.2 Biological Signals

Human speech contains involuntary biological markers that AI struggles to replicate:
- **Breath sounds**: Natural pauses between phrases
- **Micro-tremors**: Slight vocal cord vibrations
- **Pitch variance**: Natural fluctuation (±2-5 Hz)
- **Glottal pulses**: Irregular spacing in human voice

### 4.3 Temporal Patterns

| Pattern | Human | AI |
|---------|-------|----|
| Pause distribution | Varied (50-500ms) | Regular intervals |
| Speaking rate | Dynamic | Often monotonic |
| Emphasis patterns | Context-dependent | Formulaic |

---

## 5. Analysis Modes

### Single-Pass Analysis
Used for short audio clips (<12 seconds). The entire clip is analyzed as one unit.

### Segmented Analysis
Used for longer audio (>12 seconds). The audio is split into overlapping 10-second windows with 2-second overlap. Each segment receives its own verdict, and the global score is a weighted average of all segments.

This catches "partial deepfakes" where only part of the audio was AI-generated.

---

## 6. API Endpoint

### `POST /api/detect-audio`

**Input:** `multipart/form-data` with `audio` field OR JSON with base64 `audio` field  
**Supported formats:** WAV, MP3, FLAC, OGG, M4A, WEBM, AAC, WMA  
**Max file size:** 25MB

**Response:**
```json
{
  "success": true,
  "prediction": "ai_generated",
  "verdict": "likely_ai",
  "fake_probability": 87.5,
  "real_probability": 12.5,
  "confidence": 87.5,
  "analysis_mode": "segmented",
  "total_duration_seconds": 25.3,
  "segments_analyzed": 4,
  "ensemble_details": [
    {
      "name": "XLS-R 300M Expert",
      "model_id": "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification",
      "type": "wav2vec2-xlsr",
      "fake_probability": 91.2,
      "real_probability": 8.8,
      "weight": 0.30,
      "verdict": "likely_ai"
    }
  ],
  "segments": [
    {
      "start_sec": 0.0,
      "end_sec": 10.0,
      "fake_probability": 85.3,
      "real_probability": 14.7,
      "verdict": "likely_ai"
    }
  ],
  "artifacts_detected": [
    "High-frequency spectral anomalies detected",
    "Synthetic phase coherence observed across frequency bands"
  ],
  "meta": {
    "duration_seconds": 25.3,
    "sample_rate": 16000,
    "file_name": "audio.wav",
    "ensemble_size": 5,
    "segment_length_sec": 10.0,
    "segment_overlap_sec": 2.0
  }
}
```

### `GET /api/detect-audio/info`

Returns model availability and ensemble metadata.

---

## 7. Limitations

1. **Short clips (<2 seconds)**: Insufficient signal for reliable detection
2. **Music/singing**: Models trained on speech, not musical audio
3. **Poor recording quality**: Low bitrate obscures forensic signals
4. **Real-time cloning**: Newest models (VALL-E 2) approaching human-level quality
5. **Non-speech audio**: Environmental sounds, sound effects not covered
6. **Memory usage**: 5-model ensemble requires ~3-4 GB RAM

---

## 8. Future Improvements

1. **AudioSeal integration**: Meta's sample-level watermark detection
2. **Larger backbone**: Upgrade anchor model to XLS-R-1B or XLS-R-2B
3. **AASIST backend**: Add spectro-temporal graph attention classifier
4. **Speaker verification**: Cross-reference with known voice samples
5. **Multilingual validation**: Benchmark on non-English test sets
6. **Cascade mode**: Run 2 models first, escalate to 5 when uncertain

---

## References

1. ASVspoof Challenge: [asvspoof.org](https://www.asvspoof.org)
2. Wav2Vec 2.0: "Self-Supervised Speech Representations" (Facebook AI, 2020)
3. XLS-R: "Self-supervised Cross-lingual Speech Representations" (Facebook AI, 2021)
4. WavLM: "Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing" (Microsoft, 2022)
5. HuBERT: "Self-Supervised Speech Representation via Masked Prediction" (2021)
6. AudioSeal: Meta's Audio Watermarking (2024)
7. ASVspoof5: "Spoofing-Aware Speaker Verification Challenge" (2024)
