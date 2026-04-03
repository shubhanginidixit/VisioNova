# Audio Detection

## Executive Summary

VisioNova employs an **Omni-Audio Router Pipeline** that intelligently classifies the audio domain before applying specialized detection models. By differentiating between Speech and Music/Ambient audio, we eliminate hallucinated false positives. For speech, we utilize a **4-model Vanguard ensemble** combining XLS-R, WavLM, and Wav2Vec2 architectures. For music, we apply heuristic spectral analysis.

---

## 1. Scope: What We Are Detecting

We detect both **AI-Synthesized Speech** and **AI-Generated Music**.

| Type | Examples | Detection Focus |
|------|----------|-----------------| 
| **Text-to-Speech (TTS)** | ElevenLabs, XTTS, Bark, Tortoise | Synthetic vocal patterns, missing biological signals |
| **Voice Cloning / VC** | So-VITS-SVC, RVC, VALL-E | Unnatural formant transitions, spectral anomalies |
| **AI Music Generation**| Suno, Udio | High-frequency diffusion roll-offs, unnatural spectral flatness |

### Out of Scope
- Audio editing/splicing detection (traditional tampering)
- Environmental sound synthesis

---

## 2. Detection Architecture

### Omni-Audio Router Pipeline

```text
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
┌─────────────────────────────┐
│  Audio Router (Silero VAD)  │
│  Detects human vocal presence│
└────────────┬────────────────┘
             │
     Speech > 5%?
       ┌─────┴─────┐
    Yes│           │No (Music)
       ▼           ▼
┌──────────────┐ ┌──────────────┐
│ Vanguard     │ │ Music        │
│ Speech       │ │ Heuristics   │
│ Ensemble     │ │ Detector     │
└───────┬──────┘ └───────┬──────┘
        │                │
        ▼                ▼
     Final Verdict & Artifacts
```

### Why This Architecture?

| Feature | Benefit |
|---------|---------|
| **Domain Routing** | Prevents speech models from evaluating polyphonic music and hallucinating "AI" verdicts. |
| **Vanguard Models** | 4-Model setup optimized specifically against top-tier modern TTS (e.g., ElevenLabs). |
| **Spectral Heuristics**| Catches diffusion-based music generation markers (e.g., 14kHz cutoffs). |
| **Confidence Calibration**| Pulls speech scores toward 50% if the ensemble members strongly disagree. |

---

## 3. Models and Performance

### Vanguard Speech Ensemble

| # | Model | Architecture | Weight | Strength |
|---|-------|-------------|--------|----------|
| 1 | `Gustking/wav2vec2-large...` | XLS-R 300M | 35% | Cross-lingual, ASVspoof-validated |
| 2 | `DavidCombei/wavLM-base...` | WavLM-base | 25% | Denoising pre-training |
| 3 | `Vansh180/deepfake...` | Wav2Vec2-base | 20% | Explicit bonafide/spoof labels |
| 4 | `Mihaiii/wav2vec2-base...` | Wav2Vec2-base | 20% | Fine-tuned for modern AI TTS |

### Music Heuristics (For Music/Ambient)
- **Feature Analytics**: Evaluates `spectral_flatness` and `spectral_rolloff` to identify the "digital haze" typical of generative AI music platforms.

### Aggregate Performance

| Metric | Value |
|--------|-------|
| Target Sample Rate | 16000 Hz |
| Inference Time | ~5-15s per 10s segment |
| Max Audio Length | 60 seconds |

---

## 4. Key Detection Signals

### 4.1 Biological vs. Synthesized Speech

Human speech contains involuntary biological markers that AI struggles to replicate:
- **Breath sounds**: Natural pauses between phrases.
- **Glottal pulses**: Irregular spacing in the human voice.
- **Pitch variance**: Natural fluctuation compared to monotonous TTS.

### 4.2 AI Music Signatures

- **High-frequency Rolloff**: AI music models generally struggle to produce true lossless high-fidelity > 14kHz.
- **Spectral Flatness**: Artificial diffusion processes create an unnatural "synthesized haze".

---

## 5. Analysis Modes

### Single-Pass Analysis
Used for short audio clips (<12 seconds) or Music. The entire clip is analyzed as one unit.

### Segmented Analysis
Used for longer AI speech (>12 seconds). The audio is split into overlapping 10-second windows with 2-second overlap. Each segment receives its own verdict, and the global score is a weighted average of all segments.

---

## 6. API Endpoint

### `POST /api/detect-audio`

**Input:** `multipart/form-data` with `audio` field OR JSON with base64 `audio` field  
**Supported formats:** WAV, MP3, FLAC, OGG, M4A, WEBM, AAC, WMA  

**Response:**
```json
{
  "success": true,
  "prediction": "ai_generated",
  "verdict": "likely_ai",
  "fake_probability": 87.5,
  "real_probability": 12.5,
  "confidence": 87.5,
  "audio_domain": "Speech",
  "analysis_mode": "segmented",
  "total_duration_seconds": 25.3,
  "segments_analyzed": 3,
  "ensemble_details": [
    {
      "name": "XLS-R 300M Anchor",
      "model_id": "Gustking/...",
      "type": "wav2vec2-xlsr",
      "fake_probability": 91.2,
      "real_probability": 8.8,
      "weight": 0.35,
      "verdict": "likely_ai"
    }
  ],
  "artifacts_detected": [
    "High-frequency spectral anomalies detected",
    "Vocoder signature detected — strong indicator of AI synthesis"
  ],
  "meta": {
    "duration_seconds": 25.3,
    "sample_rate": 16000,
    "domain": "Speech"
  }
}
```

### `GET /api/detect-audio/info`

Returns model availability and ensemble metadata, including the status of the Audio Router.

---

## 7. Future Improvements

1. **AudioSeal integration**: Meta's sample-level watermark detection.
2. **Larger backbone**: Upgrade anchor model to XLS-R-1B.
3. **Advanced AI Music Models**: Replace current music heuristics with a zero-shot binary classifier once openly available.
4. **Speaker verification**: Cross-reference with known voice samples.

---

## References

1. ASVspoof Challenge: [asvspoof.org](https://www.asvspoof.org)
2. Wav2Vec 2.0: "Self-Supervised Speech Representations" (Facebook AI, 2020)
3. Silero VAD: Voice Activity Detection (2021)
