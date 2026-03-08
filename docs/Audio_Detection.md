# Audio Detection

## Executive Summary

VisioNova employs the state-of-the-art **NII AntiDeepfake** model (ASRU 2025) for detecting AI-generated speech. These are SSL foundation models post-trained on **74,000+ hours** of audio (56k real + 18k fake) across **100+ languages**, achieving an Equal Error Rate (EER) as low as **1.23%** on the In-the-Wild benchmark — the best publicly available deepfake audio detector as of 2025.

Paper: [Post-training for Deepfake Speech Detection](https://arxiv.org/abs/2506.21090) (Ge et al., ASRU 2025)

---

## 1. Scope: What We Are Detecting

We are specifically targeting **AI-Synthesized Speech**, often referred to as "Audio Deepfakes". This includes:

| Type | Examples | Detection Focus |
|------|----------|-----------------|
| **Text-to-Speech (TTS)** | ElevenLabs, XTTS, Bark, Tortoise TTS, F5-TTS | Synthetic vocal patterns, missing biological signals |
| **Voice Conversion (VC)** | So-VITS-SVC, RVC, VALL-E, GPT-SoVITS | Unnatural formant transitions, spectral anomalies |
| **Voice Cloning** | ElevenLabs cloning, Resemble.AI, OpenVoice | Pitch stability, breath pattern absence |
| **Codec-based synthesis** | Encodec, SoundStream, neural codecs | Codec artifact detection |

### Out of Scope
- Music generation detection (Suno, Udio)
- Audio editing/splicing detection
- Environmental sound synthesis

---

## 2. Detection Architecture

### NII AntiDeepfake Pipeline

```
Audio Input (.wav/.mp3/.flac/.ogg/.m4a/.webm/.aac/.wma)
    │
    ▼
┌─────────────────────────────────────┐
│  Preprocessing (soundfile/torchaudio│
│  • Load via soundfile (wav/flac)    │
│    or torchaudio (mp3/aac/etc.)     │
│  • Resample to 16kHz mono           │
│  • Layer normalization              │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  Segmentation (for long audio)      │
│  • Short (≤30s): single pass        │
│  • Long (>30s): 30s overlapping     │
│    chunks with 5s overlap           │
│  • Tail segment kept if ≥3s         │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  SSL Frontend (Pure PyTorch Wav2Vec2│
│  • XLS-R-1B: 48 layers, dim=1280   │
│    or                               │
│  • Wav2Vec2-Large: 24 layers, 1024 │
│  • Self-supervised features         │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  Backend Classifier                 │
│  • AdaptiveAvgPool1d                │
│  • Linear(dim, 2)                   │
│  • Softmax: [fake_score, real_score]│
└────────────┬────────────────────────┘
             │
             ▼
       Per-Segment Verdicts
   (fake_prob vs real_prob per chunk)
             │
             ▼
┌─────────────────────────────────────┐
│  Aggregation                        │
│  • Duration-weighted average of     │
│    per-segment fake probabilities   │
│  • Overall verdict from aggregate   │
└────────────┬────────────────────────┘
             │
             ▼
       Final Verdict + Segment Timeline
   (aggregate + per-segment breakdown)
```

### Why AntiDeepfake / SSL Post-training?

| Feature | Benefit |
|---------|---------|
| Post-training on 74k hours | Bridges gap between general pre-training and domain fine-tuning |
| 100+ language coverage | Robust across multilingual deepfakes (not just English) |
| Zero-shot generalization | Detects unseen TTS/VC systems without fine-tuning |
| Self-supervised features | Captures subtle artifacts invisible to hand-crafted features |
| Multiple model sizes | Auto-fallback from 1B to 315M params based on available VRAM |

---

## 3. Key Detection Signals

### 3.1 Spectral Anomalies

| Signal | Human Audio | AI Audio |
|--------|-------------|----------|
| Frequency range | Full spectrum (20Hz–20kHz) | Often truncated at 8–12kHz |
| Harmonics | Natural harmonic series | Missing or artificial harmonics |
| Background noise | Varied, natural | Uniform or absent |
| Formant transitions | Smooth, continuous | Abrupt or mechanical |

### 3.2 Biological Signals

Human speech contains involuntary biological markers that AI struggles to replicate:
- **Breath sounds**: Natural pauses between phrases
- **Micro-tremors**: Slight vocal cord vibrations
- **Pitch variance**: Natural fluctuation (±2–5 Hz)
- **Glottal pulses**: Irregular spacing in human voice

### 3.3 Temporal Patterns

| Pattern | Human | AI |
|---------|-------|----|
| Pause distribution | Varied (50–500ms) | Regular intervals |
| Speaking rate | Dynamic | Often monotonic |
| Emphasis patterns | Context-dependent | Formulaic |

---

## 4. Models and Performance

### Primary: NII AntiDeepfake (ASRU 2025)

VisioNova loads models by priority, automatically falling back if GPU memory is insufficient:

| Model | Architecture | Params | VRAM | In-the-Wild EER | FakeOrReal EER |
|-------|-------------|--------|------|-----------------|----------------|
| **XLS-R-1B AntiDeepfake** (primary) | XLS-R 1B + FC | 1B | ~4GB | **1.35%** | 5.74% |
| **Wav2Vec2-Large AntiDeepfake** (fallback) | Wav2Vec2 Large + FC | 315M | ~1.2GB | **1.91%** | 0.67% |

#### Full Benchmark Results (XLS-R-1B, zero-shot)

| Dataset | ROC AUC | Accuracy | Precision | Recall | F1 | EER |
|---------|---------|----------|-----------|--------|-----|-----|
| ADD2023 | 0.987 | 0.951 | 0.979 | 0.954 | 0.966 | 5.39% |
| DeepVoice | 0.998 | 0.926 | 0.625 | 0.994 | 0.767 | 2.47% |
| FakeOrReal | 0.984 | 0.938 | 0.946 | 0.926 | 0.936 | 5.74% |
| In-the-Wild | 0.998 | 0.986 | 0.992 | 0.986 | 0.989 | **1.35%** |
| Deepfake-Eval-2024 | 0.810 | 0.751 | 0.758 | 0.910 | 0.827 | 26.76% |

> **Note:** Models are lazy-loaded on first request. If the primary model causes an OOM error, VisioNova automatically tries the fallback model.

### License

All AntiDeepfake model weights are released under **CC BY-NC-SA 4.0** (non-commercial research use) by NII Yamagishi Lab.

---

## 5. Training Data

AntiDeepfake models were post-trained on a massive multilingual dataset:

| Dataset | Languages | Real (hours) | Fake (hours) |
|---------|-----------|-------------|-------------|
| ASVspoof 2019-LA | en | 11.85 | 97.80 |
| ASVspoof 2021 (LA+DF) | en | 37.13 | 603.10 |
| ASVspoof5 | en | 413.49 | 1,808.48 |
| Codecfake | en, zh | 129.66 | 1,469.24 |
| CVoiceFake | en, fr, de, it, zh | 315.14 | 1,561.16 |
| SpoofCeleb | Multilingual | 173.00 | 1,916.20 |
| FLEURS/FLEURS-R | 102 languages | 1,388.97 | 1,238.83 |
| MLS | 8 languages | 50,558.11 | — |
| VoxCeleb2 + Vocoded | Multilingual | 1,179.62 | 4,721.46 |
| WaveFake | en, ja | — | 198.65 |
| **Total** | **100+ languages** | **56,370** | **18,280** |

---

## 6. API Endpoint

### `POST /api/detect-audio`

**Input:** `multipart/form-data` with `audio` field, or JSON with base64 `audio` field  
**Supported formats:** WAV, MP3, FLAC, OGG, M4A, WebM, AAC, WMA  
**Max file size:** 200 MB  
**Max duration:** Unlimited (segmented analysis for files >30 seconds)

**Response:**
```json
{
  "success": true,
  "prediction": "ai_generated",
  "verdict": "likely_ai",
  "fake_probability": 92.35,
  "real_probability": 7.65,
  "confidence": 92.35,
  "analysis_mode": "segmented",
  "total_duration_seconds": 125.4,
  "segments_analyzed": 5,
  "segments": [
    {
      "start_sec": 0.0,
      "end_sec": 30.0,
      "fake_probability": 89.12,
      "real_probability": 10.88,
      "verdict": "likely_ai"
    },
    {
      "start_sec": 25.0,
      "end_sec": 55.0,
      "fake_probability": 94.50,
      "real_probability": 5.50,
      "verdict": "likely_ai"
    }
  ],
  "ensemble_details": [
    {
      "name": "XLS-R-1B AntiDeepfake",
      "model_id": "nii-yamagishilab/xls-r-1b-anti-deepfake",
      "fake_probability": 92.35,
      "weight": 1.0
    }
  ],
  "artifacts_detected": [
    "High-frequency spectral anomalies detected",
    "Synthetic phase coherence observed",
    "Vocoder analysis signature present"
  ],
  "meta": {
    "duration_seconds": 125.4,
    "sample_rate": 16000,
    "segment_length_sec": 30,
    "segment_overlap_sec": 5
  }
}
```

### `GET /api/detect-audio/info`

Returns model metadata including active model, training data size, and device.

---

## 7. Dependencies

```
torchaudio>=2.0.0        # Audio I/O, resampling, preprocessing
torch>=2.0.0             # PyTorch inference
soundfile>=0.12.0        # Audio file format support
huggingface_hub>=0.20.0  # Model weight download from HuggingFace Hub
safetensors>=0.4.0       # Efficient weight loading
# wav2vec2_arch.py       # Standalone Wav2Vec2 implementation (no fairseq needed)
```

---

## 8. Limitations

1. **Short clips (< 2 seconds)**: Insufficient signal for reliable detection
2. **Music/singing**: Models trained on speech, not musical audio
3. **Poor recording quality**: Low bitrate obscures forensic signals
4. **Newest real-time cloning**: VALL-E 2, F5-TTS approaching human-level quality
5. **Non-speech audio**: Environmental sounds, sound effects not covered
6. **Non-commercial license**: CC BY-NC-SA 4.0 restricts commercial deployment
7. **Segment boundaries**: Overlapping segments may double-count boundary regions

---

## 9. Future Improvements

1. **XLS-R-2B upgrade**: Upgrade to the 2B param model (EER 1.23%) when GPU allows
2. **Fine-tuning on Deepfake-Eval-2024**: Can reduce EER from 26.76% to ~8.28% on hardest benchmarks
3. **AudioSeal integration**: Meta's sample-level watermark detection
4. **Speaker verification**: Cross-reference with known voice samples
5. **Real-time streaming**: Socket-based detection for live audio feeds
6. **Drift detection**: NII provides drift detection utilities for monitoring model degradation

---

## References

1. Ge, W., Wang, X., Liu, X., & Yamagishi, J. (2025). "Post-training for Deepfake Speech Detection." ASRU 2025. [arXiv:2506.21090](https://arxiv.org/abs/2506.21090)
2. AntiDeepfake Models: [HuggingFace Collection](https://huggingface.co/collections/nii-yamagishilab/antideepfake)
3. AntiDeepfake Code: [GitHub Repository](https://github.com/nii-yamagishilab/AntiDeepfake)
4. ASVspoof Challenge: [asvspoof.org](https://www.asvspoof.org)
5. Wav2Vec 2.0: "Self-Supervised Speech Representations" (Facebook AI, 2020)
6. XLS-R: "XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale" (2022)
