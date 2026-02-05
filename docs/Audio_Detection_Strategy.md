# Audio Detection Strategy

## Executive Summary

This document outlines VisioNova's comprehensive approach to detecting AI-generated and manipulated audio, including synthetic speech (TTS), voice cloning, and deepfake voices. As voice AI evolves from robotic-sounding outputs to emotionally nuanced, accent-aware clones (ElevenLabs, XTTS, Tortoise TTS), our detection pipeline combines spectral analysis with state-of-the-art self-supervised models to maintain high accuracy against increasingly realistic synthetic voices.

---

## 1. The Challenge: Why Audio Detection is Getting Harder

### The Evolution of Voice Synthesis

| Era | Technology | Audio Quality | Detection Difficulty |
|-----|------------|---------------|---------------------|
| 2015-2019 | Parametric TTS (Festival, eSpeak) | Robotic, obvious artifacts | Low |
| 2019-2022 | Neural TTS (Tacotron, WaveNet) | Natural but some tells | Medium |
| 2022-2024 | Diffusion TTS (ElevenLabs, XTTS) | Near-human, emotional | High |
| 2025+ | Real-time cloning (Dia2, Maya1, MeloTTS) | Indistinguishable | Very High |

### The Scale of the Problem
- **2023**: ~500,000 deepfake audio samples in the wild
- **2025**: ~8 million deepfake audio samples (16x increase)
- Voice-based phishing scams now outpace visual deepfakes
- Detection market: $3.5+ billion by end of 2025

---

## 2. Human vs AI Voice Characteristics

### What Makes Human Speech Unique

**Physical constraints of human vocal production:**
- **Breathing patterns**: Natural pauses for respiration between phrases
- **Pitch variance**: Micro-variations in fundamental frequency (F0)
- **Prosody**: Natural stress, rhythm, and intonation patterns
- **Formant dynamics**: Smooth transitions between vowel sounds
- **Micro-tremors**: Subtle vibrations from muscle tension

### Common AI Voice Artifacts

| Artifact | Description | Detection Method |
|----------|-------------|------------------|
| **Missing breaths** | No respiratory sounds in long utterances | Silence pattern analysis |
| **Unnatural stability** | Pitch held perfectly steady | F0 variance measurement |
| **High-frequency cutoff** | Hard cutoff at 16kHz or 22kHz | Spectrogram analysis |
| **Grid artifacts** | Checkerboard patterns in spectrogram | Frequency domain analysis |
| **Metallic resonance** | Unnatural harmonic structure | Formant analysis |
| **Prosody flatness** | Mechanical rhythm without natural variance | Prosody feature extraction |

---

## 3. Detection Methods Overview

### Multi-Stage Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                      INPUT AUDIO                                  │
└────────────────────────┬─────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   Spectral    │ │   Deep        │ │  Watermark    │
│   Analysis    │ │   Learning    │ │  Detection    │
├───────────────┤ ├───────────────┤ ├───────────────┤
│ • MFCC        │ │ • Wav2Vec2    │ │ • AudioSeal   │
│ • Mel-spec    │ │ • HuBERT      │ │ • SynthID     │
│ • FFT         │ │ • WavLM       │ │ • Metadata    │
│ • Prosody     │ │ • Whisper     │ │ • LSB         │
└───────────────┘ └───────────────┘ └───────────────┘
        │                │                │
        └────────────────┼────────────────┘
                         ▼
              ┌─────────────────────┐
              │  Weighted Fusion    │
              │  + Confidence Score │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │   Final Verdict     │
              └─────────────────────┘
```

---

## 4. Spectral Analysis Methods

### 4.1 Mel-Frequency Cepstral Coefficients (MFCCs)

MFCCs are the foundational feature for speech analysis, representing the short-term power spectrum.

#### Extraction Pipeline
1. **Pre-emphasis**: Boost high frequencies to balance spectrum
2. **Framing**: Split audio into 20-25ms overlapping frames
3. **Windowing**: Apply Hamming window to reduce edge effects
4. **FFT**: Convert to frequency domain
5. **Mel-filterbank**: Apply perceptually-spaced triangular filters
6. **Log compression**: Take logarithm of filter energies
7. **DCT**: Apply Discrete Cosine Transform for final coefficients

#### Detection Signals
| Signal | Human | AI-Generated |
|--------|-------|--------------|
| MFCC variance | High dynamic variance | Flattened, more uniform |
| Delta coefficients | Natural transitions | Abrupt changes |
| Cross-correlation | Unique speaker patterns | Model-specific patterns |

#### Implementation

```python
import librosa
import numpy as np

def extract_mfcc_features(audio_path, n_mfcc=13):
    """Extract MFCC features for deepfake detection."""
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Add delta and delta-delta features
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    
    # Combine all features
    features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
    
    return features
```

---

### 4.2 Mel-Spectrogram Analysis

Mel-spectrograms provide rich time-frequency representations that often outperform MFCCs for CNN-based detection.

#### Advantages Over MFCCs
- Preserves more spectral detail
- Better for detecting subtle artifacts
- Natural input format for CNNs
- Emphasizes perceptually relevant frequencies

#### Artifact Detection
| Artifact Type | Visual Signature | Indicates |
|---------------|------------------|-----------|
| **Hard frequency cutoff** | Sharp horizontal line at 16/22 kHz | TTS model limitation |
| **Checkerboard patterns** | Grid-like artifacts in spectrogram | Vocoder upsampling |
| **Unnatural harmonics** | Missing or duplicated harmonic bands | Synthesis artifacts |
| **Silence anomalies** | Unusual gaps or no breathing sounds | Missing respiratory model |

#### Implementation

```python
import librosa
import librosa.display
import matplotlib.pyplot as plt

def generate_mel_spectrogram(audio_path, n_mels=128):
    """Generate Mel-spectrogram for visual inspection."""
    y, sr = librosa.load(audio_path, sr=22050)
    
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, fmax=11025
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db
```

---

### 4.3 Prosody Analysis

Prosody (rhythm, stress, intonation) is often where AI voices fail most noticeably.

#### Key Prosodic Features
| Feature | Description | AI Weakness |
|---------|-------------|-------------|
| **Pitch contour (F0)** | Fundamental frequency over time | Too smooth or mechanical |
| **Pitch variance** | Standard deviation of F0 | Often unnaturally low |
| **Speaking rate** | Words/syllables per second | Inconsistent pacing |
| **Pause patterns** | Distribution and timing of silences | Missing natural pauses |
| **Energy dynamics** | Loudness changes over time | Flat energy envelope |

---

## 5. Deep Learning Models

### 5.1 Wav2Vec 2.0

Wav2Vec2 is a self-supervised transformer model that learns speech representations from raw audio.

#### Architecture
- **Input**: Raw 16kHz audio waveform
- **CNN Feature Encoder**: 7 convolutional layers
- **Transformer**: 12-24 layers with self-attention
- **Output**: 768/1024-dimensional frame-level embeddings

#### Why Wav2Vec2 Works for Detection
- Learns phonemes, prosody, and speaker characteristics
- Self-supervised pretraining on 960+ hours of speech
- Captures temporal dependencies that cloning algorithms miss
- Fine-tuning enables task-specific adaptation

#### Fine-Tuning for Detection

```python
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

# Load pre-trained model
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base-960h",
    num_labels=2,
    id2label={0: "real", 1: "fake"},
    label2id={"real": 0, "fake": 1}
)

# Training configuration
training_args = {
    "learning_rate": 1e-5,
    "batch_size": 8,
    "epochs": 10,
    "warmup_ratio": 0.1,
    "max_audio_length": 10,  # seconds
}
```

---

### 5.2 HuBERT (Hidden-Unit BERT)

HuBERT uses masked prediction of discrete speech units for self-supervised learning.

#### Performance on ASVspoof
| Metric | Value |
|--------|-------|
| Equal Error Rate (EER) | 2.89% |
| min t-DCF | 0.2182 |
| Dataset | ASVspoof 2021 LA |

#### Advantages
- Extracts detailed acoustic features across languages
- Fine-tuning improves generalization
- Better at detecting unknown deepfake attacks
- Cross-lingual capabilities

---

### 5.3 WavLM

WavLM extends HuBERT with additional denoising objectives, making it more robust to noise and channel effects.

#### Key Features
- Pre-trained on 94,000 hours of speech
- Handles noise, reverberation, and codec artifacts
- State-of-the-art on ASVspoof benchmarks
- Combines with Multi-Fusion Attentive (MFA) classifier

#### Implementation with MFA Classifier

```python
from transformers import WavLMModel
import torch.nn as nn

class WavLMDeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
    
    def forward(self, input_values):
        outputs = self.wavlm(input_values).last_hidden_state
        pooled = outputs.mean(dim=1)  # Global average pooling
        return self.classifier(pooled)
```

---

### 5.4 Model Comparison

| Model | EER (ASVspoof) | Parameters | Strengths |
|-------|----------------|------------|-----------|
| **Wav2Vec2** | ~3.5% | 95M | Strong baseline, widely available |
| **HuBERT** | 2.89% | 95M | Cross-lingual, detailed features |
| **WavLM** | 2.4% | 95M | Noise robust, best benchmark results |
| **XLSR-53** | ~3.2% | 300M | Multilingual (53 languages) |

---

## 6. Watermark Detection

### 6.1 AudioSeal (Meta)

AudioSeal is a state-of-the-art proactive audio watermarking system designed for AI speech detection.

#### Key Features
| Feature | Description |
|---------|-------------|
| **Localized embedding** | Sample-level watermarks, survives editing |
| **Imperceptible** | Undetectable by human listeners |
| **Robust** | Survives compression, re-encoding, noise |
| **Fast detection** | 100x faster than existing methods |

#### How It Works
1. **Generator**: Embeds imperceptible watermark during synthesis
2. **Detector**: Single-pass detection identifies watermark fragments
3. **Bit capacity**: Typically 16-48 bits of embedded information

#### Detection Integration

```python
# Pseudo-code for AudioSeal detection
from audioseal import AudioSealDetector

detector = AudioSealDetector.load()

def detect_watermark(audio_path):
    audio = load_audio(audio_path)
    result = detector.detect(audio)
    
    return {
        "watermark_detected": result.detected,
        "confidence": result.confidence,
        "generator_id": result.payload if result.detected else None
    }
```

---

### 6.2 SynthID for Audio (Google DeepMind)

Google's SynthID embeds invisible watermarks into AI-generated audio produced by Google's models.

#### Characteristics
- Survives compression, speed changes, and minor edits
- Integrated into Gemini, Google Cloud TTS
- Detection API available through Google Cloud

---

### 6.3 Provider-Specific Detection

| Provider | Detection Method | Claimed Accuracy |
|----------|------------------|------------------|
| **ElevenLabs** | AI Speech Classifier API | 99% (unmodified), 90% (processed) |
| **Meta** | AudioSeal | High (sample-level) |
| **Google** | SynthID | High (survives transformations) |
| **Resemble.ai** | Watermark + fingerprinting | High |

---

## 7. Training Datasets

### 7.1 ASVspoof Challenge Datasets

The ASVspoof series is the gold standard for voice spoofing detection research.

#### ASVspoof 2021

| Task | Description | Attack Types |
|------|-------------|--------------|
| **LA (Logical Access)** | TTS and voice conversion | 13 spoofing algorithms |
| **PA (Physical Access)** | Replay attacks | Various playback conditions |
| **DF (Deepfake)** | Manipulated speech | Compressed deepfakes |

#### ASVspoof 5 (2024)

| Feature | Value |
|---------|-------|
| Speakers | ~2,000 crowdsourced |
| Attacks | 20+ crowdsourced attacks |
| Adversarial | 7 adversarial attack types (first time) |
| Focus | Robust deepfake detection |

### 7.2 Other Datasets

| Dataset | Size | Focus | Use Case |
|---------|------|-------|----------|
| **VCTK** | 109 speakers | Multi-speaker TTS | Training baseline |
| **LibriSpeech** | 1000 hours | Read speech | Bona fide samples |
| **FakeAVCeleb** | 490 videos | Celebrity deepfakes | Face + voice |
| **In-the-Wild** | 19.2 hours | Real-world deepfakes | Cross-domain testing |
| **ADD 2022** | Multi-lingual | Audio deepfake | General detection |

### 7.3 Building Custom Dataset

```python
from datasets import Dataset, Audio

def create_audio_detection_dataset():
    data = {
        "audio": [],
        "label": [],  # 0 = real, 1 = fake
        "source": [],
        "attack_type": []
    }
    
    # Real speech samples
    for audio_path in real_speech_paths:
        data["audio"].append(audio_path)
        data["label"].append(0)
        data["source"].append("real")
        data["attack_type"].append("none")
    
    # Synthetic samples from multiple TTS systems
    for tts_system in ["elevenlabs", "xtts", "tortoise", "bark"]:
        for audio_path in get_tts_samples(tts_system):
            data["audio"].append(audio_path)
            data["label"].append(1)
            data["source"].append(tts_system)
            data["attack_type"].append("tts")
    
    # Voice cloning samples
    for audio_path in voice_cloning_samples:
        data["audio"].append(audio_path)
        data["label"].append(1)
        data["source"].append("clone")
        data["attack_type"].append("voice_conversion")
    
    dataset = Dataset.from_dict(data)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    return dataset
```

---

## 8. Model Training Guide

### 8.1 Fine-Tuning Wav2Vec2 for Detection

```python
from transformers import (
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer
)
import torch

# Data preprocessing
def preprocess_audio(batch):
    audio = batch["audio"]
    inputs = processor(
        audio["array"],
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
        max_length=160000,  # 10 seconds
        truncation=True
    )
    batch["input_values"] = inputs.input_values[0]
    batch["labels"] = batch["label"]
    return batch

# Model setup
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base-960h",
    num_labels=2,
    problem_type="single_label_classification"
)

# Freeze feature extractor (optional, for faster training)
model.freeze_feature_extractor()

# Training
training_args = TrainingArguments(
    output_dir="./wav2vec2-deepfake-detector",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    warmup_ratio=0.1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,  # Mixed precision training
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
```

### 8.2 Data Augmentation

Critical for preventing overfitting with WavLM/HuBERT:

```python
import torch_audiomentations as ta

augmentation = ta.Compose([
    ta.AddColoredNoise(p=0.5, min_snr_in_db=10, max_snr_in_db=40),
    ta.Gain(p=0.5, min_gain_in_db=-12, max_gain_in_db=12),
    ta.PitchShift(p=0.3, min_transpose_semitones=-2, max_transpose_semitones=2),
])
```

---

## 9. Ensemble Scoring

### Current VisioNova Implementation

```python
DEFAULT_WEIGHTS = {
    "spectral": 0.25,     # MFCC, Mel-spectrogram, prosody
    "wav2vec": 0.35,      # Wav2Vec2/HuBERT primary model
    "watermark": 0.25,    # AudioSeal, SynthID, metadata
    "prosody": 0.15,      # Breathing, pitch variance, rhythm
}
```

### Score Fusion Formula

```
Final Score = Σ (weight_i × detector_score_i) / Σ weights

Confidence = base_score × agreement_factor × audio_quality_bonus
```

### Quality Adjustments
- Low bitrate audio: Reduce confidence by 10-20%
- Heavy compression: Reduce confidence by 15%
- Background noise: Apply noise-robust model weights

---

## 10. Confidence Thresholds

| AI Probability | Verdict | Recommendation |
|----------------|---------|----------------|
| 0-25% | Likely Authentic | Low concern |
| 25-45% | Uncertain | Manual review recommended |
| 45-65% | Possibly AI | Spectral analysis needed |
| 65-85% | Likely AI | High confidence synthetic |
| 85-100% | AI Generated | Very high confidence |

---

## 11. Known Limitations

### Detection Challenges

1. **Real-time cloning**: Latest models (Dia2, Maya1) use novel architectures
2. **Codec compression**: Heavy compression destroys forensic signals
3. **Background music**: Overlapping audio masks artifacts
4. **Short clips**: < 3 seconds provides insufficient signal
5. **Cross-lingual**: Models trained on English may miss non-English artifacts

### False Positive Sources

| Source | Mitigation |
|--------|------------|
| Professional voice actors | Cross-reference with known speakers |
| Heavily processed recordings | Lower confidence scores |
| Vintage recordings | Check metadata for recording era |
| Whispered speech | Apply specialized prosody analysis |

---

## 12. Future Improvements Roadmap

### Short-Term (Q1 2026)
1. **Add latest TTS systems** (Dia2, Maya1, MeloTTS) to training
2. **Implement AudioSeal** detection integration
3. **WavLM fine-tuning** on ASVspoof 5

### Medium-Term (Q2-Q3 2026)
1. **Multi-lingual detection** with XLSR-53
2. **Speaker verification** integration (is this voice Person X?)
3. **Real-time streaming** detection API

### Long-Term (Q4 2026+)
1. **Video-audio sync** detection for lip-sync deepfakes
2. **Emotional authenticity** analysis
3. **Source attribution** (identify which TTS system was used)

---

## 13. Best Practices

### For Developers
- Always use ensemble of spectral + deep learning methods
- Include data augmentation during training
- Test on out-of-domain samples (unseen TTS systems)
- Regularly update models with new TTS outputs
- Log detection failures for continuous improvement

### For Users
- Longer audio clips provide more reliable detection
- Check multiple sources if content is suspicious
- Consider context and source credibility
- Remember: No detector is 100% accurate

---

## References

1. ASVspoof Challenge: [asvspoof.org](https://www.asvspoof.org)
2. AudioSeal: Meta Research, ICML 2024
3. Wav2Vec2: "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations" (2020)
4. HuBERT: "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction" (2021)
5. WavLM: "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing" (2022)
6. SynthID: Google DeepMind (2023)
7. ElevenLabs AI Speech Classifier: [elevenlabs.io](https://elevenlabs.io)
8. XTTS: Coqui TTS (2023)
9. Tortoise TTS: [github.com/neonbjb/tortoise-tts](https://github.com/neonbjb/tortoise-tts)
