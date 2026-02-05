# Video Deepfake Detection Strategy

## Executive Summary

This document outlines VisioNova's comprehensive approach to detecting AI-generated and manipulated video content. As video generation evolves from face-swap deepfakes to fully synthetic scenes (Sora, Veo 3, Runway Gen3), our detection pipeline combines frame-level forensics, temporal analysis, audio-visual synchronization, and biological signal monitoring to identify increasingly sophisticated synthetic media.

---

## 1. The Challenge: Why Video Detection is Critical

### The Evolution of Video Synthesis

| Era | Technology | Quality | Detection Difficulty |
|-----|------------|---------|---------------------|
| 2017-2019 | Face2Face, FaceSwap | Obvious artifacts | Low |
| 2019-2022 | DeepFaceLab, GAN-based | Occasional flaws | Medium |
| 2022-2024 | Diffusion (Runway, Pika) | Near-photorealistic | High |
| 2025+ | Full video generation (Sora, Veo) | Indistinguishable | Very High |

### The Scale of the Threat (2025)
- **8 million+** deepfake videos projected (up from 500K in 2023)
- **300%** surge in AI face-swap attacks over the past year
- **0.1%** of consumers can accurately identify high-quality deepfakes
- **40%** of biometric fraud attempts will involve deepfakes

---

## 2. Types of Video Manipulation

### 2.1 Face Swap Deepfakes
Replace one person's face with another's while maintaining expressions and movements.

**Common Tools**: DeepFaceLab, FaceSwap, Reface, Runway Act One

**Detection Signals**:
- Blending boundaries at face edges
- Inconsistent lighting between face and body
- Skin texture discontinuities

### 2.2 Lip-Sync Deepfakes
Modify mouth and jaw regions to match different audio.

**Common Tools**: Wav2Lip, SadTalker, InfiniteTalk

**Detection Signals**:
- Audio-visual synchronization errors
- Unnatural mouth movements
- Missing micro-expressions around lips

### 2.3 Fully Synthetic Video
Generate entire video from text prompts or images.

**Common Tools**: Sora, Veo 3, Runway Gen3 Alpha, Kling

**Detection Signals**:
- Physics violations (gravity, object permanence)
- Temporal inconsistencies across frames
- Repeating patterns in backgrounds

### 2.4 Expression/Reenactment
Transfer expressions from one person to another.

**Common Tools**: Face2Face, NeuralTextures, First Order Motion

**Detection Signals**:
- Unnatural expression transitions
- Landmark jitter at boundaries
- Head pose inconsistencies

---

## 3. Detection Methods Overview

### Multi-Modal Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      INPUT VIDEO                                  │
└────────────────────────┬─────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────────────────┐
        ▼                ▼                            ▼
┌───────────────┐ ┌───────────────┐          ┌───────────────┐
│  Per-Frame    │ │   Temporal    │          │  Audio-Visual │
│  Analysis     │ │   Analysis    │          │  Analysis     │
├───────────────┤ ├───────────────┤          ├───────────────┤
│ • XceptionNet │ │ • LSTM/RNN    │          │ • SyncNet     │
│ • EfficientNet│ │ • Optical Flow│          │ • TrueSync    │
│ • ViT         │ │ • rPPG        │          │ • Wav2Lip Det │
│ • Face Mesh   │ │ • Blink Rate  │          │ • Phoneme Map │
└───────────────┘ └───────────────┘          └───────────────┘
        │                │                            │
        └────────────────┼────────────────────────────┘
                         ▼
              ┌─────────────────────┐
              │   Watermark &       │
              │   Metadata Check    │
              ├─────────────────────┤
              │ • C2PA Manifest     │
              │ • SynthID           │
              │ • EXIF Analysis     │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │   Weighted Fusion   │
              │   + Confidence      │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │    Final Verdict    │
              └─────────────────────┘
```

---

## 4. Frame-Level Detection (Mesoscopic Analysis)

### 4.1 XceptionNet

The foundational model for deepfake detection, pre-trained on FaceForensics++.

#### Architecture
- Deep separable convolutions for efficient texture analysis
- Focus on compression artifacts and blending boundaries
- 36-layer deep network optimized for forgery detection

#### Performance
| Dataset | Accuracy | Notes |
|---------|----------|-------|
| FaceForensics++ | 87.7% | Standard benchmark |
| Celeb-DF | 65-75% | Cross-dataset challenge |
| DFDC | 70-80% | In-the-wild videos |

### 4.2 EfficientNet-B4

Scalable architecture with better cross-dataset generalization.

#### Performance (FaceForensics++)
| Metric | Value |
|--------|-------|
| AUC | 95.59% |
| Log Loss | 0.215 |

#### Advantages
- Compound scaling (depth, width, resolution)
- Better generalization than XceptionNet
- Efficient for real-time processing

### 4.3 Vision Transformers for Video

Emerging approach using ViT-based architectures.

```python
from transformers import VideoMAEForVideoClassification

model = VideoMAEForVideoClassification.from_pretrained(
    "MCG-NJU/videomae-base",
    num_labels=2,
    id2label={0: "real", 1: "fake"}
)
```

### 4.4 Face Extraction Pipeline

```python
import cv2
import dlib

def extract_faces_from_video(video_path, sample_rate=1):
    """Extract face crops from video for analysis."""
    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(video_path)
    
    faces = []
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample at specified rate (e.g., 1 frame per second)
        if frame_count % int(fps * sample_rate) == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected = detector(rgb)
            
            for face in detected:
                x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                face_crop = rgb[y1:y2, x1:x2]
                faces.append({
                    "frame": frame_count,
                    "crop": face_crop,
                    "bbox": (x1, y1, x2, y2)
                })
        
        frame_count += 1
    
    cap.release()
    return faces
```

---

## 5. Temporal Analysis

### 5.1 Landmark Tracking with LSTM

Deepfakes often exhibit temporal inconsistencies that are imperceptible in single frames but obvious over time.

#### Key Signals
| Signal | Detection Method |
|--------|------------------|
| **Landmark jitter** | Euclidean distance of facial points between frames |
| **Superhuman motion** | Velocity exceeds biological limits |
| **Expression discontinuity** | Abrupt changes in expression trajectory |
| **Head pose stability** | Unnatural steadiness or micro-tremors |

#### Implementation

```python
import torch
import torch.nn as nn

class TemporalDeepfakeDetector(nn.Module):
    """LSTM-based temporal inconsistency detector."""
    
    def __init__(self, input_size=468*3, hidden_size=256, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,  # 468 MediaPipe landmarks * 3 (x, y, z)
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, landmarks * 3)
        lstm_out, _ = self.lstm(x)
        # Use last timestep
        last_out = lstm_out[:, -1, :]
        return self.classifier(last_out)
```

### 5.2 Optical Flow Analysis

Analyze motion patterns between consecutive frames.

#### Detection Signals
- **Inconsistent flow**: Face moves differently from background
- **Boundary artifacts**: Sharp flow discontinuities at manipulation edges
- **Temporal noise**: Irregular motion patterns in synthetic regions

### 5.3 Blink Rate Analysis

Human blinking follows predictable patterns; early deepfakes often failed to replicate this.

| Metric | Real Video | Deepfake (Low Quality) |
|--------|------------|------------------------|
| Blink rate | 15-20/minute | Often 0 or mechanical |
| Blink duration | 100-400ms | Too fast or too slow |
| Blink pattern | Random natural | Periodic or absent |

**Note**: Modern deepfakes (2025+) often incorporate natural blinking, reducing this signal's reliability.

---

## 6. Biological Signal Detection (rPPG)

### Remote Photoplethysmography

Real faces show subtle color changes synchronized with heartbeat. Deepfakes traditionally lack this signal.

#### How It Works
1. Extract skin regions from face (forehead, cheeks)
2. Analyze color channel variations over time
3. Detect periodic signal at 0.5-3 Hz (heart rate range)
4. Absence of signal suggests synthetic origin

#### Limitations (2025)
- High-quality deepfakes can now incorporate pulse signals
- rPPG quality degrades with video compression
- Environmental factors (lighting, motion) affect accuracy
- Some deepfakes inherit pulse from source video

#### Implementation

```python
import numpy as np
from scipy.signal import butter, filtfilt

def extract_rppg_signal(face_crops, fps=30):
    """Extract rPPG signal from sequence of face crops."""
    # Extract mean RGB values from skin regions
    signals = []
    for crop in face_crops:
        r_mean = np.mean(crop[:, :, 0])
        g_mean = np.mean(crop[:, :, 1])
        b_mean = np.mean(crop[:, :, 2])
        signals.append([r_mean, g_mean, b_mean])
    
    signals = np.array(signals)
    
    # Apply bandpass filter (0.5-3 Hz for heart rate)
    nyq = fps / 2
    low, high = 0.5 / nyq, 3.0 / nyq
    b, a = butter(2, [low, high], btype='band')
    
    # Use green channel (strongest PPG signal)
    filtered = filtfilt(b, a, signals[:, 1])
    
    return filtered

def detect_pulse_signal(rppg_signal, threshold=0.3):
    """Detect if valid pulse signal is present."""
    # Compute power spectral density
    from scipy.signal import welch
    freqs, psd = welch(rppg_signal, fs=30, nperseg=256)
    
    # Look for peak in heart rate range (0.7-2.5 Hz = 42-150 BPM)
    hr_mask = (freqs >= 0.7) & (freqs <= 2.5)
    hr_power = np.max(psd[hr_mask])
    total_power = np.sum(psd)
    
    pulse_ratio = hr_power / total_power if total_power > 0 else 0
    
    return pulse_ratio > threshold, pulse_ratio
```

---

## 7. Audio-Visual Synchronization

### 7.1 SyncNet

Specialized network for detecting lip-sync inconsistencies.

#### Architecture
- Two-stream network: audio encoder + visual encoder
- Contrastive learning on synchronized pairs
- Outputs synchronization confidence score

#### Performance
| Configuration | Accuracy |
|---------------|----------|
| Standard | ~76% |
| With TrueSync (CNN-LSTM hybrid) | ~95% |

### 7.2 TrueSync Framework

Combines lip-sync analysis with blink rate monitoring.

```python
def truesync_detect(video_path, audio_path):
    """TrueSync: Combined lip-sync and blink detection."""
    
    # 1. Extract lip sync score
    lip_score = syncnet_analyze(video_path, audio_path)
    
    # 2. Extract blink pattern
    blink_pattern = analyze_blinks(video_path)
    blink_score = evaluate_blink_naturalness(blink_pattern)
    
    # 3. Combine scores
    combined_score = (lip_score * 0.7) + (blink_score * 0.3)
    
    return {
        "lip_sync_score": lip_score,
        "blink_score": blink_score,
        "combined_score": combined_score,
        "verdict": "fake" if combined_score < 0.5 else "real"
    }
```

### 7.3 Phoneme-Viseme Mapping

Check if mouth shapes (visemes) match spoken sounds (phonemes).

| Phoneme | Expected Viseme | Detection Method |
|---------|-----------------|------------------|
| /p/, /b/, /m/ | Lips closed | Lip distance < threshold |
| /f/, /v/ | Lower lip under teeth | Teeth-lip contact |
| /aa/, /ah/ | Mouth open wide | Mouth aspect ratio |
| /oo/, /w/ | Lips rounded | Lip circularity |

---

## 8. Watermark and Provenance Detection

### 8.1 C2PA Content Credentials

Check for cryptographic provenance metadata.

#### Video-Specific Fields
| Field | Indicates |
|-------|-----------|
| `generator` | AI tool used (Sora, Veo, etc.) |
| `edit_history` | Manipulation chain |
| `ai_generated` | Boolean flag for synthetic content |

### 8.2 SynthID for Video

Google's invisible watermarking for AI-generated video.

#### Characteristics
- Frame-level embedding
- Survives compression and editing
- Requires Google's detection API

### 8.3 Platform-Specific Markers

| Platform | Marker | Location |
|----------|--------|----------|
| **Sora** | Bouncing cloud logo | Visible watermark |
| **OpenAI** | C2PA "issued by OpenAI" | Metadata |
| **Google Veo** | SynthID | Invisible watermark |
| **Runway** | None standardized | Manual review |

---

## 9. Training Datasets

### 9.1 Benchmark Datasets

| Dataset | Size | Manipulation Types | Focus |
|---------|------|-------------------|-------|
| **FaceForensics++** | 1000 videos × 4 methods | Deepfakes, Face2Face, FaceSwap, NeuralTextures | Standard benchmark |
| **Celeb-DF** | 5,639 videos | Celebrity face swaps | Cross-identity |
| **DFDC** | 100K+ videos | Diverse methods | In-the-wild |
| **WildDeepfake** | 7,314 videos | Real-world deepfakes | Unconstrained |
| **FakeAVCeleb** | 490 videos | Face + voice | Audio-visual |
| **OPENFAKE (2025)** | Latest | Modern generators | Sora, Veo, etc. |

### 9.2 FaceForensics++ Details

| Method | Description | Key Artifacts |
|--------|-------------|---------------|
| **Deepfakes** | Autoencoder face swap | Blending, resolution |
| **Face2Face** | Expression transfer | Texture inconsistency |
| **FaceSwap** | Graphics-based swap | Edge artifacts |
| **NeuralTextures** | Neural rendering | Temporal flicker |

### 9.3 Creating Custom Dataset

```python
import cv2
import json

def create_video_detection_dataset(real_videos, fake_videos, output_dir):
    """Create dataset for deepfake video detection."""
    
    dataset = {
        "videos": [],
        "labels": [],
        "metadata": []
    }
    
    for video_path in real_videos:
        dataset["videos"].append(video_path)
        dataset["labels"].append(0)  # Real
        dataset["metadata"].append({
            "source": "real",
            "manipulation_type": "none"
        })
    
    for video_path, manipulation_type in fake_videos:
        dataset["videos"].append(video_path)
        dataset["labels"].append(1)  # Fake
        dataset["metadata"].append({
            "source": "synthetic",
            "manipulation_type": manipulation_type
        })
    
    with open(f"{output_dir}/dataset.json", "w") as f:
        json.dump(dataset, f)
    
    return dataset
```

---

## 10. Model Training Guide

### 10.1 Frame-Level Detector Training

```python
from torchvision import models
from torch.utils.data import DataLoader

# Load pretrained EfficientNet-B4
model = models.efficientnet_b4(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

# Training configuration
training_config = {
    "learning_rate": 1e-4,
    "batch_size": 32,
    "epochs": 20,
    "image_size": 224,
    "augmentation": ["horizontal_flip", "color_jitter", "random_crop"],
    "scheduler": "cosine_annealing",
}

# Train
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    **training_config
)
trainer.train()
```

### 10.2 Temporal Model Training

```python
# LSTM training for landmark sequences
temporal_model = TemporalDeepfakeDetector(
    input_size=468*3,
    hidden_size=256,
    num_layers=2
)

# Use sequences of 30-60 frames (1-2 seconds at 30fps)
sequence_length = 60

training_config = {
    "learning_rate": 1e-4,
    "batch_size": 16,
    "epochs": 30,
    "sequence_length": sequence_length,
}
```

---

## 11. Ensemble Scoring

### Current VisioNova Implementation

```python
DEFAULT_WEIGHTS = {
    "frame_analysis": 0.30,    # XceptionNet/EfficientNet per-frame
    "temporal": 0.25,          # LSTM landmark tracking
    "audio_visual": 0.20,      # SyncNet lip-sync
    "biological": 0.10,        # rPPG pulse detection
    "watermark": 0.15,         # C2PA, SynthID, metadata
}
```

### Score Fusion Formula

```
Final Score = Σ (weight_i × detector_score_i) / Σ weights

Confidence = base_score × agreement_factor × video_quality_bonus
```

### Agreement Analysis
| Agreement Level | Confidence Modifier |
|-----------------|---------------------|
| All methods agree | 1.3× |
| Majority agree (≥75%) | 1.0× |
| Mixed signals | 0.8× |
| Strong disagreement | 0.5× |

---

## 12. Confidence Thresholds

| AI Probability | Verdict | Recommendation |
|----------------|---------|----------------|
| 0-20% | Likely Authentic | Low concern |
| 20-40% | Uncertain | Manual review recommended |
| 40-60% | Possibly Manipulated | Detailed forensic review |
| 60-80% | Likely Deepfake | High confidence synthetic |
| 80-100% | Deepfake Detected | Very high confidence |

---

## 13. Known Limitations

### Detection Challenges

1. **Sora/Veo quality**: Newest generators produce near-perfect output
2. **Compression degradation**: Social media compression destroys forensic signals
3. **Short clips (< 3 seconds)**: Insufficient temporal information
4. **Partial manipulations**: Only lips or eyes modified
5. **High-quality training data**: Deepfakes trained on target person

### False Positive Sources

| Source | Mitigation |
|--------|------------|
| Heavy video compression | Lower confidence, flag for review |
| Professional makeup/lighting | Cross-reference temporal signals |
| CGI in movies | Context-aware classification |
| Video filters (TikTok, Instagram) | Pattern-based filter detection |

---

## 14. Future Improvements Roadmap

### Short-Term (Q1 2026)
1. **Add Sora/Veo detection** module trained on latest samples
2. **Implement TrueSync** (combined lip-sync + blink)
3. **C2PA verification** integration

### Medium-Term (Q2-Q3 2026)
1. **3D face reconstruction** for depth consistency analysis
2. **Audio deepfake fusion** (joint audio + video analysis)
3. **Real-time streaming** detection API

### Long-Term (Q4 2026+)
1. **Source attribution** (identify which generator)
2. **Adversarial robustness** training
3. **Cross-modal verification** (text, audio, video consistency)

---

## 15. Best Practices

### For Developers
- Combine frame-level and temporal analysis (never rely on one method)
- Test on cross-dataset scenarios (train FaceForensics++, test Celeb-DF)
- Include audio-visual sync for lip-focused manipulations
- Update models quarterly with new generator outputs

### For Users
- Request longer video samples when possible (> 10 seconds)
- Check for C2PA/SynthID watermarks first
- Consider source credibility alongside detection results
- Remember: No detector achieves 100% accuracy

---

## References

1. FaceForensics++: [github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics)
2. XceptionNet: "Xception: Deep Learning with Depthwise Separable Convolutions" (2017)
3. EfficientNet: "EfficientNet: Rethinking Model Scaling" (2019)
4. SyncNet: "Out of Time: Automated Lip Sync in the Wild" (2016)
5. TrueSync: IJCESEN 2025
6. rPPG Survey: "Remote Photoplethysmography: A Survey" (2023)
7. Celeb-DF: "Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics" (2020)
8. DFDC: Facebook Deepfake Detection Challenge (2020)
9. C2PA Standard: [c2pa.org](https://c2pa.org)
10. SynthID: Google DeepMind (2023)
