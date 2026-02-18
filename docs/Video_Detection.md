# Video Detection

## Executive Summary

This document outlines VisioNova's strategy for detecting deepfake videos and AI-generated video content. Our approach combines face-level analysis (facial landmarks, lip-sync verification), frame-level detection (EfficientNet-B4, XceptionNet), and temporal consistency analysis to identify synthetic or manipulated video content.

---

## 1. The Challenge: Video Deepfakes in 2025-2026

### Types of Video Manipulation

| Type | Method | Difficulty to Detect |
|------|--------|---------------------|
| **Face Swap** | Replace one face with another (DeepFaceLab, FaceFusion) | Medium |
| **Face Reenactment** | Drive a face with another's expressions/movements | Medium-High |
| **Full Video Generation** | AI-generated entire video (Sora, Veo 3, Runway Gen3) | Very High |
| **Lip Sync** | Modify lip movements to match new audio (Wav2Lip) | High |
| **Body Puppeteering** | Control body movements of a target | High |

### Why Video Detection is Harder Than Image

- Each frame must be analyzed, but context spans multiple frames
- Temporal consistency matters (natural motion vs. synthetic jitter)
- Compression artifacts from video codecs obscure forensic signals
- Real-time deepfakes enable live video manipulation

---

## 2. Detection Architecture

### Multi-Layer Pipeline

```
Video Input (.mp4/.avi/.mov)
    │
    ▼
┌───────────────────────────────────┐
│  Pre-Processing                   │
│  • Extract keyframes (1-5 fps)    │
│  • Face detection & alignment     │
│  • Audio-Video separation         │
└──────────────┬────────────────────┘
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌──────────────┐
│ Frame  │ │ Face   │ │  Temporal    │
│ Level  │ │ Level  │ │  Analysis   │
├────────┤ ├────────┤ ├──────────────┤
│ CNN    │ │ Land-  │ │ Motion      │
│ per    │ │ marks  │ │ consistency │
│ frame  │ │ + Lip  │ │ + Pulse     │
│        │ │ Sync   │ │ detection   │
└────┬───┘ └───┬────┘ └──────┬───────┘
     └─────────┼─────────────┘
               ▼
    ┌─────────────────────┐
    │  Temporal Fusion    │
    │  (LSTM/Attention)   │
    └──────────┬──────────┘
               ▼
    ┌─────────────────────┐
    │   Final Verdict     │
    │  + Suspicious Frames│
    └─────────────────────┘
```

---

## 3. Detection Methods

### 3.1 Frame-Level Analysis (CNN Detectors)

Each extracted frame is analyzed by image-level classifiers:

| Model | Architecture | Performance | Best For |
|-------|-------------|-------------|----------|
| **EfficientNet-B4** | CNN | AUC 95.59% | Face-swap detection |
| **XceptionNet** | CNN (depth-separable) | AUC 99.26% (FaceForensics++) | General deepfakes |
| **CapsuleNet** | Capsule network | Good on unseen methods | Cross-method generalization |

### 3.2 Facial Landmark Analysis

Tracks 68 facial landmarks across frames to detect:

| Signal | Natural | Deepfake |
|--------|---------|----------|
| **Landmark jitter** | < 2.5 pixels between frames | > 4 pixels (inconsistent warping) |
| **Eye blink rate** | 15-20 blinks/minute | Often absent or irregular |
| **Eye reflection** | Consistent across both eyes | Asymmetric or missing |
| **Facial boundary** | Smooth transition to background | Visible seam/blending artifacts |

### 3.3 Lip-Sync Verification

Compares audio waveform with lip movements:

| Component | Method |
|-----------|--------|
| **Audio features** | MFCC extraction from speech |
| **Visual features** | Mouth region crop, landmark tracking |
| **Sync score** | Cross-correlation of audio/visual features |
| **SyncNet** | Pre-trained A/V synchronization model |

**Detection logic:**
- Sync score < 0.3: Strong lip-sync mismatch → Likely manipulated
- Sync score 0.3-0.6: Possible issues → Manual review
- Sync score > 0.6: Good sync → Likely authentic

### 3.4 Temporal Consistency Analysis

Analyzes motion patterns across sequential frames:

| Check | Description |
|-------|-------------|
| **Optical flow** | Smooth vs. erratic motion between frames |
| **Head pose estimation** | Natural head rotation vs. sudden jumps |
| **Background consistency** | Static background should remain stable |
| **Lighting consistency** | Shadows/highlights should be temporally coherent |

### 3.5 Remote Photoplethysmography (rPPG) — Pulse Detection

Real human faces show subtle color changes from blood flow:
- **Present in real video**: 60-100 BPM detectable from facial skin
- **Absent in deepfakes**: No true biological signal
- **Limitation**: Requires ~10 seconds minimum, sensitive to lighting

---

## 4. Full Video Generation Detection (Sora/Veo/Runway)

AI-generated videos (not face-swaps) require different detection strategies:

| Method | Signal |
|--------|--------|
| **Physics violations** | Objects defying gravity, impossible reflections |
| **Temporal incoherence** | Objects morphing, disappearing between frames |
| **3D consistency** | Depth estimation inconsistencies |
| **Texture repetition** | Repeating patterns in backgrounds |
| **Motion blur** | Unnatural or absent motion blur |

---

## 5. Training Datasets

| Dataset | Size | Focus |
|---------|------|-------|
| **FaceForensics++** | 1M+ frames | 5 manipulation methods (Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter) |
| **DFDC** (Deepfake Detection Challenge) | 120K+ videos | Diverse actors, environments, manipulations |
| **Celeb-DF v2** | 5,639 videos | High-quality celebrity deepfakes |
| **WildDeepfake** | 707 videos | Real-world deepfakes from the internet |
| **DeeperForensics** | 60K videos | Large-scale, quality-controlled |

---

## 6. API Endpoint

### `POST /api/detect-video`

**Input:** `multipart/form-data` with `video` field  
**Supported formats:** MP4, AVI, MOV, WebM  
**Max file size:** 100MB

**Response:**
```json
{
  "success": true,
  "deepfake_probability": 78.5,
  "verdict": "LIKELY_DEEPFAKE",
  "frames_analyzed": 45,
  "suspicious_frames": [12, 23, 34, 35, 36],
  "analysis": {
    "face_detection": {
      "faces_found": true,
      "landmark_jitter": 4.8,
      "blink_rate": "absent",
      "eye_reflection": "inconsistent"
    },
    "lip_sync": {
      "sync_score": 0.25,
      "verdict": "MISMATCH"
    },
    "temporal": {
      "motion_consistency": 0.45,
      "lighting_consistency": 0.72
    },
    "pulse_detection": {
      "signal_present": false,
      "estimated_bpm": null
    }
  },
  "frame_thumbnails": ["data:image/jpeg;base64,..."]
}
```

---

## 7. Confidence Thresholds

| Deepfake Probability | Verdict | Action |
|---------------------|---------|--------|
| 0-20% | Likely Authentic | Low concern |
| 20-45% | Uncertain | Manual review recommended |
| 45-65% | Possible Manipulation | Investigation needed |
| 65-85% | Likely Deepfake | High confidence |
| 85-100% | Deepfake Detected | Very high confidence |

---

## 8. Limitations

1. **High compression**: Heavy video codec compression destroys forensic signals
2. **No-face videos**: Face-based methods fail on non-face content
3. **Short clips (< 3 seconds)**: Insufficient temporal data
4. **Real-time deepfakes**: Live manipulation harder to capture
5. **Full AI generation**: Sora/Veo-generated videos don't have face-swap artifacts
6. **Processing time**: Full analysis can take 30-120 seconds per video

---

## 9. Future Improvements

1. **Temporal Transformer**: Replace LSTM with attention for better long-range dependency capture
2. **AI video generation detection**: Specialized models for Sora/Veo/Runway outputs
3. **Real-time detection**: WebRTC integration for live video verification
4. **Source attribution**: Identify which tool generated the deepfake
5. **Audio-visual cross-modal**: Joint analysis of audio authenticity + video manipulation
6. **3D face reconstruction**: Robust to face-quality and angle variations

---

## References

1. FaceForensics++: [github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics)
2. DFDC: [ai.meta.com/datasets/dfdc](https://ai.meta.com/datasets/dfdc/)
3. EfficientNet: "Rethinking Model Scaling for CNNs" (Google, 2019)
4. XceptionNet: "Deep Learning with Depthwise Separable Convolutions" (2017)
5. SyncNet: "Out of Time: Audio-Visual Lip Sync" (2016)
6. rPPG: "Remote Photoplethysmography for Face Anti-Spoofing" (2020)
