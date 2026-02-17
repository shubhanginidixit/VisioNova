# Audio Detection Strategy

## Executive Summary

VisioNova employs a state-of-the-art deep learning approach to detect synthetic audio. By leveraging the **Wav2Vec2** architecture finetuned on large-scale deepfake datasets (ASVspoof, WaveFake), we can effectively distinguish between bonafide human speech and AI-generated audio (TTS, Voice Conversion).

---

## 1. Scope: What We Are Detecting

We are specifically targeting **AI-Synthesized Speech**, often referred to as "Audio Deepfakes". This includes:

*   **Text-to-Speech (TTS):** Audio generated entirely from text by models like ElevenLabs, Bark, Tortoise, VALL-E.
*   **Voice Conversion (VC):** Real human speech transformed to sound like another person (e.g., RVC - Retrieval-based Voice Conversion).
*   **Replayed/Spoofed Audio:** Audio that has been artificially manipulated or re-recorded to deceive verification systems.

**We are NOT detecting:**
*   Acoustic environment analysis (background noise type).
*   Speaker identification (who is speaking), unless it's to verify a specific target voice (future scope).
*   General audio classification (music vs speech).

---

## 2. Detection Results & Outputs

The audio detection system provides the following rigorous analysis for every input file:

| Metric | Description |
| :--- | :--- |
| **Prediction** | A binary classification: **"Real"** (Bonafide) or **"Fake"** (Spoof). |
| **Confidence Score** | A percentage (0-100%) indicating how certain the model is of its prediction. |
| **AI Probability** | The raw probability score (0.0 - 1.0) that the audio is synthetic. |
| **Human Probability** | The raw probability score (0.0 - 1.0) that the audio is authentic. |
| **Model Used** | Identifier of the specific model used for inference (e.g., `MelodyMachine/Deepfake-audio-detection-V2`). |

**Example JSON Output:**

```json
{
  "prediction": "ai_generated",
  "confidence": 99.85,
  "ai_probability": 99.85,
  "human_probability": 0.15,
  "details": {
    "duration": 4.5,
    "model": "MelodyMachine/Deepfake-audio-detection-V2"
  }
}
```

---

## 3. Selected Model Architecture: The Ensemble Approach

To maximize detection accuracy and robustness, VisioNova employs a **Weighted Ensemble** of two distinct high-performance models. This "multi-expert" approach reduces the risk of a single model being fooled by a specific generator.

### The Ensemble Registry

| Model | Architecture | Weight | Role |
| :--- | :--- | :--- | :--- |
| **`MelodyMachine/Deepfake-audio-detection-V2`** | **Wav2Vec2** | **60%** | The primary expert. Extremely robust, fine-tuned on diverse datasets (In-the-Wild, WaveFake). |
| **`DavidCombei/wavLM-base-Deepfake_V2`** | **WavLM** | **40%** | The specialist. Uses a different architectural approach (WavLM) to catch edge cases that Wav2Vec2 might miss. |

### Why Ensemble?

1.  **Diversity:** Wav2Vec2 and WavLM are trained with different objectives. WavLM is specifically designed to handle "masked speech prediction and denoising," making it sensitive to different kinds of digital artifacts than Wav2Vec2.
2.  **Robustness:** If a new text-to-speech engine manages to fool one model, it is statistically unlikely to fool both simultaneously in the exact same way.
3.  **Calibration:** Averaging scores helps smooth out overconfident predictions on ambiguous audio.

### Voting Logic

```python
Final_Score = (Wav2Vec2_Prob * 0.60) + (WavLM_Prob * 0.40)
```

- **> 50%**: Classified as **Fake/AI**
- **< 50%**: Classified as **Real/Human**

---

## 4. Implementation Details

The implementation handles:
1.  **Preprocessing:** Resampling audio to **16kHz** (standard for Wav2Vec2).
2.  **Normalization:** Ensuring audio amplitude is normalized to prevent volume-based bias.
3.  **Inference:** Passing the processed waveform through the transformer network to get logit scores.
4.  **Thresholding:** While the model provides raw probabilities, we apply a threshold (typically 0.5) to determine the final `Real` vs `Fake` label, with a "Warning" zone for scores between 0.4 and 0.6.
