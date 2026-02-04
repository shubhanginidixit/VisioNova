# Audio Detection Strategy

## Simple Explanation
Human voices are produced by air passing through vocal cords and a mouth. This physical process creates limitations—we need to breathe, and our pitch changes smoothly.

AI voices are often created by math, not air.
*   **The "Robotic" Perfectness:** AI voices can hold a tone purely steady in a way no human can.
*   **Missing Breaths:** AI sometimes forgets to add breathing sounds between long sentences.
*   **Invisible High-Pitch Noises:** The math used to generate audio often creates weird, harsh sounds at very high pitches (frequencies) that human ears act as filters for, but our software can see clearly on a graph.

---

## Technical Explanation

Our Audio Forensics module focuses on **Spectral Analysis** and **Feature Extraction**.

### 1. Mel-Frequency Cepstral Coefficients (MFCCs)
*   **Technique:** We extract MFCCs to map the power spectrum of the audio.
*   **Detection:** Synthetic speech often exhibits flattened variance in MFCC trajectories compared to the dynamic variance of a human vocal tract.

### 2. Spectral Analysis (The "Spectrogram")
*   **Visualizing Audio:** We convert audio files into Spectrograms (visual heatmaps of frequency over time).
*   **Artifact Detection:**
    *   **High-Frequency Cutoffs:** Many TTS (Text-to-Speech) mode models have a hard cutoff at 16kHz or 22kHz, whereas real high-fidelity recordings roll off naturally.
    *   **Grid Artifacts:** Generative audio models (like Vocoders) often leave checkerboard-like patterns in the spectrogram due to upsampling layers.

### 3. Deep Learning Classifier (Wav2Vec 2.0)
*   **Model:** Fine-tuned Wav2Vec 2.0 or HuberT base.
*   **Input:** Raw audio waveforms.
*   **Training:** Trained on the ASVspoof dataset (benchmark for voice cloning detection).
*   **Function:** The model learns temporal dependencies and subtle prosodic cues (intonation, stress, rhythm) that cloning algorithms often miss.

### 4. Biometric Consistency (Future Scope)
*   We aim to implement analysis that verifies if the voice's pitch and resonance are consistent with the assumed speaker's physical vocal tract size (a technique used in biological forensics).
