import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

models_to_test = [
    "abhishtagatya/hubert-base-960h-itw-deepfake",
    "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification",
    "garystafford/wav2vec2-deepfake-voice-detector",
    "Vansh180/deepfake-audio-wav2vec2",
    "MelodyMachine/Deepfake-audio-detection-V2",
    "mo-thecreator/Deepfake-audio-detection"
]

# Create a fake audio signal (sine wave)
sr = 16000
t = np.linspace(0, 5, sr * 5)
audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

print("Running baseline checks on a synthetic sine wave (should trigger anomalies):\n")

for m in models_to_test:
    try:
        processor = AutoFeatureExtractor.from_pretrained(m)
        model = AutoModelForAudioClassification.from_pretrained(m)
        model.eval()

        inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1)[0]
        
        print(f"Model: {m}")
        print(f"  Probs: index 0: {probs[0]:.4f}, index 1: {probs[1]:.4f}")
        print(f"  id2label: {model.config.id2label}\n")
    except Exception as e:
        print(f"Failed {m}: {e}")
