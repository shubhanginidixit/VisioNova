import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

m = "DavidCombei/wavLM-base-Deepfake_V2"
sr = 16000
t = np.linspace(0, 5, sr * 5)
audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

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
