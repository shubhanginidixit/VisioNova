#!/usr/bin/env python3
"""Debug linguistic metrics"""

import sys
sys.path.insert(0, '/Users/adm/OneDrive/Desktop/VisioNova/backend')

from text_detector.detector import AIContentDetector

detector = AIContentDetector(use_ml_model=False)

text = "Conclusively, one must acknowledge that the implementation of sophisticated methodologies constitutes a watershed moment in technological advancement."

# Calculate linguistic metrics manually
ttr = detector._calculate_ttr(text)
entropy = detector._calculate_entropy(text)
ngram_bi = detector._calculate_ngram_uniformity(text, 2)
ngram_tri = detector._calculate_ngram_uniformity(text, 3)

print(f"Text: {text}")
print(f"\nLinguistic Metrics:")
print(f"  TTR (vocabulary diversity): {ttr:.3f}")
print(f"    -> TTR AI score (1-TTR): {1.0 - ttr:.3f}")
print(f"  Entropy: {entropy:.3f}")
print(f"    -> Entropy AI score (1-entropy): {1.0 - entropy:.3f}")
print(f"  Bigram uniformity: {ngram_bi:.3f}")
print(f"  Trigram uniformity: {ngram_tri:.3f}")

# These are the issues:
print(f"\nPROBLEM: Short formal text has:")
print(f"  - HIGH TTR (all words unique) -> low AI score from TTR")
print(f"  - LOW repetition -> low AI scores from n-grams")  
print(f"  - These metrics were designed for longer texts")
