#!/usr/bin/env python3
"""Direct detector test bypassing Flask"""

from text_detector.detector import AIContentDetector

detector = AIContentDetector(use_ml_model=False)

text = "In conclusion, the implementation of advanced algorithms constitutes a significant advancement"

result = detector.predict(text)

print(f"Direct Detector Test")
print(f"Text: {text}")
print(f"Prediction: {result['prediction']}")
print(f"Scores: {result['scores']}")
print(f"Pattern count: {result['detected_patterns']['total_count']}")

# Now test with pattern detection to see scores:
import re

patterns = detector._detect_patterns_in_text(text)
print(f"\nPatterns found: {len(patterns)}")
for p in patterns:
    print(f"  - {p['category']}: {p['pattern']}")

# Now calculate offline score
human_prob, ai_prob = detector._calculate_offline_score(text, patterns)
print(f"\nOffline scoring:")
print(f"  Human prob: {human_prob:.3f}")
print(f"  AI prob: {ai_prob:.3f}")
print(f"  AI % = {ai_prob * 100:.2f}%")
