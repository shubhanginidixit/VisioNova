#!/usr/bin/env python3
"""Direct detector test to verify scoring logic"""

import sys
sys.path.insert(0, '/Users/adm/OneDrive/Desktop/VisioNova/backend')

from text_detector.detector import AIContentDetector

detector = AIContentDetector(use_ml_model=False)

# Test text with "In conclusion" pattern
text1 = "In conclusion, the implementation of advanced algorithms constitutes a significant advancement"

result = detector.predict(text1)
print(f"Text: {text1}")
print(f"Prediction: {result['prediction']}")
print(f"Scores: {result['scores']}")
print(f"Pattern count: {result['detected_patterns']['total_count']}")
print(f"Patterns: {list(result['detected_patterns']['categories'].keys())}")
