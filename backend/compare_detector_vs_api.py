#!/usr/bin/env python3
"""Compare direct detector vs API for same text"""

import requests
from text_detector.detector import AIContentDetector

text = "In conclusion, the implementation of advanced algorithms constitutes a significant advancement"

# Direct detector
detector = AIContentDetector(use_ml_model=False)
direct_result = detector.predict(text)
print("DIRECT DETECTOR:")
print(f"  Human: {direct_result['scores']['human']}%")
print(f"  AI: {direct_result['scores']['ai_generated']}%")

# API
api_result = requests.post('http://127.0.0.1:5000/api/detect-ai', json={'text': text}).json()
print("\nAPI RESULT:")
print(f"  Human: {api_result['scores']['human']}%")
print(f"  AI: {api_result['scores']['ai_generated']}%")

print(f"\nDifference:")
print(f"  AI score diff: {float(api_result['scores']['ai_generated']) - direct_result['scores']['ai_generated']:.2f}%")
