#!/usr/bin/env python3
"""Debug detector to see what patterns are detected"""
from text_detector.detector import AIContentDetector

detector = AIContentDetector(use_ml_model=False)

# Test 1: Human text
human_text = "hey whats up buddy"
result1 = detector.predict(human_text)
print("TEST 1 - Human text:")
print(f"  Text: {human_text}")
print(f"  Prediction: {result1['prediction']}")
print(f"  Confidence: {result1['confidence']:.1f}%")
print(f"  Scores: Human={result1['scores']['human']:.1f}%, AI={result1['scores']['ai_generated']:.1f}%")
print(f"  Patterns detected: {result1['detected_patterns']}")
print()

# Test 2: AI formal text
ai_text = "In conclusion, the multifaceted implications of this phenomenon warrant comprehensive examination of underlying mechanisms and their systematic integration within existing frameworks."
result2 = detector.predict(ai_text)
print("TEST 2 - AI formal text:")
print(f"  Text: {ai_text}")
print(f"  Prediction: {result2['prediction']}")
print(f"  Confidence: {result2['confidence']:.1f}%")
print(f"  Scores: Human={result2['scores']['human']:.1f}%, AI={result2['scores']['ai_generated']:.1f}%")
print(f"  Patterns detected: {result2['detected_patterns']}")
print()

# Test 3: Simple AI text
simple_ai = "The implementation of advanced algorithms necessitates careful consideration of computational efficiency and resource allocation."
result3 = detector.predict(simple_ai)
print("TEST 3 - Simple AI text:")
print(f"  Text: {simple_ai}")
print(f"  Prediction: {result3['prediction']}")
print(f"  Confidence: {result3['confidence']:.1f}%")
print(f"  Scores: Human={result3['scores']['human']:.1f}%, AI={result3['scores']['ai_generated']:.1f}%")
print(f"  Patterns detected: {result3['detected_patterns']}")
