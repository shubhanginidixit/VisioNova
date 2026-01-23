#!/usr/bin/env python3
"""Test with new text to avoid caching"""

import requests

API_URL = 'http://127.0.0.1:5000'

# Completely new AI text that wasn't tested before
new_ai_text = "Conclusively, one must acknowledge that the implementation of sophisticated methodologies constitutes a watershed moment in technological advancement."

r = requests.post(f'{API_URL}/api/detect-ai', json={'text': new_ai_text})
result = r.json()

print(f"Text: {new_ai_text}")
print(f"Prediction: {result.get('prediction')}")
print(f"Human: {result['scores']['human']}%")
print(f"AI: {result['scores']['ai_generated']}%")
print(f"Pattern count: {result['detected_patterns']['total_count']}")
print(f"Pattern categories: {list(result['detected_patterns']['categories'].keys())}")
