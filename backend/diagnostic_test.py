#!/usr/bin/env python3
"""Diagnostic test to verify detection results"""

import requests
import json

API_URL = 'http://127.0.0.1:5000'

# Test 1: Clearly HUMAN text
print("\n" + "="*70)
print("TEST 1: CLEARLY HUMAN TEXT")
print("="*70)
human_text = "hey whats up! yeah i totally get it man. like seriously, no way, that sounds insane lol"
r = requests.post(f'{API_URL}/api/detect-ai', json={'text': human_text})
result = r.json()
print(f"Text: {human_text}")
print(f"Expected: Human-written")
print(f"Prediction: {result.get('prediction')}")
print(f"Scores: {json.dumps(result.get('scores'), indent=2)}")
print(f"Human score: {result.get('scores', {}).get('human')}%")
print(f"AI score: {result.get('scores', {}).get('ai_generated')}%")
print(f"Status: {'✓ CORRECT' if result.get('prediction') == 'human' else '✗ WRONG'}")

# Test 2: Clearly AI text
print("\n" + "="*70)
print("TEST 2: CLEARLY AI-GENERATED TEXT")
print("="*70)
ai_text = "In conclusion, the implementation of advanced algorithms constitutes a significant advancement in the field of computer science, ultimately facilitating enhanced efficiency and streamlined operations."
r = requests.post(f'{API_URL}/api/detect-ai', json={'text': ai_text})
result = r.json()
print(f"Text: {ai_text}")
print(f"Expected: AI-generated")
print(f"Prediction: {result.get('prediction')}")
print(f"Scores: {json.dumps(result.get('scores'), indent=2)}")
print(f"Human score: {result.get('scores', {}).get('human')}%")
print(f"AI score: {result.get('scores', {}).get('ai_generated')}%")
print(f"Status: {'✓ CORRECT' if result.get('prediction') == 'ai_generated' else '✗ WRONG'}")

# Test 3: Medium-length human text
print("\n" + "="*70)
print("TEST 3: MEDIUM-LENGTH HUMAN TEXT")
print("="*70)
human_medium = "I went to the store yesterday and bought some groceries. The weather was nice so I decided to take a walk afterward. I saw some old friends there and we grabbed coffee together. It was a pretty good day overall."
r = requests.post(f'{API_URL}/api/detect-ai', json={'text': human_medium})
result = r.json()
print(f"Text: {human_medium}")
print(f"Expected: Human-written")
print(f"Prediction: {result.get('prediction')}")
print(f"Scores: {json.dumps(result.get('scores'), indent=2)}")
print(f"Human score: {result.get('scores', {}).get('human')}%")
print(f"AI score: {result.get('scores', {}).get('ai_generated')}%")
print(f"Status: {'✓ CORRECT' if result.get('prediction') == 'human' else '✗ WRONG'}")

# Test 4: Medium-length AI text
print("\n" + "="*70)
print("TEST 4: MEDIUM-LENGTH AI TEXT")
print("="*70)
ai_medium = "The advancement of technology in recent years has led to significant transformations across various sectors of society. Furthermore, it is imperative to acknowledge the profound implications of these developments. In summary, the future trajectory of technological progress remains a subject of considerable scholarly interest and investigation."
r = requests.post(f'{API_URL}/api/detect-ai', json={'text': ai_medium})
result = r.json()
print(f"Text: {ai_medium}")
print(f"Expected: AI-generated")
print(f"Prediction: {result.get('prediction')}")
print(f"Scores: {json.dumps(result.get('scores'), indent=2)}")
print(f"Human score: {result.get('scores', {}).get('human')}%")
print(f"AI score: {result.get('scores', {}).get('ai_generated')}%")
print(f"Status: {'CORRECT' if result.get('prediction') == 'ai_generated' else 'WRONG'}")

print("\n" + "="*70)
