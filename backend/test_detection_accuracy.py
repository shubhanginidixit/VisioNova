#!/usr/bin/env python3
"""Test to verify AI detection accuracy"""

import requests
import json

API_URL = 'http://127.0.0.1:5000'

test_cases = [
    {
        'name': 'HUMAN: Casual conversation',
        'text': "hey whats up! yeah i totally get it man. like seriously, no way, that sounds insane lol",
        'expected': 'human'
    },
    {
        'name': 'AI: Formal paragraph',
        'text': "In conclusion, the implementation of advanced algorithms constitutes a significant advancement in the field of computer science, ultimately facilitating enhanced efficiency and streamlined operations.",
        'expected': 'ai_generated'
    },
    {
        'name': 'HUMAN: Personal story',
        'text': "I went to the store yesterday and bought some groceries. The weather was nice so I decided to take a walk afterward. I saw some old friends there and we grabbed coffee together. It was a pretty good day overall.",
        'expected': 'human'
    },
    {
        'name': 'AI: Academic tone',
        'text': "The advancement of technology in recent years has led to significant transformations across various sectors of society. Furthermore, it is imperative to acknowledge the profound implications of these developments. In summary, the future trajectory remains a subject of considerable scholarly interest.",
        'expected': 'ai_generated'
    },
    {
        'name': 'HUMAN: Thoughts',
        'text': "honestly i'm not sure what to think about all this. on one hand it's cool, but on the other hand it kinda worries me you know?",
        'expected': 'human'
    },
]

results = []
correct = 0

for test in test_cases:
    r = requests.post(f'{API_URL}/api/detect-ai', json={'text': test['text']})
    result = r.json()
    
    prediction = result.get('prediction')
    is_correct = prediction == test['expected']
    correct += 1 if is_correct else 0
    
    print(f"\nTest: {test['name']}")
    print(f"Expected: {test['expected']}")
    print(f"Predicted: {prediction}")
    print(f"Scores: Human={result['scores']['human']:.1f}%, AI={result['scores']['ai_generated']:.1f}%")
    print(f"Result: {'PASS' if is_correct else 'FAIL'}")
    
    results.append({
        'test': test['name'],
        'expected': test['expected'],
        'predicted': prediction,
        'correct': is_correct,
        'scores': result['scores']
    })

print(f"\n\n{'='*70}")
print(f"SUMMARY: {correct}/{len(test_cases)} tests passed ({100*correct//len(test_cases)}%)")
print(f"{'='*70}")

if correct < len(test_cases):
    print("\nFailed tests:")
    for r in results:
        if not r['correct']:
            print(f"  - {r['test']}")
            print(f"    Expected: {r['expected']}, Got: {r['predicted']}")
            print(f"    Scores: Human={r['scores']['human']:.1f}%, AI={r['scores']['ai_generated']:.1f}%")
