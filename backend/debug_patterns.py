#!/usr/bin/env python3
"""Debug detection to see what patterns are found"""

import requests
import json

API_URL = 'http://127.0.0.1:5000'

test_cases = [
    ("AI text 1", "In conclusion, the implementation of advanced algorithms constitutes a significant advancement"),
    ("AI text 2", "Furthermore, it is imperative to acknowledge the profound implications of these developments"),
    ("Human text", "hey whats up! yeah i totally get it man. like seriously, no way, that sounds insane lol"),
]

for name, text in test_cases:
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    print(f"Text: {text}\n")
    
    r = requests.post(f'{API_URL}/api/detect-ai', json={'text': text})
    result = r.json()
    
    print(f"Prediction: {result.get('prediction')}")
    print(f"Confidence: {result.get('confidence')}%")
    print(f"Scores: {json.dumps(result['scores'], indent=2)}")
    
    patterns = result.get('detected_patterns', {})
    print(f"\nPatterns detected: {patterns['total_count']}")
    
    if patterns['total_count'] > 0:
        print("By category:")
        for cat, info in patterns.get('categories', {}).items():
            print(f"  - {cat}: {info['count']} ({info['examples'][:2]})")
    else:
        print("  (none)")
    
    # Print metrics
    metrics = result.get('metrics', {})
    print(f"\nLinguistic metrics:")
    print(f"  Word count: {metrics.get('word_count')}")
    print(f"  Vocabulary richness: {metrics.get('vocabulary_richness')}%")
    print(f"  Burstiness: {metrics.get('burstiness', {}).get('score')}")
    print(f"  N-gram uniformity: Bigram={metrics.get('ngram_uniformity', {}).get('bigram')}, Trigram={metrics.get('ngram_uniformity', {}).get('trigram')}")
