import sys
sys.path.insert(0, '.')

from text_detector import AIContentDetector

detector = AIContentDetector(use_ml_model=False)

test_texts = {
    "human": "hey whats up buddy",
    "ai_formal": "In conclusion, the multifaceted implications of this phenomenon warrant comprehensive examination of underlying mechanisms and their systematic integration within existing frameworks."
}

for label, text in test_texts.items():
    print(f"\n{'='*70}")
    print(f"TEST: {label}")
    print(f"{'='*70}")
    result = detector.predict(text)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Scores: AI={result['scores']['ai_generated']:.1f}%, Human={result['scores']['human']:.1f}%")
    print(f"Detected Patterns: {len(result['detected_patterns'])}")
    for p in result['detected_patterns']:
        print(f"  - {p['category']}: {p['pattern']}")
    print(f"Metrics: {result['metrics']}")
