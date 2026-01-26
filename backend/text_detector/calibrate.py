"""
Quick calibration harness for the offline AIContentDetector.
Runs a small labeled sample set (embedded) and reports accuracy and confusion.
"""
from collections import Counter
import json

from .detector import AIContentDetector


import csv
import os

SAMPLES = []
# Try to load CSV dataset if present
csv_path = os.path.join(os.path.dirname(__file__), 'datasets', 'calibration_samples.csv')
if os.path.exists(csv_path):
    with open(csv_path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            SAMPLES.append((row['text'], row['label']))
else:
    # Fallback small list (kept for backward compatibility)
    SAMPLES = [
        ("In today’s rapidly evolving digital landscape, artificial intelligence has emerged as a game-changing tool for content creation. From streamlining workflows to boosting productivity, AI enables businesses to foster innovation.", 'ai'),
        ("I went to the park yesterday and my dog chased a squirrel. We had ice cream and then walked home.", 'human'),
        ("As an AI assistant, I do not have personal experiences, but I can summarize the data for you in a concise manner.", 'ai'),
        ("The recipe calls for two eggs, a cup of flour, and a half cup of milk. Mix and bake for 20 minutes.", 'human'),
        ("Our solution leverages state-of-the-art techniques to streamline workflows and boost productivity across teams.", 'ai'),
        ("She laughed and told me about the funny movie she watched last night; it made her cry and laugh at the same time.", 'human'),
        ("Hybrid approaches combining human expertise and automated systems often yield the best outcomes.", 'ai'),
        ("Can you pass the salt please?", 'human')
    ]


def run_calibration(samples=None):
    det = AIContentDetector(use_ml_model=False)
    samples = samples or SAMPLES

    results = []
    for text, label in samples:
        out = det.predict(text, detailed=False)
        pred = 'ai' if out.get('prediction') == 'ai_generated' else 'human' if out.get('prediction') == 'human' else 'uncertain'
        results.append((label, pred, out))

    # Compute simple metrics
    correct = sum(1 for t in results if t[0] == t[1])
    total = len(results)
    accuracy = correct / total if total else 0.0

    # Confusion
    conf = Counter()
    for true, pred, _ in results:
        conf[(true, pred)] += 1

    print(f"Calibration samples: {total}")
    print(f"Accuracy: {accuracy*100:.1f}% ({correct}/{total})")
    print("\nConfusion matrix (true,pred):")
    for k, v in conf.items():
        print(f"  {k}: {v}")

    print("\nDetailed results:")
    for true, pred, out in results:
        print(json.dumps({"true": true, "pred": pred, "scores": out.get('scores'), "decision": out.get('decision')}, indent=2))


if __name__ == '__main__':
    run_calibration()
