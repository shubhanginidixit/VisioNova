"""
Parameter tuning script for `AIContentDetector`.
Performs a grid search over `pattern_weight` and `uncertainty_threshold` using a small labeled sample set.
"""
import json
from itertools import product

from .detector import AIContentDetector


import csv
import os

SAMPLES = []
csv_path = os.path.join(os.path.dirname(__file__), 'datasets', 'calibration_samples.csv')
if os.path.exists(csv_path):
    with open(csv_path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            SAMPLES.append((row['text'], row['label']))
else:
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


def evaluate(params):
    pw, ut = params
    det = AIContentDetector(use_ml_model=False)
    det.PATTERN_WEIGHT = pw
    det.LINGUISTIC_WEIGHT = round(1.0 - pw, 3)
    det.UNCERTAINTY_THRESHOLD = ut

    correct = 0
    total = 0
    for text, label in SAMPLES:
        out = det.predict(text, detailed=False)
        pred = 'ai' if out.get('prediction') == 'ai_generated' else 'human' if out.get('prediction') == 'human' else 'uncertain'
        # Treat 'uncertain' as incorrect for strict accuracy
        if pred == label:
            correct += 1
        total += 1

    acc = correct / total if total else 0.0
    return acc


def grid_search():
    pattern_weights = [round(x, 2) for x in [0.45, 0.5, 0.55, 0.6, 0.65, 0.7]]
    thresholds = [round(x, 3) for x in [0.05, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]]

    best = (None, -1.0)
    results = []

    for pw, ut in product(pattern_weights, thresholds):
        acc = evaluate((pw, ut))
        results.append({'pattern_weight': pw, 'uncertainty_threshold': ut, 'accuracy': acc})
        if acc > best[1]:
            best = ((pw, ut), acc)

    print("Grid search results (top 5):")
    results.sort(key=lambda r: r['accuracy'], reverse=True)
    for r in results[:5]:
        print(json.dumps(r))

    print('\nBest found:', best)
    return best


if __name__ == '__main__':
    grid_search()
