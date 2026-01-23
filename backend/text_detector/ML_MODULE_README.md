# VisioNova ML Text Detection Module

## Quick Start

### 1. **Preprocess & Extract Features** (No model needed)
```bash
cd backend/text_detector
python demo_ml.py
```

### 2. **Train Your Own Model** (with labeled dataset)
```bash
# Prepare CSV with 'text' and 'label' columns (0=human, 1=AI)
python train_ml.py dataset.csv models/

# Output:
# - models/ai_text_model.pkl
# - models/tfidf_vectorizer.pkl
```

### 3. **Make Predictions**
```bash
# Test with demo texts
python predict_ml.py

# Or use in your code
from predict_ml import MLTextPredictor
predictor = MLTextPredictor()
result = predictor.predict("Your text here")
print(result['prediction'])  # 'human' or 'ai_generated'
```

---

## Module Overview

### `utils_ml.py` - Text Processing
```python
from utils_ml import preprocess, extra_features

# Preprocess text (lowercase, remove non-alpha, clean whitespace)
clean_text = preprocess("Hello! How are you?")
# → "hello how are you"

# Extract features (length + vocabulary diversity)
features = extra_features(clean_text)
# → [4, 1.0]  (4 words, 1.0 unique ratio)
```

### `train_ml.py` - Model Training
```bash
# Train with custom dataset
python train_ml.py my_dataset.csv output_models/

# Training uses:
# - TF-IDF vectorizer: 3000 features, 1-2 word grams
# - Random Forest: 300 trees
# - Train/test split: 80/20
```

### `predict_ml.py` - Inference
```python
from predict_ml import MLTextPredictor

predictor = MLTextPredictor()

# Single prediction
result = predictor.predict("Some text here")
# {
#   'prediction': 'human',
#   'confidence': 75.5,
#   'human_prob': 0.755,
#   'ai_prob': 0.245,
#   'text_length': 3,
#   'unique_ratio': 1.0
# }

# Batch predictions
results = predictor.predict_batch([
    "text 1",
    "text 2",
    "text 3"
])
```

### `demo_ml.py` - Testing & Examples
```bash
python demo_ml.py

# Shows:
# 1. Text preprocessing examples
# 2. Feature extraction examples  
# 3. Full ML prediction examples (if model exists)
```

---

## Dataset Format

Create a CSV file with exactly 2 columns:

```csv
text,label
"This is human written text",0
"The multifaceted implications warrant comprehensive examination.",1
"Hey what's up dude",0
"As an artificial intelligence, I must emphasize the paramount importance.",1
```

**Labels:**
- `0` = Human-written text
- `1` = AI-generated text

**Recommendations:**
- Minimum 100-200 samples (better with 1000+)
- Balanced classes (roughly equal 0s and 1s)
- Diverse text sources (different writing styles, topics, lengths)

---

## Features Used

### TF-IDF Features (3000 dimensions)
- 1-grams: Single words (e.g., "the", "implementation")
- 2-grams: Word pairs (e.g., "in conclusion", "must emphasize")
- Removes English stop words (the, and, or, etc.)
- Captures writing patterns unique to AI text

### Extra Features (2 dimensions)
1. **Text Length** - Number of words
   - AI text tends to be longer/more verbose
   
2. **Unique Word Ratio** - Vocabulary diversity
   - Human text often repeats words naturally
   - AI text tends toward more varied vocabulary

---

## Training & Performance

### Default Settings
```python
# TF-IDF
max_features = 3000
ngram_range = (1, 2)
stop_words = 'english'

# Random Forest
n_estimators = 300
random_state = 42
max_depth = 20

# Split
train_size = 0.8
test_size = 0.2
```

### Example Output
```
==================================================
MODEL PERFORMANCE
==================================================
Train Accuracy: 0.9540
Test Accuracy:  0.8920
Precision:      0.8850
Recall:         0.8960
F1-Score:       0.8905
==================================================
```

---

## Integration Examples

### Standalone Usage
```python
from predict_ml import MLTextPredictor

predictor = MLTextPredictor()
result = predictor.predict("The implementation necessitates consideration...")

if result['prediction'] == 'ai_generated':
    print(f"AI detected! Confidence: {result['confidence']}%")
```

### With Flask API
```python
from flask import Flask, request, jsonify
from predict_ml import MLTextPredictor

app = Flask(__name__)
predictor = MLTextPredictor()

@app.route('/api/detect', methods=['POST'])
def detect():
    data = request.json
    text = data.get('text', '')
    result = predictor.predict(text)
    return jsonify(result)
```

### Batch Processing
```python
texts = [
    "text 1",
    "text 2",
    "text 3"
]

predictor = MLTextPredictor()
results = predictor.predict_batch(texts)

for text, result in zip(texts, results):
    print(f"{text}: {result['prediction']} ({result['confidence']}%)")
```

---

## Troubleshooting

### Q: "Model not found"
**A:** Train the model first:
```bash
python train_ml.py dataset.csv models/
```

### Q: Low accuracy?
**A:** Check dataset quality:
- Minimum 1000 samples recommended
- Balanced classes (50/50 human/AI)
- Diverse text sources
- Proper labeling

### Q: Model takes too long to train?
**A:** Reduce features or trees:
```python
# In train_ml.py, modify:
max_features = 1000  # was 3000
n_estimators = 100   # was 300
```

### Q: Text too short warning?
**A:** Model needs minimum 2 words. Longer texts (20+ words) are more reliable.

---

## Files Structure
```
backend/text_detector/
├── utils_ml.py              # Core utilities
├── train_ml.py              # Training script
├── predict_ml.py            # Inference module
├── demo_ml.py               # Demo & tests
├── ML_INTEGRATION.md         # Integration guide (detailed)
├── ML_MODULE_README.md       # This file
└── models/                  # Auto-created after training
    ├── ai_text_model.pkl    # Trained Random Forest
    └── tfidf_vectorizer.pkl # TF-IDF transformer
```

---

## Dependencies
```
pandas >= 1.0
numpy >= 1.19
scikit-learn >= 0.24
scipy >= 1.5
joblib >= 1.0
```

Install: `pip install pandas numpy scikit-learn scipy joblib`

---

## Performance Notes

- **Training time:** 1-5 minutes (depends on dataset size)
- **Prediction time:** <10ms per text
- **Model size:** ~100-200 MB
- **Memory usage:** ~500 MB during training

---

## License & Attribution
Part of VisioNova AI Detection System
