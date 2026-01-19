# Text Detection Module Documentation

## Overview
The **Text Detection Module** is the core AI analysis engine of VisioNova, designed to distinguish between human-written and AI-generated text. It combines machine learning models with linguistic analysis (stylometry) to provide a comprehensive verdict with high confidence.

## Architecture
The system uses a hybrid approach:
1.  **Deep Learning Model:** A DistilBERT-based classifier for the primary probability score.
2.  **Linguistic Analysis:** Statistical analysis of text structure (perplexity, burstiness).
3.  **Pattern Recognition:** RegEx-based detection of common "AI-isms".

## Core Components

### 1. AI Content Detector (`detector.py`)
**Purpose:** The main engine that performs the analysis.

**Key Features:**
- **Hybrid Detection:**
    - **Model Inference:** Uses `AutoModelForSequenceClassification` (DistilBERT) to get a base "Human vs. AI" probability.
    - **Linguistic Metrics:** 
        - **Perplexity:** Measures how predictable the text is. Low perplexity often indicates AI.
        - **Burstiness:** Measures the variance in sentence length and structure. AI tends to be more uniform (low burstiness).
        - **N-gram Uniformity:** AI models often repeat specific word sequences more uniformly than humans.
    - **Pattern Matching:** Detects specific phrases often used by LLMs (e.g., "It is important to note", "In conclusion", "As an AI language model").
- **Sentence-Level Analysis:** Can break down text to flag specific suspect sentences.
- **Chunking:** Efficiently handles large documents by splitting them into chunks and aggregating the weighted results.
- **Caching:** Implements LRU caching to speed up repeated queries for the same text.

**Usage:**
```python
detector = AIContentDetector()
result = detector.predict("Text to analyze...")
# Returns: { 'prediction': 'ai_generated', 'confidence': 98.5, 'metrics': {...} }
```

### 2. Text Explainer (`explainer.py`)
**Purpose:** Translates technical detection metrics into human-readable insights.

**Key Features:**
- **LLM Integration:** Uses the Groq API (e.g., Llama 3) to generate natural language explanations.
- **Context-Aware:** Feeds the raw probability, linguistic metrics, and detected patterns into the LLM to generate a personalized explanation.
- **Actionable Feedback:** Provides specific writing suggestions (e.g., "Try varying your sentence lengths") based on the analysis.
- **Fallback Mode:** Includes a deterministic fallback generator if the LLM API is unavailable, ensuring the user always gets an explanation.

**Usage:**
```python
explainer = TextExplainer()
explanation = explainer.explain(detection_result)
# Returns: { 'summary': "...", 'key_indicators': [...], 'suggestions': [...] }
```

## Key Metrics Explained

| Metric | Definition | AI Behavior | Human Behavior |
|--------|------------|-------------|----------------|
| **Perplexity** | "Surprise" factor of the text. | **Low** (text is predictable) | **High** (text is creative/unpredictable) |
| **Burstiness** | Variation in sentence structure/length. | **Low** (uniform, monotonic) | **High** (varied length and structure) |
| **Patterns** | Specific phrases (hedging, transitions). | **Frequent** usage of formal transitions | **Rare** or organic usage |

## Usage Flow
1.  **Input:** Text is sent to `detector.predict()`.
2.  **Processing:**
    *   Model calculates probability.
    *   Linguistic attributes (Perplexity, Burstiness) are computed.
    *   Regex patterns are counted.
3.  **Result:** A JSON object with the diagnosis is returned.
4.  **Explanation:** The result is passed to `explainer.explain()` to generate the user-facing summary showing *why* the text was flagged.
