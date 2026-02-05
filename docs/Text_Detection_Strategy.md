# Text Detection Strategy

## Executive Summary

This document outlines VisioNova's comprehensive approach to detecting AI-generated text, combining statistical analysis, pattern recognition, zero-shot detection, and supervised deep learning. As LLMs evolve from simple text predictors to sophisticated reasoning engines (GPT-5, DeepSeek R1, o1), our detection pipeline must adapt to address the widening "forensic gap."

---

## 1. The Problem: Why AI Text Detection is Hard

### Human vs AI Writing Patterns

**Human writing** is characterized by:
- **Burstiness**: Sudden shifts in sentence length, vocabulary, and syntax
- **Creative variance**: Unique word choices, idiomatic expressions, grammatical "imperfections"
- **Emotional nuance**: Personal anecdotes, subjective language

**AI-generated text** tends to exhibit:
- **Statistical smoothness**: Consistent token probability distributions
- **Hedging language**: "It's important to note," "Generally speaking"
- **Formal transitions**: "Furthermore," "In conclusion," "Nevertheless"
- **Low perplexity**: Text is highly predictable to language models

### The Evolving Challenge

| Generation | Characteristics | Detection Difficulty |
|------------|-----------------|---------------------|
| GPT-2/3 Era | Statistical anomalies, obvious patterns | Low |
| GPT-4 Era | Better coherence, hallucinations, "assistant" persona | Medium |
| R1/o1/GPT-5 Era | Chain-of-thought reasoning, hyper-polished output | High |

---

## 2. Detection Methods

### 2.1 Statistical and Stylometric Analysis

These lightweight methods run on CPU and provide baseline detection without ML models.

#### Perplexity (PPL)
Measures how "surprised" a language model is by text:
- **Low perplexity** (< 30): Highly predictable → Likely AI
- **High perplexity** (> 60): Creative variance → Likely human

#### Burstiness
Measures variance in perplexity across sentences:
- **Low burstiness**: Flat, consistent rhythm → AI indicator
- **High burstiness**: Spikes and valleys → Human indicator

#### N-gram Repetition
AI text tends to repeat certain word sequences (n-grams) more frequently than human writing due to token probability maximization.

#### Type-Token Ratio (TTR)
Vocabulary diversity score:
- **High TTR** (> 0.7): Diverse vocabulary → Human-like
- **Low TTR** (< 0.5): Repetitive vocabulary → AI-like

#### Shannon Entropy
Character distribution randomness:
- **High entropy** (> 4.0): More varied → Human-like
- **Low entropy** (< 3.5): Predictable → AI-like

---

### 2.2 Pattern-Based Detection

We maintain a comprehensive library of AI writing patterns across categories:

| Category | Examples | Weight |
|----------|----------|--------|
| **Hedging Language** | "It's important to note," "Generally speaking" | High |
| **Formal Transitions** | "Furthermore," "In conclusion," "Nevertheless" | High |
| **AI Self-References** | "As an AI," "I don't have personal" | Critical |
| **Tech Jargon** | "Leverage," "Navigate," "Delve into" | Medium |
| **Filler Phrases** | "In order to," "Due to the fact that" | Medium |
| **Academic Filler** | "Research indicates," "Studies show" | Medium |
| **Overly Formal** | "Utilize," "Facilitate," "Endeavor" | Medium |
| **Passive Voice** | "Can be seen," "Has been established" | Low |

---

### 2.3 Zero-Shot Detection: Binoculars Method

The Binoculars method provides state-of-the-art accuracy without training on specific LLM outputs.

#### How It Works

1. Use two closely related LLMs:
   - **Observer Model (M1)**: Falcon-7B
   - **Performer Model (M2)**: Falcon-7B-Instruct

2. Calculate two perplexity scores:
   - **PPL1**: Perplexity of input text from M1
   - **PPL2**: Cross-perplexity (M1 observing M2's predictions)

3. Compute the ratio: `score = PPL1 / PPL2`
   - **score > 0.9**: Likely human-written
   - **score ≤ 0.9**: Likely LLM-generated

#### Performance Metrics

| Metric | Value |
|--------|-------|
| True Positive Rate | 90%+ |
| False Positive Rate | 0.01% |
| Model Agnostic | Yes (works on unseen LLMs) |
| Training Required | None (zero-shot) |

#### Requirements
- GPU with 16GB+ VRAM
- Two Falcon-7B models loaded simultaneously

---

### 2.4 Supervised Deep Learning: DeBERTa-v3

For highest accuracy, we use a fine-tuned transformer classifier.

#### Model Architecture

- **Base Model**: `microsoft/deberta-v3-base` (86M parameters)
- **Alternative**: `microsoft/deberta-v3-large` (304M parameters)
- **Architecture**: Disentangled attention + enhanced mask decoder
- **Max Sequence Length**: 512 tokens

#### Why DeBERTa Over Alternatives

| Model | ROC AUC | Advantages | Disadvantages |
|-------|---------|------------|---------------|
| **DeBERTa-v3** | 0.9665 | Best RAID benchmark, disentangled attention | Higher compute |
| RoBERTa | 0.936 | Good baseline, widely supported | Less robust to attacks |
| Longformer | ~0.91 | Handles 4096+ tokens | Struggles with some generators |
| DistilBERT | 0.89 | Fast, lightweight | Lower accuracy |

#### Fine-Tuning Configuration

```python
# Recommended training hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3
MAX_LENGTH = 512
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
```

---

### 2.5 Watermark Signal Detection

LLM providers (OpenAI, Google) embed statistical watermarks using "green/red token lists."

#### How Watermarking Works

1. Vocabulary is split into "green list" (favored) and "red list" (avoided)
2. LLM is biased to select green list tokens during generation
3. Detection counts green token frequency to identify watermarked text

#### Detection Signals

- **High green token ratio (> 55%)**: Suspected watermark
- **Low complex word ratio (< 12%)**: Avoidance of rare tokens
- **Combined signal**: Both patterns together strongly indicate AI

#### Limitations
- Easily defeated by paraphrasing
- Not all providers implement watermarking
- Open-source models typically lack watermarks

---

## 3. Training Datasets

### 3.1 Recommended Datasets for Training

| Dataset | Size | Focus | Best For |
|---------|------|-------|----------|
| **RAID** | 10M+ docs | Adversarial robustness | General robustness, attack defense |
| **WildChat** | 1M+ logs | Real interactions | Detecting "wild" prompts, safety |
| **MGTBench 2.0** | 500K+ | Academic integrity | Human vs AI-polished text |
| **LMSYS-Chat-1M** | 1M conversations | Model attribution | Identifying which AI wrote text |
| **DeepSeek R1 Distill** | 100K+ | Reasoning traces | Detecting "thinking" models |
| **FineWeb-Edu** | 1.3T tokens | Human baseline | Reducing false positives |

### 3.2 RAID Benchmark Details

The RAID (Robust AI Detection) benchmark is the gold standard for testing detector robustness.

#### Coverage
- **11 generator models**: GPT-4, LLaMA-2, Mistral, Cohere, etc.
- **12 adversarial attacks**: Homoglyphs, paraphrasing, misspellings, zero-width chars
- **11 domains**: News, Reddit, poetry, recipes, abstracts, code
- **4 decoding strategies**: Greedy, sampling, temperature variations

#### Adversarial Attack Types

| Attack | Description | Defense |
|--------|-------------|---------|
| **Homoglyph** | Replace Latin chars with visually identical chars | Character normalization |
| **Paraphrasing** | Rewrite with another AI (QuillBot) | Semantic embeddings |
| **Misspellings** | Inject deliberate typos | Spell correction preprocessing |
| **Zero-width chars** | Insert invisible Unicode | Regex stripping |
| **Synonym swap** | Replace words with synonyms | Contextual analysis |

### 3.3 Building a Custom Dataset

For domain-specific detection, create a custom dataset:

```python
# Example: Creating a balanced training dataset
from datasets import Dataset

def create_detection_dataset():
    data = {
        "text": [],
        "label": [],  # 0 = human, 1 = AI
        "source": []
    }
    
    # Add human samples from FineWeb-Edu
    for sample in human_corpus:
        data["text"].append(sample)
        data["label"].append(0)
        data["source"].append("human")
    
    # Add AI samples from multiple generators
    for model in ["gpt-4", "llama-3", "mistral", "deepseek"]:
        for sample in generate_ai_samples(model):
            data["text"].append(sample)
            data["label"].append(1)
            data["source"].append(model)
    
    return Dataset.from_dict(data)
```

---

## 4. Model Training Guide

### 4.1 Option A: Fine-Tune DeBERTa-v3

**When to use**: When you need maximum accuracy and have labeled training data.

#### Step 1: Prepare Environment

```bash
pip install torch transformers datasets accelerate
```

#### Step 2: Load and Preprocess Data

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

dataset = load_dataset("your_dataset")
tokenized_dataset = dataset.map(tokenize_function, batched=True)
```

#### Step 3: Fine-Tune

```python
from transformers import TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base",
    num_labels=2,
    id2label={0: "human", 1: "ai"},
    label2id={"human": 0, "ai": 1}
)

training_args = TrainingArguments(
    output_dir="./deberta-ai-detector",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

trainer.train()
```

### 4.2 Option B: Use Pre-trained Detector

**When to use**: Quick deployment without custom training.

#### Recommended Pre-trained Models

| Model | Hugging Face Path | Accuracy |
|-------|-------------------|----------|
| Pangram AI | `pangram/pangram-v1` | High |
| Hello-SimpleAI | `Hello-SimpleAI/chatgpt-detector-roberta` | Medium-High |
| Roberta-Base-OpenAI | `roberta-base-openai-detector` | Medium |

### 4.3 Option C: Ensemble Approach

**When to use**: Production systems requiring maximum robustness.

Combine multiple detection methods with weighted voting:

```python
def ensemble_detect(text):
    scores = {
        "deberta": deberta_detector(text),      # Weight: 0.35
        "binoculars": binoculars_detect(text),  # Weight: 0.30
        "statistical": statistical_score(text), # Weight: 0.20
        "patterns": pattern_score(text),        # Weight: 0.15
    }
    
    weights = {"deberta": 0.35, "binoculars": 0.30, 
               "statistical": 0.20, "patterns": 0.15}
    
    final_score = sum(scores[k] * weights[k] for k in scores)
    return final_score
```

---

## 5. Detecting Reasoning Models (R1/o1/GPT-5)

### The Challenge

Reasoning models like DeepSeek R1 and OpenAI o1 generate internal "chain-of-thought" traces before outputting the final answer. This makes detection harder because:

1. Final output is highly polished and logic-dense
2. Common AI "tells" (hedging, hallucinations) are self-corrected
3. Text exhibits "hyper-rationality" with perfect discourse markers

### Detection Strategies for Reasoning Models

#### 1. Structure Analysis
Reasoning models produce overly structured responses:
- Perfect "First, Second, Finally" organization
- Exhaustive enumeration of all points
- Lack of tangential thoughts or personal asides

#### 2. Hedging Deficit
Unlike standard LLMs, reasoning models **avoid** hedging:
- Fewer uses of "maybe," "possibly," "potentially"
- More confident, declarative statements

#### 3. Training on Distilled Data
Use DeepSeek R1 distilled datasets as a proxy for GPT-5:
- Contains reasoning traces in `<think>` tags
- Final outputs represent "super-polished" style
- Dataset: `gsingh1-py/train` on Hugging Face

---

## 6. Current VisioNova Implementation

### Detection Modes

| Mode | Method | Requirements | Use Case |
|------|--------|--------------|----------|
| `offline` | Statistical + patterns | CPU only | Fast, lightweight |
| `ml` | DeBERTa + statistical hybrid | Model files | Balanced accuracy/speed |
| `binoculars` | Dual Falcon-7B zero-shot | GPU (16GB+) | Maximum accuracy |

### Scoring Formula

```
Final Score = (Pattern Score × 0.60) + (Linguistic Score × 0.25) + (Watermark Score × 0.15)
```

Where:
- **Pattern Score**: Based on detected AI writing patterns
- **Linguistic Score**: TTR, entropy, burstiness, n-gram uniformity
- **Watermark Score**: Green/red token ratio analysis

### Confidence Thresholds

| AI Probability | Verdict |
|----------------|---------|
| 0-30% | Likely Human |
| 30-50% | Uncertain |
| 50-70% | Possibly AI |
| 70-100% | Likely AI |

---

## 7. Future Improvements Roadmap

### Short-Term (Q1 2026)

1. **Upgrade to DeBERTa-v3-large** for 3% accuracy improvement
2. **Add RAID adversarial training** for robustness
3. **Implement curriculum learning** (easy → hard examples)

### Medium-Term (Q2-Q3 2026)

1. **Add multilingual support** with XLM-RoBERTa
2. **Integrate reasoning model detection** (R1/o1 patterns)
3. **Build attribution classifier** (identify which AI wrote text)

### Long-Term (Q4 2026+)

1. **Develop domain-specific detectors** (academic, legal, medical)
2. **Implement real-time monitoring API** for streaming text
3. **Add "AI-polished" detection** for human-AI collaborative writing

---

## 8. Limitations and Ethical Considerations

### Known Limitations

1. **Short text (< 50 words)**: Insufficient signal for reliable detection
2. **Domain shift**: Detectors trained on essays may fail on code or poetry
3. **Adversarial evasion**: Sophisticated users can bypass detection
4. **False positives**: Non-native English speakers may trigger false alarms

### Ethical Guidelines

- **Never use detection alone** for high-stakes decisions (academic integrity, hiring)
- **Combine with human review** for all flagged content
- **Disclose detection limitations** to end users
- **Regularly audit** for demographic bias in false positive rates

---

## References

1. RAID Benchmark: [https://raid-bench.xyz](https://raid-bench.xyz)
2. Binoculars Paper: "Spotting LLM-Generated Content Using Binoculars" (2024)
3. DeBERTa-v3: Microsoft Research (2021)
4. WildChat Dataset: Hugging Face Hub
5. MGTBench 2.0: ACL 2024
6. DeepSeek R1 Technical Report (2025)
