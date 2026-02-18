# Text Detection

## Executive Summary

This document outlines VisioNova's comprehensive approach to detecting AI-generated text, combining a multi-model ensemble (DeBERTa, RoBERTa, E5), real-time perplexity analysis, zero-shot Binoculars detection, and adversarial defense mechanisms. As LLMs evolve from simple text predictors to sophisticated reasoning engines (GPT-5, DeepSeek R1, o1), our detection pipeline adapts to address the widening "forensic gap" while minimizing bias against non-native speakers.

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

Lightweight methods that run on CPU and provide baseline detection without ML models.

| Metric | AI Indicator | Human Indicator |
|--------|-------------|-----------------|
| **Perplexity (PPL)** | Low (15-40) — predictable | High (50-200+) — creative variance |
| **Burstiness** | Low — flat rhythm | High — spikes and valleys |
| **N-gram Repetition** | High — repeated sequences | Low — varied patterns |
| **Type-Token Ratio** | Low (<0.5) — repetitive vocab | High (>0.7) — diverse vocab |
| **Shannon Entropy** | Low (<3.5) — predictable | High (>4.0) — varied |

### 2.2 Pattern-Based Detection

We maintain a comprehensive library of AI writing patterns:

| Category | Examples | Weight |
|----------|----------|--------|
| **Hedging Language** | "It's important to note," "Generally speaking" | High |
| **Formal Transitions** | "Furthermore," "In conclusion," "Nevertheless" | High |
| **AI Self-References** | "As an AI," "I don't have personal" | Critical |
| **Tech Jargon** | "Leverage," "Navigate," "Delve into" | Medium |
| **Filler Phrases** | "In order to," "Due to the fact that" | Medium |
| **Overly Formal** | "Utilize," "Facilitate," "Endeavor" | Medium |

### 2.3 Adversarial Defense & Bias Correction

#### Homoglyph Normalization
Defends against evasion attacks where Latin characters are replaced with visually identical Cyrillic/Greek characters. Also strips zero-width spaces, soft hyphens, and other invisible characters.

#### ESL Bias Mitigation
- **ESL Score**: Calculates probability of ESL authorship based on common error patterns
- **Threshold Adjustment**: Dynamically raises the "AI Confidence" threshold for suspected ESL text

---

## 3. Zero-Shot Detection: Binoculars Method

The Binoculars method provides state-of-the-art accuracy without training on specific LLM outputs.

### How It Works
1. Use two closely related LLMs: **Falcon-7B** (Observer) and **Falcon-7B-Instruct** (Performer)
2. Calculate two perplexity scores: PPL1 from Observer, PPL2 as cross-perplexity
3. Compute ratio: `score = PPL1 / PPL2`
   - **score > 0.9**: Likely human
   - **score ≤ 0.9**: Likely AI

### Performance

| Metric | Value |
|--------|-------|
| True Positive Rate | 90%+ |
| False Positive Rate | 0.01% |
| Model Agnostic | Yes (works on unseen LLMs) |
| Training Required | None (zero-shot) |
| GPU Required | 14GB+ VRAM (Falcon-7B × 2) |

### Key Advantage
**Future-proof**: Works on models that don't exist yet because it detects the fundamental "predictability signature" that all AI text has, not specific model patterns.

---

## 4. Supervised Deep Learning: The Ensemble Approach

VisioNova uses a **Weighted Ensemble** of 4 diverse models:

| Model ID | Weight | Architecture | Strengths |
|----------|--------|--------------|-----------|
| **desklib/ai-text-detector-v1.01** | **35%** | DeBERTa-v3-large | #1 on RAID benchmark |
| **Oxidane/tmr-ai-text-detector** | **30%** | RoBERTa-base | Low false positive rate (0.2%), AUROC 0.99 |
| **fakespot-ai/roberta-base** | **20%** | RoBERTa-base | Robust general-purpose detector |
| **MayZhou/e5-small-lora** | **15%** | E5-small + LoRA | Lightweight, fast |

### Watermark Signal Detection
LLM providers embed statistical watermarks using "green/red token lists":
- **High green token ratio (> 55%)**: Suspected watermark
- **Low complex word ratio (< 12%)**: Avoidance of rare tokens
- Easily defeated by paraphrasing; not all providers implement watermarking

---

## 5. Detecting Reasoning Models (R1/o1/GPT-5)

Reasoning models produce internal "chain-of-thought" traces, making detection harder because:
1. Final output is highly polished and logic-dense
2. Common AI "tells" are self-corrected
3. Text exhibits "hyper-rationality" with perfect discourse markers

**Detection Strategies:**
- **Structure Analysis**: Perfect "First, Second, Finally" organization; exhaustive enumeration
- **Hedging Deficit**: Fewer "maybe," "possibly" — more confident declarations
- **Training on Distilled Data**: Use DeepSeek R1 distilled datasets as proxy

---

## 6. Detection Modes

| Mode | Method | Requirements | Use Case |
|------|--------|--------------|----------|
| `offline` | Patterns + Statistical (Real PPL) | CPU / Low RAM | Fast, privacy-preserving |
| `ml` | **Ensemble** (4 Models) + Statistical | ~2GB RAM + CPU/GPU | **Default**: Best accuracy/speed balance |
| `binoculars` | Dual Falcon-7B zero-shot | GPU (16GB+ VRAM) | Research-grade accuracy (slow) |

### Document Upload & Processing

| Feature | Details |
|---------|---------|
| **Extraction** | PyMuPDF for PDF, python-docx for DOCX, OCR fallback |
| **Chunking** | Sentence-boundary splitting into ~2000 char chunks |
| **AI Enhancement** | Optional Groq/Llama 4 Scout cleanup |
| **Max File Size** | 10MB |

### Scoring Formula

```
Final Score = (Ensemble_Prob × 0.60) + (Pattern_Score × 0.25) + (Linguistic_Score × 0.15)
```

**Adjustments:**
- **ESL Boost**: If ESL probability > 0.6, threshold increases by 10-15%
- **Uncertainty**: 40-60% range triggers "Mixed/Uncertain" verdict

### Confidence Thresholds

| AI Probability | Verdict |
|----------------|---------|
| 0-30% | Likely Human |
| 30-50% | Uncertain |
| 50-70% | Possibly AI |
| 70-100% | Likely AI |

---

## 7. API Endpoints

### `POST /api/detect-ai` — Plain Text Analysis
- **Input:** `{ "text": "...", "explain": true/false }`
- **Max:** 10,000 characters
- **Rate limit:** 10/minute

### `POST /api/detect-ai/upload` — Document Upload
- **Input:** `multipart/form-data` with `file` field (PDF/DOCX/TXT)
- **Max file size:** 10MB
- **Rate limit:** 5/minute

---

## 8. Training Datasets

| Dataset | Size | Focus |
|---------|------|-------|
| **RAID** | 10M+ docs | Adversarial robustness (11 models, 12 attacks, 11 domains) |
| **WildChat** | 1M+ logs | Real interactions |
| **MGTBench 2.0** | 500K+ | Academic integrity |
| **LMSYS-Chat-1M** | 1M conversations | Model attribution |
| **DeepSeek R1 Distill** | 100K+ | Reasoning traces |
| **FineWeb-Edu** | 1.3T tokens | Human baseline |

---

## 9. Limitations and Ethical Considerations

1. **Short text (< 50 words)**: Insufficient signal for reliable detection
2. **Domain shift**: Detectors trained on essays may fail on code or poetry
3. **Adversarial evasion**: Sophisticated users can bypass detection
4. **False positives**: Non-native English speakers may trigger false alarms
5. **Never use detection alone** for high-stakes decisions (academic integrity, hiring)

---

## 10. Future Improvements

### Short-Term (Q1 2026)
1. Upgrade to DeBERTa-v3-large for 3% accuracy improvement
2. Add RAID adversarial training for robustness

### Medium-Term (Q2-Q3 2026)
1. Add multilingual support with XLM-RoBERTa
2. Build attribution classifier (identify which AI wrote text)

### Long-Term (Q4 2026+)
1. Domain-specific detectors (academic, legal, medical)
2. Real-time monitoring API for streaming text

---

## References

1. RAID Benchmark: [https://raid-bench.xyz](https://raid-bench.xyz)
2. Binoculars Paper: "Spotting LLM-Generated Content Using Binoculars" (2024)
3. DeBERTa-v3: Microsoft Research (2021)
4. WildChat Dataset: Hugging Face Hub
5. MGTBench 2.0: ACL 2024
6. DeepSeek R1 Technical Report (2025)
