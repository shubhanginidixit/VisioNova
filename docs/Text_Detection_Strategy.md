# Text Detection Strategy

## Simple Explanation
Imagine writing is like a fingerprint. Humans write with "bursts" of creativity—we vary our sentence lengths, use unique words, and sometimes break grammar rules for effect. Experience shows that human writing is "bumpy."

AI models, on the other hand, are designed to predict the most likely next word. This makes their writing incredibly "smooth," consistent, and statistically predictable.

**How VisioNova detects it:**
We use a tool that looks at the text through the eyes of another AI. If the text looks "too perfect" and predictable to our detector, it’s likely AI. If it has the chaotic "burstiness" of human thought, it’s likely human.

---

## Technical Explanation

Our text detection pipeline utilizes a **Hybrid Ensemble Approach** combining Low-Level Statistical Analysis with High-Level Semantic Representation.

### 1. Statistical & Stylometric Analysis (The "Fingerprint")
*   **Perplexity (PPL):** We measure how "surprised" a language model is by the text. Low perplexity suggests the text is highly probable (AI-generated), while high perplexity suggests human variance.
*   **Burstiness:** A measure of the variation in perplexity over time. Human text has spikes (high burstiness); AI text is flat.
*   **n-gram Analysis:** We verify if the frequency of specific word sequences matches the training data distribution of common LLMs (Llama 3, GPT-4).

### 2. Zero-Shot Detection (Binoculars)
*   **Mechanism:** We employ the **"Binoculars"** method. This involves using two different LLMs (a "Scorer" and an "Observer") to calculate a specialized score.
*   **Logic:** By contrasting the perplexity of the input text between a highly capable model and a less capable model, we can isolate machine-generated patterns without needing to train on specific AI examples. This makes us robust against new, unseen AI models.

### 3. Supervised Deep Learning (DeBERTa)
*   **Model:** Fine-tuned `microsoft/deberta-v3-base`.
*   **Architecture:** Transformer-based classifier trained on a massive dataset of paired Human/AI essays (e.g., DAIGT dataset).
*   **Role:** While statistical methods catch "generic" AI usage, DeBERTa excels at catching sophisticated rewriting attacks where simple statistics might fail.

### 4. Decision Logic
The final verdict is a weighted average of:
1.  **Binoculars Score** (High precision, low false positives).
2.  **DeBERTa Confidence** (High recall).
3.  **Stylometric Heuristics** (Fallback for short text).
