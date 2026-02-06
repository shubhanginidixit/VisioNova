# VisioNova Exam Cheat Sheet 🎓

## 1. One-Liner Definition
**VisioNova** is a multi-modal AI credibility engine that detects whether content is **human-created or AI-generated** (Text, Image, Audio, Video) and verifies factual claims.

---

## 2. Key Statistics (Why We Built It)
| Metric | Number | Source |
|--------|--------|--------|
| **Deepfake growth** | 16× increase (2023-2025) | Sensity AI |
| **Deepfake count** | 8 Million+ online | Surfshark / Govt Stats |
| **Fraud** | 30% of ID fraud is deepfakes | Gartner 2025 |
| **Misinformation speed** | Spreads 6× faster than truth | MIT Study (Science, 2018) |
| **Human detection** | Only ~24% accuracy for videos | Sensity AI |

---

## 3. The 4 Big Gaps in Existing Tools
1.  **Single Modality:** Others check only text OR image. We check **ALL 4**.
2.  **Black Box:** Others say "90% Fake". We explain **WHY** (heatmaps, confidence).
3.  **No Fact-Check:** Others detect AI but don't check truth. We do **both**.
4.  **Slow:** Human checking takes days. We are **instant**.

---

## 4. Technical Stack
-   **Frontend:** HTML/JS (Simple, fast)
-   **Backend:** Python Flask (Great for ML)
-   **AI Engines:**
    -   **Text:** DeBERTa-v3 (Transformers)
    -   **Image:** ViT (Vision Transformer)
    -   **Audio:** HuBERT / Wav2Vec2
    -   **Video:** EfficientNet + LSTM
    -   **Reasoning:** LLaMA 3 (via Groq)

---

## 5. Performance Metrics (Memorize These!) 🌟
| Modality | Model | Performance | What it means |
|----------|-------|-------------|---------------|
| **Text** | DeBERTa-v3 | **96.2% Accuracy** | Very reliable for ChatGPT/Claude |
| **Image** | ViT (Vision Transformer) | **97.36% Accuracy** | Tested on NYUAD benchmark |
| **Audio** | HuBERT | **2.89% EER** | Equal Error Rate (Lower is better!) |
| **Video** | EfficientNet-B4 | **95.59% AUC** | Area Under Curve (High reliability) |

---

## 6. Key Algorithms (How It Works)
-   **ELA (Images):** Checks for compression inconsistency (edits glow).
-   **FFT (Images):** Checks for GAN "grid patterns" in frequency.
-   **Temporal Analysis (Fact-Check):** Extracts dates to search the right era.
-   **Vector Similarity (Search):** Matches claims to evidence.
-   **rPPG (Video):** Detects human pulse/heartbeat from face color.
-   **Lip-Sync (Video):** Matches mouth shape to audio sound.

---

## 7. Fact-Check Pipeline steps
1.  **Classify** (Is it a claim?)
2.  **Temporal Context** (When did it happen?)
3.  **Broad Search** (Google/Wiki/DDG)
4.  **Credibility Score** (Weigh sources by trust)
5.  **AI Verdict** (LLaMA 3 analyzes evidence)
