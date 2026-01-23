# Training Setup Instructions

## Prerequisites

Install required packages:

```bash
pip install transformers datasets torch scikit-learn pandas numpy accelerate
```

## Step-by-Step Training Process

### Step 1: Generate AI Samples (Using Groq API)

```bash
cd backend/text_detector
python generate_samples.py
```

**What it does:**
- Generates 1000+ AI samples using Llama 3.1, Gemma 2, Mixtral
- Uses your existing Groq API (free)
- Takes ~30-45 minutes
- Saves to `datasets/ai_samples_groq.csv`

**Customize sample count:**
Edit `generate_samples.py`, line 254:
```python
num_samples = 30000  # For full dataset (takes ~20 hours)
```

---

### Step 2: Download HC3 Dataset

```bash
python download_hc3.py
```

**What it does:**
- Downloads 87K ChatGPT vs Human samples from Hugging Face
- Takes ~10 minutes
- Saves to `datasets/hc3_dataset.csv`

---

### Step 3: Prepare Combined Dataset

```bash
python prepare_dataset.py
```

**What it does:**
- Combines HC3 + Groq samples
- Balances classes (50% human, 50% AI)
- Splits into train/val/test (80/10/10)
- Creates:
  - `datasets/train.csv`
  - `datasets/val.csv`
  - `datasets/test.csv`

---

### Step 4: Train Model

#### Option A: Local Training (if you have GPU)

```bash
python train_model.py
```

**Requirements:**
- NVIDIA GPU with 8GB+ VRAM
- CUDA installed
- Takes ~2-3 hours

#### Option B: Google Colab (Recommended - Free GPU)

1. Open the `VisioNova_Training_Colab.ipynb` notebook
2. Upload to Google Colab
3. Upload your `datasets/` folder to Colab
4. Run all cells
5. Download trained model files

**Colab advantages:**
- Free Tesla T4 GPU
- Faster training (~2 hours)
- No local setup needed

---

### Step 5: Integrate Trained Model

After training completes:

```bash
# Copy model files
cp -r model_trained/* ../text_detector/model/

# Or on Windows:
xcopy model_trained backend\text_detector\model\ /E /I /Y
```

Then update `detector.py` to use RoBERTa tokenizer.

---

## Expected Results

**Healthy Model Performance:**
- Accuracy: 85-92%
- F1 Score: 0.85-0.92
- Precision: 0.80-0.90
- Recall: 0.85-0.93

**Red Flags:**
- Accuracy > 95% = Overfitting ⚠️
- Accuracy < 80% = Undertrained ⚠️

---

## Troubleshooting

### "No module named 'datasets'"
```bash
pip install datasets
```

### "CUDA out of memory"
Reduce batch size in `train_model.py`:
```python
"batch_size": 8,  # Reduce from 16 to 8
```

### "Groq API rate limit"
The script auto-sleeps 2 seconds between requests.
For faster generation, reduce to 1 second:
```python
time.sleep(1)  # Line 195 in generate_samples.py
```

### "HC3 download fails"
Check internet connection and try again.
Hugging Face servers may be temporarily down.

---

## Quick Start (Minimum Viable Dataset)

For fast testing with smaller dataset:

1. Generate 500 samples: Edit `generate_samples.py` line 254
2. Download HC3: `python download_hc3.py`
3. Prepare: `python prepare_dataset.py`
4. Train: `python train_model.py`

This creates a ~40K sample dataset and trains in ~1 hour.

---

## Next: Integration Guide

After training, see `INTEGRATION.md` for:
- Updating `detector.py` for RoBERTa
- Adjusting hybrid detection weights
- Testing with real samples
