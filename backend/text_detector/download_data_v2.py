"""
VisioNova Dataset Downloader v2
Downloads a high-quality, multi-model AI detection dataset.

Dataset: gsingh1-py/train (Hugging Face)
- 58,000+ samples
- Human: Real New York Times articles
- AI: GPT-4-o, Gemma-2-9b, Mistral-7B, Qwen-2-72B, LLaMA-8B, Yi-Large

Usage:
    python download_data_v2.py
    python download_data_v2.py --samples 20000
"""
import os
import argparse
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# Constants
# Constants
DATASET_NAME = "artem9k/ai-text-detection-pile"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "datasets")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ai_detection_multi_model.csv")


def download_dataset(sample_size: int = 20000):
    """
    Downloads and prepares a balanced dataset for AI text detection.
    
    Logic:
    1. Load the dataset from Hugging Face (streaming to save memory)
    2. Balance samples between human (label=0) and AI (label=1)
    3. Shuffle and save as CSV
    
    Args:
        sample_size: Total number of samples to download (will be split 50/50)
    """
    print(f"=" * 60)
    print(f"VisioNova Dataset Downloader v2")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Target samples: {sample_size}")
    print(f"=" * 60)
    
    print("\n[1/4] Loading dataset from Hugging Face...")
    try:
        # Try loading the full dataset first (it's not that large)
        dataset = load_dataset(DATASET_NAME, split="train")
        print(f"      Loaded {len(dataset)} total samples")
        use_streaming = False
    except Exception as e:
        print(f"      Full load failed ({e}), using streaming mode...")
        dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
        use_streaming = True
    
    print("\n[2/4] Processing and balancing dataset...")
    
    rows = []
    counts = {0: 0, 1: 0}  # 0 = human, 1 = AI
    target_per_class = sample_size // 2
    
    # Determine iteration method
    iterator = tqdm(dataset, total=sample_size if use_streaming else len(dataset))
    
    for item in iterator:
        # Extract text - handle different column names
        text = item.get('text', item.get('content', item.get('article', '')))
        
        # Extract label - handle different column names
        label = item.get('label', None)
        
        # If label is not directly available, try to infer from other fields
        if label is None:
            if 'generated' in item:
                label = 1 if item['generated'] else 0
            elif 'source' in item:
                src = str(item['source']).lower()
                # AI sources typically have model names
                if any(x in src for x in ['gpt', 'gemma', 'mistral', 'qwen', 'llama', 'yi', 'claude', 'ai']):
                    label = 1
                else:
                    label = 0
            elif 'model' in item and item['model']:
                label = 1  # If a model is specified, it's AI-generated
            else:
                continue  # Skip if we can't determine label
        
        label = int(label)
        
        # Skip empty text
        if not text or len(text.strip()) < 50:
            continue
        
        # Balance classes
        if counts[label] < target_per_class:
            rows.append({
                'text': text.strip(),
                'label': label
            })
            counts[label] += 1
            iterator.set_postfix(human=counts[0], ai=counts[1])
        
        # Stop when both classes are full
        if counts[0] >= target_per_class and counts[1] >= target_per_class:
            break
    
    print(f"\n      Collected: {counts[0]} human + {counts[1]} AI = {len(rows)} total")
    
    print("\n[3/4] Creating and shuffling DataFrame...")
    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Show statistics
    print("\n" + "=" * 40)
    print("Dataset Statistics:")
    print("=" * 40)
    print(f"Total samples: {len(df)}")
    print(f"Human (label=0): {len(df[df['label'] == 0])}")
    print(f"AI (label=1): {len(df[df['label'] == 1])}")
    print(f"Avg text length: {df['text'].str.len().mean():.0f} chars")
    print("=" * 40)
    
    print("\n[4/4] Saving dataset...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"      Saved to: {OUTPUT_FILE}")
    
    print("\n" + "=" * 60)
    print("SUCCESS! Dataset is ready for training.")
    print("=" * 60)
    print("\nNext steps:")
    print(f"  python train.py --data_path datasets/ai_detection_multi_model.csv --epochs 3")
    print("\n")
    
    return OUTPUT_FILE


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download AI Detection Dataset")
    parser.add_argument("--samples", type=int, default=20000, 
                        help="Total number of samples (default: 20000)")
    
    args = parser.parse_args()
    download_dataset(args.samples)
