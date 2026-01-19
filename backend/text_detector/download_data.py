"""
VisioNova Dataset Downloader
Downloads and formats a modern AI detection dataset.
Dataset: artem9k/ai-text-detection-pile
"""
import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# Constants
DATASET_NAME = "artem9k/ai-text-detection-pile"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "datasets")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ai_detection_data.csv")
SAMPLE_SIZE = 10000  # Number of samples to download

def prepare_dataset():
    print(f"Downloading dataset: {DATASET_NAME} (Streaming mode)...")
    try:
        # Load dataset in streaming mode to avoid downloading terabytes
        dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return

    print(f"Processing until we have balanced samples ({SAMPLE_SIZE} total)...")
    
    rows = []
    counts = {0: 0, 1: 0}
    target_per_class = SAMPLE_SIZE // 2
    
    for item in tqdm(dataset):
        # Determine format. Usually 'text' and 'label'
        text = item.get('text', '')
        # Check for label. Some datasets use 'label', 'generated', 'source'
        label_val = item.get('label')

        # Fallback logic for datasets without explicit 'label' key
        if label_val is None:
             if 'source' in item:
                src = str(item['source']).lower()
                label_val = 1 if 'ai' in src or 'gpt' in src else 0
             elif 'generated' in item:
                 label_val = int(item['generated'])
             else:
                 continue
                 
        label_val = int(label_val)
        
        # Check if we need more of this class
        if counts[label_val] < target_per_class:
            if text:
                rows.append({
                    'text': text,
                    'label': label_val
                })
                counts[label_val] += 1
        
        # Stop if both full
        if counts[0] >= target_per_class and counts[1] >= target_per_class:
            break
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Show stats
    print("\nDataset Statistics:")
    print("-" * 20)
    print(df['label'].value_counts())
    print(f"Total samples: {len(df)}")
    
    # Ensure directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save
    print(f"\nSaving to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    print("Done! You can now run the training script:")
    print(f"python train.py --data_path datasets/ai_detection_data.csv")

if __name__ == "__main__":
    prepare_dataset()
