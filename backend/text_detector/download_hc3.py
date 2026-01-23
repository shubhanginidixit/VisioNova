"""
Download and prepare HC3 dataset (Human ChatGPT Comparison Corpus)
HC3 contains 87K high-quality human and ChatGPT generated samples.
"""
import os
from pathlib import Path
from datasets import load_dataset
import pandas as pd


def download_hc3_dataset(output_dir: str = "datasets", sample_size: int = None):
    """
    Download HC3 dataset from Hugging Face.
    
    Args:
        output_dir: Directory to save the dataset
        sample_size: Number of samples to download (None = all ~87K)
    """
    print("="*60)
    print("Downloading HC3 Dataset (Human ChatGPT Comparison Corpus)")
    print("="*60)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    try:
        print("\nLoading dataset from Hugging Face...")
        # HC3 dataset with multiple domains
        dataset = load_dataset("Hello-SimpleAI/HC3", "all")
        
        print(f"Dataset loaded successfully!")
        print(f"Total samples: {len(dataset['train'])}")
        
        # Extract human and ChatGPT answers
        samples = []
        
        for i, item in enumerate(dataset['train']):
            # Human answers
            if 'human_answers' in item and item['human_answers']:
                for answer in item['human_answers'][:1]:  # Take first human answer
                    if answer and len(answer.strip()) > 50:  # Filter very short texts
                        samples.append({
                            'text': answer.strip(),
                            'label': 'human',
                            'source': 'hc3',
                            'question': item.get('question', '')[:100]
                        })
            
            # ChatGPT answers
            if 'chatgpt_answers' in item and item['chatgpt_answers']:
                for answer in item['chatgpt_answers'][:1]:  # Take first ChatGPT answer
                    if answer and len(answer.strip()) > 50:
                        samples.append({
                            'text': answer.strip(),
                            'label': 'ai',
                            'model': 'chatgpt',
                            'source': 'hc3',
                            'question': item.get('question', '')[:100]
                        })
            
            if sample_size and len(samples) >= sample_size:
                break
            
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1} items, extracted {len(samples)} samples")
        
        # Create DataFrame
        df = pd.DataFrame(samples)
        
        # Balance classes
        human_df = df[df['label'] == 'human']
        ai_df = df[df['label'] == 'ai']
        
        min_count = min(len(human_df), len(ai_df))
        
        balanced_df = pd.concat([
            human_df.sample(n=min_count, random_state=42),
            ai_df.sample(n=min_count, random_state=42)
        ]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save to CSV
        output_file = output_path / "hc3_dataset.csv"
        balanced_df.to_csv(output_file, index=False)
        
        print("\n" + "="*60)
        print("✅ HC3 Dataset Downloaded Successfully!")
        print("="*60)
        print(f"📁 Saved to: {output_file}")
        print(f"📊 Total samples: {len(balanced_df)}")
        print(f"   - Human: {len(balanced_df[balanced_df['label'] == 'human'])}")
        print(f"   - AI: {len(balanced_df[balanced_df['label'] == 'ai'])}")
        
        return output_file
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Install datasets: pip install datasets")
        print("2. Check internet connection")
        print("3. Try again in a few minutes (Hugging Face may be down)")
        raise


if __name__ == "__main__":
    # Download full dataset (or specify sample_size for testing)
    download_hc3_dataset(sample_size=None)  # None = download all
