"""
Prepare combined dataset from multiple sources and split into train/val/test sets.
Combines: HC3 (ChatGPT), Groq samples (Llama/Gemma/Mixtral), and human samples.
"""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def combine_datasets(datasets_dir: str = "datasets"):
    """Combine all dataset sources into one balanced dataset."""
    
    datasets_path = Path(datasets_dir)
    
    print("="*60)
    print("Combining Datasets for Training")
    print("="*60)
    
    all_data = []
    
    # Load HC3 dataset (ChatGPT samples)
    hc3_file = datasets_path / "hc3_dataset.csv"
    if hc3_file.exists():
        print(f"\n✓ Loading HC3 dataset...")
        hc3_df = pd.read_csv(hc3_file)
        print(f"  Loaded {len(hc3_df)} samples from HC3")
        all_data.append(hc3_df)
    else:
        print(f"\n⚠ HC3 dataset not found at {hc3_file}")
        print("  Run download_hc3.py first")
    
    # Load Groq-generated samples (Llama, Gemma, Mixtral)
    groq_file = datasets_path / "ai_samples_groq.csv"
    if groq_file.exists():
        print(f"\n✓ Loading Groq-generated samples...")
        groq_df = pd.read_csv(groq_file)
        print(f"  Loaded {len(groq_df)} samples from Groq API")
        print(f"  Models: {groq_df['model'].unique().tolist()}")
        all_data.append(groq_df)
    else:
        print(f"\n⚠ Groq samples not found at {groq_file}")
        print("  Run generate_samples.py first")
    
    # Load additional human samples if available
    human_file = datasets_path / "human_samples.csv"
    if human_file.exists():
        print(f"\n✓ Loading additional human samples...")
        human_df = pd.read_csv(human_file)
        print(f"  Loaded {len(human_df)} human samples")
        all_data.append(human_df)
    
    if not all_data:
        raise ValueError("No datasets found! Please run download_hc3.py and generate_samples.py first")
    
    # Combine all datasets
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Ensure consistent columns
    required_cols = ['text', 'label']
    combined_df = combined_df[required_cols]
    
    # Remove duplicates and very short texts
    print(f"\nCleaning data...")
    original_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['text'])
    combined_df = combined_df[combined_df['text'].str.len() >= 50]
    print(f"  Removed {original_count - len(combined_df)} duplicates/short texts")
    
    # Balance classes
    print(f"\nBalancing classes...")
    human_df = combined_df[combined_df['label'] == 'human']
    ai_df = combined_df[combined_df['label'] == 'ai']
    
    print(f"  Original: {len(human_df)} human, {len(ai_df)} AI")
    
    min_count = min(len(human_df), len(ai_df))
    
    balanced_df = pd.concat([
        human_df.sample(n=min_count, random_state=42),
        ai_df.sample(n=min_count, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"  Balanced: {len(balanced_df)} total ({min_count} each class)")
    
    return balanced_df


def split_dataset(df: pd.DataFrame, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Split dataset into train/validation/test sets."""
    
    print("\n" + "="*60)
    print("Splitting Dataset (80/10/10)")
    print("="*60)
    
    # First split: train + (val+test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=(val_ratio + test_ratio),
        random_state=42,
        stratify=df['label']
    )
    
    # Second split: val and test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=42,
        stratify=temp_df['label']
    )
    
    print(f"\n📊 Dataset splits:")
    print(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    # Verify balance
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        human_count = (split_df['label'] == 'human').sum()
        ai_count = (split_df['label'] == 'ai').sum()
        print(f"  {name}: {human_count} human, {ai_count} AI")
    
    return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df, output_dir: str = "datasets"):
    """Save train/val/test splits to CSV files."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("Saving Dataset Splits")
    print("="*60)
    
    train_file = output_path / "train.csv"
    val_file = output_path / "val.csv"
    test_file = output_path / "test.csv"
    
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"\n✅ Splits saved successfully!")
    print(f"  📁 Train: {train_file}")
    print(f"  📁 Val:   {val_file}")
    print(f"  📁 Test:  {test_file}")
    
    return train_file, val_file, test_file


def main():
    """Main preparation pipeline."""
    
    print("\n" + "🚀 "*20)
    print("Dataset Preparation Pipeline")
    print("🚀 "*20 + "\n")
    
    try:
        # Step 1: Combine datasets
        combined_df = combine_datasets()
        
        # Step 2: Split into train/val/test
        train_df, val_df, test_df = split_dataset(combined_df)
        
        # Step 3: Save splits
        train_file, val_file, test_file = save_splits(train_df, val_df, test_df)
        
        print("\n" + "="*60)
        print("✅ Dataset Preparation Complete!")
        print("="*60)
        print("\nNext steps:")
        print("1. Review the datasets in the 'datasets' folder")
        print("2. Run training script: python train_model.py")
        print("3. Evaluate on test set after training")
        
    except Exception as e:
        print(f"\n❌ Error during preparation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
