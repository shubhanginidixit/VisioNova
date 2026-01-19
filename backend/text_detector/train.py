"""
VisioNova Model Trainer
Script to fine-tune the AI detection model on new data.

Usage:
    python train.py --data_path datasets/my_data.csv --epochs 3

Data Format (CSV):
    text,label
    "Sample human text...",0
    "Sample AI text...",1
"""
import os
import argparse
import pandas as pd
import numpy as np

# Fix for Windows Tokenizer Deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
# Force flush on all prints
import functools
print = functools.partial(print, flush=True)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATASET_DIR = os.path.join(BASE_DIR, "datasets")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train(data_path, output_dir=None, epochs=3, batch_size=8, learning_rate=2e-5):
    """
    Fine-tune the model.
    """
    if not output_dir:
        output_dir = MODEL_DIR
        
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # specific check for VisioNova column names
    if 'generated' in df.columns:
        df['label'] = df['generated'].astype(int) # 1 for AI
    elif 'label' not in df.columns:
        print("Error: CSV must have 'text' and 'label' columns (or 'generated')")
        return
        
    # Ensure text is string
    df['text'] = df['text'].astype(str)
    
    # Split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    dataset_train = Dataset.from_pandas(train_df)
    dataset_val = Dataset.from_pandas(val_df)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    # Load Model & Tokenizer
    print(f"Loading model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=2)
    
    print("Tokenizing manually to avoid map crash...", flush=True)
    
    def process_data(df):
        texts = df['text'].astype(str).tolist()
        labels = df['label'].tolist()
        
        # Tokenize all at once
        encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=512)
        
        # Convert to dataset
        dataset_dict = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'label': labels
        }
        return Dataset.from_dict(dataset_dict)

    tokenized_train = process_data(train_df)
    print("Training set tokenized.", flush=True)
    
    tokenized_val = process_data(val_df)
    print("Validation set tokenized.", flush=True)
    
    print("Dataset preparation complete.", flush=True)
    
    print("Dataset mapping complete. Creating DataCollator...", flush=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    print("Defining TrainingArguments...", flush=True)
    try:
        # Training Arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(BASE_DIR, "checkpoints"),
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            eval_strategy="epoch", # Renamed from evaluation_strategy in 4.42+
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
            # report_to="none",  # Commenting out potential problematic args
            dataloader_num_workers=0,  # Critical for Windows
            save_total_limit=2,  # Save space
            logging_steps=100,
            no_cuda=True, # Force CPU for debugging stability
            use_mps_device=False,
        )
        print("TrainingArguments defined successfully.", flush=True)
    except Exception as e:
        print(f"CRITICAL ERROR in TrainingArguments: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return
    
    print("Initializing Trainer...", flush=True)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    print("Trainer initialized successfully.", flush=True)
    
    print("Starting training...")
    try:
        trainer.train()
        print(f"Saving fine-tuned model to {output_dir}...")
        trainer.save_model(output_dir)
        print("Done!")
    except Exception as e:
        print(f"\nCRITICAL ERROR DURING TRAINING: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VisioNova AI Detector")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    
    args = parser.parse_args()
    
    if os.path.exists(args.data_path):
        train(args.data_path, epochs=args.epochs)
    else:
        print(f"File not found: {args.data_path}")
