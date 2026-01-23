"""
VisioNova Enhanced Model Trainer v2
Fine-tunes AI text detection with modern best practices.

Key Improvements:
- Gradient accumulation for larger effective batch sizes
- Learning rate warmup and decay
- Early stopping to prevent overfitting
- Mixed precision training (when GPU available)
- Better logging and progress tracking

Usage:
    python train_v2.py --data_path datasets/ai_detection_multi_model.csv
    python train_v2.py --data_path datasets/ai_detection_multi_model.csv --epochs 5 --batch_size 16
"""
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# Fix for Windows Tokenizer Deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import functools
print = functools.partial(print, flush=True)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOGS_DIR = os.path.join(BASE_DIR, "training_logs")


def compute_metrics(pred):
    """
    Computes evaluation metrics for the model.
    
    Returns accuracy, F1, precision, recall.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    acc = accuracy_score(labels, preds)
    
    # Confusion matrix for detailed analysis
    cm = confusion_matrix(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'true_negatives': int(cm[0][0]) if len(cm) > 0 else 0,
        'false_positives': int(cm[0][1]) if len(cm) > 0 else 0,
        'false_negatives': int(cm[1][0]) if len(cm) > 1 else 0,
        'true_positives': int(cm[1][1]) if len(cm) > 1 else 0,
    }


def train_v2(
    data_path: str,
    output_dir: str = None,
    model_name: str = "distilbert-base-uncased",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    gradient_accumulation_steps: int = 4,
    early_stopping_patience: int = 3,
    use_gpu: bool = True
):
    """
    Enhanced training function with modern best practices.
    
    Logic Flow:
    1. Load and validate the CSV dataset
    2. Split into train/validation sets (80/20)
    3. Tokenize text with the model's tokenizer
    4. Configure training with warmup, early stopping, and logging
    5. Train and save the best model
    
    Args:
        data_path: Path to CSV file with 'text' and 'label' columns
        output_dir: Where to save the fine-tuned model
        epochs: Number of training epochs
        batch_size: Batch size per device
        learning_rate: Initial learning rate
        max_length: Maximum token length for text
        warmup_ratio: Fraction of steps for learning rate warmup
        weight_decay: Weight decay for regularization
        gradient_accumulation_steps: Accumulate gradients for larger effective batch
        early_stopping_patience: Stop if no improvement for N evaluations
        use_gpu: Whether to use GPU if available
    """
    if not output_dir:
        output_dir = MODEL_DIR
    
    # Create directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"visionova_train_{timestamp}"
    
    print("=" * 70)
    print("VisioNova Enhanced Model Trainer v2")
    print("=" * 70)
    print(f"Run ID: {run_name}")
    print(f"Data: {data_path}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size} (effective: {batch_size * gradient_accumulation_steps})")
    print(f"Learning Rate: {learning_rate}")
    print(f"Max Length: {max_length}")
    print("=" * 70)
    
    # =========================================================================
    # STEP 1: Load and validate data
    # =========================================================================
    print("\n[1/5] Loading dataset...")
    
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"ERROR: Could not read CSV file: {e}")
        return None
    
    # Handle different column naming conventions
    if 'generated' in df.columns and 'label' not in df.columns:
        df['label'] = df['generated'].astype(int)
    
    if 'text' not in df.columns or 'label' not in df.columns:
        print("ERROR: CSV must have 'text' and 'label' columns")
        print(f"       Found columns: {list(df.columns)}")
        return None
    
    # Clean data
    df['text'] = df['text'].astype(str).str.strip()
    df['label'] = df['label'].astype(int)
    df = df[df['text'].str.len() > 50]  # Remove very short texts
    
    print(f"       Total samples: {len(df)}")
    print(f"       Human (0): {len(df[df['label'] == 0])}")
    print(f"       AI (1): {len(df[df['label'] == 1])}")
    
    # =========================================================================
    # STEP 2: Split data
    # =========================================================================
    print("\n[2/5] Splitting into train/validation sets...")
    
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42,
        stratify=df['label']  # Maintain class balance in splits
    )
    
    print(f"       Training samples: {len(train_df)}")
    print(f"       Validation samples: {len(val_df)}")
    
    # =========================================================================
    # STEP 3: Load model and tokenize
    # =========================================================================
    print("\n[3/5] Loading model and tokenizing...")
    
    # Check if we should load from a local directory or HuggingFace
    load_path = model_name
    if os.path.isdir(os.path.join(BASE_DIR, model_name)):
        load_path = os.path.join(BASE_DIR, model_name)
    
    print(f"       Loading from: {load_path}")
    tokenizer = AutoTokenizer.from_pretrained(load_path)
    model = AutoModelForSequenceClassification.from_pretrained(load_path, num_labels=2)
    
    # Check device
    if use_gpu and torch.cuda.is_available():
        device = "cuda"
        print(f"       Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print(f"       Using CPU")
    
    def tokenize_batch(df):
        """Tokenize a DataFrame of texts."""
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        
        encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors=None
        )
        
        return Dataset.from_dict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'label': labels
        })
    
    print("       Tokenizing training set...")
    train_dataset = tokenize_batch(train_df)
    
    print("       Tokenizing validation set...")
    val_dataset = tokenize_batch(val_df)
    
    print(f"       Tokenization complete!")
    
    # =========================================================================
    # STEP 4: Configure training
    # =========================================================================
    print("\n[4/5] Configuring training...")
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        run_name=run_name,
        
        # Training hyperparameters
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        
        # Evaluation and saving
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=3,
        
        # Logging
        logging_dir=os.path.join(LOGS_DIR, run_name),
        logging_steps=50,
        report_to="none",
        
        # Performance
        dataloader_num_workers=0,  # Critical for Windows
        no_cuda=(device == "cpu"),
        fp16=(device == "cuda"),  # Mixed precision on GPU
        
        # Misc
        push_to_hub=False,
        seed=42,
    )
    
    # Early stopping callback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=0.001
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping]
    )
    
    print("       Training configuration complete!")
    
    # =========================================================================
    # STEP 5: Train!
    # =========================================================================
    print("\n[5/5] Starting training...")
    print("-" * 70)
    
    try:
        trainer.train()
        
        print("-" * 70)
        print("\nTraining complete! Evaluating final model...")
        
        # Final evaluation
        eval_results = trainer.evaluate()
        
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        print(f"Accuracy:  {eval_results.get('eval_accuracy', 0):.4f}")
        print(f"F1 Score:  {eval_results.get('eval_f1', 0):.4f}")
        print(f"Precision: {eval_results.get('eval_precision', 0):.4f}")
        print(f"Recall:    {eval_results.get('eval_recall', 0):.4f}")
        print("=" * 70)
        
        # Save the model
        print(f"\nSaving fine-tuned model to {output_dir}...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save training log
        log_file = os.path.join(LOGS_DIR, f"{run_name}_results.txt")
        with open(log_file, 'w') as f:
            f.write(f"Run: {run_name}\n")
            f.write(f"Data: {data_path}\n")
            f.write(f"Samples: {len(df)}\n")
            f.write(f"Epochs: {epochs}\n")
            f.write(f"Results:\n")
            for k, v in eval_results.items():
                f.write(f"  {k}: {v}\n")
        
        print(f"Training log saved to: {log_file}")
        print("\n✅ SUCCESS! Model is ready to use.")
        
        return eval_results
        
    except Exception as e:
        print(f"\n❌ ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VisioNova AI Detector v2")
    
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to CSV dataset")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size per device (default: 8)")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate (default: 2e-5)")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum token length (default: 512)")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased",
                        help="Base model name or path (default: distilbert-base-uncased)")
    parser.add_argument("--no_gpu", action="store_true",
                        help="Force CPU training")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"ERROR: File not found: {args.data_path}")
        exit(1)
    
    train_v2(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        model_name=args.model_name,
        use_gpu=not args.no_gpu
    )
