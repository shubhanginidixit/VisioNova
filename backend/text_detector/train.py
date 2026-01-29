
import os
import argparse
import numpy as np
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Constants
MODEL_NAME = "microsoft/deberta-v3-base"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
DEFAULT_EPOCHS = 3
BATCH_SIZE = 8
MAX_LENGTH = 512

def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train(args):
    """
    Main training function.
    """
    print(f"Starting training with model: {MODEL_NAME}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # 1. Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 2. Load Dataset
    # Ideally, this should load from RAID, WildChat, or a local JSON file.
    # For now, we support loading a local JSON dataset if provided, else dummy data for verification.
    if args.dataset_path and os.path.exists(args.dataset_path):
        print(f"Loading dataset from {args.dataset_path}...")
        extension = args.dataset_path.split('.')[-1]
        data_files = {"train": args.dataset_path}
        dataset = load_dataset(extension, data_files=data_files)
        # Assuming dataset has 'text' and 'label' columns
        if 'train' in dataset:
            dataset = dataset['train'].train_test_split(test_size=0.2)
    else:
        print("No dataset provided or file not found. Using DUMMY data for verification.")
        dummy_data = {
            "text": [
                "This is a human written text example.", 
                "As an AI language model, I cannot help with that.",
                "The quick brown fox jumps over the lazy dog.",
                "I apologize, but I am unable to fulfill this request."
            ] * 25, # 100 samples
            "label": [0, 1, 0, 1] * 25
        }
        dataset = Dataset.from_dict(dummy_data).train_test_split(test_size=0.2)
        
    print(f"Dataset Loaded: {dataset}")

    # 3. Preprocess Dataset
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 4. Initialize Model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2,
        id2label={0: "human", 1: "ai"},
        label2id={"human": 0, "ai": 1}
    )

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        logging_dir='./logs',
        no_cuda=args.no_cuda,  # Support CPU-only training
        report_to="none"
    )

    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 7. Train
    if args.dry_run:
        print("Dry Run: verifying pipeline setup...")
        # Just run evaluation to check model loading and data processing
        trainer.evaluate()
        print("Dry run complete. Pipeline is functional.")
    else:
        print("Starting training loop...")
        trainer.train()
        print("Training complete.")
        
        # 8. Save Model
        print(f"Saving model to {OUTPUT_DIR}...")
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print("Model saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeBERTa-v3 for AI Text Detection")
    parser.add_argument("--dataset_path", type=str, help="Path to local JSON/CSV dataset")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--dry-run", action="store_true", help="Run verification without full training")
    parser.add_argument("--no-cuda", action="store_true", help="Force CPU training")
    
    args = parser.parse_args()
    
    try:
        train(args)
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()