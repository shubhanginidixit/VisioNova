"""
Train RoBERTa-based AI text detection model.
This model is designed to work on GPT-5 and all future AI models by learning
universal AI characteristics across multiple model architectures.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch


# Training configuration
CONFIG = {
    "model_name": "roberta-base",  # Better than DistilBERT for classification
    "num_labels": 2,  # Binary: human vs AI
    "max_length": 512,
    "learning_rate": 2e-5,
    "batch_size": 16,  # Adjust based on GPU memory
    "num_epochs": 3,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "fp16": True,  # Mixed precision training (faster)
    "early_stopping_patience": 3,
    "eval_steps": 500,
    "save_steps": 500,
    "logging_steps": 100,
}


class AIDetectionTrainer:
    """Trainer for AI text detection model."""
    
    def __init__(self, data_dir: str = "datasets", output_dir: str = "model_trained"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def load_datasets(self):
        """Load train/val/test datasets."""
        
        print("="*60)
        print("Loading Datasets")
        print("="*60)
        
        train_df = pd.read_csv(self.data_dir / "train.csv")
        val_df = pd.read_csv(self.data_dir / "val.csv")
        test_df = pd.read_csv(self.data_dir / "test.csv")
        
        print(f"\n✓ Train: {len(train_df)} samples")
        print(f"✓ Val:   {len(val_df)} samples")
        print(f"✓ Test:  {len(test_df)} samples")
        
        # Convert labels to integers (0=human, 1=ai)
        label_map = {"human": 0, "ai": 1}
        train_df['label'] = train_df['label'].map(label_map)
        val_df['label'] = val_df['label'].map(label_map)
        test_df['label'] = test_df['label'].map(label_map)
        
        # Create HuggingFace datasets
        dataset_dict = DatasetDict({
            'train': Dataset.from_pandas(train_df[['text', 'label']]),
            'validation': Dataset.from_pandas(val_df[['text', 'label']]),
            'test': Dataset.from_pandas(test_df[['text', 'label']])
        })
        
        return dataset_dict
    
    def prepare_model(self):
        """Initialize tokenizer and model."""
        
        print("\n" + "="*60)
        print("Preparing Model")
        print("="*60)
        
        print(f"\n✓ Loading tokenizer: {CONFIG['model_name']}")
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
        
        print(f"✓ Loading model: {CONFIG['model_name']}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            CONFIG['model_name'],
            num_labels=CONFIG['num_labels']
        )
        
        # Label mapping
        self.model.config.label2id = {"human": 0, "ai": 1}
        self.model.config.id2label = {0: "human", 1: "ai"}
        
        print(f"✓ Model loaded with {self.model.num_parameters():,} parameters")
        
    def tokenize_function(self, examples):
        """Tokenize text for model input."""
        return self.tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=CONFIG['max_length']
        )
    
    def compute_metrics(self, eval_pred):
        """Calculate evaluation metrics."""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )
        acc = accuracy_score(labels, predictions)
        
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, dataset_dict):
        """Train the model."""
        
        print("\n" + "="*60)
        print("Training Configuration")
        print("="*60)
        
        for key, value in CONFIG.items():
            print(f"  {key}: {value}")
        
        # Tokenize datasets
        print("\n✓ Tokenizing datasets...")
        tokenized_datasets = dataset_dict.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['text']
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            evaluation_strategy="steps",
            eval_steps=CONFIG['eval_steps'],
            save_strategy="steps",
            save_steps=CONFIG['save_steps'],
            learning_rate=CONFIG['learning_rate'],
            per_device_train_batch_size=CONFIG['batch_size'],
            per_device_eval_batch_size=CONFIG['batch_size'],
            num_train_epochs=CONFIG['num_epochs'],
            weight_decay=CONFIG['weight_decay'],
            warmup_steps=CONFIG['warmup_steps'],
            logging_steps=CONFIG['logging_steps'],
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            fp16=CONFIG['fp16'],
            push_to_hub=False,
            save_total_limit=2,  # Keep only 2 best checkpoints
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation'],
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=CONFIG['early_stopping_patience'])]
        )
        
        # Train
        print("\n" + "="*60)
        print("🚀 Starting Training...")
        print("="*60)
        print("\n⏱ This will take 2-3 hours on Colab free GPU")
        print("⏱ Or 5-10 hours on CPU\n")
        
        train_result = self.trainer.train()
        
        print("\n" + "="*60)
        print("✅ Training Complete!")
        print("="*60)
        
        return train_result
    
    def evaluate(self, dataset_dict):
        """Evaluate on test set."""
        
        print("\n" + "="*60)
        print("📊 Final Evaluation on Test Set")
        print("="*60)
        
        # Tokenize test set
        tokenized_test = dataset_dict['test'].map(
            self.tokenize_function,
            batched=True,
            remove_columns=['text']
        )
        
        # Evaluate
        eval_results = self.trainer.evaluate(tokenized_test)
        
        print("\n🎯 Test Set Results:")
        print(f"  Accuracy:  {eval_results['eval_accuracy']:.4f} ({eval_results['eval_accuracy']*100:.2f}%)")
        print(f"  F1 Score:  {eval_results['eval_f1']:.4f}")
        print(f"  Precision: {eval_results['eval_precision']:.4f}")
        print(f"  Recall:    {eval_results['eval_recall']:.4f}")
        
        # Check for overfitting
        if eval_results['eval_accuracy'] > 0.95:
            print("\n⚠️  WARNING: Accuracy > 95% - Possible overfitting!")
            print("   Consider reducing epochs or adding regularization")
        elif eval_results['eval_accuracy'] < 0.80:
            print("\n⚠️  WARNING: Accuracy < 80% - Model may need more training")
            print("   Consider increasing epochs or adjusting learning rate")
        else:
            print("\n✅ Accuracy in healthy range (80-95%)")
        
        return eval_results
    
    def save_model(self, final_dir: str = "model"):
        """Save the trained model."""
        
        print("\n" + "="*60)
        print("💾 Saving Model")
        print("="*60)
        
        save_path = Path(final_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        
        self.trainer.save_model(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))
        
        # Save training results
        results_file = save_path / "training_results.json"
        import json
        with open(results_file, 'w') as f:
            json.dump(self.trainer.state.log_history[-1], f, indent=2)
        
        print(f"\n✅ Model saved to: {save_path}")
        print(f"   Files: config.json, model.safetensors, tokenizer files")
        
        return save_path


def main():
    """Main training pipeline."""
    
    print("\n" + "🤖 "*20)
    print("VisioNova AI Text Detection Model Training")
    print("Future-Proof Multi-Model Architecture")
    print("🤖 "*20 + "\n")
    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Device: {device.upper()}")
    if device == "cpu":
        print("⚠️  Training on CPU will be slow (~10 hours)")
        print("   Consider using Google Colab for free GPU\n")
    
    try:
        trainer = AIDetectionTrainer()
        
        # Step 1: Load datasets
        dataset_dict = trainer.load_datasets()
        
        # Step 2: Prepare model
        trainer.prepare_model()
        
        # Step 3: Train
        train_result = trainer.train(dataset_dict)
        
        # Step 4: Evaluate
        eval_results = trainer.evaluate(dataset_dict)
        
        # Step 5: Save model
        model_path = trainer.save_model()
        
        print("\n" + "="*60)
        print("🎉 Training Pipeline Complete!")
        print("="*60)
        print(f"\n📁 Model saved to: {model_path}")
        print(f"🎯 Test Accuracy: {eval_results['eval_accuracy']*100:.2f}%")
        print(f"📊 Test F1 Score: {eval_results['eval_f1']:.4f}")
        
        print("\n✨ Next steps:")
        print("1. Copy the model files to backend/text_detector/model/")
        print("2. Update detector.py to use RoBERTa")
        print("3. Test with real GPT-4, Claude, and human samples")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
