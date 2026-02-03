
import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --- DEVICE CONFIGURATION ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ GPU NOT DETECTED! Switching to CPU mode.")
    print("Training will be significantly slower. For real training, use a GPU runtime.")
    device = torch.device("cpu")

# --- CONFIG ---
DATASET_NAME = "artem9k/ai-text-detection-pile"
MODEL_ID = "microsoft/deberta-v3-base"
SAVE_DIR_NAME = f"{DATASET_NAME.split('/')[-1]}_DeBERTa_v3"
OUTPUT_DIR = os.path.join("./VisioNova_Models", SAVE_DIR_NAME)

# --- HYPERPARAMETERS ---
EPOCHS = 3
# Reduce batch size for CPU compatibility
BATCH_SIZE = 8 if torch.cuda.is_available() else 2
LEARNING_RATE = 2e-5
MAX_LEN = 512

print(f"Model will be saved to: {OUTPUT_DIR}")
print(f"Batch Size: {BATCH_SIZE}")

# 1. LOAD & PREPARE DATASET
print(f"Loading dataset: {DATASET_NAME}...")
try:
    dataset = load_dataset(DATASET_NAME)
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Check your internet connection or HuggingFace login.")
    exit(1)

# Debug info
print(f"Dataset splits: {list(dataset.keys())}")
if 'train' in dataset:
    split_dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)
else:
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

print(f"Training Samples: {len(split_dataset['train'])}")
print(f"Validation Samples: {len(split_dataset['test'])}")

# 2. CREATE LABELS
HUMAN_SOURCES = {'human', 'Human', 'wikipedia', 'reddit', 'news', 'books'}

def add_labels(example):
    """Add binary label based on source column."""
    source = example.get('source', '').lower()
    is_human = any(h.lower() in source for h in HUMAN_SOURCES)
    example['labels'] = 0 if is_human else 1
    return example

print("Adding labels based on 'source' column...")
split_dataset = split_dataset.map(add_labels)

# 3. TOKENIZATION
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=MAX_LEN
    )

print("Tokenizing dataset...")
tokenized_datasets = split_dataset.map(tokenize_function, batched=True)

# Set columns
required_cols = ["input_ids", "attention_mask", "labels"]
if "token_type_ids" in tokenized_datasets['train'].column_names:
    required_cols.insert(1, "token_type_ids")

tokenized_datasets.set_format(type="torch", columns=required_cols)

# 4. MODEL & TRAINER
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, 
    num_labels=2,
    id2label={0: "HUMAN", 1: "AI"},
    label2id={"HUMAN": 0, "AI": 1}
)

training_args = TrainingArguments(
    output_dir="./checkpoints",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(), # Disable fp16 on CPU
    no_cuda=not torch.cuda.is_available(),
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

# 5. TRAIN
print("Starting training...")
trainer.train()

print(f"\nSaving model to: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save metrics
metrics = trainer.evaluate()
print(f"Final Metrics: {metrics}")
with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
    f.write(str(metrics))

print("\n✅ Training Complete!")
