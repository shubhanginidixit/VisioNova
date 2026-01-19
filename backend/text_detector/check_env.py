"""
Diagnostic script to check environment health.
"""
print("IMPORTS START", flush=True)

try:
    import torch
    print(f"Torch imported: {torch.__version__}", flush=True)
except ImportError as e:
    print(f"Torch failed: {e}")

try:
    from transformers import Trainer, TrainingArguments
    print(f"Transformers imported: Trainer found", flush=True)
except ImportError as e:
    print(f"Transformers failed: {e}")

try:
    from accelerate import Accelerator
    print(f"Accelerate imported", flush=True)
except ImportError as e:
    print(f"Accelerate failed: {e}")
    
try:
    from datasets import Dataset
    print(f"Datasets imported", flush=True)
except ImportError as e:
    print(f"Datasets failed: {e}")

print("IMPORTS END", flush=True)
