"""
VisioNova — Download Top 5 Image Detection Models

Run this script ONCE to pre-download all 5 HuggingFace model weights
to your local cache (~/.cache/huggingface/). After downloading, the
models load from disk without internet access.

Usage:
    python download_image_models.py

Estimated total download size: ~2-3 GB
Estimated time: 5-15 minutes on a typical connection
"""

import sys
import time


# Top 5 models used by VisioNova's image detection ensemble
MODELS = [
    # 1. SigLIP2+DINOv2 Ensemble (BEST OVERALL — 99.97% AUC)
    {
        "id": "Bombek1/ai-image-detector-siglip-dinov2",
        "name": "SigLIP2+DINOv2 Ensemble",
        "accuracy": "99.97% AUC",
        "params": "~740M",
        "note": "Custom architecture — downloads pytorch_model.pt + backbones",
    },
    # 2. Ateeqq SigLIP2 (99.23% accuracy)
    {
        "id": "Ateeqq/ai-vs-human-image-detector",
        "name": "Ateeqq SigLIP2",
        "accuracy": "99.23%",
        "params": "92.9M",
    },
    # 3. dima806 Deepfake ViT (98.25% accuracy)
    {
        "id": "dima806/deepfake_vs_real_image_detection",
        "name": "Deepfake ViT",
        "accuracy": "98.25%",
        "params": "86M",
    },
    # 4. Organika SDXL Swin Transformer (98.1% accuracy)
    {
        "id": "Organika/sdxl-detector",
        "name": "SDXL Swin Transformer",
        "accuracy": "98.1%",
        "params": "87M",
    },
    # 5. WpythonW DINOv2 Deepfake (degradation-resilient, ~95%)
    {
        "id": "WpythonW/dinoV2-deepfake-detector",
        "name": "DINOv2 Deepfake",
        "accuracy": "~95%",
        "params": "~86M",
    },
]


def download_standard_model(model_info: dict) -> bool:
    """Download a standard HuggingFace AutoModel."""
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    
    model_id = model_info["id"]
    print(f"  Downloading processor...")
    AutoImageProcessor.from_pretrained(model_id)
    print(f"  Downloading model weights...")
    AutoModelForImageClassification.from_pretrained(model_id)
    return True


def download_bombek1_model(model_info: dict) -> bool:
    """Download the Bombek1 custom model (needs special handling)."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoProcessor
    import timm
    
    repo_id = model_info["id"]
    
    # Download the custom checkpoint
    print(f"  Downloading pytorch_model.pt...")
    hf_hub_download(repo_id=repo_id, filename="pytorch_model.pt")
    
    # Download the SigLIP2 backbone
    print(f"  Downloading SigLIP2 backbone (google/siglip2-so400m-patch14-384)...")
    AutoProcessor.from_pretrained("google/siglip2-so400m-patch14-384")
    
    # Download the DINOv2 backbone via timm
    print(f"  Downloading DINOv2 backbone (vit_large_patch14_dinov2.lvd142m)...")
    timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
    
    return True


def main():
    print("=" * 60)
    print("VisioNova — Top 5 AI Image Detection Model Downloader")
    print("=" * 60)
    print(f"\nTotal models to download: {len(MODELS)}")
    print("Estimated size: ~2-3 GB\n")
    
    successes = 0
    failures = []
    
    for i, model in enumerate(MODELS, 1):
        model_id = model["id"]
        name = model["name"]
        accuracy = model["accuracy"]
        
        print(f"\n[{i}/{len(MODELS)}] {name}")
        print(f"  Model ID: {model_id}")
        print(f"  Accuracy: {accuracy}, Params: {model['params']}")
        
        start = time.time()
        try:
            # Bombek1 needs special download logic
            if "Bombek1" in model_id:
                download_bombek1_model(model)
            else:
                download_standard_model(model)
            
            elapsed = time.time() - start
            print(f"  ✅ Downloaded in {elapsed:.1f}s")
            successes += 1
            
        except Exception as e:
            elapsed = time.time() - start
            print(f"  ❌ FAILED after {elapsed:.1f}s: {e}")
            failures.append((name, str(e)))
    
    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {successes}/{len(MODELS)} models downloaded successfully")
    
    if failures:
        print(f"\n[WARN] {len(failures)} model(s) failed:")
        for name, error in failures:
            print(f"  - {name}: {error}")
        print("\nYou can re-run this script to retry failed downloads.")
    else:
        print("\n🎉 All models downloaded! You can now run VisioNova offline.")
    
    print("=" * 60)
    
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
