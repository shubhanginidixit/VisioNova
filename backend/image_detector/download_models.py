"""
VisioNova Model Downloader
Downloads pre-trained AI detection models (DIRE + UniversalFakeDetect)
Total size: ~2.6GB
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm
import torch

# Model directory
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODELS = {
    "dire": {
        "name": "DIRE (Diffusion Reconstruction Error)",
        "url": "https://huggingface.co/DIRE/DIRE/resolve/main/aire_model.pth",
        "filename": "dire_model.pth",
        "size": "1.8GB",
        "description": "Best for Stable Diffusion, DALL-E 3, Midjourney v6"
    },
    "universal": {
        "name": "UniversalFakeDetect",
        "url": "https://huggingface.co/umm-maybe/AI-image-detector/resolve/main/model.pth",
        "filename": "universal_detector.pth",
        "size": "850MB",
        "description": "Works across all AI generators"
    }
}


def download_file(url: str, filepath: Path, description: str = "Downloading"):
    """Download file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=description) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"✓ Downloaded: {filepath.name}")
        return True
        
    except Exception as e:
        print(f"✗ Error downloading {filepath.name}: {e}")
        if filepath.exists():
            filepath.unlink()
        return False


def check_gpu():
    """Check if CUDA is available."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ GPU detected: {gpu_name}")
        print(f"  CUDA version: {torch.version.cuda}")
        return True
    else:
        print("⚠ No GPU detected. Models will run on CPU (slower).")
        return False


def download_models(models_to_download=None):
    """
    Download specified models.
    
    Args:
        models_to_download: List of model keys to download, or None for all
    """
    print("=" * 60)
    print("VisioNova AI Detection Model Downloader")
    print("=" * 60)
    
    # Check GPU
    check_gpu()
    print()
    
    # Determine which models to download
    if models_to_download is None:
        models_to_download = list(MODELS.keys())
    
    total_downloaded = 0
    total_failed = 0
    
    for model_key in models_to_download:
        if model_key not in MODELS:
            print(f"⚠ Unknown model: {model_key}")
            continue
        
        model_info = MODELS[model_key]
        filepath = MODEL_DIR / model_info["filename"]
        
        print(f"\n📦 {model_info['name']}")
        print(f"   Size: {model_info['size']}")
        print(f"   Description: {model_info['description']}")
        
        # Check if already downloaded
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"   ✓ Already downloaded ({size_mb:.1f}MB)")
            print(f"   Location: {filepath}")
            total_downloaded += 1
            continue
        
        # Download
        print(f"   Downloading from Hugging Face...")
        if download_file(model_info["url"], filepath, f"Downloading {model_info['name']}"):
            total_downloaded += 1
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"   ✓ Download complete ({size_mb:.1f}MB)")
            print(f"   Location: {filepath}")
        else:
            total_failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"✓ Successfully downloaded: {total_downloaded}")
    if total_failed > 0:
        print(f"✗ Failed: {total_failed}")
    print(f"\nModels saved to: {MODEL_DIR}")
    print("\nNext steps:")
    print("1. Models are ready to use")
    print("2. Restart your Flask server to load the models")
    print("3. Test with an AI-generated image")
    print("=" * 60)


def verify_models():
    """Verify downloaded models can be loaded."""
    print("\n🔍 Verifying models...")
    
    for model_key, model_info in MODELS.items():
        filepath = MODEL_DIR / model_info["filename"]
        
        if not filepath.exists():
            print(f"✗ {model_info['name']}: Not found")
            continue
        
        try:
            # Try loading with torch
            state_dict = torch.load(filepath, map_location='cpu')
            print(f"✓ {model_info['name']}: Valid PyTorch model")
        except Exception as e:
            print(f"✗ {model_info['name']}: Error loading - {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download AI detection models")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()) + ["all"],
        default="all",
        help="Models to download (default: all)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify downloaded models"
    )
    
    args = parser.parse_args()
    
    # Determine models to download
    if args.models == "all" or "all" in args.models:
        models_to_download = None
    else:
        models_to_download = args.models
    
    # Download
    download_models(models_to_download)
    
    # Verify if requested
    if args.verify:
        verify_models()
