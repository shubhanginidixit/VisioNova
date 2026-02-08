"""
VisioNova ML Models Setup Script
Uses HuggingFace models (no manual downloads needed)
Requires Python 3.10 environment (.venv310)
"""

import sys
import subprocess
from pathlib import Path

print("=" * 70)
print("VisioNova ML Models Setup (Python 3.10 + CUDA)")
print("=" * 70)
print()

# Check if in Python 3.10 virtual environment
venv_path = Path(__file__).parent.parent / ".venv310"
if venv_path.exists():
    print(f"✓ Python 3.10 virtual environment found: {venv_path}")
else:
    print("⚠ Python 3.10 virtual environment not found!")
    print("  Please activate: .venv310\\Scripts\\Activate.ps1")
    print("  Or create it: py -3.10 -m venv .venv310")
    sys.exit(1)

# Verify Python version
python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
if sys.version_info.major != 3 or sys.version_info.minor != 10:
    print(f"⚠ Wrong Python version: {python_version}")
    print("  This script requires Python 3.10")
    print("  Please activate .venv310")
    sys.exit(1)

print(f"✓ Python version: {python_version}")
print()

# Step 1: Check CUDA availability
print("Step 1: Checking PyTorch and CUDA...")
print("-" * 70)

try:
    import torch
    print(f"  ✓ PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available: YES")
        print(f"  ✓ CUDA version: {torch.version.cuda}")
        print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  ⚠ CUDA not available - models will run on CPU")
        print("    Install: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
except ImportError:
    print("  ✗ PyTorch not installed!")
    print("    Install: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)

print()

# Step 2: Install HuggingFace dependencies
print("Step 2: Checking HuggingFace dependencies...")
print("-" * 70)

required_packages = {
    'transformers': 'transformers>=4.30.0',
    'pillow': 'pillow>=10.0.0',
    'numpy': 'numpy',
    'cv2': 'opencv-python'
}

missing = []
for module, package in required_packages.items():
    try:
        __import__(module.replace('-', '_'))
        print(f"  ✓ {module}")
    except ImportError:
        print(f"  ✗ {module} - MISSING")
        missing.append(package)

if missing:
    print()
    print(f"Installing: {', '.join(missing)}")
    subprocess.run([sys.executable, "-m", "pip", "install"] + missing)
else:
    print("  ✓ All dependencies installed")

print()

# Step 3: Download Stable Signature watermark decoder
print("Step 3: Downloading Meta Stable Signature model...")
print("-" * 70)
print("This decoder detects 48-bit watermarks from Meta AI generators")
print()

try:
    import urllib.request
    import os
    
    model_dir = Path(__file__).parent.parent / "backend" / "image_detector" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "stable_signature_decoder.pt"
    
    if model_path.exists():
        print(f"  \u2713 Stable Signature model already exists: {model_path}")
    else:
        model_url = "https://dl.fbaipublicfiles.com/ssl_watermarking/dec_48b_whit.torchscript.pt"
        print(f"  Downloading from: {model_url}")
        print(f"  Saving to: {model_path}")
        
        urllib.request.urlretrieve(model_url, str(model_path))
        
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"  \u2713 Model downloaded successfully ({size_mb:.1f} MB)")
        else:
            print("  \u2717 Download failed")
            
except Exception as e:
    print(f"  \u2717 Error downloading Stable Signature model: {e}")
    print("  Model will not be available for watermark detection")

print()

# Step 4: Verify watermark libraries
print("Step 4: Verifying watermark detection libraries...")
print("-" * 70)

try:
    from imwatermark import WatermarkDecoder
    test_decoder = WatermarkDecoder('bytes', 32)
    print("  \u2713 invisible-watermark library installed and working")
except ImportError:
    print("  \u2717 invisible-watermark not installed")
    print("    Install with: pip install invisible-watermark")
except Exception as e:
    print(f"  \u26a0 invisible-watermark installed but initialization failed: {e}")

try:
    import c2pa
    print("  \u2713 c2pa-python library installed")
except ImportError:
    print("  \u26a0 c2pa-python not installed (optional)")
    print("    Install with: pip install c2pa-python")

print()

# Step 5: Download HuggingFace models
print("Step 5: Downloading HuggingFace AI detection models...")
print("-" * 70)
print("Models will be cached in: ~/.cache/huggingface/")
print()
print("Models to download:")
print("  • NYUAD AI Detector (ViT-based, 97.36% accuracy)")
print("  • CLIP ViT-L/14 (for UniversalFakeDetect)")
print("  • dima806 AI Detector (98.25% accuracy, alternative)")
print()

response = input("Download models now? (y/n): ")
if response.lower() in ['y', 'yes']:
    print()
    
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification, CLIPProcessor, CLIPModel
        
        # NYUAD model
        print("[1/3] Downloading NYUAD AI Detector...")
        try:
            processor1 = AutoImageProcessor.from_pretrained("NYUAD-ComNets/NYUAD_AI-generated_images_detector")
            model1 = AutoModelForImageClassification.from_pretrained("NYUAD-ComNets/NYUAD_AI-generated_images_detector")
            print("  ✓ NYUAD model cached")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
        
        # CLIP model
        print("\n[2/3] Downloading CLIP (for UniversalFakeDetect)...")
        try:
            processor2 = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            model2 = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            print("  ✓ CLIP model cached")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
        
        # dima806 model
        print("\n[3/3] Downloading dima806 AI Detector...")
        try:
            processor3 = AutoImageProcessor.from_pretrained("dima806/ai_vs_real_image_detection")
            model3 = AutoModelForImageClassification.from_pretrained("dima806/ai_vs_real_image_detection")
            print("  ✓ dima806 model cached")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
        
        print("\n✓ Model download complete!")
        
    except ImportError as e:
        print(f"  ✗ Error: {e}")
        print("  Install transformers: pip install transformers")
    except Exception as e:
        print(f"  ✗ Error downloading models: {e}")
else:
    print("Skipped. Models will auto-download on first use.")

print()

# Step 6: Test ML detectors
print("Step 6: Testing ML detector integration...")
print("-" * 70)

try:
    from image_detector.ml_detector import create_ml_detectors
    
    print("Loading detectors...")
    detectors = create_ml_detectors(device="auto", load_all=False)
    
    loaded_count = 0
    
    if detectors.get('nyuad') and detectors['nyuad'].model_loaded:
        print("  ✓ NYUAD detector loaded (97.36% accuracy)")
        loaded_count += 1
    else:
        print("  ⚠ NYUAD detector not loaded")
    
    if detectors.get('clip') and detectors['clip'].model_loaded:
        print("  ✓ CLIP detector loaded (UniversalFakeDetect)")
        loaded_count += 1
    else:
        print("  ⚠ CLIP detector not loaded")
    
    if loaded_count > 0:
        print(f"\n✓ Setup complete! {loaded_count} model(s) loaded")
    else:
        print("\n⚠ No models loaded - will use statistical analysis")
    
except Exception as e:
    print(f"  ✗ Error loading detectors: {e}")
    import traceback
    traceback.print_exc()
    print()
    print("Troubleshooting:")
    print("1. Make sure PyTorch is installed: pip list | findstr torch")
    print("2. Check transformers: pip install transformers")
    print("3. Verify Python 3.10: python --version")

print()
print("=" * 70)
print("Setup Complete!")
print("=" * 70)
print("Next Steps:")
print("  1. Activate Python 3.10: .venv310\\Scripts\\Activate.ps1")
print("  2. Start Flask server: python backend/app.py")
print("  3. Upload an AI image to test")
print()
print("Expected log message:")
print("  ✓ NYUAD ML model loaded - 97% accuracy")
print("=" * 70)
