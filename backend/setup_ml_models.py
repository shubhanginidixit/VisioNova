"""
VisioNova ML Models Setup Script
Automates downloading and testing DIRE + UniversalFakeDetect models
"""

import sys
import subprocess
from pathlib import Path

print("=" * 70)
print("VisioNova ML Models Setup")
print("=" * 70)
print()

# Check if in virtual environment
venv_path = Path(__file__).parent.parent / ".venv"
if venv_path.exists():
    print(f"✓ Virtual environment found: {venv_path}")
else:
    print("⚠ Warning: Virtual environment not detected at .venv/")
    print("  Make sure you've activated your virtual environment!")
print()

# Step 1: Install/update dependencies
print("Step 1: Checking dependencies...")
print("-" * 70)

required_packages = [
    'torch',
    'torchvision', 
    'tqdm',
    'requests',
    'pillow'
]

missing_packages = []
for package in required_packages:
    try:
        __import__(package.replace('-', '_'))
        print(f"  ✓ {package}")
    except ImportError:
        print(f"  ✗ {package} - MISSING")
        missing_packages.append(package)

if missing_packages:
    print()
    print(f"Installing missing packages: {', '.join(missing_packages)}")
    subprocess.run([sys.executable, "-m", "pip", "install"] + missing_packages)
else:
    print()
    print("✓ All dependencies installed")

print()

# Step 2: Download models
print("Step 2: Downloading AI detection models")
print("-" * 70)
print("This will download ~2.6GB of model weights")
print("Models:")
print("  • DIRE (1.8GB) - Best for Stable Diffusion, DALL-E 3, Midjourney v6")
print("  • UniversalFakeDetect (850MB) - Works across all AI generators")
print()

response = input("Download models now? (y/n): ")
if response.lower() in ['y', 'yes']:
    print()
    from image_detector.download_models import download_models, verify_models
    
    download_models()
    print()
    verify_models()
else:
    print("Skipping download. Run 'python image_detector/download_models.py' later.")

print()

# Step 3: Test models
print("Step 3: Testing model integration")
print("-" * 70)

try:
    from image_detector.ml_detector import create_ml_detectors
    
    print("Loading detectors...")
    detectors = create_ml_detectors(device="auto", load_all=False)
    
    if detectors.get('dire') and detectors['dire'].model_loaded:
        print("  ✓ DIRE detector loaded successfully")
    else:
        print("  ⚠ DIRE detector not loaded (download models first)")
    
    if detectors.get('nyuad') and detectors['nyuad'].model_loaded:
        print("  ✓ NYUAD detector loaded successfully")
    else:
        print("  ⚠ NYUAD detector not loaded")
    
    print()
    print("✓ Setup complete!")
    
except Exception as e:
    print(f"  ✗ Error loading detectors: {e}")
    print()
    print("Troubleshooting:")
    print("1. Make sure you downloaded the models (Step 2)")
    print("2. Check that torch is installed: pip install torch torchvision")
    print("3. Try running: python image_detector/download_models.py --verify")

print()
print("=" * 70)
print("Next Steps:")
print("  1. Restart your Flask server: python backend/app.py")
print("  2. Upload an AI-generated image to test detection")
print("  3. Check the results - you should see ML model scores!")
print("=" * 70)
