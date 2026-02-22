import sys
import os
import json

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
try:
    from image_detector.ensemble_detector import EnsembleDetector
except ImportError:
    print("Cannot import EnsembleDetector")
    sys.exit(1)

def run():
    img_path = sys.argv[1]
    with open(img_path, 'rb') as f:
        img_bytes = f.read()
    
    detector = EnsembleDetector(use_gpu=False, load_ml_models=True)
    result = detector.detect(img_bytes, os.path.basename(img_path))
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    run()
