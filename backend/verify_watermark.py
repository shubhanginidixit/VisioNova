
import sys
import os
import logging

# Add backend to path
sys.path.append(os.path.abspath("e:/Personal Projects/VisioNova/backend"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    print("Testing imports...")
    try:
        from image_detector.watermark_detector import WatermarkDetector
        from image_detector.ensemble_detector import EnsembleDetector
        print("Imports successful!")
        return True
    except Exception as e:
        print(f"Import failed: {e}")
        return False

def test_instantiation():
    print("\nTesting instantiation...")
    try:
        from image_detector.watermark_detector import WatermarkDetector
        wd = WatermarkDetector()
        print("WatermarkDetector instantiated.")
        
        # Create a dummy image to test basic flow
        from PIL import Image
        import io
        img = Image.new('RGB', (100, 100), color='white')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_data = img_byte_arr.getvalue()
        
        # Test analyze (it won't find anything, but should run)
        print("Running analyze on dummy image...")
        result = wd.analyze(img_data)
        print(f"Analysis result: {result}")
        
    except Exception as e:
        print(f"Instantiation/Execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if test_imports():
        test_instantiation()
