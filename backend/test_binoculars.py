"""
Test script for Binoculars integration
Run this to verify Binoculars is working correctly
"""
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

from text_detector.text_detector_service import AIContentDetector, DETECTION_MODE_BINOCULARS, DETECTION_MODE_OFFLINE

def test_binoculars():
    """Test Binoculars detection mode."""
    print("=" * 60)
    print("BINOCULARS INTEGRATION TEST")
    print("=" * 60)
    
    # Test texts
    ai_text = """
    It's important to note that artificial intelligence has fundamentally transformed 
    the landscape of modern technology. Furthermore, machine learning algorithms 
    leverage vast datasets to optimize performance metrics. In conclusion, the 
    integration of neural networks represents a paradigm shift in computational capabilities.
    """
    
    human_text = """
    I was walking down the street yesterday when I saw the funniest thing - a dog 
    wearing sunglasses! Made my whole day honestly. Sometimes it's the little 
    unexpected moments that matter most, you know? Life's weird like that.
    """
    
    print("\n1. Testing OFFLINE mode (baseline)...")
    print("-" * 60)
    
    detector_offline = AIContentDetector(detection_mode=DETECTION_MODE_OFFLINE)
    
    result = detector_offline.predict(ai_text, detailed=False)
    print(f"AI Text (Offline): {result['prediction']} - {result['confidence']:.1f}% confidence")
    print(f"  Scores: Human={result['scores']['human']:.1f}% AI={result['scores']['ai_generated']:.1f}%")
    print(f"  Method: {result.get('detection_method', 'N/A')}")
    
    result = detector_offline.predict(human_text, detailed=False)
    print(f"Human Text (Offline): {result['prediction']} - {result['confidence']:.1f}% confidence")
    print(f"  Scores: Human={result['scores']['human']:.1f}% AI={result['scores']['ai_generated']:.1f}%")
    print(f"  Method: {result.get('detection_method', 'N/A')}")
    
    print("\n2. Testing BINOCULARS mode...")
    print("-" * 60)
    
    try:
        detector_binoculars = AIContentDetector(detection_mode=DETECTION_MODE_BINOCULARS)
        
        # Check if Binoculars is actually available
        if detector_binoculars.detection_mode != DETECTION_MODE_BINOCULARS:
            print("⚠️  Binoculars not available - fell back to offline mode")
            print("   Reason: GPU with 14GB+ VRAM required")
            print("   Current mode:", detector_binoculars.detection_mode)
            return
        
        print("✓ Binoculars loaded successfully!")
        
        # Get info
        if detector_binoculars.binoculars:
            info = detector_binoculars.binoculars.get_info()
            print(f"  GPU: {info.get('gpu', 'N/A')}")
            print(f"  VRAM: {info.get('vram_gb', 0):.1f}GB")
            print()
        
        print("Testing AI text...")
        result = detector_binoculars.predict(ai_text, detailed=False)
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.1f}%")
        print(f"  Scores: Human={result['scores']['human']:.1f}% AI={result['scores']['ai_generated']:.1f}%")
        print(f"  Method: {result.get('detection_method', 'N/A')}")
        print()
        
        print("Testing Human text...")
        result = detector_binoculars.predict(human_text, detailed=False)
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.1f}%")
        print(f"  Scores: Human={result['scores']['human']:.1f}% AI={result['scores']['ai_generated']:.1f}%")
        print(f"  Method: {result.get('detection_method', 'N/A')}")
        
        print("\n✓ Binoculars integration successful!")
        
    except Exception as e:
        print(f"❌ Error testing Binoculars: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_binoculars()
