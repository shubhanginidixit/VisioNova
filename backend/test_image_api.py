"""
Quick test to verify the /api/detect-image endpoint is working
"""
import requests
import base64

# Read a simple 1x1 pixel test image (red pixel)
test_image_base64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

print("Testing /api/detect-image endpoint...")
print("Sending request to: http://localhost:5000/api/detect-image")

try:
    response = requests.post(
        'http://localhost:5000/api/detect-image',
        json={
            'image': test_image_base64,
            'filename': 'test.png'
        },
        timeout=30
    )
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        print("✅ SUCCESS! API is working")
        result = response.json()
        print("\nResponse:")
        print(f"  Prediction: {result.get('prediction', 'N/A')}")
        print(f"  Confidence: {result.get('confidence', 'N/A')}")
        print(f"  Has metadata: {result.get('metadata', {}).get('has_metadata', 'N/A')}")
    else:
        print("❌ FAILED!")
        print(f"Response: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("❌ CONNECTION ERROR!")
    print("Flask server is not running on http://localhost:5000")
    print("Start it with: python app.py")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
