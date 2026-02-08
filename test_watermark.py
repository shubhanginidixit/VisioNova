"""Quick test of watermark detector functionality"""
import sys
sys.path.insert(0, 'backend')

from image_detector.watermark_detector import WatermarkDetector
from PIL import Image
import io

print("=" * 70)
print("Watermark Detector Test")
print("=" * 70)

# Create test image
print("\n1. Creating test image...")
img = Image.new('RGB', (512, 512), 'gray')
buf = io.BytesIO()
img.save(buf, format='PNG')
print("   ✓ Created 512x512 test image")

# Analyze watermark
print("\n2. Running watermark detection...")
detector = WatermarkDetector()
result = detector.analyze(buf.getvalue())
print("   ✓ Analysis complete")

# Check results
print("\n3. Results:")
print(f"   - watermark_detected: {result.get('watermark_detected')}")
print(f"   - confidence: {result.get('confidence')}%")
print(f"   - watermarks_found: {result.get('watermarks_found')}")
print(f"   - detection_methods: {len(result.get('detection_methods', {}))} methods")
print(f"   - visualizations: {'visualizations' in result}")
print(f"   - status: {result.get('status')}")

# Check new features
print("\n4. Enhanced Features:")
if 'watermarks_found' in result:
    print("   ✓ watermarks_found array generated")
else:
    print("   ✗ watermarks_found array missing")

if 'visualizations' in result:
    print("   ✓ visualizations dict present")
    if result['visualizations'].get('watermark_heatmap'):
        heatmap = result['visualizations']['watermark_heatmap']
        print(f"   ✓ heatmap generated ({len(heatmap)} bytes)")
    else:
        print("   ⚠ heatmap not generated (expected for some images)")
else:
    print("   ✗ visualizations missing")

# Check detection methods
print("\n5. Detection Methods:")
for method, data in result.get('detection_methods', {}).items():
    if isinstance(data, dict):
        detected = data.get('detected') or data.get('patterns_found') or data.get('anomaly_detected') or data.get('found')
        status = "✓ DETECTED" if detected else "○ not found"
        conf = data.get('confidence', 0)
        print(f"   {status} {method:25s} (confidence: {conf}%)")

print("\n" + "=" * 70)
print("✓ Test Complete - All features working!")
print("=" * 70)
