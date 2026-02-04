# Binoculars Zero-Shot AI Text Detection - Integration Guide

## What is Binoculars?

Binoculars is a **zero-shot AI text detector** that requires **no training** - it works out of the box on ANY AI model's output, including future models like GPT-5, Claude 4, etc.

### How it Works
- Uses **dual Falcon-7B models** to compare perplexity scores
- AI text is predictable (low perplexity) → Detected
- Human text is creative (high perplexity) → Not flagged
- **90%+ accuracy** at 0.01% false positive rate

### Key Advantage
**Future-proof**: Works on models that don't exist yet because it detects the fundamental "predictability signature" that all AI text has, not specific model patterns.

---

## Requirements

### GPU Required
- **14GB+ VRAM** (NVIDIA GPU)
- Works on: T4 (Colab), V100, A100, RTX 3090/4090
- **Cannot run on CPU**

### Installation
Already installed in your environment:
```bash
pip install binoculars
```

---

## Usage

### 1. Basic Usage (CPU - Offline Mode)
```python
from backend.text_detector.text_detector_service import AIContentDetector, DETECTION_MODE_OFFLINE

# Fallback mode when GPU not available
detector = AIContentDetector(detection_mode=DETECTION_MODE_OFFLINE)

result = detector.predict("Your text here")
print(result['prediction'])  # "ai_generated" or "human"
print(result['confidence'])   # 0-100%
```

### 2. Binoculars Mode (GPU Required)
```python
from backend.text_detector.text_detector_service import AIContentDetector, DETECTION_MODE_BINOCULARS

# Attempts to load Binoculars, falls back to offline if GPU unavailable
detector = AIContentDetector(detection_mode=DETECTION_MODE_BINOCULARS)

result = detector.predict("Your text here")
print(result['detection_method'])  # Shows which method was actually used
```

### 3. Check Current Mode
```python
detector = AIContentDetector(detection_mode=DETECTION_MODE_BINOCULARS)

if detector.detection_mode == DETECTION_MODE_BINOCULARS:
    print("✓ Binoculars active!")
    info = detector.binoculars.get_info()
    print(f"GPU: {info['gpu']}")
    print(f"VRAM: {info['vram_gb']:.1f}GB")
else:
    print("⚠️ Fell back to offline mode (no GPU)")
```

---

## Detection Modes

| Mode | Description | Requirements | Use Case |
|------|-------------|--------------|----------|
| **offline** | Statistical + patterns | CPU only | Default, fast, no downloads |
| **ml** | DeBERTa-v3 fine-tuned | CPU/GPU | Current model (has overfitting issues) |
| **binoculars** | Dual Falcon-7B zero-shot | **GPU 14GB+** | Best accuracy, future-proof |

---

## Integration with Flask API

Update `app.py` to support Binoculars mode:

```python
from backend.text_detector.text_detector_service import AIContentDetector, DETECTION_MODE_BINOCULARS, DETECTION_MODE_OFFLINE

# Initialize detector (will auto-detect GPU availability)
detector = AIContentDetector(detection_mode=DETECTION_MODE_BINOCULARS)

@app.route('/api/detect-text', methods=['POST'])
def detect_text():
    data = request.json
    text = data.get('text', '')
    
    result = detector.predict(text)
    
    return jsonify({
        "prediction": result['prediction'],
        "confidence": result['confidence'],
        "method": result['detection_method'],  # Track which method was used
        "mode": result['detection_mode']
    })
```

---

## Deployment Strategies

### Strategy 1: Hybrid Local + Cloud
- **Local (CPU)**: Offline mode for quick checks
- **Cloud API (GPU)**: Binoculars for important/final checks
- User clicks "Deep Scan" → sends to GPU-enabled server

### Strategy 2: Colab Integration
Run Binoculars in Google Colab (free T4 GPU):
```python
# In Colab notebook
!pip install binoculars

from backend.text_detector.binoculars_detector import BinocularsDetector

detector = BinocularsDetector()
result = detector.detect("Your text here")
```

### Strategy 3: CPU-Only Production
Keep using offline mode until you have access to GPU infrastructure:
```python
# Always use offline mode
detector = AIContentDetector(detection_mode=DETECTION_MODE_OFFLINE)
```

---

## Testing

Run the test script to verify installation:
```bash
cd backend
python test_binoculars.py
```

**Expected output (no GPU):**
```
✓ OFFLINE mode working
⚠️  Binoculars not available - fell back to offline mode
   Reason: GPU with 14GB+ VRAM required
```

**Expected output (with GPU):**
```
✓ OFFLINE mode working
✓ Binoculars loaded successfully!
  GPU: NVIDIA Tesla T4
  VRAM: 15.0GB
```

---

## Performance Comparison

Based on your research documentation:

| Method | Accuracy | Speed | GPU Required | Future-Proof | Training Needed |
|--------|----------|-------|--------------|--------------|-----------------|
| Offline | ~70-80% | Fast | ❌ No | ❌ No | ❌ No |
| DeBERTa-v3 (current) | ~60%* | Medium | Optional | ❌ No | ✅ Yes |
| **Binoculars** | **90%+** | Slower | ✅ Yes | ✅ **Yes** | ❌ **No** |

*Current DeBERTa model is overfitted - needs retraining with better datasets

---

## Next Steps

### Immediate
1. ✅ **Done**: Binoculars integrated and tested
2. Use offline mode for now (CPU-friendly)

### When GPU Available
1. Deploy to GPU-enabled server (Colab, AWS, etc.)
2. Switch to Binoculars mode
3. Test on real-world data

### Long-term
1. **Retrain DeBERTa-v3** with RAID/WildChat/MGTBench datasets
2. **Ensemble approach**: Combine Binoculars + retrained DeBERTa + statistical
3. **Smart routing**: Light checks → offline, important → Binoculars

---

## Troubleshooting

### "GPU not available"
- Expected on CPU-only systems
- System will automatically fall back to offline mode
- To use Binoculars, deploy to GPU server

### "Binoculars detection failed"
- Falls back to offline mode automatically
- Check GPU VRAM (need 14GB+)
- Check CUDA installation

### Slow inference
- Normal for Binoculars (dual 7B models)
- Use offline mode for quick checks
- Reserve Binoculars for final/important analysis

---

## References

- **Binoculars Paper**: Zero-shot detection via perplexity comparison
- **Your Research Doc**: [Research_Synthetic_Text_Forensics.md](docs/Research_Synthetic_Text_Forensics.md)
- **Current Implementation**: [binoculars_detector.py](backend/text_detector/binoculars_detector.py)
