import os
import sys

def download_audio_models():
    print("Initiating pre-download of Audio Deepfake Models (this may take a while)...")
    try:
        from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
        import torch
    except ImportError:
        print("Error: Missing required packages. Please run 'pip install transformers torch'")
        return

    models = [
        "nii-yamagishilab/wav2vec-large-anti-deepfake",
        "DavidCombei/wavLM-base-Deepfake_V2",
        "mo-thecreator/Deepfake-audio-detection",
        "facebook/wav2vec2-base" # Fallback feature extractor
    ]

    for model_id in models:
        print(f"\n--- Downloading: {model_id} ---")
        try:
            # We only want to cache them, so loading them is enough
            try:
                print("1/2: Downloading Feature Extractor...")
                AutoFeatureExtractor.from_pretrained(model_id)
            except Exception as e:
                print(f"Skipping feature extractor for {model_id} (Expected for some models: {e})")
            
            print("2/2: Downloading Model Weights...")
            AutoModelForAudioClassification.from_pretrained(model_id)
            print(f"✓ Successfully cached {model_id}")
            
        except Exception as e:
            print(f"✗ Error caching {model_id}: {e}")

    print("\nAll models have been processed. You can now use the Audio Detector in VisioNova without downloading delays.")

if __name__ == "__main__":
    download_audio_models()
