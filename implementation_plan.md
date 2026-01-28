# Implementation Plan - Modernize Text Detector (DeBERTa-v3)

## Goal Description
Upgrade the current `AIContentDetector` from the legacy `DistilBERT` architecture to the state-of-the-art `microsoft/deberta-v3-base` (as recommended in the "Synthetic Text Forensics" research). This involves creating a new training script and updating the detector to handle the new model.

This directly answers the user's question: "before training is there any change we have to make?" -> **YES, we must switch the architecture to DeBERTa.**

## User Review Required
> [!IMPORTANT]
> **Performance Trade-off**: DeBERTa-v3 is significantly more accurate but slower and larger than DistilBERT.
> *   **DistilBERT**: ~67M params, very fast.
> *   **DeBERTa-v3-Base**: ~184M params, slower.
> *   **Impact**: The "Offline Mode" (statistical only) becomes more valuable for quick checks. The ML mode will be high-accuracy but heavier.

## Proposed Changes

### Component: Text Detector (`backend/text_detector/`)

#### [NEW] [train.py](file:///e:/Personal%20Projects/VisioNova/backend/text_detector/train.py)
Create a modern HuggingFace training script:
*   **Base Model**: `microsoft/deberta-v3-base`
*   **Dataset Handling**: Support for loading `RAID`, `WildChat` (or custom JSON/CSV datasets).
*   **Training Arguments**: Optimized for forensic detection (learning rate ~2e-5, weight decay 0.01).
*   **Output**: Saves model and tokenizer to `backend/text_detector/model/`.

#### [MODIFY] [detector.py](file:///e:/Personal%20Projects/VisioNova/backend/text_detector/detector.py)
*   Update `_load_model` to print confirmation of DeBERTa architecture.
*   Update `_cached_inference` to handle potential tokenizer differences (mostly `AutoTokenizer` handles this, but DeBERTa requires `sentencepiece`).

#### [MODIFY] [requirements.txt](file:///e:/Personal%20Projects/VisioNova/backend/requirements.txt)
*   Add `sentencepiece` (Required for DeBERTa tokenizer).
*   Add `datasets` and `scikit-learn` (Required for training).

## Verification Plan

### Automated Verification
1.  **Training Dry Run**: Run `python backend/text_detector/train.py --dry-run` to ensure the script downloads the model and starts the training loop (even with dummy data).
2.  **Inference Test**: Run `detector.py` main block (or a test script) to load the new model and predict on a sample string.

### Manual Verification
*   **User Action**: Run the training command provided in the plan.
*   **User Action**: Restart backend and test `Text Checker` in the UI to confirm it doesn't crash with the new model.
