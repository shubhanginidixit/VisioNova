# Scripts

Utility and setup scripts for VisioNova.

## Files

- **setup_ml_models.py** - Download and configure ML models for image detection (requires Python 3.10 + .venv310)
- **download_models.py** - Legacy model downloader (deprecated, use setup_ml_models.py)
- **train.py** - Train custom text detection models
- **train_deberta.py** - Train DeBERTa model for text classification

## Usage

### Setup ML Models (Image Detection)
```powershell
# Activate Python 3.10 environment
.venv310\Scripts\Activate.ps1

# Run setup
python scripts/setup_ml_models.py
```

### Train Text Detection Model
```powershell
# Activate main environment
.venv\Scripts\Activate.ps1

# Run training
python scripts/train_deberta.py
```
