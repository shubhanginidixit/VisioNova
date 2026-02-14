# AI Detection Models

High-accuracy image detection requires large pre-trained models. The automatic downloader script failed due to authentication/LFS restrictions. Please download these files manually.

## 1. DIRE (Diffusion Reconstruction Error)
- **Description**: Detects images from Stable Diffusion, DALL-E 3, Midjourney.
- **Link**: [Download checkpoint_dire.pth](https://huggingface.co/XPixelGroup/DIRE/resolve/main/checkpoint_dire.pth)
- **Action**: 
  1. Download the file.
  2. Rename it to `dire_model.pth`.
  3. Place it in this folder: `backend/image_detector/models/`

## 2. UniversalFakeDetect
- **Description**: General-purpose detector for various AI generators.
- **Link**: [Download fc_weights.pth](https://github.com/Yuheng-Li/UniversalFakeDetect/raw/main/pretrained_weights/fc_weights.pth)
  - *Note*: If the link downloads a small 4KB file, you need to use Git LFS or find the "Raw" download button on GitHub.
- **Action**:
  1. Download the file.
  2. Rename it to `universal_detector.pth`.
  3. Place it in this folder: `backend/image_detector/models/`

## Troubleshooting
If you cannot download the files, the system will continue to work using **Basic Detection** (Metadata, ELA, Noise Analysis) but will have lower accuracy for high-quality AI images.
