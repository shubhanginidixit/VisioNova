# Image Detection Strategy

## Simple Explanation
When an AI generates an image, it paints pixel-by-pixel. While it looks good at a glance, it often makes mistakes that a camera never would.

*   **The "Melting" Effect:** AI often struggles with complex textures. Jewelry might melt into skin, or text might look like alien symbols.
*   **Invisible Noise:** Cameras leave a specific pattern of "noise" (grain) on a photo. AI images often have completely different, unnatural noise patterns that are invisible to the naked eye but obvious to our scanners.
*   **Compression:** When you edit a photo and save it, you leave "digital scars." We look for these scars to see if an image is original or modified.

---

## Technical Explanation

VisioNova employs a **Multi-Stage Ensemble Pipeline** for image forensics.

### 1. Error Level Analysis (ELA)
*   **Concept:** JPEG works on 8x8 pixel grids. When an image is modified and resaved, the compression artifacts in the edited areas differ from the original image.
*   **Detection:** We re-compress the image at a known quality (e.g., 95%) and subtract natural difference. High-frequency changes in specific regions indicate tampering (e.g., a face pasted onto another body).

### 2. Frequency Domain Analysis (DCT)
*   **Method:** We convert the image from the spatial domain to the frequency domain using Discrete Cosine Transform (DCT).
*   **Indicator:** AI generators (GANs and Diffusion models) leave specific "fingerprints" in the high-frequency spectrum. We look for anomalous peaks in the Fourier Transform that don't exist in natural photography.

### 3. Vision Transformers (ViT) & CNNs
*   **Model:** Ensemble of EfficientNet-B4 and a fine-tuned Vision Transformer.
*   **Training:** Trained on datasets like CIFAKE and vast libraries of Midjourney/DALL-E outputs.
*   **Focus:** These models learn to spot semantic inconsistencies—lighting direction errors, warping of geometric shapes, and textural anomalies in hair/eyes.

### 4. Metadata & Content Credentials (C2PA)
*   **Metadata:** We parse EXIF data for contradictions (e.g., "iPhone 13" tag but image resolution matches DALL-E default).
*   **C2PA/IPTC:** We check for the presence of "Content Credentials" signatures, which major AI providers are starting to embed to voluntarily identify AI media.
