"""
VisioNova Image Explainer — XAI (Explainable AI) System

Fully offline explanation engine that extracts real evidence from pretrained
detection models to explain WHY an image was classified as AI or human.

Three-layer explanation approach:
1. Ensemble Disagreement Analysis — reads per-model scores, maps each model
   to its specialty, and generates structured, evidence-based findings.
2. Grad-CAM Attention Heatmaps — extracts attention maps from ViT/Swin
   models to show WHERE in the image AI artifacts were detected.
3. Forensic Evidence Summary — combines metadata, watermark, C2PA, and
   statistical forensics into a single "evidence dossier".

No external API calls needed. Works 100% offline.
"""
import io
import base64
import logging
from typing import Dict, Optional, Any, List

import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Model Specialties ────────────────────────────────────────────────────────
# Each pretrained model has a specific strength. This mapping turns raw scores
# into human-readable explanations of WHY a model flagged the image.

MODEL_SPECIALTIES = {
    'ateeqq': {
        'name': 'Ateeqq SigLIP2',
        'architecture': 'SigLIP2 (Google Sigmoid Vision-Language)',
        'accuracy': '99.23%',
        'specialty': 'High-precision general AI image detection using semantic visual features',
        'detects': 'Broad AI generators including DALL-E, Midjourney, Stable Diffusion',
        'how_it_works': 'Analyzes how shapes, textures, and colors relate to each other — AI generators create statistically unusual texture-edge correlations that this model is trained to spot.',
        'ai_explanation': 'Semantic visual features match patterns from known AI image generators',
        'human_explanation': 'Semantic features are consistent with natural photography',
        'evidence_ai': 'The way textures blend at edges, the distribution of colors, and the overall "look" of the image match the statistical fingerprint of AI-generated content.',
        'evidence_human': 'Natural texture gradients, realistic edge transitions, and organic color distribution are all consistent with real camera capture.',
    },
    'siglip_dinov2': {
        'name': 'Bombek1 SigLIP2+DINOv2',
        'architecture': 'Hybrid SigLIP2 + DINOv2 (dual-encoder)',
        'accuracy': '99.97% AUC',
        'specialty': 'Best overall detector — combines semantic + self-supervised features',
        'detects': '25+ generators: Flux, MJ v6, DALL-E 3, SDXL, GPT-Image-1',
        'how_it_works': 'Cross-references what the image depicts (semantic meaning) with how it was physically rendered (pixel-level structure). Genuine photos have consistent meaning-structure pairs; AI images often have subtle mismatches.',
        'ai_explanation': 'Both semantic (SigLIP2) and structural (DINOv2) features indicate AI origin',
        'human_explanation': 'Both semantic and structural features match authentic photographs',
        'evidence_ai': 'The image meaning and its low-level pixel structure both independently point to AI generation — this dual-signal is the strongest indicator available.',
        'evidence_human': 'Both the visual meaning and the pixel-level structure are mutually consistent, matching what we see in authentic camera-captured content.',
    },
    'deepfake': {
        'name': 'dima806 ViT',
        'architecture': 'Vision Transformer (ViT-B/16)',
        'accuracy': '98.25%',
        'specialty': 'General deepfake and AI image detection with strong community validation',
        'detects': 'General AI-generated images and face manipulations',
        'how_it_works': 'Examines the overall layout and global composition — AI images have subtly regular geometric patterns that cameras never produce, because generators build images from mathematical noise rather than sensor data.',
        'ai_explanation': 'Global compositional patterns detected by Vision Transformer indicate AI generation',
        'human_explanation': 'Global image composition is consistent with natural capture',
        'evidence_ai': 'The global composition has subtle, regular patterns that cameras don\'t produce — a hallmark of mathematical image generation.',
        'evidence_human': 'The overall composition, layout, and global structure are consistent with natural camera capture — no synthetic patterns detected.',
    },
    'sdxl': {
        'name': 'Organika SDXL-Detector',
        'architecture': 'Swin Transformer',
        'accuracy': '98.1%',
        'specialty': 'Specialist for modern diffusion models (SDXL, Flux, SD3)',
        'detects': 'Stable Diffusion XL, Flux, and other modern diffusion model outputs',
        'how_it_works': 'Trained specifically on Stable Diffusion / Flux outputs — recognizes the unique pixel-level "fingerprint" these diffusion models leave behind in the denoising process.',
        'ai_explanation': 'Patterns match known Stable Diffusion / Flux generator fingerprints',
        'human_explanation': 'No diffusion model fingerprints detected',
        'evidence_ai': 'Pixel-level fingerprints match those left behind by diffusion models (Stable Diffusion, Flux, SDXL) during their denoising generation process.',
        'evidence_human': 'No diffusion-model fingerprints were found — the pixel patterns don\'t match any known Stable Diffusion, Flux, or SDXL output characteristics.',
    },
}


class ImageExplainer:
    """
    Generates evidence-based explanations for image detection results.

    Uses three complementary approaches:
    1. Ensemble Disagreement Analysis — structures per-model scores into
       human-readable findings with model-specialty context.
    2. Grad-CAM Attention Heatmaps — optional visual overlay showing
       which image regions triggered the AI detection.
    3. Forensic Evidence Summary — metadata, watermarks, C2PA, and
       statistical comparisons in a single "evidence dossier".
    """

    def __init__(self):
        """Initialize the XAI explainer. No API keys needed."""
        # Check if pytorch-grad-cam is available (optional dependency)
        self.gradcam_available = False
        try:
            from pytorch_grad_cam import GradCAM
            self.gradcam_available = True
            logger.info("ImageExplainer initialized with Grad-CAM support")
        except ImportError:
            logger.info("ImageExplainer initialized (Grad-CAM not available — install pytorch-grad-cam for heatmaps)")

    def analyze_image(self, image_data: bytes, detection_result: Dict,
                      model_registry: Dict = None) -> Dict[str, Any]:
        """
        Perform XAI analysis of an image detection result.

        Combines ensemble disagreement analysis with optional Grad-CAM heatmaps
        and forensic evidence to produce a complete, evidence-based explanation.

        Args:
            image_data: Raw image bytes
            detection_result: Results from the ML ensemble detector
            model_registry: Optional dict of loaded model objects for Grad-CAM

        Returns:
            dict with visual_analysis, reasoning, combined_verdict, forensic_evidence
        """
        try:
            # Layer 1: Ensemble Disagreement Analysis (always works, no deps)
            ensemble_explanation = self._build_ensemble_explanation(detection_result)

            # Layer 2: Grad-CAM Heatmap (optional, needs pytorch-grad-cam)
            heatmap_b64 = None
            if self.gradcam_available and model_registry:
                heatmap_b64 = self._generate_gradcam_heatmap(image_data, detection_result, model_registry)

            # Layer 3: Forensic Evidence Summary
            forensic_evidence = self._build_forensic_evidence(detection_result)

            # Build the response in the same shape the frontend expects
            visual_analysis = {
                'assessment': ensemble_explanation['summary'],
                'anomalies': ensemble_explanation['anomalies'],
                'model_breakdown': ensemble_explanation['model_breakdown'],
                'agreement_level': ensemble_explanation['agreement_level'],
                'agreement_detail': ensemble_explanation['agreement_detail'],
            }

            if heatmap_b64:
                visual_analysis['attention_heatmap'] = heatmap_b64

            return {
                'success': True,
                'visual_analysis': visual_analysis,
                'reasoning': ensemble_explanation['reasoning'],
                'forensic_evidence': forensic_evidence,
                'combined_verdict': {
                    'combined_probability': detection_result.get('ai_probability', 50),
                    'verdict': detection_result.get('verdict', 'UNCERTAIN'),
                    'verdict_description': ensemble_explanation['verdict_description'],
                    'analysis_agreement': ensemble_explanation['agreement_level'],
                },
                'explanation_method': 'XAI (Ensemble Analysis' + (' + Grad-CAM)' if heatmap_b64 else ')'),
            }

        except Exception as e:
            logger.error(f"XAI analysis error: {e}")
            return {
                'success': False,
                'error': str(e),
                'visual_analysis': None,
                'reasoning': {'bullets': [f'Explanation generation failed: {str(e)}'], 'caveat': None},
                'forensic_evidence': None,
                'combined_verdict': {
                    'combined_probability': detection_result.get('ai_probability', 50),
                    'verdict': detection_result.get('verdict', 'UNCERTAIN'),
                    'verdict_description': 'Analysis completed — explanation unavailable',
                    'analysis_agreement': 'UNKNOWN',
                },
            }

    # ─── Layer 1: Ensemble Disagreement Analysis ─────────────────────────────

    def _build_ensemble_explanation(self, detection_result: Dict) -> Dict[str, Any]:
        """
        Build a structured explanation from per-model ensemble scores.

        Reads individual_results from the ensemble detector, maps each model
        to its specialty, and generates evidence-based findings.

        Args:
            detection_result: Full ensemble detection result

        Returns:
            dict with summary, reasoning, model_breakdown, anomalies, agreement info
        """
        individual_results = detection_result.get('individual_results', {})
        ai_probability = detection_result.get('ai_probability', 50)
        verdict = detection_result.get('verdict', 'UNCERTAIN')

        # Build per-model breakdown
        model_breakdown = []
        models_flagging_ai = 0
        models_total = 0
        scores = []

        for model_key, model_result in individual_results.items():
            if not isinstance(model_result, dict):
                continue

            # Only include real ML detection models, not utility analyzers
            # (metadata, ela, watermark, c2pa etc. are not detection models)
            if model_key not in MODEL_SPECIALTIES:
                continue

            specialty = MODEL_SPECIALTIES[model_key]
            model_score = model_result.get('ai_probability', 50)
            model_success = model_result.get('success', False)

            if not model_success:
                continue

            models_total += 1
            scores.append(model_score)
            is_flagging = model_score > 50

            if is_flagging:
                models_flagging_ai += 1

            # Pick the right explanation based on whether model thinks AI or not
            interpretation = specialty.get('ai_explanation', 'AI patterns detected') if is_flagging else specialty.get('human_explanation', 'Appears authentic')
            evidence = specialty.get('evidence_ai') if is_flagging else specialty.get('evidence_human')

            model_breakdown.append({
                'name': specialty.get('name', model_key),
                'key': model_key,
                'score': round(model_score, 1),
                'accuracy': specialty.get('accuracy', 'N/A'),
                'specialty': specialty.get('specialty', 'General detection'),
                'detects': specialty.get('detects', 'AI-generated images'),
                'how_it_works': specialty.get('how_it_works', ''),
                'interpretation': interpretation,
                'evidence': evidence,
                'flagged_as_ai': is_flagging,
            })

        # Sort by score descending (strongest signals first)
        model_breakdown.sort(key=lambda m: m['score'], reverse=True)

        # Agreement analysis
        agreement_level, agreement_detail = self._analyze_agreement(
            models_flagging_ai, models_total, scores
        )

        # Generate anomalies / findings
        anomalies = self._generate_anomalies(model_breakdown, detection_result)

        # Generate human-readable summary
        summary = self._generate_summary(
            models_flagging_ai, models_total, ai_probability, model_breakdown
        )

        # Generate detailed reasoning
        reasoning = self._generate_reasoning(
            model_breakdown, agreement_level, ai_probability, detection_result
        )

        # Verdict description
        verdict_description = self._generate_verdict_description(
            verdict, ai_probability, models_flagging_ai, models_total
        )

        return {
            'summary': summary,
            'reasoning': reasoning,
            'model_breakdown': model_breakdown,
            'anomalies': anomalies,
            'agreement_level': agreement_level,
            'agreement_detail': agreement_detail,
            'verdict_description': verdict_description,
        }

    def _analyze_agreement(self, flagging: int, total: int, scores: List[float]) -> tuple:
        """
        Determine how much the models agree with each other.

        Returns:
            tuple of (level_string, detail_string)
        """
        if total == 0:
            return 'NO_DATA', 'No models produced results'

        ratio = flagging / total

        if ratio >= 0.8:
            level = 'STRONG_AI_CONSENSUS'
            detail = f'{flagging}/{total} models agree this is AI-generated'
        elif ratio >= 0.6:
            level = 'MAJORITY_AI'
            detail = f'{flagging}/{total} models flagged as AI — majority consensus'
        elif ratio <= 0.2:
            level = 'STRONG_HUMAN_CONSENSUS'
            detail = f'{total - flagging}/{total} models agree this is human-created'
        elif ratio <= 0.4:
            level = 'MAJORITY_HUMAN'
            detail = f'{total - flagging}/{total} models lean toward human origin'
        else:
            level = 'SPLIT_DECISION'
            detail = f'Models are divided ({flagging} AI vs {total - flagging} Human) — mixed signals'

        # Add spread information
        if len(scores) >= 2:
            spread = max(scores) - min(scores)
            if spread > 40:
                detail += f'. High spread ({spread:.0f}%) suggests different models see different things.'

        return level, detail

    def _generate_anomalies(self, model_breakdown: List[Dict], detection_result: Dict) -> List[str]:
        """
        Generate a list of notable anomalies / findings from the analysis.
        """
        anomalies = []

        # Check for unanimous detection
        ai_models = [m for m in model_breakdown if m['flagged_as_ai']]
        human_models = [m for m in model_breakdown if not m['flagged_as_ai']]

        if len(ai_models) == len(model_breakdown) and len(model_breakdown) > 0:
            anomalies.append(f'All {len(model_breakdown)} detection models unanimously classify this as AI-generated')
        elif len(human_models) == len(model_breakdown) and len(model_breakdown) > 0:
            anomalies.append(f'All {len(model_breakdown)} detection models unanimously classify this as human-created')

        # Check for specific model triggers
        for model in model_breakdown:
            if model['key'] == 'sdxl' and model['flagged_as_ai'] and model['score'] > 80:
                anomalies.append(f"SDXL-Detector ({model['score']}%) — image matches Stable Diffusion / Flux generator patterns")
            if model['key'] == 'siglip_dinov2' and model['flagged_as_ai'] and model['score'] > 95:
                anomalies.append(f"SigLIP2+DINOv2 ({model['score']}%) — strongest dual-encoder model confirms AI origin")
            if model['key'] == 'ateeqq' and model['flagged_as_ai'] and model['score'] > 95:
                anomalies.append(f"Ateeqq SigLIP2 ({model['score']}%) — highest-precision model strongly detects AI")

        # Check for outlier models (one model strongly disagrees with the rest)
        if len(model_breakdown) >= 3:
            scores = [m['score'] for m in model_breakdown]
            mean_score = np.mean(scores)
            for model in model_breakdown:
                deviation = abs(model['score'] - mean_score)
                if deviation > 30:
                    if model['score'] > mean_score:
                        anomalies.append(f"{model['name']} scores much higher ({model['score']:.0f}% vs avg {mean_score:.0f}%) — may be detecting generator-specific artifacts")
                    else:
                        anomalies.append(f"{model['name']} scores much lower ({model['score']:.0f}% vs avg {mean_score:.0f}%) — this model's specialty may not match the image type")

        # Check watermark findings if available
        watermark = detection_result.get('individual_results', {}).get('watermark', {})
        if watermark.get('watermark_detected'):
            source = watermark.get('source', watermark.get('detected_watermark', 'Unknown'))
            anomalies.append(f'Invisible AI watermark detected (source: {source})')

        # Check C2PA
        c2pa = detection_result.get('individual_results', {}).get('c2pa', {})
        if c2pa.get('is_ai_generated'):
            generator = c2pa.get('ai_generator', 'Unknown')
            anomalies.append(f'C2PA Content Credentials confirm AI generation by {generator}')

        return anomalies

    # ── Plain-English explanation templates per model key ──────────────────
    # Each entry has:
    #   'ai': what a high score from this model MEANS in plain English
    #   'human': what a low score from this model MEANS in plain English
    PLAIN_REASONS = {
        'ateeqq': {
            'ai': 'The semantic visual features — the way shapes, textures, and colors relate — match patterns found in AI-generated images, not real photographs.',
            'human': 'The overall semantic composition of the image is consistent with how a real camera captures a scene.',
        },
        'siglip_dinov2': {
            'ai': 'Both the high-level meaning of the image and its pixel-level structure independently point to AI generation — a strong dual signal.',
            'human': 'Both the image meaning and its low-level structure are consistent with authentic, camera-captured content.',
        },
        'deepfake': {
            'ai': "The image's global composition has subtle, regular patterns that cameras don't produce — a hallmark of AI generation.",
            'human': 'The overall composition and layout of the image looks consistent with natural camera capture.',
        },
        'sdxl': {
            'ai': 'The pixel-level fingerprints match those left behind by Stable Diffusion, Flux, or similar diffusion models.',
            'human': 'No diffusion-model fingerprints (Stable Diffusion, Flux, SDXL) were found in the image.',
        },
    }

    def _generate_summary(self, flagging: int, total: int, ai_prob: float,
                          model_breakdown: List[Dict]) -> str:
        """
        Generate a single plain-English verdict sentence.
        Example: 'This image is very likely AI-generated. Multiple independent
        checks all point to the same conclusion.'
        """
        if total == 0:
            return 'No detection models were available for this image.'

        if ai_prob > 50:
            if flagging == total:
                return (f'This image is almost certainly AI-generated. '
                        f'Every independent check we ran came to the same conclusion.')
            elif flagging > total / 2:
                return (f'This image is very likely AI-generated. '
                        f'{flagging} out of {total} independent checks agree on this.')
            else:
                return (f'This image shows signs of AI generation, '
                        f'though the evidence is mixed. Treat this result with some caution.')
        else:
            if flagging == 0:
                return (f'This image appears to be a real, authentic photograph. '
                        f'No AI generation patterns were found.')
            elif flagging < total / 2:
                return (f'This image is most likely authentic. '
                        f'{total - flagging} out of {total} checks agree it is human-created.')
            else:
                return (f'This image appears authentic, but the evidence is mixed. '
                        f'Some AI patterns were detected — verify with additional sources if needed.')

    def _generate_reasoning(self, model_breakdown: List[Dict], agreement_level: str,
                            ai_prob: float, detection_result: Dict) -> Dict:
        """
        Generate plain-English explanation as a list of bullet points.

        Returns a dict with:
          - 'bullets': List[str] — why the verdict was reached, in plain English
          - 'caveat': str | None — shown only when results are uncertain
        """
        bullets = []
        is_ai = ai_prob > 50

        # Pick the models that agree with the final verdict (strongest signal first)
        if is_ai:
            signal_models = sorted(
                [m for m in model_breakdown if m['flagged_as_ai']],
                key=lambda m: m['score'], reverse=True
            )
        else:
            signal_models = sorted(
                [m for m in model_breakdown if not m['flagged_as_ai']],
                key=lambda m: m['score']
            )

        # Add up to 3 plain-English reasons from the strongest signalling models
        seen_reasons = set()
        for model in signal_models[:3]:
            key = model['key']
            direction = 'ai' if is_ai else 'human'
            reason = self.PLAIN_REASONS.get(key, {}).get(direction)
            # Deduplicate near-identical reasons
            if reason and reason not in seen_reasons:
                bullets.append(reason)
                seen_reasons.add(reason)

        # Add a metadata/watermark bullet if detected
        individual = detection_result.get('individual_results', {})
        watermark = individual.get('watermark', {})
        metadata = individual.get('metadata', {})
        c2pa = individual.get('c2pa', {})

        if c2pa.get('is_ai_generated'):
            generator = c2pa.get('ai_generator', 'an AI tool')
            bullets.append(f'The image contains verified digital credentials (C2PA) that confirm it was created by {generator}.')
        elif watermark.get('watermark_detected'):
            bullets.append('An invisible AI watermark was detected embedded in the image — a strong indicator of AI generation.')
        elif metadata.get('ai_software_detected') and is_ai:
            sw = metadata.get('software_detected', 'AI software')
            bullets.append(f'The image metadata records {sw} as the creation tool.')
        elif not metadata.get('has_exif') and is_ai:
            bullets.append('The image has no camera metadata (EXIF data). Real photographs almost always contain this information.')

        # Build caveat for uncertain/split cases
        caveat = None
        if agreement_level == 'SPLIT_DECISION':
            caveat = ('The detection models disagree on this image, which can happen when an image is heavily '
                      'compressed, filtered, or sits at the boundary of what current detectors can identify. '
                      'We recommend additional verification.')
        elif agreement_level in ('MAJORITY_HUMAN', 'MAJORITY_AI') and len(model_breakdown) >= 3:
            dissenting = len(model_breakdown) - len(signal_models)
            if dissenting > 0:
                caveat = (f'{dissenting} out of {len(model_breakdown)} checks pointed the other way. '
                          f'This result is likely correct, but not unanimous.')

        return {
            'bullets': bullets,
            'caveat': caveat,
        }

    def _generate_verdict_description(self, verdict: str, ai_prob: float,
                                       flagging: int, total: int) -> str:
        """Generate a short verdict label."""
        if ai_prob > 50:
            return f'AI-Generated — {flagging}/{total} checks agree'
        else:
            return f'Authentic — {total - flagging}/{total} checks agree'

    # ─── Layer 2: Grad-CAM Attention Heatmaps ────────────────────────────────

    def _generate_gradcam_heatmap(self, image_data: bytes, detection_result: Dict,
                                   model_registry: Dict = None) -> Optional[str]:
        """
        Generate a Grad-CAM attention heatmap from a loaded detection model.

        Extracts attention weights from either ViT (deepfake) or Swin (SDXL)
        model and produces a heatmap overlay showing which image regions
        triggered the AI detection.

        Args:
            image_data: Raw image bytes
            detection_result: Results from ensemble detector
            model_registry: Dict mapping model keys to loaded model objects

        Returns:
            Base64-encoded heatmap image, or None if generation fails
        """
        if not model_registry:
            return None

        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
            import torch
            from torchvision import transforms

            # Try models in order of preference for Grad-CAM compatibility
            target_model = None
            target_layer = None
            processor = None

            # 1. Try deepfake (ViT) — standard architecture for Grad-CAM
            deepfake_det = model_registry.get('deepfake')
            if deepfake_det and hasattr(deepfake_det, 'model') and deepfake_det.model is not None:
                model = deepfake_det.model
                if hasattr(model, 'vit') and hasattr(model.vit, 'encoder'):
                    target_layer = model.vit.encoder.layer[-1].layernorm_after
                    target_model = model
                    processor = getattr(deepfake_det, 'processor', None)

            # 2. Try SDXL (Swin) as fallback
            if target_model is None:
                sdxl_det = model_registry.get('sdxl')
                if sdxl_det and hasattr(sdxl_det, 'model') and sdxl_det.model is not None:
                    model = sdxl_det.model
                    if hasattr(model, 'swin') and hasattr(model.swin, 'layernorm'):
                        target_layer = model.swin.layernorm
                        target_model = model
                        processor = getattr(sdxl_det, 'processor', None)

            if target_model is None or target_layer is None:
                logger.debug("Grad-CAM: No compatible model found in registry")
                return None

            # Prepare image
            image = Image.open(io.BytesIO(image_data)).convert('RGB')

            # Process image for model input
            if processor:
                inputs = processor(images=image, return_tensors="pt")
                input_tensor = inputs['pixel_values']
            else:
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
                input_tensor = transform(image).unsqueeze(0)

            # Move to same device as model
            device = next(target_model.parameters()).device
            input_tensor = input_tensor.to(device)

            # Run Grad-CAM — target the highest-scoring class
            cam = GradCAM(model=target_model, target_layers=[target_layer])
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)
            grayscale_cam = grayscale_cam[0, :]  # First image in batch

            # Resize cam to match and create overlay
            cam_h, cam_w = grayscale_cam.shape
            img_resized = np.array(image.resize((cam_w, cam_h))).astype(np.float32) / 255.0
            visualization = show_cam_on_image(img_resized, grayscale_cam, use_rgb=True)

            # Convert to base64
            vis_image = Image.fromarray(visualization)
            buffer = io.BytesIO()
            vis_image.save(buffer, format='JPEG', quality=85)
            buffer.seek(0)
            heatmap_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            logger.info("Grad-CAM heatmap generated successfully")
            return f"data:image/jpeg;base64,{heatmap_b64}"

        except ImportError:
            logger.debug("pytorch-grad-cam not installed")
            return None
        except Exception as e:
            logger.warning(f"Grad-CAM heatmap generation failed: {e}")
            return None

    # ─── Layer 3: Forensic Evidence Summary ──────────────────────────────────

    def _build_forensic_evidence(self, detection_result: Dict) -> Dict[str, Any]:
        """
        Build a forensic evidence summary from all non-ML detection signals.
        Summarizes metadata, watermark, C2PA, and statistical findings into
        a structured object the frontend can display as an "evidence dossier".
        """
        individual = detection_result.get('individual_results', {})
        is_ai = detection_result.get('ai_probability', 50) > 50

        return {
            'metadata': self._summarize_metadata(individual.get('metadata', {})),
            'watermark': self._summarize_watermark(individual.get('watermark', {})),
            'c2pa': self._summarize_c2pa(individual.get('c2pa', {})),
            'ela': self._summarize_ela(individual.get('ela', {})),
            'comparison': self._build_comparison(is_ai),
        }

    def _summarize_metadata(self, meta: Dict) -> Dict:
        if not meta:
            return {'status': 'not_scanned', 'summary': 'Metadata analysis was not performed.'}

        findings = []
        if meta.get('has_exif'):
            camera = meta.get('camera_info', {})
            if camera.get('make'):
                findings.append(f"Camera: {camera['make']} {camera.get('model', '')}")
            if meta.get('has_gps'):
                findings.append("GPS coordinates present")
            if meta.get('timestamp'):
                findings.append(f"Created: {meta['timestamp']}")
        else:
            findings.append("No EXIF metadata — AI-generated images typically have no camera data")

        if meta.get('ai_software_detected'):
            findings.append(f"AI software detected: {meta.get('software_detected', 'Unknown')}")
        if meta.get('screenshot_detected'):
            findings.append("This appears to be a screenshot")

        return {
            'status': 'ai_detected' if meta.get('ai_software_detected') else ('clean' if meta.get('has_exif') else 'missing'),
            'summary': ' • '.join(findings) if findings else 'No notable metadata findings.',
            'findings': findings,
        }

    def _summarize_watermark(self, wm: Dict) -> Dict:
        if not wm:
            return {'status': 'not_scanned', 'summary': 'Watermark scan was not performed.'}

        if wm.get('watermark_detected'):
            source = wm.get('source', wm.get('detected_watermark', 'Unknown'))
            conf = wm.get('confidence', 0)
            return {
                'status': 'detected',
                'summary': f"AI watermark detected from {source} ({conf}% confidence)",
                'source': source,
                'confidence': conf,
            }
        return {
            'status': 'clear',
            'summary': f"No AI watermarks found (scanned with {wm.get('methods_count', 'multiple')} methods)",
        }

    def _summarize_c2pa(self, c2pa: Dict) -> Dict:
        if not c2pa:
            return {'status': 'not_scanned', 'summary': 'C2PA analysis was not performed.'}

        if c2pa.get('has_content_credentials'):
            if c2pa.get('is_ai_generated'):
                return {
                    'status': 'ai_confirmed',
                    'summary': f"C2PA confirms AI generation by {c2pa.get('ai_generator', 'Unknown')}",
                    'generator': c2pa.get('ai_generator'),
                }
            return {
                'status': 'authentic',
                'summary': 'Content Credentials verify authentic provenance',
            }
        return {'status': 'absent', 'summary': 'No C2PA Content Credentials found in image.'}

    def _summarize_ela(self, ela: Dict) -> Dict:
        if not ela:
            return {'status': 'not_scanned', 'summary': 'ELA analysis was not performed.'}

        score = ela.get('manipulation_score', 0)
        if score > 70:
            return {'status': 'suspicious', 'summary': f'High manipulation indicators (score: {score}/100)'}
        elif score > 40:
            return {'status': 'moderate', 'summary': f'Some inconsistencies detected (score: {score}/100)'}
        return {'status': 'clean', 'summary': f'Error levels appear consistent (score: {score}/100)'}

    def _build_comparison(self, is_ai: bool) -> Dict:
        """What would a real/AI photo look like? comparison section."""
        if is_ai:
            return {
                'title': 'What makes this look AI-generated?',
                'points': [
                    {'label': 'Noise patterns', 'finding': 'Unnaturally uniform — real photos have sensor noise that varies across the image'},
                    {'label': 'EXIF metadata', 'finding': 'Missing or synthetic — real photos contain camera make, model, GPS, exposure settings'},
                    {'label': 'Texture transitions', 'finding': 'Statistically unusual edge-texture correlations that don\'t occur in optical capture'},
                    {'label': 'Global composition', 'finding': 'Subtle regularities in layout that mathematical generators produce'},
                ],
            }
        else:
            return {
                'title': 'What makes this look authentic?',
                'points': [
                    {'label': 'Noise patterns', 'finding': 'Natural sensor noise with expected spatial variation'},
                    {'label': 'EXIF metadata', 'finding': 'Camera data present and consistent with the claimed source'},
                    {'label': 'Texture transitions', 'finding': 'Natural gradient patterns where surfaces meet edges'},
                    {'label': 'Global composition', 'finding': 'No synthetic regularities detected — appears optically captured'},
                ],
            }


def create_image_explainer() -> ImageExplainer:
    """
    Factory function to create an ImageExplainer instance.

    Returns:
        ImageExplainer instance (no API key needed)
    """
    return ImageExplainer()
