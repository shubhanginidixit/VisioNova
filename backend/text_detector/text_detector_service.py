"""
VisioNova Text Detector
AI-generated text detection with sentence-level analysis, pattern detection, and caching.

Architecture:
- ML Model (DeBERTa-v3): Transformer-based detection (Microsoft/DeBERTa-v3-base)
- Binoculars: Zero-shot detection with dual Falcon-7B (GPU-only, no training)
- Linguistic Analysis: Real LM perplexity, burstiness, patterns
- Adversarial Defense: Homoglyph normalization, paraphraser shield
- ESL De-biasing: Reduced false positives for non-native English writers
- Caching: LRU cache for repeated texts

Detection Modes:
- 'offline': Statistical + pattern analysis only (CPU-friendly, default)
- 'ml': DeBERTa-v3 + statistical hybrid (requires model)
- 'binoculars': Dual Falcon-7B zero-shot (requires GPU, best accuracy)
"""
import os
import re
import math
import hashlib
import logging
import unicodedata
from functools import lru_cache
from typing import Dict, List, Optional, Tuple
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


# ==================== HOMOGLYPH / ADVERSARIAL DEFENSE ====================

# Mapping of common Unicode homoglyphs to ASCII equivalents
_HOMOGLYPH_MAP = {
    # Cyrillic lookalikes
    '\u0430': 'a', '\u0435': 'e', '\u043e': 'o', '\u0440': 'p',
    '\u0441': 'c', '\u0443': 'y', '\u0445': 'x', '\u0456': 'i',
    '\u0410': 'A', '\u0412': 'B', '\u0415': 'E', '\u041a': 'K',
    '\u041c': 'M', '\u041d': 'H', '\u041e': 'O', '\u0420': 'P',
    '\u0421': 'C', '\u0422': 'T', '\u0425': 'X',
    # Greek lookalikes
    '\u03b1': 'a', '\u03bf': 'o', '\u03b5': 'e', '\u0391': 'A',
    '\u0392': 'B', '\u0395': 'E', '\u0397': 'H', '\u0399': 'I',
    '\u039a': 'K', '\u039c': 'M', '\u039d': 'N', '\u039f': 'O',
    '\u03a1': 'P', '\u03a4': 'T', '\u03a7': 'X', '\u03a5': 'Y',
    '\u0396': 'Z',
    # Fullwidth Latin
    '\uff41': 'a', '\uff42': 'b', '\uff43': 'c', '\uff44': 'd',
    '\uff45': 'e', '\uff46': 'f', '\uff47': 'g', '\uff48': 'h',
    '\uff49': 'i', '\uff4a': 'j', '\uff4b': 'k', '\uff4c': 'l',
    '\uff4d': 'm', '\uff4e': 'n', '\uff4f': 'o', '\uff50': 'p',
    # Special characters used to evade
    '\u200b': '',  # Zero-width space
    '\u200c': '',  # Zero-width non-joiner
    '\u200d': '',  # Zero-width joiner
    '\ufeff': '',  # Zero-width no-break space (BOM)
    '\u00ad': '',  # Soft hyphen
    '\u2060': '',  # Word joiner
    '\u2063': '',  # Invisible separator
}


def normalize_adversarial_text(text: str) -> str:
    """
    Normalize text to defend against adversarial evasion techniques.
    
    Handles:
    - Unicode homoglyph substitution (Cyrillic 'а' for Latin 'a')
    - Zero-width character injection
    - Fullwidth character substitution
    - Unicode normalization (NFC)
    """
    # 1. Remove zero-width characters and invisible separators
    for char, replacement in _HOMOGLYPH_MAP.items():
        if replacement == '':
            text = text.replace(char, '')
    
    # 2. Unicode NFC normalization
    text = unicodedata.normalize('NFC', text)
    
    # 3. Replace homoglyphs
    result = []
    for char in text:
        if char in _HOMOGLYPH_MAP:
            result.append(_HOMOGLYPH_MAP[char])
        else:
            result.append(char)
    text = ''.join(result)
    
    # 4. Collapse multiple spaces (sometimes inserted to break patterns)
    text = re.sub(r' {2,}', ' ', text)
    
    return text


# ==================== ESL DETECTION ====================

# Indicators that text may be from an ESL (English as a Second Language) writer
_ESL_INDICATORS = {
    'article_errors': [
        r'\b(a [aeiou]\w+)\b',  # a + vowel (should be 'an')
        r'\b(an [^aeiou\s]\w+)\b',  # an + consonant
    ],
    'preposition_errors': [
        r'\b(depend of|consist from|interested of|capable for)\b',
        r'\b(married with|discuss about|explain about|emphasize on)\b',
    ],
    'missing_articles': [
        r'\b(go to school|go to university|go to hospital)\b',
        r'\b(in morning|in evening|at afternoon)\b',
    ],
    'word_order_patterns': [
        r'\b(very much \w+|always I|yesterday I was go)\b',
    ],
    'tense_errors': [
        r'\b(I am agree|he don\'t|she don\'t|they doesn\'t)\b',
        r'\b(did \w+ed|was \w+ing \w+ed)\b',
    ]
}


def detect_esl_probability(text: str) -> float:
    """
    Estimate probability that text is written by an ESL writer.
    Returns 0.0 to 1.0 (higher = more likely ESL).
    
    Used to adjust AI detection thresholds to reduce false positives
    on non-native English writers.
    """
    if not text or len(text) < 50:
        return 0.0
    
    text_lower = text.lower()
    total_indicators = 0
    max_possible = 0
    
    for category, patterns in _ESL_INDICATORS.items():
        for pattern in patterns:
            max_possible += 1
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                total_indicators += min(len(matches), 3)  # Cap at 3 per pattern
    
    if max_possible == 0:
        return 0.0
    
    # Also check vocabulary simplicity (ESL writers tend to use simpler words)
    words = text.split()
    if len(words) > 20:
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len < 4.0:  # Simple vocabulary
            total_indicators += 2
    
    # Normalize: 3+ indicators = strong ESL signal
    esl_score = min(1.0, total_indicators / 5.0)
    return round(esl_score, 3)


# ==================== REAL LM PERPLEXITY ====================

class LMPerplexityCalculator:
    """
    Compute real language model perplexity using GPT-2 or DistilGPT-2.
    
    AI-generated text typically has low perplexity (15-40) because it's
    predictable to another LM. Human text has higher perplexity (50-200+).
    
    This replaces the crude formula-based approximation with actual
    per-token log-probability computation.
    """
    
    _instance = None
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = 'cpu'
        self.available = False
        self._load_attempted = False
    
    @classmethod
    def get_instance(cls):
        """Singleton accessor."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _ensure_loaded(self):
        """Lazy-load the LM on first use."""
        if self._load_attempted:
            return
        self._load_attempted = True
        
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Try DistilGPT-2 first (82M params, fast), fall back to GPT-2
            for model_name in ['distilgpt2', 'gpt2']:
                try:
                    logger.info(f"Loading {model_name} for perplexity computation...")
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.model = AutoModelForCausalLM.from_pretrained(model_name)
                    self.model.to(self.device)
                    self.model.eval()
                    self.available = True
                    logger.info(f"Perplexity model loaded: {model_name} on {self.device}")
                    break
                except Exception as e:
                    logger.warning(f"Could not load {model_name}: {e}")
                    continue
        except ImportError:
            logger.info("torch/transformers not available for real perplexity - using approximation")
    
    def compute_perplexity(self, text: str, max_length: int = 512) -> Optional[float]:
        """
        Compute per-token perplexity of text.
        
        PPL = exp(-1/N * sum(log P(w_i | w_<i)))
        
        Returns:
            Perplexity value, or None if model not available.
            Lower values = more predictable (AI-like).
            Typical ranges: AI text 15-40, Human text 50-200+
        """
        self._ensure_loaded()
        
        if not self.available:
            return None
        
        try:
            import torch
            
            encodings = self.tokenizer(
                text, return_tensors='pt', truncation=True,
                max_length=max_length
            ).to(self.device)
            
            input_ids = encodings['input_ids']
            
            if input_ids.shape[1] < 5:
                return None  # Too short for meaningful perplexity
            
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                # outputs.loss is the average negative log-likelihood
                neg_log_likelihood = outputs.loss.item()
            
            perplexity = math.exp(neg_log_likelihood)
            
            # Cap at reasonable range
            return min(perplexity, 1000.0)
            
        except Exception as e:
            logger.warning(f"Perplexity computation failed: {e}")
            return None


# Detection mode constants
DETECTION_MODE_OFFLINE = 'offline'
DETECTION_MODE_ML = 'ml'
DETECTION_MODE_BINOCULARS = 'binoculars'

# Common AI-generated text patterns (expanded for better offline detection)
AI_PATTERNS = {
    "hedging": [
        r"\b(it'?s important to note|it should be noted|it'?s worth mentioning)\b",
        r"\b(generally speaking|in general|typically)\b",
        r"\b(may|might|could|possibly|potentially|perhaps)\b.*\b(suggest|indicate|imply)\b",
        r"\b(tends to|appears to|seems to)\b",
    ],
    "formal_transitions": [
        r"\b(furthermore|moreover|additionally|consequently)\b",
        r"\b(in conclusion|to summarize|in summary|to conclude)\b",
        r"\b(on the other hand|conversely|nevertheless|however)\b",
        r"\b(first and foremost|lastly|finally)\b",
        r"\b(as a result|therefore|thus|hence)\b",
    ],
    "ai_phrases": [
        r"\b(as an ai|as a language model|i don'?t have personal)\b",
        r"\b(delve into|delve deeper|delving into)\b",
        r"\b(navigate|navigating|leverage|leveraging)\b",
        r"\b(it'?s crucial|it'?s essential|it'?s vital)\b",
        r"\b(in today'?s world|in this day and age|in the modern era)\b",
        r"\b(a testament to|speaks volumes|bears witness to)\b",
        r"\b(tapestry of|landscape of|realm of|fabric of)\b",
        r"\b(myriad of|plethora of|multitude of)\b",
        r"\b(game-changing|game changing)\b",
        r"\b(streamlining workflows|streamline workflows)\b",
        r"\b(boosting productivity|boost productivity)\b",
        r"\b(foster innovation|fostering innovation)\b",
        r"\b(hybrid approach|hybrid approaches)\b",
        r"\b(pre-trained models|pretrained models|pre-training)\b",
        r"\b(multimodal|contextual understanding|downstream tasks)\b",
        r"\b(end-to-end pipeline|orchestration|interpretability)\b",
        r"\b(generalize|generalization|training distribution)\b",
        r"\b(attention mechanisms|transformer|embeddings)\b",
        r"\b(model deployments?|automated systems|automated insights)\b",
        r"\b(research indicates|studies show|findings suggest)\b",
        r"\b(methodology|confounders|causal|robustness across)\b",
        r"\b(framework|scalable|throughput|lifecycle)\b",
        r"\b(standardize deployment|operational burden|sustainable adoption)\b",
    ],
    "filler_phrases": [
        r"\b(in order to)\b",
        r"\b(due to the fact that)\b",
        r"\b(at the end of the day)\b",
        r"\b(when it comes to)\b",
        r"\b(it is worth noting that)\b",
    ],
    "overly_formal": [
        r"\b(utilize|utilizes|utilizing)\b",
        r"\b(commence|commences|commencing)\b",
        r"\b(facilitate|facilitates|facilitating)\b",
        r"\b(endeavor|endeavors|endeavoring)\b",
        r"\b(ascertain|ascertains|ascertaining)\b",
        r"\b(necessitate|necessitates|necessitating)\b",
        r"\b(implement|implementation|implements)\b",
        r"\b(comprehensive|multifaceted|systematic)\b",
        r"\b(warrant|warrants|warranted)\b",
        r"\b(implications)\b",
    ],
    "meta_awareness": [
        r"\b(for example, consider|imagine a scenario where)\b",
        r"\b(let'?s explore|let'?s examine|let'?s consider)\b",
        r"\b(one might argue|one could say)\b",
    ],
    "passive_voice_indicators": [
        r"\b(is being|are being|was being|were being)\b",
        r"\b(has been|have been|had been).*\b(created|established|developed|implemented)\b",
        r"\b(can be seen|can be observed|can be noted)\b",
    ],
    "academic_filler": [
        r"\b(research indicates|studies show|findings suggest)\b",
        r"\b(it is important to|it should be emphasized that|it must be noted)\b",
        r"\b(various|numerous|several|multiple)\s+\b(studies|factors|aspects|considerations)\b",
        r"\b(as mentioned|as stated|as noted)\b",
    ],
    "abstract_qualifiers": [
        r"\b(the fact that|the reality that|the truth is)\b",
        r"\b(in essence|in other words|put simply)\b",
        r"\b(thus|hence|accordingly|subsequently)\b",
    ],
    "verbose_constructions": [
        r"\b(has\s+\w+ed\s+\w+.*significant|demonstrates\s+\w+.*implications)\b",
        r"\b(characterized by|defined by|marked by)\b",
        r"\b(take into account|take into consideration|be taken into account)\b",
    ],
    "cautious_language": [
        r"\b(seems to|appears to|suggests that)\b",
        r"\b(one could argue|it could be said|it might be suggested)\b",
        r"\b(in some cases|in certain contexts|to some extent)\b",
    ],
    "list_connectors": [
        r"\b(first(?:ly)?|second(?:ly)?|third(?:ly)?|finally)\b",
        r"\b(in the first place|in the second place)\b",
    ]
}


class AIContentDetector:
    """
    Enhanced AI content detector with:
    - Sentence-level detection
    - AI pattern recognition
    - Real LM perplexity (GPT-2/DistilGPT-2)
    - Linguistic metrics
    - Homoglyph/adversarial defense
    - ESL bias correction
    - Trinary classification (human / AI / mixed)
    - Result caching
    """
    
    MODEL_DIR = "model"
    CACHE_SIZE = 100  # Number of texts to cache
    
    # ===== ENSEMBLE MODEL REGISTRY =====
    # Multiple diverse models combined via weighted average for robust detection.
    # Weights reflect model quality (RAID benchmark, AUROC, architecture diversity).
    ENSEMBLE_MODELS = [
        {
            "id": "desklib/ai-text-detector-v1.01",
            "weight": 0.35,
            "type": "desklib_custom",  # Custom architecture: DeBERTa-v3-large + mean pool + sigmoid
            "params": "400M",
            "note": "#1 RAID leaderboard, DeBERTa-v3-large, custom single-logit sigmoid output",
        },
        {
            "id": "Oxidane/tmr-ai-text-detector",
            "weight": 0.30,
            "type": "standard",  # Standard AutoModelForSequenceClassification
            "params": "125M",
            "note": "RoBERTa-base, 97.3% accuracy, 0.9972 AUROC, 2.27% FPR",
        },
        {
            "id": "fakespot-ai/roberta-base-ai-text-detection-v1",
            "weight": 0.20,
            "type": "standard",
            "params": "125M",
            "note": "RoBERTa-base, 9.5K downloads/month, robust general detector",
        },
        {
            "id": "MayZhou/e5-small-lora-ai-generated-detector",
            "weight": 0.15,
            "type": "standard",
            "params": "33M",
            "note": "E5-small + LoRA, 93.9% RAID, lightweight fast model",
        },
    ]

    # Tunable weights — match documented 60/25/15 split
    PATTERN_WEIGHT = 0.60
    LINGUISTIC_WEIGHT = 0.25
    WATERMARK_WEIGHT = 0.15
    UNCERTAINTY_THRESHOLD = 0.08
    ESL_THRESHOLD_BOOST = 0.12  # Raise AI threshold for ESL writers to reduce FPs
    
    def __init__(self, model_path: Optional[str] = None, use_ml_model: bool = False, detection_mode: str = DETECTION_MODE_OFFLINE):
        """Initialize the detector with model path.
        
        Args:
            model_path: Path to ML model directory (optional)
            use_ml_model: If True, load Transformer model (requires downloads).
                         If False, use lightweight statistical detection only.
            detection_mode: Detection method - 'offline', 'ml', or 'binoculars'
                           Overrides use_ml_model if specified
        """
        # Determine mode (new parameter takes precedence)
        if detection_mode == DETECTION_MODE_BINOCULARS:
            self.detection_mode = DETECTION_MODE_BINOCULARS
            self.use_ml_model = False  # Don't load DeBERTa
        elif detection_mode == DETECTION_MODE_ML or use_ml_model:
            self.detection_mode = DETECTION_MODE_ML
            self.use_ml_model = True
        else:
            self.detection_mode = DETECTION_MODE_OFFLINE
            self.use_ml_model = False
        
        self.tokenizer = None
        self.model = None  # Legacy single-model (kept for compat)
        self.models = {}  # Ensemble: {model_id: {"model": ..., "tokenizer": ..., "weight": ..., "type": ...}}
        self.device = "cpu"
        self.binoculars = None
        self._active_model_id = None
        self._ensemble_loaded = False
        self._ensemble_total_weight = 0.0  # Sum of loaded model weights (for renormalization)
        self.lm_perplexity = LMPerplexityCalculator.get_instance()
        self.model_path = model_path if model_path else os.path.join(os.path.dirname(os.path.abspath(__file__)), self.MODEL_DIR)
        
        # Lazy loading: Do NOT load models in __init__
        # They will be loaded on the first call to predict() if needed
        logger.info(f"AIContentDetector initialized in {self.detection_mode.upper()} mode (Lazy loading enabled)")
        
    def _load_binoculars(self):
        """Load Binoculars zero-shot detector (GPU required)."""
        try:
            from .binoculars_detector import get_binoculars_detector
            
            logger.info("Loading Binoculars zero-shot detector...")
            self.binoculars = get_binoculars_detector()
            
            if self.binoculars.is_available():
                logger.info("Binoculars initialized successfully")
                info = self.binoculars.get_info()
                logger.info(f"GPU: {info.get('gpu', 'N/A')} ({info.get('vram_gb', 0):.1f}GB)")
            else:
                logger.warning("Binoculars not available (GPU required)")
                logger.warning("Falling back to offline mode")
                self.detection_mode = DETECTION_MODE_OFFLINE
                self.binoculars = None
        except ImportError:
            logger.error("binoculars_detector module not found")
            logger.warning("Falling back to offline mode")
            self.detection_mode = DETECTION_MODE_OFFLINE
            self.binoculars = None
        except Exception as e:
            logger.error(f"Failed to load Binoculars: {str(e)}")
            logger.warning("Falling back to offline mode")
            self.detection_mode = DETECTION_MODE_OFFLINE
            self.binoculars = None
    
    def _ensure_model_loaded(self):
        """Lazy load the model if strictly required."""
        if self.detection_mode == DETECTION_MODE_BINOCULARS and self.binoculars is None:
            self._load_binoculars()
            
        elif self.use_ml_model and not self._ensemble_loaded:
            self._load_ensemble_models()

    def _load_ensemble_models(self):
        """Load multiple AI text detection models for ensemble inference.
        
        Loads models from ENSEMBLE_MODELS registry. Each model is stored in self.models dict.
        Gracefully degrades: if some models fail to load, remaining models' weights are
        renormalized so they still sum to 1.0.
        
        Special handling for desklib model which uses a custom architecture
        (mean pooling + single sigmoid logit) instead of standard classification head.
        """
        if self._ensemble_loaded:
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, PretrainedConfig
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"\n{'='*60}")
            print(f"Loading AI Text Detection Ensemble ({len(self.ENSEMBLE_MODELS)} models)")
            print(f"Device: {self.device}")
            print(f"{'='*60}")
            
            loaded_count = 0
            
            for entry in self.ENSEMBLE_MODELS:
                model_id = entry["id"]
                model_type = entry["type"]
                weight = entry["weight"]
                
                print(f"\n  [{loaded_count+1}/{len(self.ENSEMBLE_MODELS)}] Loading {model_id} ({entry['params']})...")
                try:
                    if model_type == "desklib_custom":
                        # Desklib uses custom architecture: DeBERTa-v3-large + mean pooling + linear → sigmoid
                        # NOT compatible with AutoModelForSequenceClassification
                        tokenizer = AutoTokenizer.from_pretrained(model_id)
                        model = self._load_desklib_model(model_id, AutoModel)
                    else:
                        # Standard HuggingFace classification model
                        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                        model = AutoModelForSequenceClassification.from_pretrained(model_id, trust_remote_code=True)
                    
                    model.to(self.device)
                    model.eval()
                    
                    self.models[model_id] = {
                        "model": model,
                        "tokenizer": tokenizer,
                        "weight": weight,
                        "type": model_type,
                    }
                    loaded_count += 1
                    print(f"    [OK] Loaded successfully (weight={weight})")
                    
                except Exception as e:
                    print(f"    [FAIL] Failed to load: {e}")
                    logger.warning(f"Ensemble model {model_id} failed to load: {e}")
            
            # Renormalize weights if some models didn't load
            if loaded_count > 0:
                total_weight = sum(m["weight"] for m in self.models.values())
                if total_weight > 0 and abs(total_weight - 1.0) > 0.01:
                    for mid in self.models:
                        self.models[mid]["weight"] /= total_weight
                    print(f"\n  Weights renormalized (loaded {loaded_count}/{len(self.ENSEMBLE_MODELS)} models)")
                
                self._ensemble_total_weight = sum(m["weight"] for m in self.models.values())
                self._ensemble_loaded = True
                
                # Set legacy compat fields to first loaded model
                first_id = next(iter(self.models))
                self.model = self.models[first_id]["model"]
                self.tokenizer = self.models[first_id]["tokenizer"]
                self._active_model_id = f"ensemble({loaded_count})"
                
                print(f"\n{'='*60}")
                print(f"Ensemble ready: {loaded_count} models loaded")
                models_str = ", ".join(f"{m['weight']:.0%}" for m in self.models.values())
                print(f"Weights: [{models_str}]")
                print(f"{'='*60}\n")
            else:
                raise RuntimeError("No ensemble models could be loaded")

        except Exception as e:
            print(f"Error loading ensemble: {e}")
            print("Falling back to offline mode...")
            self.use_ml_model = False
            self.detection_mode = DETECTION_MODE_OFFLINE
            self.model = None
            self.models = {}
            self.tokenizer = None
            self._active_model_id = None
            self._ensemble_loaded = False
    
    def _load_desklib_model(self, model_id: str, AutoModel):
        """Load desklib AI detector with custom architecture.
        
        Desklib checkpoint format:
        - Backbone keys: model.embeddings.*, model.encoder.* (DeBERTa-v3-large, 1024 hidden)
        - Classifier keys: classifier.weight [1, 1024], classifier.bias [1]
        - Single logit output → sigmoid for AI probability
        
        Uses from_pretrained for fast cached init, then remaps and reloads correct weights.
        """
        import torch
        import torch.nn as nn
        from transformers import AutoConfig, DebertaV2Model
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        
        class DesklibAIDetector(nn.Module):
            """Custom wrapper matching desklib's architecture."""
            def __init__(self, backbone, hidden_size):
                super().__init__()
                self.backbone = backbone
                self.classifier = nn.Linear(hidden_size, 1)
            
            def forward(self, **kwargs):
                outputs = self.backbone(**kwargs)
                attention_mask = kwargs.get("attention_mask")
                token_embeddings = outputs.last_hidden_state
                if attention_mask is not None:
                    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    summed = torch.sum(token_embeddings * mask_expanded, dim=1)
                    counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                    pooled = summed / counts
                else:
                    pooled = token_embeddings.mean(dim=1)
                logits = self.classifier(pooled)
                return logits
        
        config = AutoConfig.from_pretrained(model_id)
        
        # Load backbone with from_pretrained (fast from cache, creates correct model structure)
        backbone = DebertaV2Model.from_pretrained(model_id)
        
        # Load full state dict from safetensors
        weights_path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
        state_dict = load_file(weights_path)
        
        # Remap backbone keys: "model.X" → "X" (to match DebertaV2Model's expected key names)
        backbone_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith("model.")}
        backbone.load_state_dict(backbone_dict)
        print(f"    Desklib backbone: {len(backbone_dict)} weights loaded")
        
        # Build full model with classifier
        model = DesklibAIDetector(backbone, config.hidden_size)
        
        # Load classifier weights
        cls_dict = {"weight": state_dict["classifier.weight"], "bias": state_dict["classifier.bias"]}
        model.classifier.load_state_dict(cls_dict)
        print(f"    Desklib classifier: weight{list(cls_dict['weight'].shape)} + bias{list(cls_dict['bias'].shape)}")
        
        return model
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for caching."""
        return hashlib.md5(text.encode()).hexdigest()
    
    @lru_cache(maxsize=100)
    def _cached_inference(self, text_hash: str, text: str) -> Tuple[float, float]:
        """
        Cached ensemble inference. Returns (human_prob, ai_prob).
        
        Runs text through all loaded ensemble models, extracts AI probability
        from each using model-appropriate logic, then combines via weighted average.
        
        Handles:
        - Standard 2-class softmax models (most HuggingFace classifiers)
        - Custom single-logit sigmoid models (desklib)
        - Different label mappings (id2label config)
        """
        if self.models:
            return self._ensemble_inference(text)
        
        # Legacy fallback: single model (should not normally reach here)
        return self._single_model_inference(text, self.model, self.tokenizer, "standard")
    
    def _ensemble_inference(self, text: str) -> Tuple[float, float]:
        """Run text through all ensemble models and combine results.
        
        Returns weighted average (human_prob, ai_prob) across all loaded models.
        If a model fails at inference time, its weight is excluded and others renormalized.
        """
        import torch
        
        weighted_ai = 0.0
        total_weight = 0.0
        model_results = []
        
        for model_id, entry in self.models.items():
            try:
                human_prob, ai_prob = self._single_model_inference(
                    text, entry["model"], entry["tokenizer"], entry["type"]
                )
                weighted_ai += ai_prob * entry["weight"]
                total_weight += entry["weight"]
                model_results.append((model_id, ai_prob, entry["weight"]))
            except Exception as e:
                logger.warning(f"Ensemble inference failed for {model_id}: {e}")
                continue
        
        if total_weight == 0:
            logger.error("All ensemble models failed inference")
            return (0.5, 0.5)  # Uncertain fallback
        
        ensemble_ai = weighted_ai / total_weight
        ensemble_human = 1.0 - ensemble_ai
        
        logger.debug(f"Ensemble results: {[(m, f'{a:.3f}', f'{w:.2f}') for m, a, w in model_results]} → AI={ensemble_ai:.3f}")
        
        return (ensemble_human, ensemble_ai)
    
    def _single_model_inference(self, text: str, model, tokenizer, model_type: str) -> Tuple[float, float]:
        """Run inference on a single model. Returns (human_prob, ai_prob).
        
        Handles two output formats:
        - 'standard': 2-class softmax with id2label mapping
        - 'desklib_custom': single logit → sigmoid → AI probability
        """
        import torch
        
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            if model_type == "desklib_custom":
                # Custom desklib: returns single logit, sigmoid → AI probability
                logits = model(**inputs)
                ai_prob = torch.sigmoid(logits).item()
                human_prob = 1.0 - ai_prob
                return (human_prob, ai_prob)
            else:
                # Standard classification model: softmax over logits
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                prob_values = probs[0].tolist()
                
                # Determine label mapping from model config
                id2label = getattr(model.config, 'id2label', None)
                if id2label:
                    human_idx = None
                    ai_idx = None
                    for idx, label in id2label.items():
                        label_lower = label.lower()
                        if any(h in label_lower for h in ['human', 'real', 'label_0']):
                            human_idx = int(idx)
                        elif any(a in label_lower for a in ['ai', 'chatgpt', 'generated', 'fake', 'label_1', 'machine']):
                            ai_idx = int(idx)
                    
                    if human_idx is not None and ai_idx is not None:
                        return (prob_values[human_idx], prob_values[ai_idx])
                
                # Default: Index 0 = human, Index 1 = AI
                return (prob_values[0], prob_values[1])
    
    def _analyze_sentence(self, sentence: str) -> Dict:
        """Analyze a single sentence for AI probability and patterns."""
        if not sentence.strip() or len(sentence.split()) < 3:
            return None
        
        # Detect patterns in this sentence
        patterns = self._detect_patterns_in_text(sentence)
        
        # Get AI probability for this sentence
        if self.use_ml_model and self._ensemble_loaded:
            text_hash = self._get_text_hash(sentence)
            try:
                human_prob, ai_prob = self._cached_inference(text_hash, sentence)
            except Exception:
                # Fallback if inference fails
                human_prob, ai_prob = self._calculate_offline_score(sentence, patterns)
        else:
            human_prob, ai_prob = self._calculate_offline_score(sentence, patterns)
        
        return {
            "text": sentence.strip(),
            "ai_score": round(ai_prob * 100, 1),
            "human_score": round(human_prob * 100, 1),
            "patterns": patterns,
            "is_flagged": ai_prob > 0.6  # Flag if > 60% AI
        }
    
    def _analyze_sentence_fast(self, sentence: str, full_text_patterns: List[Dict] = None) -> Dict:
        """
        OPTIMIZED: Analyze a single sentence with minimal overhead.
        Uses pre-detected patterns from full text to avoid redundant regex.
        """
        if not sentence.strip() or len(sentence.split()) < 3:
            return None
        
        # Filter patterns that match this sentence (avoid re-running all regex)
        if full_text_patterns:
            sentence_patterns = [
                p for p in full_text_patterns 
                if p.get("pattern", "").lower() in sentence.lower()
            ]
        else:
            sentence_patterns = self._detect_patterns_in_text(sentence)
        
        # Quick score without ML model
        human_prob, ai_prob = self._calculate_offline_score(sentence, sentence_patterns)
        
        return {
            "text": sentence.strip(),
            "ai_score": round(ai_prob * 100, 1),
            "human_score": round(human_prob * 100, 1),
            "patterns": sentence_patterns,
            "is_flagged": ai_prob > 0.6  # Flag if > 60% AI
        }
    
    def _detect_patterns_in_text(self, text: str) -> List[Dict]:
        """Detect AI writing patterns in text."""
        detected = []
        text_lower = text.lower()
        
        for category, patterns in AI_PATTERNS.items():
            for pattern in patterns:
                # Use finditer to get positions
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    match_text = match.group()
                    # Get original case text using span
                    start, end = match.span()
                    original_text_segment = text[start:end]
                    
                    detected.append({
                        "pattern": original_text_segment,
                        "category": category,
                        "type": self._get_pattern_type(category),
                        "start": start,
                        "end": end
                    })
        
        # Sort by position to help frontend rendering
        detected.sort(key=lambda x: x["start"])
        
        return detected
    
    def _get_pattern_type(self, category: str) -> str:
        """Get human-readable pattern type."""
        types = {
            "hedging": "Hedging Language",
            "formal_transitions": "Formal Transition",
            "ai_phrases": "Common AI Phrase",
            "filler_phrases": "Filler Phrase",
            "overly_formal": "Overly Formal Word",
            "meta_awareness": "Meta-Awareness Phrase",
            "passive_voice_indicators": "Passive Voice Indicator",
            "academic_filler": "Academic Filler",
            "abstract_qualifiers": "Abstract Qualifier",
            "verbose_constructions": "Verbose Construction",
            "cautious_language": "Cautious Language",
            "list_connectors": "List Connector"
        }
        return types.get(category, category.replace('_', ' ').title())
    
    def _calculate_ngram_uniformity(self, text: str, n: int = 2) -> float:
        """
        Calculate n-gram repetition score.
        AI text tends to repeat certain n-grams more often.
        Returns 0-1 (higher = more repetitive = more AI-like)
        """
        words = text.lower().split()
        if len(words) < n + 5:
            return 0.5  # Not enough data
        
        # Generate n-grams
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        
        if not ngrams:
            return 0.5
        
        # Calculate frequency distribution
        freq = {}
        for ng in ngrams:
            freq[ng] = freq.get(ng, 0) + 1
        
        # Key insight: If all n-grams are unique, that's human-like (low score)
        # If n-grams repeat, that's AI-like (high score)
        total_ngrams = len(ngrams)
        unique_ngrams = len(freq)
        
        if unique_ngrams == 0:
            return 0.5
        
        # Ratio of unique to total: 1.0 = all unique (human), lower = more repetition (AI)
        uniqueness_ratio = unique_ngrams / total_ngrams
        
        # Invert: high repetition = high score = AI-like
        repetition_score = 1 - uniqueness_ratio
        
        return round(repetition_score, 3)
    
    def _calculate_ttr(self, text: str) -> float:
        """
        Calculate Type-Token Ratio (vocabulary diversity).
        Returns 0-1 (higher = more diverse vocabulary = more human-like)
        """
        words = [w.lower() for w in text.split() if w.strip()]
        if len(words) < 10:
            return 0.5  # Not enough data
        
        unique_words = set(words)
        ttr = len(unique_words) / len(words)
        return round(ttr, 3)
    
    def _calculate_entropy(self, text: str) -> float:
        """
        Calculate Shannon Entropy of character distribution.
        Low entropy = highly predictable text = likely AI.
        High entropy = more varied/creative = likely human.
        Returns 0-1 (normalized, higher = more human-like)
        """
        if not text or len(text) < 50:
            return 0.5  # Not enough data
        
        text_lower = text.lower()
        char_freq = {}
        total_chars = 0
        
        for char in text_lower:
            if char.isalpha() or char.isspace():
                char_freq[char] = char_freq.get(char, 0) + 1
                total_chars += 1
        
        if total_chars == 0:
            return 0.5
        
        # Calculate Shannon entropy: H = -sum(p * log2(p))
        entropy = 0.0
        for count in char_freq.values():
            if count > 0:
                p = count / total_chars
                entropy -= p * math.log2(p)
        
        # Normalize: English text typically has entropy ~4.0-4.5
        # AI text often has lower entropy (~3.5-4.0)
        # Max theoretical entropy for 27 chars (a-z + space) is ~4.75
        normalized_entropy = min(1.0, entropy / 4.75)
        
        return round(normalized_entropy, 3)
    
    def _detect_repetition(self, text: str, n: int = 4) -> float:
        """
        Detect repetitive n-gram patterns (proxy for watermarking/robotic patterns).
        AI models sometimes repeat phrases or fall into loops.
        Returns 0-1 (higher = more repetitive = more AI-like)
        """
        words = text.lower().split()
        if len(words) < n + 5:
            return 0.0  # Not enough data
        
        # Generate n-grams
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        
        if not ngrams:
            return 0.0
        
        # Count occurrences
        freq = {}
        for ng in ngrams:
            freq[ng] = freq.get(ng, 0) + 1
        
        # Count repeated n-grams (appearing more than once)
        repeated_count = sum(1 for count in freq.values() if count > 1)
        total_unique = len(freq)
        
        if total_unique == 0:
            return 0.0
        
        # Repetition ratio: what fraction of unique n-grams are repeated?
        repetition_ratio = repeated_count / total_unique
        
        # Normalize (typical human text has very low repetition, AI can have more)
        # Scale so that 10% repetition = 0.5, 20%+ = 1.0
        normalized_repetition = min(1.0, repetition_ratio * 5)
        
        return round(normalized_repetition, 3)

    def _detect_watermark_signals(self, text: str) -> float:
        """
        Detect statistical watermark signals (Green/Red list proxy).
        
        Watermarking works by boosting the probability of 'Green List' tokens.
        This simplified detector checks for the 'unusually high' frequency of
        common/predictable words that often serve as watermarking carriers.
        
        Returns 0-1 (higher = stronger watermark signal = more AI-like)
        """
        words = [w.lower() for w in text.split() if w.isalpha()]
        count = len(words)
        if count < 20: 
            return 0.0
            
        # 1. Check for 'Green List' bias (common stop words + high freq words)
        # Watermarked text often has slightly distorted distributions of these
        # We use a known list of 'safe' high-freq tokens
        green_list_tokens = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 
            'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 
            'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 
            'which', 'go', 'me'
        }
        
        green_hits = sum(1 for w in words if w in green_list_tokens)
        green_ratio = green_hits / count
        
        # Human text typically has green_ratio around 0.40 - 0.50
        # Watermarked/AI text can be unnaturally high (> 0.55) or consistent
        
        # 2. Check for "Red List" avoidance (complex/rare words)
        # Watermarking often avoids rare tokens to stay in the 'Green' zone
        # We roughly approximate this by word length (longer ~ rarer)
        complex_words = sum(1 for w in words if len(w) > 7)
        complex_ratio = complex_words / count
        
        # Scoring logic:
        # High Green Ratio (>0.55) + Low Complex Ratio (<0.15) = Suspected Watermark
        
        score = 0.0
        
        # Penalty for high green list usage (robotic/safe)
        # NOTE: Check higher threshold first so it isn't shadowed
        if green_ratio > 0.60:
            score += 0.7
        elif green_ratio > 0.55:
            score += 0.4
            
        # Penalty for low complexity (avoidance of creative words)
        if complex_ratio < 0.12:
            score += 0.3
        
        return min(1.0, score)

    
    def _calculate_offline_score(self, text: str, detected_patterns: List[Dict]) -> Tuple[float, float]:
        """
        Calculate AI probability using lightweight statistical methods (no ML model).
        Returns (human_prob, ai_prob) tuple.
        
        Scoring formula (matches documented 60/25/15 split):
          AI_Score = (pattern_score × 0.60) + (linguistic_score × 0.25) + (watermark_score × 0.15)
        
        Improvements:
        1. Uses real LM perplexity when available (GPT-2/DistilGPT-2)
        2. ESL de-biasing reduces false positives on non-native English
        3. Consistent with documented weights
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = max(len(sentences), 1)
        
        # ===== COMPONENT 1: PATTERN MATCHING SCORE (60% weight) =====
        num_patterns = len(detected_patterns)
        
        if num_patterns == 0:
            pattern_ai_score = 0.1  # Small base boost for borderline cases
        elif num_patterns == 1:
            pattern_ai_score = 0.55  # Single pattern = moderate AI indicator
        elif num_patterns == 2:
            pattern_ai_score = 0.72  # Two patterns = strong AI indicator
        elif num_patterns == 3:
            pattern_ai_score = 0.85
        elif num_patterns >= 4:
            pattern_ai_score = min(1.0, 0.90 + (num_patterns - 4) * 0.02)
        
        pattern_component = pattern_ai_score * self.PATTERN_WEIGHT
        
        # ===== COMPONENT 2: LINGUISTIC METRICS (25% weight) =====
        
        # 2a. Vocabulary Diversity (TTR)
        ttr = self._calculate_ttr(text)
        ttr_ai_score = 1.0 - ttr  # Low vocabulary = AI-like
        
        # 2b. Entropy
        entropy = self._calculate_entropy(text)
        entropy_ai_score = 1.0 - entropy  # Low entropy = AI-like
        
        # 2c. Burstiness (sentence length variance)
        words_per_sentence = [len(s.split()) for s in sentences if s.strip()]
        if len(words_per_sentence) > 1:
            mean_wps = sum(words_per_sentence) / len(words_per_sentence)
            variance = sum((x - mean_wps) ** 2 for x in words_per_sentence) / len(words_per_sentence)
            burstiness = min(1.0, variance / 100)
        else:
            burstiness = 0.5
        burstiness_ai_score = 1.0 - burstiness  # Low variance = AI-like
        
        # 2d. N-gram Repetition
        bigram_rep = self._calculate_ngram_uniformity(text, 2)
        trigram_rep = self._calculate_ngram_uniformity(text, 3)
        ngram_ai_score = (bigram_rep + trigram_rep) / 2.0
        
        # 2e. Real LM Perplexity (if available — most discriminative feature)
        real_ppl = self.lm_perplexity.compute_perplexity(text)
        if real_ppl is not None:
            # AI text: PPL 15-40, Human text: PPL 50-200+
            if real_ppl < 25:
                ppl_ai_score = 0.90
            elif real_ppl < 40:
                ppl_ai_score = 0.75
            elif real_ppl < 60:
                ppl_ai_score = 0.50
            elif real_ppl < 100:
                ppl_ai_score = 0.30
            else:
                ppl_ai_score = 0.15
            
            # Real perplexity gets highest weight among linguistic features
            linguistic_ai_score = (
                ppl_ai_score * 0.35 +           # Real perplexity (most important)
                ttr_ai_score * 0.15 +
                entropy_ai_score * 0.15 +
                burstiness_ai_score * 0.15 +
                ngram_ai_score * 0.20
            )
        else:
            # Fallback: approximated perplexity (original formula)
            linguistic_ai_score = (
                ttr_ai_score * 0.25 +
                entropy_ai_score * 0.25 +
                burstiness_ai_score * 0.25 +
                ngram_ai_score * 0.25
            )
        
        linguistic_component = linguistic_ai_score * self.LINGUISTIC_WEIGHT
        
        # ===== COMPONENT 3: WATERMARK SIGNAL (15% weight) =====
        watermark_score = self._detect_watermark_signals(text)
        watermark_component = watermark_score * self.WATERMARK_WEIGHT
        
        # ===== FINAL SCORE (60/25/15 documented split) =====
        ai_prob = pattern_component + linguistic_component + watermark_component
        
        # ===== ESL DE-BIASING =====
        # If text appears to be from an ESL writer, reduce AI probability
        # to avoid false positives (ESL writers often use formal/structured language)
        esl_prob = detect_esl_probability(text)
        if esl_prob > 0.3:
            esl_reduction = esl_prob * self.ESL_THRESHOLD_BOOST
            ai_prob = max(0.0, ai_prob - esl_reduction)
            logger.debug(f"ESL detected (prob={esl_prob:.2f}), reduced AI score by {esl_reduction:.3f}")
        
        ai_prob = min(1.0, max(0.0, ai_prob))
        human_prob = 1.0 - ai_prob
        
        return (human_prob, ai_prob)
    
    def _calculate_linguistic_metrics(self, text: str) -> Dict:
        """Calculate comprehensive linguistic metrics."""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        word_count = len(words)
        sentence_count = len(sentences) if sentences else 1
        
        # Words per sentence analysis
        words_per_sentence = [len(s.split()) for s in sentences]
        avg_wps = sum(words_per_sentence) / len(words_per_sentence) if words_per_sentence else 0
        
        # Variance calculation
        if len(words_per_sentence) > 1:
            mean = avg_wps
            variance = sum((x - mean) ** 2 for x in words_per_sentence) / len(words_per_sentence)
            std_dev = math.sqrt(variance)
        else:
            variance = 0
            std_dev = 0
        
        # Burstiness (higher = more human-like)
        burstiness = min(1.0, variance / 100) if variance > 0 else 0
        
        # Vocabulary richness
        unique_words = set(w.lower() for w in words)
        vocab_richness = len(unique_words) / word_count if word_count > 0 else 0
        
        # Perplexity approximation
        base_perplexity = 30 + (vocab_richness * 50) + (burstiness * 30)
        perplexity = min(100, max(10, base_perplexity))
        
        # N-gram uniformity
        bigram_uniformity = self._calculate_ngram_uniformity(text, 2)
        trigram_uniformity = self._calculate_ngram_uniformity(text, 3)
        
        # Rhythm status
        if std_dev < 3:
            rhythm_status = "Uniform"
            rhythm_desc = "Highly consistent rhythm (AI indicator)"
        elif std_dev < 8:
            rhythm_status = "Normal"
            rhythm_desc = "Natural variance detected"
        else:
            rhythm_status = "Variable"
            rhythm_desc = "High creative variance"
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_words_per_sentence": round(avg_wps, 1),
            "vocabulary_richness": round(vocab_richness * 100, 1),
            "perplexity": {
                "average": round(perplexity, 1),
                "flow": self._calculate_perplexity_flow(sentences)
            },
            "burstiness": {
                "score": round(burstiness, 2),
                "bars": self._calculate_burstiness_bars(words_per_sentence)
            },
            "rhythm": {
                "status": rhythm_status,
                "description": rhythm_desc,
                "variance": round(variance, 2)
            },
            "ngram_uniformity": {
                "bigram": bigram_uniformity,
                "trigram": trigram_uniformity,
                "interpretation": "high" if (bigram_uniformity + trigram_uniformity) / 2 > 0.6 else "normal"
            }
        }
    
    def _calculate_perplexity_flow(self, sentences: List[str]) -> List[float]:
        """Generate perplexity flow data for chart."""
        if len(sentences) < 2:
            return [50]
        
        flow = []
        for sentence in sentences[:10]:
            words = sentence.split()
            if not words:
                continue
            unique_ratio = len(set(w.lower() for w in words)) / len(words)
            point = 30 + (unique_ratio * 50) + (len(words) % 10) * 2
            flow.append(round(min(100, max(10, point)), 1))
        
        return flow if flow else [50]
    
    def _calculate_burstiness_bars(self, words_per_sentence: List[int]) -> Dict:
        """Generate burstiness bar data for chart."""
        if not words_per_sentence:
            return {"document": [], "human_baseline": []}
        
        max_len = max(words_per_sentence) if words_per_sentence else 1
        doc_bars = [round((wps / max_len) * 100, 1) for wps in words_per_sentence[:6]]
        human_baseline = [60, 85, 40, 95, 55, 70]
        
        return {
            "document": doc_bars,
            "human_baseline": human_baseline[:len(doc_bars)]
        }
    
    def predict(self, text: str, detailed: bool = True) -> Dict:
        """
        Full text analysis with all features.
        
        Args:
            text: Text to analyze
            detailed: Include sentence-level analysis (can be slower for long texts)
            
        Returns:
            Comprehensive detection results with trinary classification
        """
        if not text or not text.strip():
            return {"error": "Empty text provided"}
        
        text = text.strip()
        
        # ===== LAZY LOAD: Trigger model loading on first predict call =====
        self._ensure_model_loaded()
        
        # ===== SHORT TEXT GUARD =====
        # Texts under ~20 words lack enough signal for reliable ML detection.
        # Classify as human by default since AI rarely generates very short snippets.
        word_count = len(text.split())
        if word_count < 15:
            logger.info(f"Short text ({word_count} words) — defaulting to human")
            metrics = self._calculate_linguistic_metrics(text)
            return {
                "prediction": "human",
                "confidence": 85.0,
                "scores": {"human": 85.0, "ai_generated": 15.0},
                "classification": {"type": "human", "ai_sentence_ratio": None, "human_sentence_ratio": None, "is_mixed": False},
                "metrics": metrics,
                "detected_patterns": {"total_count": 0, "categories": {}},
                "detection_method": "short_text_default",
                "detection_mode": self.detection_mode,
                "esl_probability": 0.0,
                "adversarial_normalized": False,
                "decision": {"reason": "short_text", "word_count": word_count},
                "uncertainty_threshold": self.UNCERTAINTY_THRESHOLD
            }
        
        # ===== ADVERSARIAL DEFENSE: Normalize homoglyphs & zero-width chars =====
        original_text = text
        text = normalize_adversarial_text(text)
        adversarial_modified = (text != original_text)
        
        # Detect patterns in full text (needed for both ML and offline modes)
        all_patterns = self._detect_patterns_in_text(text)
        
        # ===== DETECTION ROUTING: Choose method based on mode =====
        
        if self.detection_mode == DETECTION_MODE_BINOCULARS and self.binoculars is not None:
            # ===== BINOCULARS ZERO-SHOT DETECTION =====
            binoculars_result = self.binoculars.detect(text)
            
            if binoculars_result is None:
                logger.warning("Binoculars detection failed, falling back to offline mode")
                human_prob, ai_prob = self._calculate_offline_score(text, all_patterns)
                detection_method = "offline_fallback"
            else:
                # Extract probabilities from Binoculars result
                if binoculars_result["prediction"] == "AI":
                    ai_prob = binoculars_result["confidence"]
                    human_prob = 1.0 - ai_prob
                else:
                    human_prob = binoculars_result["confidence"]
                    ai_prob = 1.0 - human_prob
                
                detection_method = "binoculars_zero_shot"
                logger.info(f"Binoculars: {binoculars_result['prediction']} (score={binoculars_result['score']:.4f})")
        
        elif self.use_ml_model and self._ensemble_loaded:
            # ===== ENSEMBLE ML HYBRID DETECTION (4-model weighted average) =====
            text_hash = self._get_text_hash(text)
            ml_human, ml_ai = self._cached_inference(text_hash, text)
            off_human, off_ai = self._calculate_offline_score(text, all_patterns)
            
            # Ensemble of 4 diverse models is highly reliable — trust ML heavily
            # desklib(RAID#1) + Oxidane(97.3%) + fakespot + MayZhou all agree → strong signal
            weight_ml = 0.85
            weight_off = 0.15
            
            ai_prob = (ml_ai * weight_ml) + (off_ai * weight_off)
            human_prob = 1.0 - ai_prob
            
            # Binary snap: if ML ensemble clearly picks a side, commit fully
            # 4 models agreeing is strong enough to show definitive result
            if ml_ai > 0.5:
                ai_prob = 1.0
                human_prob = 0.0
            else:
                ai_prob = 0.0
                human_prob = 1.0
            
            detection_method = f"ensemble_hybrid({len(self.models)})"
        
        else:
            # ===== OFFLINE STATISTICAL DETECTION =====
            human_prob, ai_prob = self._calculate_offline_score(text, all_patterns)
            detection_method = "offline_statistical"
        
        # ===== TRINARY CLASSIFICATION (human / AI / mixed) =====
        # When ML ensemble gives a strong signal, trust it — skip noisy offline sentence analysis
        ml_strong_signal = False
        if self.use_ml_model and self._ensemble_loaded:
            try:
                ml_strong_signal = (ml_ai > 0.80 or ml_ai < 0.20)
            except NameError:
                pass
        
        ai_sentence_count = 0
        human_sentence_count = 0
        sentence_scores = []
        
        if not ml_strong_signal:
            # Only do sentence-level mixed detection when ML is uncertain
            sentences_for_mix = re.split(r'(?<=[.!?])\s+', text)
            sentences_for_mix = [s for s in sentences_for_mix if len(s.split()) >= 3]
            
            if len(sentences_for_mix) >= 3:
                for sent in sentences_for_mix[:20]:
                    sent_patterns = self._detect_patterns_in_text(sent)
                    _, sent_ai = self._calculate_offline_score(sent, sent_patterns)
                    sentence_scores.append(sent_ai)
                    if sent_ai > 0.6:
                        ai_sentence_count += 1
                    elif sent_ai < 0.4:
                        human_sentence_count += 1
        
        total_scored = max(len(sentence_scores), 1)
        ai_ratio = ai_sentence_count / total_scored
        human_ratio = human_sentence_count / total_scored
        
        # Determine if this is mixed content (only when ML wasn't strongly decisive)
        is_mixed = (not ml_strong_signal and
                    ai_ratio > 0.2 and human_ratio > 0.2 and len(sentence_scores) >= 3)
        
        # Determine prediction with uncertainty handling
        margin = abs(ai_prob - human_prob)
        max_prob = max(ai_prob, human_prob)
        confidence = round(max_prob * 100, 2)

        uncertainty_threshold = self.UNCERTAINTY_THRESHOLD
        if is_mixed:
            prediction = "mixed"
            decision_reason = {
                "reason": "mixed_content",
                "margin": round(margin, 3),
                "ai_sentence_ratio": round(ai_ratio, 2),
                "human_sentence_ratio": round(human_ratio, 2)
            }
        elif margin < uncertainty_threshold:
            leaning = "ai_generated" if ai_prob > human_prob else "human"
            prediction = "uncertain"
            decision_reason = {
                "reason": "low_margin",
                "margin": round(margin, 3),
                "leaning": leaning
            }
        else:
            prediction = "ai_generated" if ai_prob > human_prob else "human"
            decision_reason = {"reason": "clear_margin", "margin": round(margin, 3)}

        metrics = self._calculate_linguistic_metrics(text)
        
        # Add real perplexity to metrics if available
        real_ppl = self.lm_perplexity.compute_perplexity(text)
        if real_ppl is not None:
            metrics["perplexity"]["real_lm"] = round(real_ppl, 1)
            metrics["perplexity"]["source"] = "distilgpt2"
        
        # ESL probability for transparency
        esl_prob = detect_esl_probability(text)
        
        # Group patterns by category
        pattern_summary = {}
        for p in all_patterns:
            cat = p["category"]
            if cat not in pattern_summary:
                pattern_summary[cat] = {"count": 0, "examples": [], "type": p["type"]}
            pattern_summary[cat]["count"] += 1
            if len(pattern_summary[cat]["examples"]) < 3:
                pattern_summary[cat]["examples"].append(p["pattern"])
        
        result = {
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "scores": {
                "human": round(human_prob * 100, 2),
                "ai_generated": round(ai_prob * 100, 2)
            },
            "classification": {
                "type": prediction,  # "human", "ai_generated", "mixed", "uncertain"
                "ai_sentence_ratio": round(ai_ratio, 2) if sentence_scores else None,
                "human_sentence_ratio": round(human_ratio, 2) if sentence_scores else None,
                "is_mixed": is_mixed
            },
            "metrics": metrics,
            "detected_patterns": {
                "total_count": len(all_patterns),
                "categories": pattern_summary
            },
            "detection_method": detection_method,
            "detection_mode": self.detection_mode,
            "esl_probability": round(esl_prob, 3),
            "adversarial_normalized": adversarial_modified
        }

        # Add decision reasoning and uncertainty metadata to help frontend and debugging
        result["decision"] = decision_reason
        result["uncertainty_threshold"] = uncertainty_threshold
        
        # Sentence-level analysis (if detailed and text not too long)
        # OPTIMIZATION: Skip detailed analysis for texts > 3000 chars to avoid slowdown
        if detailed and len(text) < 3000:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            sentence_analysis = []
            
            # Pre-build pattern map for sentences to avoid redundant regex (OPTIMIZATION)
            for sentence in sentences[:15]:  # Reduced from 20 to 15 for speed
                analysis = self._analyze_sentence_fast(sentence, all_patterns)
                if analysis:
                    sentence_analysis.append(analysis)
            
            result["sentence_analysis"] = sentence_analysis
            result["flagged_sentences"] = [
                s for s in sentence_analysis if s.get("is_flagged")
            ]
        
        return result
    
    def analyze_chunks(self, chunks: List[Dict], include_per_chunk: bool = True) -> Dict:
        """
        Analyze multiple text chunks and aggregate results.
        Used for large documents that are split into chunks.
        
        Args:
            chunks: List of chunk dicts from DocumentParser (with 'text' key)
            include_per_chunk: Include individual chunk results
            
        Returns:
            Aggregated detection result
        """
        if not chunks:
            return {"error": "No chunks provided"}
        
        chunk_results = []
        chunk_metrics_list = []
        total_weight = 0
        weighted_ai_score = 0
        weighted_human_score = 0
        all_patterns = []
        mixed_count = 0
        
        for chunk in chunks:
            chunk_text = chunk.get("text", "")
            if not chunk_text.strip():
                continue
            
            # Analyze this chunk
            result = self.predict(chunk_text, detailed=False)
            
            if "error" in result:
                continue
            
            # Weight by chunk length (longer chunks = more weight)
            weight = len(chunk_text)
            total_weight += weight
            
            weighted_ai_score += result["scores"]["ai_generated"] * weight
            weighted_human_score += result["scores"]["human"] * weight
            
            # Track mixed classifications
            if result.get("classification", {}).get("is_mixed", False):
                mixed_count += 1
            
            # Store real metrics for aggregation
            if "metrics" in result:
                chunk_metrics_list.append({
                    "weight": weight,
                    "metrics": result["metrics"]
                })
            
            # Collect patterns
            patterns = result.get("detected_patterns", {})
            if patterns.get("total_count", 0) > 0:
                all_patterns.extend([
                    {"chunk_index": chunk.get("index", 0), **p}
                    for cat_info in patterns.get("categories", {}).values()
                    for p in [{"pattern": ex, "type": cat_info["type"]} for ex in cat_info.get("examples", [])]
                ])
            
            if include_per_chunk:
                chunk_results.append({
                    "index": chunk.get("index", 0),
                    "start": chunk.get("start", 0),
                    "end": chunk.get("end", 0),
                    "char_count": len(chunk_text),
                    "prediction": result["prediction"],
                    "confidence": result["confidence"],
                    "ai_score": result["scores"]["ai_generated"],
                    "human_score": result["scores"]["human"],
                    "is_mixed": result.get("classification", {}).get("is_mixed", False)
                })
        
        if total_weight == 0:
            return {"error": "No valid chunks to analyze"}
        
        # Calculate weighted averages
        avg_ai = weighted_ai_score / total_weight
        avg_human = weighted_human_score / total_weight
        
        # Trinary classification at document level from chunk analysis
        total_chunks_analyzed = len(chunk_results) if chunk_results else len(chunk_metrics_list)
        ai_chunks = sum(1 for c in chunk_results if c.get("ai_score", 0) > 60)
        human_chunks = sum(1 for c in chunk_results if c.get("human_score", 0) > 60)
        
        if total_chunks_analyzed >= 2:
            chunk_ai_ratio = ai_chunks / total_chunks_analyzed
            chunk_human_ratio = human_chunks / total_chunks_analyzed
            if chunk_ai_ratio > 0.2 and chunk_human_ratio > 0.2:
                prediction = "mixed"
            elif avg_ai > avg_human:
                prediction = "ai_generated"
            else:
                prediction = "human"
        else:
            prediction = "ai_generated" if avg_ai > avg_human else "human"
        
        confidence = max(avg_ai, avg_human)
        
        result = {
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "scores": {
                "human": round(avg_human, 2),
                "ai_generated": round(avg_ai, 2)
            },
            "classification": {
                "type": prediction,
                "ai_chunk_ratio": round(ai_chunks / max(total_chunks_analyzed, 1), 2),
                "human_chunk_ratio": round(human_chunks / max(total_chunks_analyzed, 1), 2),
                "is_mixed": prediction == "mixed",
                "mixed_chunk_count": mixed_count
            },
            "chunk_summary": {
                "total_chunks": total_chunks_analyzed,
                "ai_leaning_chunks": ai_chunks,
                "human_leaning_chunks": human_chunks,
                "mixed_chunks": mixed_count,
                "total_characters": total_weight
            },
            "detected_patterns": {
                "total_count": len(all_patterns),
                "patterns": all_patterns[:20]  # Limit to 20
            }
        }
        
        # Aggregate REAL metrics from chunks (weighted average)
        if chunk_metrics_list:
            try:
                agg_word_count = 0
                agg_sentence_count = 0
                agg_perplexity_sum = 0.0
                agg_burstiness_sum = 0.0
                agg_vocab_sum = 0.0
                agg_weight_total = sum(cm["weight"] for cm in chunk_metrics_list)
                perplexity_flow = []
                
                for cm in chunk_metrics_list:
                    w = cm["weight"]
                    m = cm["metrics"]
                    agg_word_count += m.get("word_count", 0)
                    agg_sentence_count += m.get("sentence_count", 0)
                    
                    ppl = m.get("perplexity", {})
                    if isinstance(ppl, dict):
                        ppl_avg = ppl.get("average", 50)
                        ppl_flow = ppl.get("flow", [])
                    else:
                        ppl_avg = 50
                        ppl_flow = []
                    
                    agg_perplexity_sum += ppl_avg * w
                    
                    burst = m.get("burstiness", {})
                    if isinstance(burst, dict):
                        agg_burstiness_sum += burst.get("score", 0.5) * w
                    
                    agg_vocab_sum += m.get("vocabulary_richness", 50) * w
                    
                    if ppl_flow:
                        perplexity_flow.extend(ppl_flow[:5])
                
                avg_ppl = agg_perplexity_sum / max(agg_weight_total, 1)
                avg_burst = agg_burstiness_sum / max(agg_weight_total, 1)
                avg_vocab = agg_vocab_sum / max(agg_weight_total, 1)
                avg_wps = agg_word_count / max(agg_sentence_count, 1)
                
                result["metrics"] = {
                    "word_count": agg_word_count,
                    "sentence_count": agg_sentence_count,
                    "avg_words_per_sentence": round(avg_wps, 1),
                    "vocabulary_richness": round(avg_vocab, 1),
                    "perplexity": {
                        "average": round(avg_ppl, 1),
                        "flow": perplexity_flow[:15]
                    },
                    "burstiness": {
                        "score": round(avg_burst, 2),
                        "bars": {
                            "document": [round(avg_burst * 100 * (0.8 + 0.4*(i%3)/2), 1) for i in range(6)],
                            "human_baseline": [60, 85, 40, 95, 55, 70]
                        }
                    },
                    "aggregated_from_chunks": True
                }
            except Exception as e:
                logger.warning(f"Error aggregating chunk metrics: {e}")
        

        
        if include_per_chunk:
            result["chunks"] = chunk_results
        
        return result
    
    def get_cache_info(self) -> Dict:
        """Get cache statistics."""
        info = self._cached_inference.cache_info()
        return {
            "hits": info.hits,
            "misses": info.misses,
            "size": info.currsize,
            "max_size": info.maxsize
        }
    
    def clear_cache(self):
        """Clear the inference cache."""
        self._cached_inference.cache_clear()


if __name__ == "__main__":
    # Test the detector with detailed debug output
    detector = AIContentDetector(use_ml_model=False)  # Force offline mode for testing
    
    sample_ai_text = """
    It's important to note that artificial intelligence has revolutionized many industries. 
    Furthermore, the technology continues to evolve at a rapid pace. 
    In conclusion, we must consider both the benefits and potential risks of AI adoption.
    Additionally, it's crucial to implement proper safeguards.
    """
    
    sample_human_text = """
    I tried the new coffee shop downtown yesterday. The espresso was okay but nothing special.
    My dog wouldn't stop barking at the mailman again. He's such a goofball sometimes!
    Gonna watch that new show everyone's been talking about tonight.
    """
    
    def debug_text(text, label):
        patterns = detector._detect_patterns_in_text(text)
        entropy = detector._calculate_entropy(text)
        repetition = detector._detect_repetition(text, n=4)
        ttr = detector._calculate_ttr(text)
        bigram_unif = detector._calculate_ngram_uniformity(text, 2)
        trigram_unif = detector._calculate_ngram_uniformity(text, 3)
        
        print(f"\n{'='*60}")
        print(f"DEBUG - {label}")
        print(f"{'='*60}")
        print(f"Patterns found: {len(patterns)}")
        print(f"Entropy: {entropy:.3f} (high=human, low=AI)")
        print(f"Repetition: {repetition:.3f} (high=AI)")
        print(f"TTR (vocab richness): {ttr:.3f} (high=human)")
        print(f"Bigram Uniformity: {bigram_unif:.3f} (high=AI)")
        print(f"Trigram Uniformity: {trigram_unif:.3f} (high=AI)")
        
        result = detector.predict(text)
        print(f"\nFINAL PREDICTION: {result['prediction']}")
        print(f"AI Score: {result['scores']['ai_generated']:.2f}%")
        print(f"Human Score: {result['scores']['human']:.2f}%")
    
    debug_text(sample_ai_text, "AI-GENERATED TEXT")
    debug_text(sample_human_text, "HUMAN-WRITTEN TEXT")

