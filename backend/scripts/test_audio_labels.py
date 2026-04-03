import json
from transformers import AutoConfig

ENSEMBLE_MODELS = [
    {
        "id": "abhishtagatya/hubert-base-960h-itw-deepfake",
        "fake_labels": ["spoof", "fake", "deepfake", "generated", "ai", "label_1"],
    },
    {
        "id": "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification",
        "fake_labels": ["spoof", "fake", "deepfake", "generated", "ai"],
    },
    {
        "id": "garystafford/wav2vec2-deepfake-voice-detector",
        "fake_labels": ["spoof", "fake", "deepfake", "generated", "ai", "label_1"],
    },
    {
        "id": "Vansh180/deepfake-audio-wav2vec2",
        "fake_labels": ["spoof", "fake", "deepfake", "generated", "ai"],
    },
]

def _resolve_fake_index(config_obj, config_dict):
    id2label = getattr(config_obj, 'id2label', None)
    if id2label:
        fake_keywords = config_dict.get('fake_labels', [])
        for idx_str, label in id2label.items():
            label_lower = label.lower()
            for keyword in fake_keywords:
                if keyword in label_lower:
                    return int(idx_str)
    return 1

with open('model_labels_output_v2.txt', 'w', encoding='utf-8') as f:
    for m in ENSEMBLE_MODELS:
        try:
            config = AutoConfig.from_pretrained(m['id'])
            id2label = getattr(config, 'id2label', {})
            idx = _resolve_fake_index(config, m)
            f.write(f"Model: {m['id']}\n")
            f.write(f"  id2label: {json.dumps(id2label)}\n")
            f.write(f"  resolved_fake_idx: {idx}\n\n")
        except Exception as e:
            f.write(f"Error for {m['id']}: {e}\n\n")
