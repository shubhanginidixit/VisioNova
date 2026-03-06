"""
Standalone Wav2Vec 2.0 architecture (pure PyTorch, no fairseq dependency).

Matches the fairseq Wav2Vec2Model state dict layout exactly so that
NII AntiDeepfake checkpoints can be loaded directly without installing fairseq.

Only the inference path is implemented — no quantization, masking, or training logic.

Reference: fairseq/models/wav2vec/wav2vec2.py (MIT License)
"""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper modules
# ---------------------------------------------------------------------------

class TransposeLast(nn.Module):
    """Transpose the last two dimensions."""
    def forward(self, x):
        return x.transpose(-2, -1)


class Fp32LayerNorm(nn.LayerNorm):
    """LayerNorm computed in float32 for numerical stability (mixed-precision safe)."""
    def forward(self, x):
        output = F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(x)


class SamePad(nn.Module):
    """Remove one element from the right when kernel_size is even."""
    def __init__(self, kernel_size):
        super().__init__()
        self.remove = kernel_size % 2 == 0

    def forward(self, x):
        if self.remove:
            x = x[:, :, :-1]
        return x


# ---------------------------------------------------------------------------
# 7-layer CNN Feature Extractor
# ---------------------------------------------------------------------------

# Default wav2vec 2.0 / XLS-R conv configuration
DEFAULT_CONV_LAYERS = [
    (512, 10, 5),   # (out_channels, kernel_size, stride)
    (512, 3, 2),
    (512, 3, 2),
    (512, 3, 2),
    (512, 3, 2),
    (512, 2, 2),
    (512, 2, 2),
]


class ConvFeatureExtractionModel(nn.Module):
    """7-layer CNN that converts raw waveform [B, T] → latent features [B, 512, T']."""

    def __init__(self, conv_layers=None, conv_bias=True):
        super().__init__()
        if conv_layers is None:
            conv_layers = DEFAULT_CONV_LAYERS

        self.conv_layers = nn.ModuleList()
        in_d = 1  # raw waveform has 1 channel
        for out_d, k, stride in conv_layers:
            # Structure matches fairseq layer_norm mode:
            #   [0] Conv1d   [1] Dropout   [2] Sequential(Transpose, LN, Transpose)   [3] GELU
            # State dict indices: .0 = Conv1d, .2.1 = LayerNorm
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_d, out_d, k, stride=stride, bias=conv_bias),
                    nn.Dropout(p=0.0),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(out_d, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            )
            in_d = out_d

    def forward(self, x):
        # x: [B, T] raw waveform
        x = x.unsqueeze(1)  # [B, 1, T]
        for conv in self.conv_layers:
            x = conv(x)
        return x  # [B, 512, T']


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention (fairseq style: separate q/k/v projections)
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    """Multi-head self-attention with separate Q, K, V projections.
    
    State dict keys: k_proj, v_proj, q_proj, out_proj (matching fairseq).
    """

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: [T, B, C]  (fairseq convention: sequence-first)
        T, B, C = x.size()

        q = self.q_proj(x) * self.scaling
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head: [B*heads, T, head_dim]
        q = q.contiguous().view(T, B * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(T, B * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(T, B * self.num_heads, self.head_dim).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn = torch.bmm(attn_probs, v)

        # Merge heads: [T, B, C]
        attn = attn.transpose(0, 1).contiguous().view(T, B, C)
        return self.out_proj(attn)


# ---------------------------------------------------------------------------
# Transformer Encoder Layer (pre-norm: layer_norm_first=True)
# ---------------------------------------------------------------------------

class TransformerEncoderLayer(nn.Module):
    """Pre-norm transformer layer matching fairseq's TransformerSentenceEncoderLayer."""

    def __init__(self, embed_dim, ffn_embed_dim, num_heads):
        super().__init__()
        self.self_attn = SelfAttention(embed_dim, num_heads)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [T, B, C]
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x
        return x


# ---------------------------------------------------------------------------
# Transformer Encoder
# ---------------------------------------------------------------------------

class TransformerEncoder(nn.Module):
    """Transformer encoder with convolutional positional embedding."""

    def __init__(self, embed_dim, ffn_embed_dim, num_heads, num_layers,
                 conv_pos=128, conv_pos_groups=16):
        super().__init__()

        # Positional conv with weight_norm (creates weight_g, weight_v in state dict)
        pos_conv = nn.Conv1d(
            embed_dim, embed_dim,
            kernel_size=conv_pos, padding=conv_pos // 2,
            groups=conv_pos_groups,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress weight_norm deprecation
            pos_conv = nn.utils.weight_norm(pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(pos_conv, SamePad(conv_pos), nn.GELU())

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, ffn_embed_dim, num_heads)
            for _ in range(num_layers)
        ])

        # Final layer norm (applied after all layers when layer_norm_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, T, C]

        # Add positional embedding
        x_conv = self.pos_conv(x.transpose(1, 2))  # [B, C, T]
        x_conv = x_conv.transpose(1, 2)             # [B, T, C]
        x = x + x_conv

        # Transpose to [T, B, C] for transformer layers (fairseq convention)
        x = x.transpose(0, 1)
        for layer in self.layers:
            x = layer(x)

        # Back to [B, T, C]
        x = x.transpose(0, 1)

        # Final layer norm
        x = self.layer_norm(x)
        return x


# ---------------------------------------------------------------------------
# Full Wav2Vec2 Model
# ---------------------------------------------------------------------------

class Wav2Vec2Model(nn.Module):
    """
    Wav2Vec 2.0 model (inference-only).

    State dict keys match fairseq's Wav2Vec2Model exactly, enabling direct
    loading of NII AntiDeepfake checkpoints from HuggingFace Hub.

    Architecture:
        raw waveform → CNN extractor (512-d) → LayerNorm → Linear projection
        → Transformer encoder → output features
    """

    def __init__(self, encoder_layers, encoder_embed_dim, encoder_ffn_embed_dim,
                 encoder_attention_heads, conv_bias=True):
        super().__init__()

        # CNN feature extractor: waveform → 512-dim features
        self.feature_extractor = ConvFeatureExtractionModel(conv_bias=conv_bias)

        # Project 512-dim CNN output to encoder dimension
        self.layer_norm = nn.LayerNorm(512)
        self.post_extract_proj = nn.Linear(512, encoder_embed_dim)

        # Transformer encoder
        self.encoder = TransformerEncoder(
            embed_dim=encoder_embed_dim,
            ffn_embed_dim=encoder_ffn_embed_dim,
            num_heads=encoder_attention_heads,
            num_layers=encoder_layers,
        )

        # Present in fairseq checkpoints (unused during inference)
        self.mask_emb = nn.Parameter(torch.zeros(encoder_embed_dim))

    def forward(self, source, mask=False, features_only=True):
        """
        Args:
            source: [B, T] raw waveform at 16 kHz
            mask: unused (inference only)
            features_only: must be True

        Returns:
            dict with 'x': encoder output [B, T', encoder_embed_dim]
        """
        features = self.feature_extractor(source)    # [B, 512, T']
        features = features.transpose(1, 2)           # [B, T', 512]
        features = self.layer_norm(features)
        features = self.post_extract_proj(features)   # [B, T', D]
        x = self.encoder(features)                    # [B, T', D]
        return {"x": x}
