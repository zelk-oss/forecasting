#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2025}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Tuple

# External modules
import torch
import torch.nn.functional as F
from einops import rearrange, einsum

import numpy as np
try:
    from scipy.special import sph_harm_y
except Exception:
    sph_harm_y = None

# Fallback for flash attention
try:
    from flash_attn import flash_attn_func
    USE_FLASH_ATTN = True
except ImportError:
    USE_FLASH_ATTN = False
    flash_attn_func = None

# Internal modules

main_logger = logging.getLogger(__name__)

__all__ = [
    "Transformer"
]


def self_attention(
        q_proj: torch.Tensor,
        k_proj: torch.Tensor,
        v_proj: torch.Tensor
) -> torch.Tensor:
    """
    Self-attention function for flash attention, falling back to
    `scaled_dot_product_attention` if not available.
    """
    if USE_FLASH_ATTN:
        return flash_attn_func(
            q_proj.to(torch.bfloat16),
            k_proj.to(torch.bfloat16),
            v_proj.to(torch.bfloat16),
            dropout_p=0.,
            softmax_scale=None,
            causal=False
        ).to(q_proj)
    else:
        attn_out = F.scaled_dot_product_attention(
            q_proj.transpose(1, 2),
            k_proj.transpose(1, 2),
            v_proj.transpose(1, 2),
            attn_mask=None,
            dropout_p=0.,
            scale=None,
            is_causal=False
        )
        return attn_out.transpose(1, 2)


# self attention layer: each token looks at all others 
# Standard multi-head attention with RoPE and RMSNorm.
# REPLACED SelfAttentionLayer with periodic-friendly RoPE or absolute positional embeddings
class SelfAttentionLayer(torch.nn.Module):
    def __init__(
            self,
            n_features: int = 512,
            n_heads: int = 8,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_features_head = int(n_features // n_heads)
        self.n_heads = n_heads

        self.qkv_layer = torch.nn.Linear(
            n_features,
            self.n_features_head * n_heads * 3,
            bias=False
        )
        self.q_norm = torch.nn.RMSNorm(self.n_features_head)
        self.k_norm = torch.nn.RMSNorm(self.n_features_head)
        
        self.out_layer = torch.nn.Linear(
            n_features, n_features, bias=False
        )
        torch.nn.init.normal_(self.out_layer.weight, std=1E-6)

    # Fallback RoPE: identity mapping when RoPE not configured/available.
    def rope_layer(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(
            self,
            in_tensor: torch.Tensor
    ) -> torch.Tensor:
        B, T, _ = in_tensor.shape

        # Project layers
        qkv_proj = self.qkv_layer(in_tensor)
        qkv_proj = qkv_proj.view(B, T, 3, self.n_heads, self.n_features_head)
        q_proj, k_proj, v_proj = qkv_proj.unbind(dim=2)

        # QK normalization and apply RoPE
        q_proj = self.q_norm(q_proj)
        k_proj = self.k_norm(k_proj)

        # Apply spherical RoPE
        q_proj = self.rope_layer(q_proj)
        k_proj = self.rope_layer(k_proj)

        # Apply self attention
        out = self_attention(q_proj, k_proj, v_proj)
        out = out.reshape(B, T, -1)
        return self.out_layer(out)


# Multi Layer Perceptron: nonlinear processing of each token 
# feed-forward block with gating 
class MLPLayer(torch.nn.Module):
    def __init__(
            self,
            n_features,
            mult: int = 1,
    ):
        super().__init__()
        self.hidden_features = n_features * mult
        self.in_layer = torch.nn.Linear(
            n_features, self.hidden_features*2, bias=False
        )
        self.activation = torch.nn.SiLU()
        self.out_layer = torch.nn.Linear(
            self.hidden_features, n_features, bias=False
        )
        torch.nn.init.normal_(self.out_layer.weight, std=1E-6)

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        branch, gate = self.in_layer(in_tensor).chunk(2, dim=-1)
        out_tensor = self.out_layer(
            branch * self.activation(gate)
        )
        return out_tensor


# Wraps attention + MLP with residual connections and RMSNorm.
class TransformerBlock(torch.nn.Module):
    def __init__(
            self,
            n_features: int,
            n_heads: int = 8,
            n_embedding: int = 0,
            mult: int = 2,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.mult = mult
        self.n_heads = n_heads

        if n_embedding > 0:
            self.gate_layer = torch.nn.Linear(
                n_embedding, n_features*2, bias=True
            )
            self.attn_norm = torch.nn.RMSNorm(
                n_features, elementwise_affine=False
            )
            self.ffn_norm = torch.nn.RMSNorm(
                n_features, elementwise_affine=False
            )
        else:
            self.gate_layer = None
            self.attn_norm = torch.nn.RMSNorm(n_features)
            self.ffn_norm = torch.nn.RMSNorm(n_features)

        self.attn = SelfAttentionLayer(
            n_features=n_features,
            n_heads=n_heads
        )
        self.ffn = MLPLayer(
            n_features=n_features,
            mult=mult
        )

    def forward(
            self,
            in_tensor: torch.Tensor,
            embedding: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Apply gating from embedding
        # Time controls how much each operation matters
        if self.gate_layer is not None:
            gate_tensor = self.gate_layer(embedding).unsqueeze(1)
            gate_attn, gate_ffn = gate_tensor.chunk(2, dim=-1)
        else:
            gate_attn = gate_ffn = 1.0

        # Attention block
        attn_input = self.attn_norm(in_tensor) * gate_attn
        out_tensor = in_tensor + self.attn(attn_input)

        # Feed-forward block
        ffn_input = self.ffn_norm(out_tensor) * gate_ffn
        out_tensor = out_tensor + self.ffn(ffn_input)
        return out_tensor


# Converts 2D input field to tokens.
class Tokenizer(torch.nn.Module):
    def __init__(
            self,
            n_channels: int,
            n_features: int,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_features = n_features

        self.in_layer = torch.nn.Conv2d(
            n_channels + 1,
            n_features,
            kernel_size=2,
            stride=2,
            bias=False,
            padding=0
        )

    def forward(
            self,
            in_tensor: torch.Tensor
    ) -> torch.Tensor:
        in_with_bias = torch.cat(
            [in_tensor, torch.ones_like(in_tensor[:, :1, :, :])],
            dim=1
        )
        out_tensor = self.in_layer(in_with_bias)
        out_tensor = rearrange(
            out_tensor,
            "b c w h -> b (w h) c"
        )
        return out_tensor


# Reconstructs 2D output from tokens.
# takes the N=32x32 processed tokens 
# upsamples back to the spatial field 
# outcome is the predicted velocity 
class Head(torch.nn.Module):
    def __init__(
            self,
            n_features: int,
            n_output: int,
            n_embedding: int = 0,
            token_downsample_factor: int = 8,  # added this 
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_output = n_output
        self.token_grid_size = 256 // token_downsample_factor

        if n_embedding > 0:
            self.gate_layer = torch.nn.Linear(
                n_embedding, n_features, bias=False
            )
            self.in_norm = torch.nn.RMSNorm(
                n_features, elementwise_affine=False
            )
        else:
            self.gate_layer = None
            self.in_norm = torch.nn.RMSNorm(n_features)
        self.out_layer = torch.nn.Linear(
            n_features, n_output*4, bias=False
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        weights = torch.empty(self.n_output, self.n_features)
        # So that expected output variance = 1
        torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")
        # Initialize as nearest neighbor interpolation
        weights = weights.repeat_interleave(4, dim=0)
        # Copy the weights to the output layer
        self.out_layer.weight.data.copy_(weights)

    def forward(
            self,
            in_tensor: torch.Tensor,
            embedding: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.gate_layer is not None:
            gate_tensor = self.gate_layer(embedding).unsqueeze(1)
            in_normed = self.in_norm(in_tensor) * gate_tensor
        else:
            in_normed = self.in_norm(in_tensor)
        out_tensor = self.out_layer(in_normed)
        # CHANGED: h=16, w=32 → h=32, w=32 (for 32×32 token grid after pooling)
        out_tensor = rearrange(
            out_tensor,
            "b (w h) (c w2 h2) -> b c (w w2) (h h2)",
            h=self.token_grid_size,  # CHANGEED FROM h=32
            w=self.token_grid_size, 
            h2=2, w2=2
        )
        return out_tensor

# Time/lead-time conditioning.
class RandomFourierEmbedding(torch.nn.Module):
    def __init__(self, n_output=256, n_features=256, wave_length=0.1):
        super().__init__()
        # Initialise random waves
        self.fourier_weights = torch.nn.Parameter(
            torch.randn(1, n_features//2), requires_grad=False
        )

        # Small MLP for combination of Fourier features
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(n_features, n_output),
            torch.nn.SELU(),
            torch.nn.Linear(n_output, n_output)
        )
        # Wave length of features
        self.wave_length = wave_length

    def forward(self, pseudo_time):
        # Extract features at wave length
        scaled_time = pseudo_time / self.wave_length
        # Extract random Fourier features
        features = scaled_time@self.fourier_weights
        features = torch.cat((
            features.sin(), features.cos()
        ), dim=1)
        # Apply small MLP to combine Fourier features
        return self.mlp(features)


class Transformer(torch.nn.Module):
    def __init__(
            self,
            token_downsample_factor: int = 8,
            n_input: int = 1, # Only 1 variable for SQG 
            n_output: int = 1, # Only 1 variable for SQG 
            n_features: int = 512,
            n_blocks: int = 8,
            n_heads: int = 8,
            n_embedding: int = 0,
            mult: int = 2,
            wave_length: float = 0.07,
    ) -> None:
        super().__init__()
        self.tokenizer = Tokenizer(
            n_channels=n_input,
            n_features=n_features,
        )
        # NEW: after tokenizing 512×512 → 256×256 tokens,
        # further reduce to 32×32
        self.token_pool = torch.nn.AvgPool2d(
            kernel_size=token_downsample_factor, 
            stride=token_downsample_factor
        )
        self.blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    n_features=n_features,
                    n_heads=n_heads,
                    n_embedding=n_embedding,
                    mult=mult,
                )
                for _ in range(n_blocks)
            ]
        )
        self.head = Head(
            n_features=n_features,
            n_output=n_output,
            n_embedding=n_embedding,
            token_downsample_factor=token_downsample_factor,
        )
        if n_embedding > 0:
            # Define embedding
            self.embedding_layer = RandomFourierEmbedding(
                n_embedding, n_features, wave_length
            )
        else:
            self.embedding_layer = None

    # NEW: changing this to agree with downsampling operation 
    def forward(
            self,
            in_tensor: torch.Tensor,
            pseudo_time: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.embedding_layer is not None:
            embedding = self.embedding_layer(pseudo_time)
        else:
            embedding = None
        
        # Tokenize: (B, 1, 512, 512) -> (B, 256, 256, n_features)
        tokens = self.tokenizer(in_tensor)  # shape: (B, 256*256, n_features)
        
        # Reshape to 2D, pool, reshape back to sequence
        B, _, C = tokens.shape  # B = batch, _ = 256*256, C = n_features
        tokens_2d = tokens.reshape(B, 256, 256, C).permute(0, 3, 1, 2)  # (B, C, 256, 256)
        tokens_2d = self.token_pool(tokens_2d)  # (B, C, 32, 32) with factor=8
        tokens = tokens_2d.permute(0, 2, 3, 1).reshape(B, -1, C)  # (B, 32*32, C)
        
        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens, embedding)
        
        # Head expects (B, 32*32, C) and outputs (B, n_output, H, W)
        out_tensor = self.head(tokens, embedding)
        return out_tensor


def get_net(
    n_input=1, n_output=1, n_blocks=8, n_features=512, n_heads=8,
    mult=2,
    n_embedding=0, wave_length=0.07,
    token_downsample_factor=8, 
    device=None, dtype=torch.float32
):
    # More flexibility than with torch.nn.sequential
    transformer = Transformer(
        n_input=n_input, n_output=n_output, n_blocks=n_blocks,
        n_features=n_features, n_heads=n_heads,
        mult=mult,
        n_embedding=n_embedding, wave_length=wave_length,
        token_downsample_factor=token_downsample_factor,
    )
    if device is None:
        device = torch.device("cpu")
    transformer = transformer.to(device=device, dtype=dtype)
    return transformer
