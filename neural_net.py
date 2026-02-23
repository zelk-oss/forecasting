#!/bin/env python
# -*- coding: utf-8 -*-
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2025}  {Tobias Sebastian Finn}

# System modules
import logging
import math 
from typing import Tuple

# External modules
import torch
import torch.nn.functional as F
from einops import rearrange, einsum

import numpy as np

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
            patch_size = 16
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_features = n_features

        self.in_layer = torch.nn.Conv2d(
            n_channels + 1,
            n_features,
            kernel_size=patch_size,
            stride=patch_size,
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
# outcome is the predicted velocity 
class Head(torch.nn.Module):
    def __init__(
            self,
            n_features: int,
            n_output: int,
            n_embedding: int = 0,
            patch_size = 2, 
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_output = n_output
        self.patch_size = patch_size

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
            n_features, n_output*patch_size * patch_size, bias=False
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        weights = torch.empty(self.n_output, self.n_features)
        # So that expected output variance = 1
        torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")
        # Initialize as nearest neighbor interpolation
        weights = weights.repeat_interleave(self.patch_size**2, dim=0)
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
        
        # Use provided token_grid_size or calculate from tensor shape
        # Infer grid size from in_tensor shape: [batch, height*width, features]
        token_grid_size = int(math.sqrt(in_tensor.shape[1]))
        
        # Reshape to 2D spatial grid with 2x2 upsampling
        out_tensor = rearrange(
            out_tensor,
            "b (w h) (c w2 h2) -> b c (w w2) (h h2)",
            h=token_grid_size,
            w=token_grid_size, 
            h2=self.patch_size, w2=self.patch_size
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
            n_input: int = 1, # Only 1 variable for SQG 
            n_output: int = 1, # Only 1 variable for SQG 
            n_features: int = 512,
            n_blocks: int = 8,
            n_heads: int = 8,
            n_embedding: int = 0,
            patch_size: int = 16,
            mult: int = 2,
            wave_length: float = 0.07,
    ) -> None:
        super().__init__()
        self.tokenizer = Tokenizer(
            n_channels=n_input,
            n_features=n_features,
            patch_size = patch_size, 
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
            patch_size = patch_size
        )
        if n_embedding > 0:
            # Define embedding
            self.embedding_layer = RandomFourierEmbedding(
                n_embedding, n_features, wave_length
            )
        else:
            self.embedding_layer = None

    def forward(
            self,
            in_tensor: torch.Tensor,
            pseudo_time: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.embedding_layer is not None:
            embedding = self.embedding_layer(pseudo_time)
        else:
            embedding = None
        
        # Tokenize: (B, C, H, W) -> (B, H/2, W/2, n_features) then (B, (H/2)*(W/2), n_features)
        tokens = self.tokenizer(in_tensor)  # shape: (B, (H/2)*(W/2), n_features)
        
        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens, embedding)
        
        out_tensor = self.head(tokens, embedding)
        return out_tensor


def get_net(
    n_input=1, n_output=1, n_blocks=8, n_features=512, 
    patch_size: int = 16, n_heads=8,
    mult=2,
    n_embedding=0, wave_length=0.07,
    device=None, dtype=torch.float32
):
    # More flexibility than with torch.nn.sequential
    transformer = Transformer(
        n_input=n_input, n_output=n_output, n_blocks=n_blocks,
        n_features=n_features, n_heads=n_heads,
        mult=mult,
        n_embedding=n_embedding, wave_length=wave_length,
        patch_size=patch_size,
    )
    if device is None:
        device = torch.device("cpu")
    transformer = transformer.to(device=device, dtype=dtype)
    return transformer