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
from scipy.special import sph_harm_y

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

_COLATS = [
    3.04353538, 2.84740298, 2.65121971, 2.45496025,
    2.25861084, 2.06217142, 1.86565561, 1.66908873,
    1.47250392, 1.27593704, 1.07942123, 0.88298181,
    0.68663241, 0.49037295, 0.29418967, 0.09805727
]
_LONS = [
    0.04908739, 0.24543693, 0.44178647, 0.63813601, 0.83448555,
    1.03083509, 1.22718463, 1.42353417, 1.61988371, 1.81623325,
    2.01258279, 2.20893233, 2.40528188, 2.60163142, 2.79798096,
    2.9943305 , 3.19068004, 3.38702958, 3.58337912, 3.77972866,
    3.9760782 , 4.17242774, 4.36877728, 4.56512682, 4.76147637,
    4.95782591, 5.15417545, 5.35052499, 5.54687453, 5.74322407,
    5.93957361, 6.13592315
]


def real_spherical_harmonics(l, m, theta, phi):
    """
    Real-valued spherical harmonics.
    theta: colatitude [0, pi]
    phi: longitude [0, 2pi)
    """
    if m > 0:
        return np.sqrt(2) * (-1)**m * np.real(sph_harm_y(l, m, theta, phi))
    elif m < 0:
        return np.sqrt(2) * (-1)**m * np.imag(sph_harm_y(l, m, theta, phi))
    else:
        return np.real(sph_harm_y(l, 0, theta, phi))


def get_spherical_features(max_l: int = 7) -> torch.Tensor:
    """
    Get spherical mesh features based on colatitudes and longitudes.
    Args:
        max_l (int, optional): Bandwidth for spherical features.
            Defaults to 7.
    Returns:
        torch.Tensor: Spherical mesh features of shape (M, 2),
            where M is the number of mesh points.
    """
    lons2d, colats2d = np.meshgrid(_LONS, _COLATS, indexing="ij")
    lons2d = lons2d.flatten()
    colats2d = colats2d.flatten()

    features = []
    for l in range(max_l+1):
        for m in range(-l, l+1):
            Y_lm = real_spherical_harmonics(
                l, m, colats2d, lons2d
            )
            features.append(Y_lm)
    features = np.stack(features, axis=-1)
    features_norm = np.sqrt((features**2).mean(axis=0, keepdims=True))
    features = features / features_norm
    return torch.tensor(features, dtype=torch.float32)


class SphericalRopeLayer(torch.nn.Module):
    """
    A spherical Rotary Position Embedding (RoPE) layer for neural networks.
    This layer applies rotational embeddings to features based on mesh features,
    enabling position-aware transformations in a spherical coordinate system.
    It computes rotation angles from mesh features and applies 2D rotations to
    pairs of feature dimensions.
    Attributes:
        n_features (int): Dimensionality of input features. Must be even.
        n_heads (int): Number of independent rotation heads.
        n_mesh_features (int): Dimensionality of mesh features.
        alpha (float): Scaling factor for rotation angles.
        weights (torch.nn.Parameter): Learnable weight matrix of shape
            (n_heads, n_mesh_features, n_features // 2) that projects
            mesh features to rotation angles.
    Args:
        n_features (int, optional): Number of input features. Defaults to 64.
        n_heads (int, optional): Number of rotation heads. Defaults to 8.
        n_mesh_features (int, optional): Number of mesh features. Defaults to 64.
        alpha (float, optional): Rotation angle scaling factor. Defaults to 1.0.
    Returns:
        torch.nn.Module: An initialized SphericalRopeLayer instance.
    Example:
        >>> layer = SphericalRopeLayer(n_features=64, n_heads=8)
        >>> features = torch.randn(batch_size, time_steps, 64)
        >>> mesh_features = torch.randn(batch_size, time_steps, mesh_points, 64)
        >>> output = layer(features, mesh_features)
    """
    def __init__(
            self,
            n_features: int = 64,
            n_heads: int = 8,
            max_l: int = 7,
            alpha: float = 1.0
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_heads = n_heads
        self.n_rope_features = self.n_features // self.n_heads
        self.max_l = max_l
        self.n_mesh_features = (max_l + 1)**2
        self.alpha = alpha

        self.register_buffer(
            "mesh_features",
            get_spherical_features(max_l=max_l)
        )
        self.weights = torch.nn.Parameter(
            torch.randn(n_heads, self.n_mesh_features, self.n_rope_features // 2)
        )

    @property
    def angles(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes and returns rotation angles based on mesh features
        and learned weights.
        Returns:
            torch.Tensor: Rotation angles of shape (n_grid, n_heads, n_features // 2).
        """
        angles: torch.Tensor = einsum(
            self.mesh_features, self.weights, "t m, h m f -> t h f"
        )
        angles = angles * self.alpha
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        return cos_angles, sin_angles

    def _rotate_tensor(
            self,
            tensor: torch.Tensor,
            cos_angles: torch.Tensor,
            sin_angles: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies rotation to the input tensor using provided cosine and sine angles.
        The input tensor is split into even and odd parts, which are then rotated
        and concatenated to produce the output.
        Arguments:
            tensor (torch.Tensor): Input tensor of shape (..., 2 * F),
                where F is the number of feature pairs.
            cos_angles (torch.Tensor): Cosine of rotation angles of shape (..., F).
            sin_angles (torch.Tensor): Sine of rotation angles of shape (..., F).
        Returns:
            torch.Tensor: Rotated tensor of the same shape as the input `tensor`.
        """
        d = tensor.shape[-1]
        if (d % 2) != 0:
            raise ValueError("Last dimension (head_dim) must be even for RoPE pairs.")

        # cast angles to match tensor dtype/device for safe arithmetic
        cos = cos_angles.to(dtype=tensor.dtype, device=tensor.device)
        sin = sin_angles.to(dtype=tensor.dtype, device=tensor.device)

        # split into interleaved pairs and rotate (x_even = x[..., 0::2], x_odd = x[..., 1::2])
        x_even = tensor[..., 0::2]
        x_odd = tensor[..., 1::2]
        r_even = x_even * cos - x_odd * sin
        r_odd = x_even * sin + x_odd * cos

        # interleave back to original ordering
        rotated = torch.stack([r_even, r_odd], dim=-1).reshape_as(tensor)
        return rotated

    def forward(
            self,
            features: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies a learned rotation to the input features using mesh
        features and learned weights. Computes rotation angles from mesh
        features and learned weights, scales the result by a learnable parameter
        alpha, and applies the resulting rotation to the input features. The
        input features are split into even and odd parts, which are then rotated
        and concatenated to produce the output.

        Arguments:
            features (torch.Tensor): Input feature tensor of shape (..., 2 * F),
                where F is the number of feature pairs.

        Returns:
            torch.Tensor: Rotated feature tensor of the same shape as the
                input `features`.
        """
        cos_angles, sin_angles = self.angles
        rotated_features = self._rotate_tensor(
            features,
            cos_angles,
            sin_angles
        )
        return rotated_features


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


class SelfAttentionLayer(torch.nn.Module):
    def __init__(
            self,
            n_features: int = 512,
            n_heads: int = 8,
            rope_max_l: int = 7,
            rope_alpha: float = 1.0
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
        self.rope_layer = SphericalRopeLayer(
            n_features=n_features,
            n_heads=n_heads,
            max_l=rope_max_l,
            alpha=rope_alpha
        )
        self.out_layer = torch.nn.Linear(
            n_features, n_features, bias=False
        )
        torch.nn.init.normal_(self.out_layer.weight, std=1E-6)

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


class TransformerBlock(torch.nn.Module):
    def __init__(
            self,
            n_features: int,
            n_heads: int = 8,
            n_embedding: int = 0,
            mult: int = 2,
            rope_max_l: int = 9,
            rope_alpha: float = 0.025,
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
            n_heads=n_heads,
            rope_max_l=rope_max_l,
            rope_alpha=rope_alpha
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


class Head(torch.nn.Module):
    def __init__(
            self,
            n_features: int,
            n_output: int,
            n_embedding: int = 0,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_output = n_output

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
        out_tensor = rearrange(
            out_tensor,
            "b (w h) (c w2 h2) -> b c (w w2) (h h2)",
            h=16, w=32, h2=2, w2=2
        )
        return out_tensor


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
            n_input: int = 7,
            n_output: int = 7,
            n_features: int = 512,
            n_blocks: int = 8,
            n_heads: int = 8,
            n_embedding: int = 0,
            mult: int = 2,
            rope_max_l: int = 9,
            rope_alpha: float = 0.025,
            wave_length: float = 0.07,
    ) -> None:
        super().__init__()
        self.tokenizer = Tokenizer(
            n_channels=n_input,
            n_features=n_features,
        )
        self.blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    n_features=n_features,
                    n_heads=n_heads,
                    n_embedding=n_embedding,
                    mult=mult,
                    rope_max_l=rope_max_l,
                    rope_alpha=rope_alpha,
                )
                for _ in range(n_blocks)
            ]
        )
        self.head = Head(
            n_features=n_features,
            n_output=n_output,
            n_embedding=n_embedding,
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
        tokens = self.tokenizer(in_tensor)
        for block in self.blocks:
            tokens = block(tokens, embedding)
        out_tensor = self.head(tokens, embedding)
        return out_tensor


def get_net(
    n_input=7, n_output=7, n_blocks=8, n_features=512, n_heads=8,
    mult=2, rope_max_l=9, rope_alpha=0.025,
    n_embedding=0, wave_length=0.07,
    device=None, dtype=torch.float32
):
    # More flexibility than with torch.nn.sequential
    transformer = Transformer(
        n_input=n_input, n_output=n_output, n_blocks=n_blocks,
        n_features=n_features, n_heads=n_heads,
        mult=mult, rope_max_l=rope_max_l, rope_alpha=rope_alpha,
        n_embedding=n_embedding, wave_length=wave_length,
    )
    if device is None:
        device = torch.device("cpu")
    transformer = transformer.to(device=device, dtype=dtype)
    return transformer
