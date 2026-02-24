"""
Conditional ConvNet for 2-D fields, built from ConvNeXt-style residual blocks and optionally conditioned on time (or any scalar “pseudo-time”) using random Fourier feature embeddings and conditional normalization.

Random Fourier Embedding --> encodes a scalar pseudo-time into a high-dimensional smooth embedding. It allows smooth interpolation across time. 
1) t' = t/lambda 
2) [ sin(t'W), cos(t'W)] 
3) MLP projection to embedding dimension 

!! look at original citation for the net: https://arxiv.org/abs/2201.03545
"""
import torch
import torch.nn


# conditioning = time embedding 
class ConditionalLayerNorm(torch.nn.Module):
    def __init__(self, n_channels=256, n_embedding=256):
        super().__init__()
        # Layer normalisation without affine parameters (gamma, beta) Use of GroupNorm as more efficient
        self.layer_norm = torch.nn.GroupNorm(num_groups=1, num_channels=n_channels, affine=False)
        # Linear layer to project from embedding to affine parameters
        self.affine_embedding = torch.nn.Linear(n_embedding, n_channels * 2)

    def forward(self, in_tensor, embedding):
        # Get affine parameters by linearly projecting embedding
        gamma_beta = self.affine_embedding(embedding)
        # Add height, width dimension to embedding
        gamma_beta = gamma_beta.unsqueeze(-1).unsqueeze(-1)
        # Split affine parameters
        gamma, beta = gamma_beta.chunk(2, dim=1)
        # Apply layer normalization without affine
        normed_tensor = self.layer_norm(in_tensor)
        # Apply affine from embedding
        return gamma * normed_tensor + beta


class ConvNeXtBlock(torch.nn.Module):
    def __init__(self, n_features, mult=2, n_embedding=256):
        super().__init__()
        # replace with z
        self.lon_pad = torch.nn.CircularPad2d((0, 0, 3, 3))
        self.lat_pad = torch.nn.CircularPad2d((3, 3, 0, 0))
        self.conv_in = torch.nn.Conv2d(
            n_features, n_features, kernel_size=7,
            padding=0, groups=n_features
        )
        # Use of conditional layer normalisation
        self.norm = ConditionalLayerNorm(n_features, n_embedding)
        self.branch_1 = torch.nn.Conv2d(n_features, n_features*mult, kernel_size=1)
        self.branch_2 = torch.nn.Sequential(
            torch.nn.Conv2d(n_features, n_features*mult, kernel_size=1),
            torch.nn.ReLU()
        )
        self.out_branch = torch.nn.Conv2d(n_features*mult, n_features, kernel_size=1)
        self.gamma = torch.nn.Parameter(torch.ones(n_features, 1, 1)) # changed to ones instead of zeroes

    def forward(self, in_tensor, embedding):
        padded_tensor = self.lon_pad(in_tensor)
        padded_tensor = self.lat_pad(padded_tensor)
        spatial_tensor = self.conv_in(padded_tensor)
        normed_tensor = self.norm(spatial_tensor, embedding)
        branch_tensor = self.branch_1(normed_tensor) * self.branch_2(normed_tensor)
        out_tensor = self.out_branch(branch_tensor)
        return in_tensor + self.gamma * out_tensor

# pseudo time is encoded in high-dimensional embedding 
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


class ConvNet(torch.nn.Module):
    def __init__(self, n_input=7, n_output=7, n_layers=4, n_features=256, mult=2, use_time=False, n_embedding=256, wave_length=0.1):
        super().__init__()
        self.in_layer = torch.nn.Conv2d(n_input, n_features, kernel_size=1)
        self.blocks = torch.nn.ModuleList([
            ConvNeXtBlock(n_features, mult, n_embedding) for _ in range(n_layers)
        ])
        self.n_embedding = n_embedding
        if use_time:
            # Define embedding
            self.embedding = RandomFourierEmbedding(n_embedding, 256, wave_length)
        else:
            self.embedding = None

        # Use of LayerNorm conditioned on embedding
        self.out_norm = ConditionalLayerNorm(n_features, n_embedding)
        self.out_layer = torch.nn.Conv2d(n_features, n_output, kernel_size=1)

    def forward(self, in_tensor, pseudo_time=None):
        # use embedding if pseudo time conditioning
        pseudo_time_conditioning = (
            self.embedding is not None                  # Check if embedding is defined
        ) and (
            pseudo_time is not None                     # Check if pseudo time is given 
        )
        if pseudo_time_conditioning:
            embedding = self.embedding(pseudo_time)
        else:
            embedding = torch.zeros(
                (in_tensor.shape[0], self.n_embedding),
                device=in_tensor.device, dtype=in_tensor.dtype
            )
        # Apply input projection, expanding channels
        feature_tensor = self.in_layer(in_tensor)
        for b in self.blocks:
            # Non-linearly extract features
            feature_tensor = b(feature_tensor, embedding)
        # Norm before output
        feature_tensor = self.out_norm(feature_tensor, embedding)
        # Output projection, shrinking channels 
        return self.out_layer(feature_tensor)


def get_net(
    n_input=7, n_output=7, n_layers=4, n_features=256, mult=2,
    use_time=False, n_embedding=256, wave_length=0.1,
    device=None, dtype=torch.float32
):
    # More flexibility than with torch.nn.sequential
    convnet = ConvNet(
        n_input=n_input, n_output=n_output, n_layers=n_layers, n_features=n_features, mult=mult,
        use_time=use_time, n_embedding=n_embedding, wave_length=wave_length
    )
    if device is None:
        device = torch.device("cpu")
    convnet = convnet.to(device=device, dtype=dtype)
    return convnet
