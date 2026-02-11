import torch


__all__ = ["in_mean", "in_std", "res_mean", "res_std", "weights_lat"]


# Normalisation constants for the doubly-periodic single-channel dataset
in_mean = torch.tensor([0.0])[:, None, None]
in_std = torch.tensor([0.351026])[:, None, None]
res_mean = torch.tensor([0.0])[:, None, None]
res_std = torch.tensor([0.190455])[:, None, None]
# Dataset is now doubly-periodic and no latitude-longitude reweighting
# is required for training. Use scalar unity so losses remain unchanged
# but broadcasting is trivial in the training loop.
weights_lat = torch.tensor(1.0)