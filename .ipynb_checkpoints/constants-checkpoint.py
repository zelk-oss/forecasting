"""
Constants obtained from compute_norms.py which reads dataset 
"""
import torch


__all__ = ["in_mean", "in_std", "res_mean", "res_std"]


# Normalisation constants for the doubly-periodic single-channel dataset
# coming from dataset with correct model time - days conversion 
in_mean = torch.tensor([-0.000000])[:, None, None]
in_std  = torch.tensor([0.302901])[:, None, None]
res_mean= torch.tensor([-0.000000])[:, None, None]
res_std = torch.tensor([0.124247])[:, None, None]
# Dataset is now doubly-periodic and no latitude-longitude reweighting
# is required for training. Use scalar unity so losses remain unchanged
# but broadcasting is trivial in the training loop.
