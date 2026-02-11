import torch


__all__ = ["in_mean", "in_std", "res_mean", "res_std", "weights_lat"]


in_mean = torch.tensor([55508.282441,281.117870,0.003168,14.505203,-0.043664,287.388006,0.000728])[:, None, None]
in_std = torch.tensor([2728.556341,12.240092,0.002550,17.783869,12.363526,15.208583,0.001348])[:, None, None]
res_mean = torch.tensor([0.002414,0.000011,0.000000,-0.000041,0.000005,0.000020,0.000000])[:, None, None]
res_std = torch.tensor([213.410905,1.165583,0.000497,3.851449,5.196615,2.521460,0.000860])[:, None, None]
# Dataset is now doubly-periodic and no latitude-longitude reweighting
# is required for training. Use scalar unity so losses remain unchanged
# but broadcasting is trivial in the training loop.
weights_lat = torch.tensor(1.0)