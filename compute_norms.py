
import xarray as xr
import numpy as np
import torch

ds = xr.open_zarr("../simu_pyqg_512_1.zarr")["q"].drop_vars("time")
ds_in = ds[:-1]
ds_res = ds[1:] - ds[:-1]

input_mean = ds_in.mean(["time","longitude","latitude"]).compute().values
input_std  = np.sqrt(((ds_in - input_mean)**2).mean(["time","longitude","latitude"]).compute().values)

res_mean = ds_res.mean(["time","longitude","latitude"]).compute().values
res_std  = np.sqrt(((ds_res - res_mean)**2).mean(["time","longitude","latitude"]).compute().values)

def to_torch_str(arr):
    return "torch.tensor([" + ", ".join(f"{v:.6f}" for v in arr) + "])[:, None, None]"

print("in_mean =", to_torch_str(input_mean))
print("in_std  =", to_torch_str(input_std))
print("res_mean=", to_torch_str(res_mean))
print("res_std =", to_torch_str(res_std))