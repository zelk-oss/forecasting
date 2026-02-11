# run_training_one_epoch.py
# Minimal training driver to run one epoch on 1 GPU (or CPU)
import os
import torch
from torch.utils.data import DataLoader
import xarray as xr

from neural_net import get_net
from constants import in_mean, in_std, res_mean, res_std, weights_lat

# Settings (tune for server)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32         # use float32 for portability; change to torch.bfloat16 if your GPU supports it
batch_size = 8                # small to avoid OOM; raise if memory allows
n_epochs = 1
n_layers = 2
n_features = 64
lr = 1e-3

# Prepare constants on device/dtype
in_mean = in_mean.to(device=device, dtype=dtype)
in_std = in_std.to(device=device, dtype=dtype)
res_mean = res_mean.to(device=device, dtype=dtype)
res_std = res_std.to(device=device, dtype=dtype)
weights_lat = torch.as_tensor(weights_lat, device=device, dtype=dtype).squeeze()

# Load small dataset (uses parent ../data)
train_zarr = "../data/sqg_train.zarr"
val_zarr = "../data/sqg_val.zarr"

ds_train = xr.open_zarr(train_zarr)["q"].compute(num_workers=4)
ds_val = xr.open_zarr(val_zarr)["q"].compute(num_workers=4)

# Build normalized torch arrays: (time, channel, H, W)
train_vals = torch.as_tensor(ds_train.values, dtype=dtype, device=device)
val_vals   = torch.as_tensor(ds_val.values,   dtype=dtype, device=device)

train_in  = (train_vals[:-1] - in_mean) / in_std
train_res = (train_vals[1:] - train_vals[:-1] - res_mean) / res_std

train_data = torch.cat((train_in, train_res), dim=1)

val_in  = (val_vals[:-1] - in_mean) / in_std
val_res = (val_vals[1:] - val_vals[:-1] - res_mean) / res_std
val_data = torch.cat((val_in, val_res), dim=1)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)

# Build model + optimizer
model = get_net(n_input=1, n_output=1, n_blocks=n_layers, n_features=n_features, device=device, dtype=dtype)
optim = torch.optim.Adam(model.parameters(), lr=lr)

# Training loop: one epoch
model.train()
for epoch in range(n_epochs):
    running_loss = 0.0
    for i, batch in enumerate(train_loader):
        batch = batch.to(device=device, dtype=dtype)
        data_in, data_target = batch.split((1, 1), dim=1)  # single-channel
        optim.zero_grad()
        pred = model(data_in)
        loss = (weights_lat * (pred - data_target).pow(2)).mean()
        loss.backward()
        optim.step()
        running_loss += loss.item()
        if (i + 1) % 10 == 0:
            print(f"epoch {epoch+1} step {i+1}/{len(train_loader)} loss {running_loss / (i+1):.6f}")

# quick validation pass
model.eval()
mse_sum = 0.0
n_samples = 0
with torch.no_grad():
    for batch in val_loader:
        batch = batch.to(device=device, dtype=dtype)
        data_in, data_target = batch.split((1, 1), dim=1)
        pred = model(data_in)
        per_sample_mse = (weights_lat * (pred - data_target).pow(2)).mean(dim=(1,2,3))
        mse_sum += per_sample_mse.sum().item()
        n_samples += per_sample_mse.shape[0]

val_mse = mse_sum / n_samples if n_samples else float("inf")
print("Validation MSE:", val_mse)

# save a checkpoint (CPU copy)
state_dict = model.cpu().state_dict()
torch.save(state_dict, os.path.join("..", "data", "best_model_test.ckpt"))
print("Saved checkpoint to ../data/best_model_test.ckpt")