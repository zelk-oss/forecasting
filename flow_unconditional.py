"""
Training a flow matching model on SQG dynamics. 
The model takes the intermediate state xt and the time t as inputs. 
Outputs learnt velocity field. 
"""
import os
from copy import deepcopy

import torch
import torch.nn
import torch.optim
from torch.utils.data import DataLoader

import xarray as xr
import numpy

from tqdm import tqdm 

from neural_net import get_net
from constants import *

import matplotlib.pyplot as plt

torch.manual_seed(42)

# settings 
device = torch.device("cuda")
dtype = torch.bfloat16

batch_size = 32
n_epochs = 20 

n_blocks = 6
n_features = 64
n_heads = 4      # must divide n_features
n_embedding = 16

wave_length = 0.1
lr = 3e-4

# load data 
ds_train = xr.open_zarr("../data/sqg_train.zarr")["q"].compute(num_workers=2)

train_data = ((torch.as_tensor(ds_train.values[:-1], dtype=dtype)-in_mean) / in_std)
print(train_data.shape)

del ds_train
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

ds_val = xr.open_zarr("../data/sqg_val.zarr")["q"].compute(num_workers=16) # validation data 
val_data = (torch.as_tensor(ds_val.values[:-1], dtype=dtype)-in_mean) / in_std

del ds_val

val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
print(f"len train loader: {len(val_loader)}")

# checking mean and sted 
from constants import in_mean, in_std   # make the import explicit
print("=== Normalization audit ===")
print(f"  in_mean : {in_mean}")
print(f"  in_std  : {in_std}")

# Verify on a raw batch
raw_sample = torch.as_tensor(
    xr.open_zarr("../data/sqg_train.zarr")["q"].values[:200], dtype=torch.float32
)
print(f"\n  Raw data   mean={raw_sample.mean():.4f}  std={raw_sample.std():.4f}")
normalized  = (raw_sample - in_mean) / in_std
print(f"  Normalized mean={normalized.mean():.4f}  std={normalized.std():.4f}")
print(f"  → Should be ≈ 0.0 and ≈ 1.0. If not, in_mean/in_std are wrong.")
del raw_sample, normalized


# define the NN 
model = get_net(
    # Input: only intermediate state: unconditional model
    n_input=1,
    n_output=1, n_blocks=n_blocks, n_features=n_features, mult=2,
    n_heads = n_heads, 
    # Activation of pseudo time
    n_embedding=n_embedding, wave_length=wave_length,
    device=device, dtype=dtype
)
optim = torch.optim.Adam(model.parameters(), lr=lr)

print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

pbar_epoch = tqdm(range(n_epochs))

mse_val = torch.inf
best_mse = torch.inf
best_model = None

history = {"train_mse": [], "val_mse": []}

train_mse_epoch = 0.0
train_samples = 0

for _ in pbar_epoch:
    pbar_train = tqdm(iter(train_loader), total=len(train_loader), leave=True)
    # Training loop
    model = model.train()
    for batch in pbar_train:        
        batch = batch.to(device=device, dtype=dtype)

        # unconditional: target = full batch 
        data_target = batch
        noise = torch.randn_like(data_target)

        ## Stratified sampling of pseudo time to reduce variance during optimisation (Kingma et al., 2022)
        ### Take pseudo time between [0, 1)
        pseudo_time = torch.linspace(0, 1, batch.shape[0]+1, device=device, dtype=dtype)[:-1, None]
        ### Introduce random shift between 0 and 1
        time_shift = torch.rand(1, device=device, dtype=dtype)
        pseudo_time = pseudo_time + time_shift
        ### Ensure that pseudo time is between 0 and 1
        pseudo_time = pseudo_time%1

        ## Construct intermediate state (z_t) with linear interpolant
        intermediate_state = pseudo_time[..., None, None] * data_target \
            + (1-pseudo_time[..., None, None]) * noise
        target_velocity = data_target-noise

        ## Neural network input = [intermediate state, initial conditions]
        input_tensor = intermediate_state

        optim.zero_grad()

        ## Neural network predicts velocity now
        prediction = model(input_tensor, pseudo_time)
        error = (prediction - target_velocity).pow(2)

        mse_train = (error).mean()
        rmse_train = torch.sqrt(error.mean(dim=(1,2,3)))  # per sample
        mse_train.backward()
        optim.step()

        # accumulate values for loss-tracking 
        train_mse_epoch = (train_mse_epoch * train_samples + mse_train.item() * len(batch)) / (train_samples + len(batch))
        train_samples += len(batch)

        
        pbar_train.set_postfix(mse_train=mse_train.item(), mse_val=mse_val)

    history["train_mse"].append(train_mse_epoch)
    
    mse_val = 0
    samples_val = 0
    pbar_val = tqdm(enumerate(val_loader), total=len(val_loader), leave=False)

    
    # Validation loop
    model = model.eval()
    for k, batch in pbar_val:        
        batch = batch.to(device=device, dtype=dtype)
        data_target = batch

        # Change for flow matching model
        ## Sample epsilon (z_0) as random draw from a normal distribution
        noise = torch.randn_like(data_target)

        ## Stratified sampling of pseudo time to reduce variance during optimisation (Kingma et al., 2022)
        ### Take pseudo time between [0, 1)
        pseudo_time = torch.linspace(0, 1, batch.shape[0]+1, device=device, dtype=dtype)[:-1, None]
        ### Introduce random shift between 0 and 1
        time_shift = torch.rand(1, device=device, dtype=dtype)
        pseudo_time = pseudo_time + time_shift
        ### Ensure that pseudo time is between 0 and 1
        pseudo_time = pseudo_time%1

        ## Construct intermediate state (z_t) with linear interpolant
        intermediate_state = pseudo_time[..., None, None] * data_target \
            + (1-pseudo_time[..., None, None]) * noise
        target_velocity = data_target-noise

        ## Neural network input = [intermediate state, initial conditions]
        input_tensor = intermediate_state
        
        with torch.no_grad():
            prediction = model(input_tensor, pseudo_time)
        error = (prediction - target_velocity).pow(2)
        curr_se = (error).mean(dim=(1, 2, 3)).sum().item()
        mse_val = mse_val * samples_val + curr_se
        samples_val = samples_val + len(batch)
        mse_val = mse_val / samples_val

    history["val_mse"].append(mse_val)
    
    pbar_train.set_postfix(mse_train=mse_train.item(), mse_val=mse_val)
        
    # Check if new model is better
    if mse_val < 0.999 * best_mse: # 0.999 to get rid of randomness 
        best_mse = mse_val
        best_model = deepcopy(model).cpu()


import json, matplotlib.pyplot as plt

with open("../data/loss_history.json", "w") as f:
    json.dump(history, f)

fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(history["train_mse"], label="train MSE")
ax.semilogy(history["val_mse"],   label="val MSE")
ax.set_xlabel("epoch"); ax.set_ylabel("MSE (log scale)")
ax.legend(); ax.grid(ls=":", alpha=0.5)
plt.tight_layout()
plt.savefig("../data/loss_curve.png", dpi=150)

# store best model 
if best_model is not None:
    # Save best model checkpoint
    ckpt = {
        "model_state_dict": best_model.state_dict(),  # Save best
        "optimizer_state_dict": optim.state_dict(),
        "best_mse": best_mse,  # Track best MSE
        "config": {
            "n_input": 1,  
            "n_output": 1,
            "n_features": n_features,
            "n_blocks": n_blocks,
            "mult": 2,
            "n_embedding": n_embedding,      
            "wave_length": wave_length,
        }
    }
    torch.save(ckpt, os.path.join("..", "data", "best_flowmodel_test.ckpt"))
    print(f"Saved best checkpoint (MSE={best_mse:.6f}) to ../data/best_flowmodel_test.ckpt")
else:
    print("Warning: No model was saved (validation never improved)")


# After training completes:
print(f"  in_mean={float(in_mean):.5f}  in_std={float(in_std):.5f}")
# ✓ in_std should be ≈ 0.296. If it's 1.0 or something else → root cause found

# The sample_std column tells you everything:
# sample_std ≈ 1.0  at epoch 1 and stays there → model not transporting (your current situation)
# sample_std drops toward 0.3 over epochs      → normalization is working
# sample_std oscillates wildly                 → LR too high

# The sample_RMSE column:
# stays at ≈ 1.41 (√2) → pure noise, model not learning
# drops toward ≈ 1.0   → model is transporting but distribution is too wide  
# drops toward < 1.0   → model is learning the distribution correctly

# Situation                MSE         RMSE            Meaning
# Untrained / pure noise   ≈ 2.0       ≈ 1.41       model predicts 0, target has variance 2
# Learning something      < 1.5        < 1.22       already meaningful
# Decent model          0.1 – 0.5    0.32 – 0.71    depends on field complexity
# Very good model         < 0.1        < 0.32