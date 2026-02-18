# For debugging
import os
from copy import deepcopy

import torch
import torch.nn
import torch.optim
from torch.utils.data import DataLoader

import xarray as xr
import numpy

from tqdm.notebook import tqdm 

from neural_net import get_net
from constants import *

import matplotlib.pyplot as plt

torch.manual_seed(42)

# settings 
device = torch.device("cuda")
dtype = torch.bfloat16
batch_size = 64
n_epochs = 1
n_blocks = 4
n_features = 64 

# load data 
ds_train = xr.open_zarr("../data/sqg_train.zarr")["q"].compute(num_workers=4)
print(f"ds_train.shape: {ds_train.shape}")
print(f"ds_train.dims: {ds_train.dims}")

train_data = ((torch.as_tensor(ds_train.values[:-1], dtype=dtype)-in_mean) / in_std)
print(f"train_data.shape: {train_data.shape}")
print(f"train_data.dtype: {train_data.dtype}")
print(f"Number of channels: {train_data.shape[1] if len(train_data.shape) > 1 else 'N/A'}")

print("\n=== DATASET DEBUG ===")
print("train_data.shape:", train_data.shape)
print("Expected: (N, 1, H, W) for unconditional FM")

# Check if channel dimension missing
if train_data.ndim == 3:
    print("⚠️ Missing channel dimension! Should be 4D.")
elif train_data.ndim == 4:
    print("✅ Channel dimension present.")
else:
    print("❌ Unexpected tensor rank:", train_data.ndim)

del ds_train
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
print(f"len train loader: {len(train_loader)}")

# Check actual batch shapes
print("\nFirst 3 batches:")
for i, batch in enumerate(train_loader):
    print(f"  Batch {i}: {batch.shape}")
    if i >= 2:
        break

ds_val = xr.open_zarr("../data/sqg_val.zarr")["q"].compute(num_workers=16) # validation data 
val_data = (torch.as_tensor(ds_val.values[:-1], dtype=dtype)-in_mean) / in_std

print("\n=== DATASET DEBUG ===")
print("val_data.shape:", val_data.shape)
print("Expected: (N, 1, H, W) for unconditional FM")

# Check if channel dimension missing
if val_data.ndim == 3:
    print("⚠️ Missing channel dimension! Should be 4D.")
elif val_data.ndim == 4:
    print("✅ Channel dimension present.")
else:
    print("❌ Unexpected tensor rank:", val_data.ndim)
del ds_val

val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
print(f"len train loader: {len(val_loader)}")

# Check actual batch shapes
print("\nFirst 3 batches:")
for i, batch in enumerate(val_loader):
    print(f"  Batch {i}: {batch.shape}")
    if i >= 2:
        break


# define the NN 
# CHANGE THIS IF NOT CONDITIONED 
n_embedding = 32
wave_length = 0.1
lr = 1e-3
model = get_net(
    # Input: only intermediate state: unconditional model
    n_input=1,
    n_output=1, n_blocks=n_blocks, n_features=n_features, mult=2,
    # Activation of pseudo time
    n_embedding=n_embedding, wave_length=wave_length,
    device=device, dtype=dtype
)
optim = torch.optim.Adam(model.parameters(), lr=lr)

pbar_epoch = tqdm(range(n_epochs))

mse_val = torch.inf
best_mse = torch.inf
best_model = None

for _ in pbar_epoch:
    pbar_train = tqdm(iter(train_loader), total=len(train_loader), leave=True)
    # Training loop
    model = model.train()
    for batch in pbar_train:        
        batch = batch.to(device=device, dtype=dtype)

        if _ == 0 and pbar_train.n == 0:  # only first batch first epoch
            print("\n=== FIRST TRAIN BATCH DEBUG ===")
            print("batch.shape:", batch.shape)

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

        if _ == 0 and pbar_train.n == 0:
            print("\n=== INTERPOLANT DEBUG ===")
            print("intermediate_state.shape:", intermediate_state.shape)
            print("pseudo_time.shape:", pseudo_time.shape)
        ## Neural network predicts velocity now
        prediction = model(input_tensor, pseudo_time)
        error = (prediction - target_velocity).pow(2)

        mse_train = (error).mean()
        mse_train.backward()
        optim.step()

        pbar_train.set_postfix(mse_train=mse_train.item(), mse_val=mse_val)

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

    pbar_train.set_postfix(mse_train=mse_train.item(), mse_val=mse_val)
        
    # Check if new model is better
    if mse_val < 0.999 * best_mse: # 0.999 to get rid of randomness 
        best_mse = mse_val
        best_model = deepcopy(model).cpu()


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

    
