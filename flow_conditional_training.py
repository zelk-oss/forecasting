# run_training_one_epoch.py
# Flow matching training with all fixes
import os
import torch
from torch.utils.data import DataLoader
import xarray as xr
from pathlib import Path
from tqdm import tqdm 
from copy import deepcopy  

from neural_net import get_net
from constants import in_mean, in_std, res_mean, res_std, weights_lat

torch.manual_seed(42)

# Settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# PARAMS 
batch_size = 64
n_epochs = 1
n_features = 64
lr = 1e-3
n_blocks = 4
n_heads = 8
n_embedding = 32    
wave_length = 0.07  

# Cast normalization constants to device/dtype
in_mean = in_mean.to(device=device, dtype=dtype)
in_std = in_std.to(device=device, dtype=dtype)
res_mean = res_mean.to(device=device, dtype=dtype)
res_std = res_std.to(device=device, dtype=dtype)
weights_lat = weights_lat.to(device=device, dtype=dtype)

# Load datasets
datasets = {}
split_to_file = {
    "train": "sqg_train_small.zarr",  # different name
    "val": "sqg_val.zarr"
}
for split, fname in split_to_file.items():
    ds = xr.open_zarr(f"../data/{fname}")["q"].compute(num_workers=4)
    vals = torch.as_tensor(ds.values, dtype=dtype, device=device)
    
    input_norm = (vals[:-1] - in_mean) / in_std
    residual_norm = (vals[1:] - vals[:-1] - res_mean) / res_std
    datasets[split] = torch.cat((input_norm, residual_norm), dim=1)
train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=0)
val_loader   = DataLoader(datasets["val"],   batch_size=batch_size, shuffle=False, num_workers=0)

# Instantiate the model
model = get_net(
    n_input=2,  # 2 channels (intermediate_state + data_in)
    n_output=1,
    n_features=n_features,
    n_blocks=n_blocks,
    n_heads=n_heads,
    mult=2,
    n_embedding = n_embedding, 
    wave_length = wave_length,
    device=device,
    dtype=dtype
).to(device)

optim = torch.optim.Adam(model.parameters(), lr=lr)

pbar_epoch = tqdm(range(n_epochs))

mse_val = torch.inf
best_mse = torch.inf
best_model = None

for epoch in pbar_epoch:  # OPTIONAL: Use epoch variable for logging
    # Training loop
    pbar_train = tqdm(train_loader, total=len(train_loader), leave=True)
    
    model = model.train()  
    
    for batch in pbar_train:        
        batch = batch.to(device=device, dtype=dtype)
        
        # Split batch into input/conditioning and target
        data_in, data_target = batch.split((1, 1), dim=1)

        # Sample epsilon (z_0) as random draw from a normal distribution
        noise = torch.randn_like(data_target)

        # Stratified sampling of pseudo time
        pseudo_time = torch.linspace(0, 1, batch.shape[0]+1, device=device, dtype=dtype)[:-1, None]
        time_shift = torch.rand(1, device=device, dtype=dtype)
        pseudo_time = (pseudo_time + time_shift) % 1

        # Construct intermediate state (z_t) with linear interpolant
        intermediate_state = pseudo_time[..., None, None] * data_target \
            + (1 - pseudo_time[..., None, None]) * noise
        target_velocity = data_target - noise

        # Neural network input = [intermediate state, initial conditions]
        input_tensor = torch.cat((intermediate_state, data_in), dim=1)

        optim.zero_grad()
        # Neural network predicts velocity
        prediction = model(input_tensor, pseudo_time) 
    

        error = (prediction - target_velocity).pow(2)
        mse_train = error.mean()
        mse_train.backward()
        optim.step()

        pbar_train.set_postfix(mse_train=mse_train.item(), mse_val=mse_val)

    # Validation loop with DETERMINISTIC seed
    torch.manual_seed(42)  # Reproducible validation
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    mse_val = 0
    samples_val = 0
    pbar_val = tqdm(val_loader, total=len(val_loader), leave=False)
    
    model = model.eval()  
    
    for batch in pbar_val: 
        batch = batch.to(device=device, dtype=dtype)
        data_in, data_target = batch.split((1, 1), dim=1)

        # Sample epsilon (z_0) - now deterministic due to seed
        noise = torch.randn_like(data_target)

        # Stratified sampling of pseudo time
        pseudo_time = torch.linspace(0, 1, batch.shape[0]+1, device=device, dtype=dtype)[:-1, None]
        time_shift = torch.rand(1, device=device, dtype=dtype)
        pseudo_time = (pseudo_time + time_shift) % 1

        # Construct intermediate state
        intermediate_state = pseudo_time[..., None, None] * data_target \
            + (1 - pseudo_time[..., None, None]) * noise
        target_velocity = data_target - noise

        # Neural network input
        input_tensor = torch.cat((intermediate_state, data_in), dim=1)
        
        with torch.no_grad():
            prediction = model(input_tensor, pseudo_time) 
            
        
        error = (prediction - target_velocity).pow(2)
        curr_se = error.mean(dim=(1, 2, 3)).sum().item()
        mse_val = (mse_val * samples_val + curr_se) / (samples_val + len(batch))
        samples_val = samples_val + len(batch)

    pbar_train.set_postfix(mse_train=mse_train.item(), mse_val=mse_val)
    
    # Check if new model is better
    if mse_val < 0.999 * best_mse:
        best_mse = mse_val
        best_model = deepcopy(model).cpu()
        print(f"\n New best model at epoch {epoch+1}. MSE: {best_mse:.6f}")

print(f"\nFinal validation MSE: {mse_val:.6f}")
print(f"Best validation MSE: {best_mse:.6f}")

# Check generalization
if best_model is not None:
    # Save best model checkpoint
    ckpt = {
        "model_state_dict": best_model.state_dict(),  # Save best
        "optimizer_state_dict": optim.state_dict(),
        "best_mse": best_mse,  # Track best MSE
        "config": {
            "n_input": 2,  
            "n_output": 1,
            "n_features": n_features,
            "n_blocks": n_blocks,
            "n_heads": n_heads,
            "mult": 2,
            "n_embedding": n_embedding,      
            "wave_length": wave_length,
        }
    }
    torch.save(ckpt, os.path.join("..", "data", "best_flowmodel_test.ckpt"))
    print(f"Saved best checkpoint (MSE={best_mse:.6f}) to ../data/best_flowmodel_test.ckpt")
else:
    print("Warning: No model was saved (validation never improved)")