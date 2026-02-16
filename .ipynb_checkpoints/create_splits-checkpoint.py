"""
Takes the whole dataset and splits it into train, test and validate 
"""
import xarray as xr
import os

src = "../data/simu_pyqg_512_3.zarr"
out_dir = "../data"
os.makedirs(out_dir, exist_ok=True)

ds = xr.open_zarr(src)
print('Opened source:', src)
print('Data variables:', list(ds.data_vars))

# assume variable 'q' exists and has time dim
varname = 'q'
if varname not in ds:
    # try first data var
    varname = list(ds.data_vars)[0]
    print('Using', varname)

# Select the variable and keep dims
da = ds[varname]

# Determine split indices (first 80% train, next 10% val, rest test)
nt = da.sizes['time']
train_end = int(nt * 0.8)
val_end = train_end + int(nt * 0.1)

print('nt, train_end, val_end:', nt, train_end, val_end)

train = da.isel(time=slice(0, train_end))
val = da.isel(time=slice(train_end, val_end))
test = da.isel(time=slice(val_end, None))

# Convert to dataset and rename variable to 'geopotential' for compatibility
train_ds = train.to_dataset(name='q')
val_ds = val.to_dataset(name='q')
test_ds = test.to_dataset(name='q')

# Ensure var_name coordinate is a list of strings to avoid zarr encoding issues
if 'var_name' in train_ds.coords:
    try:
        train_ds = train_ds.assign_coords(var_name=[str(v) for v in train_ds['var_name'].values])
    except Exception:
        train_ds = train_ds.assign_coords(var_name=['q'])

if 'var_name' in val_ds.coords:
    try:
        val_ds = val_ds.assign_coords(var_name=[str(v) for v in val_ds['var_name'].values])
    except Exception:
        val_ds = val_ds.assign_coords(var_name=['q'])

if 'var_name' in test_ds.coords:
    try:
        test_ds = test_ds.assign_coords(var_name=[str(v) for v in test_ds['var_name'].values])
    except Exception:
        test_ds = test_ds.assign_coords(var_name=['q'])

# remove problematic encoding entries
for ds_out in (train_ds, val_ds, test_ds):
    for k in list(ds_out.encoding.keys()):
        if k in ('chunks', 'preferred_chunks'):
            del ds_out.encoding[k]

# write to zarr
train_ds.to_zarr(os.path.join(out_dir, 'sqg_train.zarr'), mode='w')
val_ds.to_zarr(os.path.join(out_dir, 'sqg_val.zarr'), mode='w')
test_ds.to_zarr(os.path.join(out_dir, 'sqg_test.zarr'), mode='w')

print('Wrote ../data/sqg_train.zarr, ../data/sqg_val.zarr, ../data/sqg_test.zarr')
