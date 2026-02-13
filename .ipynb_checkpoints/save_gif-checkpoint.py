import numpy as np 
import xarray as xr

src = "../data/simu_pyqg_512_3_second_run.zarr"

ds = xr.open_zarr(src)
print('Opened source:', src)
print('Data variables:', list(ds.data_vars))



