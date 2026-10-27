import xarray as xr
import numpy as np
import os
import pandas as pd


def monthly_conversion(input_dir=None, output_dir=None, chunksize=31):
    """
    Monthly files of the daily GSWP3-W5E5 forcing (precipitation, temperature, short- and long-wave radiation) that
    the model reads (<data root>/Input_data/monthly_climate_forcing).
    input_dir  : folder with the sub-folders precipitation/, temperature/, rad_shortwave/, rad_longwave/ holding the
                 daily ISIMIP files gswp3-w5e5_obsclim_<var>_global_daily_*.nc
    output_dir : folder of the monthly files
    chunksize  : time chunk of the output (days)
    """

    # ==== output directory ====
    if input_dir is None or output_dir is None:
        raise ValueError('monthly_conversion: input_dir and output_dir must be given')
    os.makedirs(output_dir, exist_ok=True)

    # =========================================================
    # ✅ Input file lists (one list per variable)
    # =========================================================
    pr_files = os.path.join(input_dir, 'precipitation', 'gswp3-w5e5_obsclim_pr_global_daily_*.nc')
    tp_files = os.path.join(input_dir, 'temperature', 'gswp3-w5e5_obsclim_tas_global_daily_*.nc')
    rs_files = os.path.join(input_dir, 'rad_shortwave', 'gswp3-w5e5_obsclim_rsds_global_daily_*.nc')
    rl_files = os.path.join(input_dir, 'rad_longwave', 'gswp3-w5e5_obsclim_rlds_global_daily_*.nc')

    # =========================================================
    # ✅ STEP 1: Load datasets
    # =========================================================
    ds_pr = xr.open_mfdataset(pr_files, combine="by_coords")
    ds_tp = xr.open_mfdataset(tp_files, combine="by_coords")
    ds_rs = xr.open_mfdataset(rs_files, combine="by_coords")
    ds_rl = xr.open_mfdataset(rl_files, combine="by_coords")

    # Decode time
    ds_pr = xr.decode_cf(ds_pr)
    ds_tp = xr.decode_cf(ds_tp)
    ds_rs = xr.decode_cf(ds_rs)
    ds_rl = xr.decode_cf(ds_rl)

    # =========================================================
    # ✅ STEP 2: Sort + Align time (CRITICAL)
    # =========================================================
    datasets = [ds_pr, ds_tp, ds_rs, ds_rl]

    # Sort time
    datasets = [ds.sortby("time") for ds in datasets]

    # Align all datasets to same time axis
    ds_pr, ds_tp, ds_rs, ds_rl = xr.align(*datasets, join="inner")

    # Remove duplicates (just in case)
    _, idx = np.unique(ds_pr.time.values, return_index=True)
    ds_pr = ds_pr.isel(time=idx)
    ds_tp = ds_tp.isel(time=idx)
    ds_rs = ds_rs.isel(time=idx)
    ds_rl = ds_rl.isel(time=idx)

    # =========================================================
    # Check missing dates
    # =========================================================
    # Convert time to pandas DatetimeIndex
    time_index = pd.DatetimeIndex(ds_pr.time.values)

    # Create expected full daily time range
    expected_time = pd.date_range(
        start=time_index.min(),
        end=time_index.max(),
        freq="D"
    )

    # Find missing timestamps
    missing_time = expected_time.difference(time_index)

    # Print results
    if len(missing_time) == 0:
        print("✅ No missing days detected.")
    else:
        print(f"⚠️ Missing {len(missing_time)} days!")
        print("Example missing days:", missing_time[:10])

    # =========================================================
    # ✅ STEP 3: Merge everything
    # =========================================================
    ds = xr.merge([ds_pr, ds_tp, ds_rs, ds_rl])

    # =========================================================
    # ✅ STEP 4: Add monthly label
    # =========================================================
    ds = ds.assign_coords(month=ds.time.dt.strftime('%Y-%m'))

    # =========================================================
    # ✅ STEP 5: Split by month + write
    # =========================================================
    for key, data in ds.groupby("month"):

        print(f"Processing {key} ...")

        # Load into memory to avoid dask write issues
        data = data.load()

        encoding = {}

        for var in data.data_vars:
            dims = data[var].dims
            shape = data[var].shape

            chunks = []

            for dim, size in zip(dims, shape):
                if dim == "time":
                    chunks.append(min(chunksize, size))  # monthly chunk
                else:
                    chunks.append(size)           # full spatial chunk

            encoding[var] = {
                "chunksizes": tuple(chunks),
                "zlib": True,
                "complevel": 4
            }

        outfile = os.path.join(output_dir, f"{key}.nc")

        data.to_netcdf(
            outfile,
            format="NETCDF4",
            engine="netcdf4",
            encoding=encoding
        )

    print("✅ All done!")


def check_chunksize(fn=None):
    """print the netCDF chunk sizes of every variable of a monthly forcing file written by monthly_conversion"""
    if fn is None:
        raise ValueError('check_chunksize: fn must be given')
    test_ds = xr.open_dataset(filename_or_obj=fn)

    print(f"Checking file: {fn}")

    for var in test_ds.data_vars:
        print(var, "->", test_ds[var].encoding.get("chunksizes"))

    print()
