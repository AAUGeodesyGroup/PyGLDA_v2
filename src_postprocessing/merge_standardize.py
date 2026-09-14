
import gc
import sys
import numpy as np
from pathlib import Path

import os
import numpy as np
import pandas as pd
import xarray as xr
import h5py
from tqdm import tqdm
from src_DA.EnumDA import Stage


class yearly_merge:
    """
    This class is used to post-process the model output: cropping the area and assembling daily output in yearly basis.
    """

    def __init__(self, basin_mask_dir='', basin_name='Brahmaputra', case_name='test', state_dir='', output_dir='',
                 stage=Stage.OL):
        bm = h5py.File(name=Path(basin_mask_dir) /basin_name /('%s_res_0.5.h5' % basin_name),
                       mode='r')['basin'][:]  # this has to be 0.5 degree

        # 1. Convert the 0/1 numpy array to an xarray DataArray with boolean values
        # Ensure the dimensions and coordinates match the original dataset
        res = 0.5
        err = res / 10
        lat_coords = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
        lon_coords = np.arange(-180 + res / 2, 180 - res / 2 + err, res)
        self.__mask = xr.DataArray(
            bm.astype(bool),
            coords={"lat": lat_coords, "lon": lon_coords},
            dims=["lat", "lon"],
            name="region_mask"
        )

        if stage==Stage.DA:
            self.__state_dir = Path(state_dir)/case_name
        else:
            self.__state_dir = Path(state_dir)

        dp1= Path(output_dir) / case_name
        dp1.mkdir(parents=False, exist_ok=True)
        dp2 = dp1/ stage.name
        dp2.mkdir(exist_ok=True)
        self.__output_dir = dp2
        self.__stage = stage
        # self.__case = case_name

        pass

    def configure_ensID(self, ens_id):
        self.__ens_id = ens_id
        self.__state_dir = self.__state_dir / ('Ens_%s' % self.__ens_id)
        self.__output_dir = self.__output_dir / ('Ens_%s' % self.__ens_id)
        # Create output directory if it doesn't exist
        if not os.path.exists(self.__output_dir):
            os.makedirs(self.__output_dir)
        return self

    def configure_date(self, begin_date, end_date):
        full_date_range = pd.date_range(start=begin_date, end=end_date)
        self.__years_to_process = full_date_range.year.unique()
        self.begin_date, self.end_date = begin_date, end_date
        return self

    def merge_and_crop_by_year_with_mask(self):
        """
        Merge daily .nc files into yearly files based on a specific date range,
        embed a mask, crop the spatial extent, and save with Zlib compression.

        Parameters:
        - data_dir: Directory containing the original daily files.
        - output_dir: Directory to save the processed yearly files.
        - mask_matrix: 0/1 matrix (numpy array) indicating the region of interest.
        - lat_coords: Latitude array corresponding to the mask.
        - lon_coords: Longitude array corresponding to the mask.
        - start_date: Starting date (e.g., '2000-01-01').
        - end_date: Ending date (e.g., '2019-12-31').
        """

        # ========================================================
        # 1. Auto-calculate the Bounding Box for early slicing
        # ========================================================
        # drop=True automatically removes all completely False (0) outer edges
        mask_bbox = self.__mask.where(self.__mask, drop=True)

        # Extract the exact min/max coordinates of the target region
        lat_min, lat_max = float(mask_bbox.lat.min()), float(mask_bbox.lat.max())
        lon_min, lon_max = float(mask_bbox.lon.min()), float(mask_bbox.lon.max())

        # Live bar for the rank whose output reaches the screen; a quiet bar
        # (redraw every 10 s) for ranks whose stdout Regional_DA has swapped for
        # a log file, so the file does not fill with carriage returns.
        # The untouched stdout object always reports name '<stdout>' regardless of
        # whether fd 1 is a tty, a pipe (mpiexec) or a file - so test the name.
        to_logfile = getattr(sys.stdout, "name", "<stdout>") not in ("<stdout>", "<stderr>")

        print(f"Merging daily output into yearly files | Ens_{self.__ens_id} | "
              f"{self.begin_date} to {self.end_date} | "
              f"crop {mask_bbox.sizes['lat']}x{mask_bbox.sizes['lon']} cells")
        print(f"  -> {self.__output_dir}")

        for year in self.__years_to_process:

            # 2. Construct the date range and gather available file paths for the current year
            year_start = f"{year}-01-01"
            year_end = f"{year}-12-31"

            if year == self.__years_to_process[0]:
                year_start = self.begin_date
            if year == self.__years_to_process[-1]:
                year_end = self.end_date

            date_range = pd.date_range(start=year_start, end=year_end)

            file_paths = []
            for date in date_range:
                filename = f"daily_output_{date.strftime('%Y-%m-%d')}.nc"
                full_path = os.path.join(self.__state_dir, filename)
                # print(full_path)
                if os.path.exists(full_path):
                    file_paths.append(full_path)

            # Skip the year if no files are found
            if not file_paths:
                print(f"  {year}: no daily files found, skipped")
                continue

            # 3. Preprocess function: Coarse Crop (Bounding Box) -> load -> Time Injection
            #    ORDER MATTERS. On a lazily opened (non-dask) dataset, expand_dims()
            #    forces the FULL global grid of every variable into memory, and a
            #    later .sel() only returns views into those global arrays - so each
            #    "small" cropped frame silently pins the whole daily file (~tens of MB
            #    x 365 days x every rank). Crop first while still lazy so only the
            #    bounding box is read from disk, load that, then add the time axis.
            def preprocess(ds):
                # Extract the file path from xarray encoding
                path = ds.encoding.get("source")
                # Parse the date string from the filename
                date_str = os.path.basename(path).replace("daily_output_", "").replace(".nc", "")
                time_val = pd.to_datetime(date_str)

                # EARLY SLICING while lazy: reads only the bounding box from disk
                # Check if latitude is ascending or descending to apply slice correctly
                if ds.lat[0] < ds.lat[-1]:
                    ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
                else:
                    ds = ds.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))

                ds = ds.load()  # bbox only: a few hundred cells per variable

                # Add the time dimension AFTER loading the cropped data
                if "time" not in ds.dims:
                    ds = ds.expand_dims(time=[time_val])

                return ds

            # 4. Read the daily files ONE AT A TIME (crop -> load -> close), then
            #    concatenate the year. open_mfdataset on ~365 global files kept up
            #    to 128 files open at once, each with its own HDF5 chunk cache, plus
            #    a large dask graph; with several MPI ranks doing this concurrently
            #    the machine ran out of memory and the kernel killed a rank (signal 9).
            frames = []
            bar = tqdm(total=len(file_paths), desc=f"  {year}", unit="file", colour="cyan",
                       file=sys.stdout, mininterval=10.0 if to_logfile else 0.5)
            for fp in file_paths:
                with xr.open_dataset(fp, cache=False) as ds_day:
                    frames.append(preprocess(ds_day))
                bar.update(1)
            ds_year = xr.concat(frames, dim="time", coords="minimal", compat="override")
            del frames

            # 5. Attach the mask and crop
            # Embed the mask as a variable in the dataset for future reference
            ds_year['region_mask'] = self.__mask.copy()

            # Use .where() to crop. drop=True removes outer rows/cols that are entirely NaN
            ds_masked = ds_year.where(self.__mask, drop=True)

            # ========================================================
            # 8. Save: Fast writing with low-level Zlib compression
            # ========================================================
            output_filename = os.path.join(self.__output_dir, f"daily_output_{year}.nc")

            # Configure Zlib compression for all data variables (including the mask)
            # complevel=5 provides a good balance between compression ratio and speed
            encoding = {var: {"zlib": True, "complevel": 1} for var in ds_masked.data_vars}

            bar.set_postfix_str("writing...")
            ds_masked.to_netcdf(output_filename, encoding=encoding)
            bar.set_postfix_str(f"{os.path.basename(output_filename)}  "
                                f"{os.path.getsize(output_filename) / 1e6:.1f} MB")
            bar.close()

            # Close datasets and release memory explicitly so nothing carries over to the next year
            ds_year.close()
            ds_masked.close()
            del ds_year, ds_masked
            gc.collect()

    def merge_by_year(self):
        """
        Merge daily .nc files into yearly files based on a specific date range,
        embed a mask, crop the spatial extent, and save with Zlib compression.

        Parameters:
        - data_dir: Directory containing the original daily files.
        - output_dir: Directory to save the processed yearly files.
        - mask_matrix: 0/1 matrix (numpy array) indicating the region of interest.
        - lat_coords: Latitude array corresponding to the mask.
        - lon_coords: Longitude array corresponding to the mask.
        - start_date: Starting date (e.g., '2000-01-01').
        - end_date: Ending date (e.g., '2019-12-31').
        """

        # ========================================================
        # 1. Auto-calculate the Bounding Box for early slicing
        # ========================================================
        # drop=True automatically removes all completely False (0) outer edges
        mask_bbox = self.__mask.where(self.__mask, drop=True)

        # Extract the exact min/max coordinates of the target region
        lat_min, lat_max = float(mask_bbox.lat.min()), float(mask_bbox.lat.max())
        lon_min, lon_max = float(mask_bbox.lon.min()), float(mask_bbox.lon.max())

        for year in self.__years_to_process:
            print(f"--- Processing year: {year} ---")

            # 2. Construct the date range and gather available file paths for the current year
            year_start = f"{year}-01-01"
            year_end = f"{year}-12-31"

            if year == self.__years_to_process[0]:
                year_start = self.begin_date
            if year == self.__years_to_process[-1]:
                year_end = self.end_date

            date_range = pd.date_range(start=year_start, end=year_end)

            file_paths = []
            for date in date_range:
                filename = f"daily_output_{date.strftime('%Y-%m-%d')}.nc"
                full_path = os.path.join(self.__state_dir, filename)
                # print(full_path)
                if os.path.exists(full_path):
                    file_paths.append(full_path)

            # Skip the year if no files are found
            if not file_paths:
                print(f"No files found for year {year}. Skipping.")
                continue

            # 3. Preprocess function: Coarse Crop (Bounding Box) & Time Injection
            def preprocess(ds):
                # Extract the file path from xarray encoding
                path = ds.encoding.get("source")
                # Parse the date string from the filename
                date_str = os.path.basename(path).replace("daily_output_", "").replace(".nc", "")
                time_val = pd.to_datetime(date_str)

                # Add the time dimension if it doesn't exist
                if "time" not in ds.dims:
                    ds = ds.expand_dims(time=[time_val])

                # # EARLY SLICING: Dramatically reduces I/O bottleneck
                # # Check if latitude is ascending or descending to apply slice correctly
                # if ds.lat[0] < ds.lat[-1]:
                #     ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
                # else:
                #     ds = ds.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))

                return ds

            # 4. Open and concatenate all daily files for the year using Dask
            # parallel=True speeds up the loading process significantly
            ds_year = xr.open_mfdataset(
                file_paths,
                preprocess=preprocess,
                combine="nested",
                concat_dim="time",
                compat="override",
                coords="minimal",
                parallel=False
            )

            # 5. Attach the mask and crop
            # Embed the mask as a variable in the dataset for future reference
            ds_year['region_mask'] = self.__mask.copy()

            # Use .where() to crop. drop=True removes outer rows/cols that are entirely NaN
            ds_masked = ds_year.where(self.__mask, drop=True)

            # ========================================================
            # 7. Execution: Load data into memory (Real disk I/O happens here)
            # ========================================================
            # print(f"Reading and computing data from disk for {year} (this takes time)...")
            # ds_masked.load()

            # ========================================================
            # 8. Save: Fast writing with low-level Zlib compression
            # ========================================================
            output_filename = os.path.join(self.__output_dir, f"daily_output_{year}.nc")

            # Configure Zlib compression for all data variables (including the mask)
            # complevel=5 provides a good balance between compression ratio and speed
            encoding = {var: {"zlib": True, "complevel": 1} for var in ds_year.data_vars}

            print(f"Saving cropped and compressed dataset to: {output_filename} ...")
            ds_year.to_netcdf(output_filename, encoding=encoding)

            # Close datasets to free up memory
            ds_year.close()
            print(f"Year {year} completed successfully.\n")


def demo1():
    ps = yearly_merge(basin_mask_dir='/media/user/My Book/Fan/WaterGap/Basin/mask', basin_name='Brahmaputra',
                      state_dir='/media/user/My Book/Fan/WaterGap/OL_output',
                      output_dir='/media/user/My Book/Fan/WaterGap/Res/', stage=Stage.OL)

    ps.configure_ensID(ens_id=0).configure_date(begin_date='2002-01-01', end_date='2005-04-30')

    ps.merge_and_crop_by_year_with_mask()
    pass

def demo2():
    from mpi4py import MPI
    """Parallel execution using MPI. Each rank will have its own log file."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    ps = yearly_merge(basin_mask_dir='/media/user/My Book/Fan/WaterGap/Basin/mask', basin_name='Brahmaputra',
                      state_dir='/media/user/My Book/Fan/WaterGap/OL_output',
                      output_dir='/media/user/My Book/Fan/WaterGap/Res/', stage=Stage.OL)

    ps.configure_ensID(ens_id=rank).configure_date(begin_date='2002-01-01', end_date='2005-04-30')

    ps.merge_and_crop_by_year_with_mask()


if __name__ == '__main__':
    demo2()
