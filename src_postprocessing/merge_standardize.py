
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
        with h5py.File(name=Path(basin_mask_dir) / basin_name / ('%s_res_0.5.h5' % basin_name), mode='r') as f:
            bm = f['basin'][:]  # this has to be 0.5 degree
            '''global unit mask (global_shp2mask): the yearly files keep the full grid, see _merge_year_files_global'''
            self.__global = f.attrs.get('extent') == 'global'

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
        OL stage. The daily files are GLOBAL: crop each one to the basin bounding box while
        reading, concatenate the year, mask everything outside the basin (NaN) and save the
        masked, compressed yearly file with the basin mask embedded.
        """
        self._merge_year_files_global() if self.__global else self._merge_year_files(apply_mask=True)

    def merge_by_year(self):
        """
        DA stage. The daily files are already cropped to the basin bounding box by the DA
        run (DailyModelRun(is_crop_save=True)). Concatenate the year, mask everything outside
        the basin (NaN) and save the compressed yearly file with the basin mask embedded -
        the same form as the OL stage, so both can be read identically downstream.
        The crop is still applied - on a file that already spans exactly the box it is a
        no-op, and it protects against a DA run configured without is_crop_save.
        """
        self._merge_year_files_global() if self.__global else self._merge_year_files(apply_mask=True)

    # ------------------------------------------------------------------------------------
    def _merge_year_files(self, apply_mask: bool):
        """
        Shared implementation for both stages.

        Memory design (this replaced an open_mfdataset version that kept up to 128 global
        files open at once and, across several MPI ranks, ran the machine out of memory):
          * daily files are opened ONE AT A TIME
          * each is cropped to the bounding box while still lazy, so only the box is read
          * only then is it loaded, closed, and given its time coordinate
          * the year is concatenated from these small in-memory frames
        ORDER MATTERS inside `_read_one_day`: expand_dims() on a lazily opened (non-dask)
        dataset forces the FULL global grid of every variable into memory, and a later
        .sel() returns views into those arrays, so every "small" frame would pin a whole
        daily file. Crop first, load, then add the time axis.
        """
        # 1. bounding box of the basin, for the early crop
        mask_bbox = self.__mask.where(self.__mask, drop=True)
        lat_min, lat_max = float(mask_bbox.lat.min()), float(mask_bbox.lat.max())
        lon_min, lon_max = float(mask_bbox.lon.min()), float(mask_bbox.lon.max())

        # progress bar: live for the rank whose output reaches the screen, quiet (one redraw
        # every 10 s) for ranks whose stdout Regional_DA has swapped for a log file. The
        # untouched stdout object reports name '<stdout>' whatever fd 1 is (tty or mpiexec
        # pipe); a redirected one reports the log path.
        to_logfile = getattr(sys.stdout, "name", "<stdout>") not in ("<stdout>", "<stderr>")

        print(f"Merging daily output into yearly files | {self.__stage.name} | Ens_{self.__ens_id} | "
              f"{self.begin_date} to {self.end_date} | "
              f"crop {mask_bbox.sizes['lat']}x{mask_bbox.sizes['lon']} cells"
              f"{' | mask outside basin' if apply_mask else ''}")
        print(f"  -> {self.__output_dir}")

        def _read_one_day(path):
            """crop (lazy) -> load -> close -> add time. Returns a small in-memory Dataset."""
            date_str = os.path.basename(path).replace("daily_output_", "").replace(".nc", "")
            time_val = pd.to_datetime(date_str)
            with xr.open_dataset(path, cache=False) as ds:
                if ds.lat[0] < ds.lat[-1]:
                    ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
                else:
                    ds = ds.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
                ds = ds.load()
            if "time" not in ds.dims:
                ds = ds.expand_dims(time=[time_val])
            return ds

        for year in self.__years_to_process:

            # 2. daily files of this year that exist on disk
            year_start = self.begin_date if year == self.__years_to_process[0] else f"{year}-01-01"
            year_end = self.end_date if year == self.__years_to_process[-1] else f"{year}-12-31"
            file_paths = [os.path.join(self.__state_dir, f"daily_output_{d.strftime('%Y-%m-%d')}.nc")
                          for d in pd.date_range(start=year_start, end=year_end)]
            file_paths = [f for f in file_paths if os.path.exists(f)]
            if not file_paths:
                print(f"  {year}: no daily files found, skipped")
                continue

            # 3. read one day at a time and concatenate
            frames = []
            bar = tqdm(total=len(file_paths), desc=f"  {year}", unit="file", colour="cyan",
                       file=sys.stdout, mininterval=10.0 if to_logfile else 0.5)
            for fp in file_paths:
                frames.append(_read_one_day(fp))
                bar.update(1)
            ds_year = xr.concat(frames, dim="time", coords="minimal", compat="override")
            del frames

            # 4. embed the mask and blank everything outside the basin (both stages)
            ds_year['region_mask'] = self.__mask.copy()
            ds_out = ds_year.where(self.__mask, drop=True) if apply_mask else ds_year

            # 5. save with zlib compression (level 1: good ratio, fast)
            output_filename = os.path.join(self.__output_dir, f"daily_output_{year}.nc")
            encoding = {var: {"zlib": True, "complevel": 1} for var in ds_out.data_vars}
            bar.set_postfix_str("writing...")
            ds_out.to_netcdf(output_filename, encoding=encoding)
            bar.set_postfix_str(f"{os.path.basename(output_filename)}  "
                                f"{os.path.getsize(output_filename) / 1e6:.1f} MB")
            bar.close()

            # 6. release memory explicitly so nothing carries over to the next year
            ds_year.close()
            ds_out.close()
            del ds_year, ds_out
            gc.collect()

    # ------------------------------------------------------------------------------------
    def _merge_year_files_global(self, complevel=1, tile=(36, 72)):
        """
        Global unit mask (attribute extent = 'global' in the mask file, see global_shp2mask / shp2mask.load_mask):
        the yearly files keep the FULL grid and the model values outside the units, with the unit mask embedded as
        region_mask (1 = cell updated by the DA, 0 = model only), because the global product uses WaterGAP where no
        assimilation is done. A year of global fields (~13 GB) cannot be held in memory on 31 ranks, so the days are
        written one at a time into a NETCDF4 file with an unlimited time axis (zlib `complevel`, lat/lon `tile` as
        the daily writer). The file layout (time, lat, lon; time as days since 1900-01-01; region_mask) is that of the
        regional yearly files, so that every reader (statistical_analysis, state_envelope, export_product) is unchanged.
        """
        import netCDF4 as nc
        to_logfile = getattr(sys.stdout, "name", "<stdout>") not in ("<stdout>", "<stderr>")
        region_mask = self.__mask.values.astype(np.int8)
        print(f"Merging daily output into yearly files | {self.__stage.name} | Ens_{self.__ens_id} | "
              f"{self.begin_date} to {self.end_date} | full grid {region_mask.shape[0]}x{region_mask.shape[1]} cells "
              f"(global unit mask, model values kept outside the units)", flush=True)

        for year in self.__years_to_process:
            year_start = self.begin_date if year == self.__years_to_process[0] else f"{year}-01-01"
            year_end = self.end_date if year == self.__years_to_process[-1] else f"{year}-12-31"
            days = [d for d in pd.date_range(start=year_start, end=year_end)
                    if os.path.exists(os.path.join(self.__state_dir, f"daily_output_{d.strftime('%Y-%m-%d')}.nc"))]
            if not days:
                print(f"  {year}: no daily files found, skipped")
                continue
            output_filename = os.path.join(self.__output_dir, f"daily_output_{year}.nc")
            first = os.path.join(self.__state_dir, f"daily_output_{days[0].strftime('%Y-%m-%d')}.nc")
            with nc.Dataset(first) as s0, nc.Dataset(output_filename, 'w', format='NETCDF4') as out:
                lat, lon = s0['lat'][:], s0['lon'][:]
                if region_mask.shape != (len(lat), len(lon)):
                    raise ValueError(f"{os.path.basename(first)} is {(len(lat), len(lon))} but the unit mask is "
                                     f"{region_mask.shape}: the output of a global case must be on the full grid "
                                     f"(mask attribute extent = global, see shp2mask.load_mask)")
                out.createDimension('time', None)
                out.createDimension('lat', len(lat))
                out.createDimension('lon', len(lon))
                tv = out.createVariable('time', 'f8', ('time',))
                tv.units, tv.calendar = 'days since 1900-01-01', 'proleptic_gregorian'
                out.createVariable('lat', 'f8', ('lat',))[:] = lat
                out.createVariable('lon', 'f8', ('lon',))[:] = lon
                names = [v for v in s0.variables if v not in ('lat', 'lon', 'time')]
                for v in names:
                    var = out.createVariable(v, s0[v].dtype, ('time', 'lat', 'lon'), zlib=True, complevel=complevel,
                                             chunksizes=(1, min(tile[0], len(lat)), min(tile[1], len(lon))),
                                             fill_value=np.nan)
                    for att in s0[v].ncattrs():
                        if att not in ('_FillValue', 'missing_value'):
                            var.setncattr(att, s0[v].getncattr(att))
                rm = out.createVariable('region_mask', 'i1', ('lat', 'lon'), zlib=True, complevel=complevel)
                rm[:] = region_mask
                rm.setncattr('description', 'unit mask: 1 = cell updated by the data assimilation, 0 = model only')
                bar = tqdm(total=len(days), desc=f"  {year}", unit="file", colour="cyan", file=sys.stdout,
                           mininterval=10.0 if to_logfile else 0.5)
                for n, d in enumerate(days):
                    with nc.Dataset(os.path.join(self.__state_dir, f"daily_output_{d.strftime('%Y-%m-%d')}.nc")) as s:
                        for v in names:
                            out[v][n, :, :] = s[v][:]
                    tv[n] = nc.date2num(d.to_pydatetime(), tv.units, tv.calendar)
                    bar.update(1)
            bar.set_postfix_str(f"{os.path.basename(output_filename)}  "
                                f"{os.path.getsize(output_filename) / 1e6:.1f} MB")
            bar.close()
            gc.collect()


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
