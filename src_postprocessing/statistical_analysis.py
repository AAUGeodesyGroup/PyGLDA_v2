
import warnings
import numpy as np
from pathlib import Path
import h5py
from datetime import datetime, timedelta
from src_auxiliary.GeoMathKit import GeoMathKit
from src_DA.EnumDA import Stage, WaterGap_storage_variables
import xarray as xr


class BasinAverageAnalysis:
    """
    This script is for real time analysis to extract basin time series signal for each ensemble member
    """

    def __init__(self, basin_mask_path, state_dir='', stage=Stage.OL, ens_id=0, case_name = ''):
        self.ds = None
        self.__variable_list = None
        self.bm = h5py.File(name=basin_mask_path, mode='r')
        self.__state_dir = Path(state_dir) / case_name / stage.name / f'Ens_{ens_id}'
        self.__stage = stage

    def select_variables(self, variable_list=None):
        if variable_list is None:
            self.__variable_list = [var.name for var in WaterGap_storage_variables]
        else:
            self.__variable_list = variable_list

        return self

    def get_basin_average(self):

        mask_global = self.bm['basin'][:]  # this has to be 0.5 degree
        res = 0.5
        err = res / 10
        lat_coords = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
        lon_coords = np.arange(-180 + res / 2, 180 - res / 2 + err, res)
        mask_global = xr.DataArray(
            mask_global.astype(bool),
            coords={"lat": lat_coords, "lon": lon_coords},
            dims=["lat", "lon"],
            name="region_mask"
        )

        lon_mesh_global, lat_mesh_global = np.meshgrid(lon_coords, lat_coords)
        mask_whole_basin = mask_global.where(mask_global, drop=True)
        mask_whole_basin = mask_whole_basin.notnull().values

        res = {}
        for key in self.__variable_list:
            res[key] ={}
            for basin in self.bm.keys():
                '''produce the 2D mask for each basin'''
                mask_sub_basin =  self.bm[basin][:].astype(bool)  # this has to be 0.5 degree
                lat_local = lat_mesh_global[mask_sub_basin]
                mask_global = self.bm['basin'][:].astype(bool)
                mask_bbox = mask_whole_basin.copy()
                mask_bbox[mask_whole_basin] = mask_sub_basin[mask_global]

                '''load nc file and calculate the basin average'''
                a = self.ds[key].values[:, mask_bbox]
                b = np.cos(np.deg2rad(lat_local))
                c = np.sum(a*b,1)/np.sum(b)
                res[key][basin] = c.copy()
                pass

        hf = h5py.File(str(Path(self.__state_dir) / ('basin_ts_%s.h5'%self.__stage.name)), 'w')
        # print(str(Path(self.__state_dir) / ('basin_ts_%s.h5'%self.__stage.name)))
        for gr, bb in res.items():
            dict_group = hf.create_group(gr)
            for key, vv in bb.items():
                dict_group.create_dataset(key, data=np.array(vv))
        hf.close()

        # close the (dask-backed) dataset explicitly: deleting the reference alone leaves the
        # netCDF handles in xarray's file cache until interpreter exit, where they race the
        # library teardown and print "Exception ignored in CachingFileManager.__del__ ...
        # NetCDF: Not a valid ID" on whichever rank loses the race
        self.ds.close()
        del self.ds
        pass

    def load_nc_files(self, start_date="2002-01-01", end_date="2005-04-30"):
        """
        Load the NetCDF files of given date range
        """
        import os
        import pandas as pd
        import xarray as xr

        # 2. Generate the date range using pandas
        date_range = pd.date_range(start=start_date, end=end_date)
        years_to_process = date_range.year.unique()

        # 3. Build the list of file paths based on the date sequence
        file_paths = []

        for year in years_to_process:
            # Construct the filename matching your naming convention: daily_output_YYYY-MM-DD.nc
            filename = f"daily_output_{year}.nc"
            full_path = os.path.join(self.__state_dir, filename)

            # Check if the file exists to avoid errors
            if os.path.exists(full_path):
                file_paths.append(full_path)
            else:
                print(f"Warning: File not found and skipped -> {filename}")

        # 4. Use xarray's open_mfdataset to read and merge multiple netcdf files at once
        if file_paths:
            # combine='by_coords' automatically concatenates the datasets along the coordinate dimensions (like time)
            ds = xr.open_mfdataset(file_paths, combine="by_coords", parallel=True, chunks={"time": 365})

            # print("Successfully loaded and merged dataset:")

            self.ds = ds

            # print(ds)

            # Example: Access a specific variable for further processing
            # data_var = ds['your_variable_name']
        else:
            raise FileNotFoundError("No matching files found for the specified date range.")

        return self


class BasinAverageAnalysis_post:
    """
    This is also for the basin signal analysis. However, it is different from the above one at:
    1. this is a post-processing step to collect all the ensemble results together for further analysis
    2. in addition to the states, this script also deals with GRACE data for a collective analysis.
    """

    def __init__(self, ens, case: str, basin: str, date_begin='2002-04-01', date_end='2002-04-02'):
        self.ens = ens
        self.case = case
        self.basin = basin

        """deal with the time tag"""
        daylist = GeoMathKit.dayListByDay(begin=date_begin, end=date_end)
        day_first = daylist[0].year + (daylist[0].month - 1) / 12 + daylist[0].day / 365.25
        self.time_tag = day_first + np.arange(len(daylist)) / 365.25

        self.__states = None
        self.__GRACE = None
        pass

    def get_states(self, dir: str, stage: Stage):

        post_fix = stage.name

        temp_dict = {}
        for ens in range(self.ens + 1):
            fn = h5py.File(Path(dir) / self.case/ post_fix/ ('Ens_%s' % ens) / ('basin_ts_%s.h5' % post_fix), 'r')
            ens_dict = {}
            for basin, item in fn.items():
                state_dict = {}
                for state, item2 in item.items():
                    state_dict[state] = item2[:]

                # state_dict['TWS'] = np.sum(np.array(list(state_dict.values())), axis=0)

                ens_dict[basin] = state_dict

            temp_dict[ens] = ens_dict

        '''reformulate for easier analysis'''
        state = fn.keys()
        basins = list(item.keys())

        ff = {}
        for basin in basins:
            dd = {}
            for st in state:
                ss = {}
                for ens in range(self.ens + 1):
                    ss[ens] = temp_dict[ens][st][basin]
                dd[st] = ss
            ff[basin] = dd

        assert len(self.time_tag) == len(ss[ens]), 'Time tag is incorrect!'

        ff['time'] = self.time_tag
        self.__states = ff
        return ff

    def get_GRACE(self, obs_dir: str):
        kk = {}

        gr = h5py.File(Path(obs_dir) / ('%s_obs_GRACE.hdf5' % self.basin), 'r')

        str_time = list(gr['time_epoch'][:].astype(str))
        fraction_time = []
        for tt in str_time:
            if len(tt.split('-')) == 2:
                tt = tt + '-15'
            da = datetime.strptime(tt, '%Y-%m-%d')
            fraction_time.append(da.year + (da.month - 1) / 12 + da.day / 365.25)

        kk['time'] = np.array(fraction_time)
        kk['original'] = {}
        kk['ens_mean'] = {}

        a = gr['ens_1'][:]

        '''calculate how many ensemble member it has'''
        ens_num = -1  # because ens= 0 is included
        for dd in gr.keys():
            if 'ens_' in dd:
                ens_num += 1
        # print(ens_num)
        try:
            basin_num = np.shape(gr['cov'][0])[0]
        except Exception:
            basin_num = 1

        for ens_id in range(2, ens_num + 1):
            a += gr['ens_%s' % ens_id]

        ens_mean = a / ens_num
        original = gr['ens_0'][:]

        for basin in range(1, 1 + basin_num):
            kk['original']['basin_%s' % basin] = original[:, basin - 1]
            kk['ens_mean']['basin_%s' % basin] = ens_mean[:, basin - 1]

        # kk['original']['basin_0'] = np.mean(original, axis=1)
        # kk['ens_mean']['basin_0'] = np.mean(ens_mean, axis=1)

        '''Area-weighted'''
        sub_basin_area = gr['sub_basin_area'][:]
        kk['original']['basin_0'] = np.sum(original * sub_basin_area[None, :], axis=1) / np.sum(sub_basin_area)
        kk['ens_mean']['basin_0'] = np.sum(ens_mean * sub_basin_area[None, :], axis=1) / np.sum(sub_basin_area)

        self.__GRACE = kk

        return kk

    def save_states(self, save_dir='../temp', prefix='1'):
        ff = self.__states

        hf = h5py.File(Path(save_dir) / ('Res_%s.h5df' % prefix), 'w')

        for ii, jj in ff.items():
            if ii == 'time':
                hf.create_dataset(name=ii, data=jj)
                continue
            dict_group1 = hf.create_group(ii)
            for kk, ll in jj.items():
                dict_group2 = dict_group1.create_group(kk)
                for mm, nn in ll.items():
                    dict_group2.create_dataset(name=str(mm), data=nn)
                    pass

        hf.close()

        pass

    def save_GRACE(self, save_dir='../temp', prefix='1'):

        ff = self.__GRACE

        hf = h5py.File(Path(save_dir) / ('GRACE_%s.h5df' % prefix), 'w')

        for ii, jj in ff.items():
            if ii == 'time':
                hf.create_dataset(name=ii, data=jj)
                continue
            dict_group1 = hf.create_group(ii)
            for kk, ll in jj.items():
                dict_group1.create_dataset(name=kk, data=ll)
                pass

        hf.close()
        pass

    def load_states(self, load_dir='../temp', prefix='1'):
        """
        This is simply for loading states from pre-saved results. No calculation has been done.
        It is mostly used for plotting a figure
        """
        ff = {}

        hf = h5py.File(Path(load_dir) / ('Res_%s.h5df' % prefix), 'r')

        for ii, jj in hf.items():
            if ii == 'time':
                ff[ii] = jj[:]
                continue
            dict_group1 = {}
            for kk, ll in jj.items():
                dict_group2 = {}
                for mm, nn in ll.items():
                    dict_group2[int(mm)] = nn[:]
                    pass
                dict_group1[kk] = dict_group2
            ff[ii] = dict_group1
        return ff

    def load_GRACE(self, load_dir='../temp', prefix='1'):
        """
        This is simply for loading GRACE from pre-saved results. No calculation has been done.
        It is mostly used for plotting a figure
        """
        ff = {}

        hf = h5py.File(Path(load_dir) / ('GRACE_%s.h5df' % prefix), 'r')

        for ii, jj in hf.items():
            if ii == 'time':
                ff[ii] = jj[:]
                continue
            dict_group1 = {}
            for kk, ll in jj.items():
                dict_group1[kk] = ll[:]

            ff[ii] = dict_group1

        return ff


class HarmonicMapAnalysis:
    """
    Per-grid-cell harmonic decomposition of the yearly basin files written by the collect step
    (Res/<case>/<stage>/Ens_k/daily_output_YYYY.nc: daily fields on the basin bounding box, NaN
    outside the basin). For every ensemble member and every requested variable the daily series
    of each cell is fitted with  bias + trend*t + annual + semi-annual  (src_auxiliary.ts), and
    the results are stored as 2-D maps.

    Output: one netCDF per stage, Res/<case>/Harmonic_<stage>.nc, with
        (the GRACE counterpart, Harmonic_GRACE.nc, is produced by run_GRACE with maps on (lat, lon) only)
        <var>_<quantity>            dims (ens, lat, lon)   one map per ensemble member (ens 0..N;
                                                           ens 0 is the unperturbed member)
        <var>_<quantity>_ensmean    dims (lat, lon)        fit of the ENSEMBLE-MEAN series of
                                                           members 1..N (the proper way to
                                                           average an amplitude/phase)
        <var>_<quantity>_ensspread  dims (lat, lon)        std across members 1..N (circular
                                                           std for phases)
    quantities: bias, trend, trend_std, annual_amp, annual_amp_std, annual_phase (calendar,
                deg), annual_phase_std, annual_peak_doy, semi_annual_amp, semi_annual_phase,
                semi_annual_peak_doy, residual_std, n_valid
    """

    def __init__(self, res_dir, case: str, ens: int, date_begin: str, date_end: str,
                 variables=(WaterGap_storage_variables.tws.name,), components=None):
        from src_auxiliary.ts import decomposition
        self.res_dir = Path(res_dir)
        self.case = case
        self.ens = ens
        self.date_begin, self.date_end = date_begin, date_end
        self.variables = list(variables)
        self.components = components or [decomposition.trend, decomposition.annual, decomposition.semi_annual]

    # ------------------------------------------------------------------ loading
    def _member_dir(self, stage: Stage, ens_id: int) -> Path:
        return self.res_dir / self.case / stage.name / ('Ens_%s' % ens_id)

    def _load_member(self, stage: Stage, ens_id: int) -> xr.Dataset:
        """daily (time, lat, lon) cube of the requested variables over [date_begin, date_end],
        read one yearly file at a time (small files, but keep the memory pattern uniform)."""
        import pandas as pd
        years = pd.date_range(self.date_begin, self.date_end).year.unique()
        frames = []
        for year in years:
            fn = self._member_dir(stage, ens_id) / f"daily_output_{year}.nc"
            if not fn.exists():
                print(f"  [warn] missing {fn.name} for Ens_{ens_id} ({stage.name}), skipped")
                continue
            with xr.open_dataset(fn, cache=False) as ds:
                frames.append(ds[self.variables].load())
        if not frames:
            raise FileNotFoundError(f"no yearly files for {stage.name} Ens_{ens_id} under {self._member_dir(stage, ens_id)}")
        ds = xr.concat(frames, dim='time').sortby('time').sel(time=slice(self.date_begin, self.date_end))
        return ds

    @staticmethod
    def _year_fraction(time_index) -> np.ndarray:
        """datetime64 axis -> year fraction with mid-day epochs (year + (doy - 0.5) / days_in_year)"""
        import pandas as pd
        t = pd.DatetimeIndex(time_index)
        ndays = np.where(t.is_leap_year, 366.0, 365.0)
        return t.year.values + (t.dayofyear.values - 0.5) / ndays

    # ------------------------------------------------------------------ fitting
    def _fit_cube(self, cube: xr.DataArray, tfrac: np.ndarray) -> dict:
        """
        cube : (time, lat, lon) of one variable. Every grid cell is one series; all cells are
        fitted in one least-squares call. Returns a dict quantity -> 2-D array (lat, lon).
        """
        from src_auxiliary.ts import ts, decomposition
        n_t, n_lat, n_lon = cube.shape
        obs = cube.values.reshape(n_t, n_lat * n_lon)           # columns = cells
        tsd = ts().set_period(tfrac).setDecomposition(self.components)
        sig = tsd.getSignal(obs)
        err = tsd.getError()

        def m(a):                                               # back to (lat, lon)
            return np.asarray(a, dtype=float).reshape(n_lat, n_lon)

        out = {'bias': m(sig['bias']), 'bias_std': m(err['bias_std']),
               'residual_std': m(err['residual_std']), 'n_valid': m(err['n_valid'])}
        if decomposition.trend in self.components:
            out['trend'] = m(sig['trend']); out['trend_std'] = m(err['trend_std'])
        for ss in (decomposition.annual, decomposition.semi_annual, decomposition.quarter_annual):
            if ss not in self.components:
                continue
            nm = ss.name
            out[f'{nm}_amp'] = m(sig[nm][0]);                   out[f'{nm}_amp_std'] = m(err[nm + '_std'][0])
            out[f'{nm}_phase'] = m(sig[nm + '_calendar'][1]);  out[f'{nm}_phase_std'] = m(err[nm + '_std'][1])
            out[f'{nm}_peak_doy'] = m(sig[nm + '_peak_doy'])
        return out

    @staticmethod
    def _units(q: str) -> str:
        if q.endswith('phase') or q.endswith('phase_std'):
            return 'degree'
        if q.endswith('peak_doy'):
            return 'day of year (1 Jan = 1)'
        if q == 'n_valid':
            return 'count'
        if q.startswith('trend'):
            return 'mm/yr'
        return 'mm'

    @staticmethod
    def _circular_std_deg(deg, axis=0):
        """std of angles in degrees (circular), NaN-aware"""
        r = np.deg2rad(deg)
        c, s = np.nanmean(np.cos(r), axis=axis), np.nanmean(np.sin(r), axis=axis)
        R = np.clip(np.hypot(c, s), 1e-12, 1.0)
        return np.rad2deg(np.sqrt(-2.0 * np.log(R)))

    # ------------------------------------------------------------------ driver
    def run(self, stage: Stage, save=True) -> xr.Dataset:
        print(f"Harmonic map analysis | {stage.name} | Ens_0..{self.ens} | {self.date_begin} to {self.date_end} | "
              f"{', '.join(self.variables)} | {', '.join(c.name for c in self.components)}")
        member_maps = {v: [] for v in self.variables}           # var -> list over ens of dict(q -> 2D)
        mean_series = {v: None for v in self.variables}         # var -> running sum of members 1..N
        lat = lon = tfrac = None
        for ens_id in range(self.ens + 1):
            ds = self._load_member(stage, ens_id)
            if lat is None:
                lat, lon = ds['lat'].values, ds['lon'].values
                tfrac = self._year_fraction(ds['time'].values)
            for v in self.variables:
                cube = ds[v].transpose('time', 'lat', 'lon')
                member_maps[v].append(self._fit_cube(cube, tfrac))
                if ens_id >= 1:                                 # ensemble mean excludes the unperturbed member 0
                    mean_series[v] = cube.values.copy() if mean_series[v] is None else mean_series[v] + cube.values
            print(f"  Ens_{ens_id}: fitted {len(tfrac)} days x {lat.size * lon.size} cells")

        data_vars = {}
        for v in self.variables:
            quantities = member_maps[v][0].keys()
            for q in quantities:
                stack = np.stack([mm[q] for mm in member_maps[v]], axis=0)          # (ens, lat, lon)
                data_vars[f'{v}_{q}'] = (('ens', 'lat', 'lon'), stack, {'units': self._units(q)})
                members = stack[1:] if stack.shape[0] > 1 else stack
                with warnings.catch_warnings():                                    # all-NaN cells outside the basin
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    if q.endswith('phase'):
                        spread = self._circular_std_deg(members, axis=0)
                    else:
                        spread = np.nanstd(members, axis=0, ddof=1) if members.shape[0] > 1 else np.full_like(members[0], np.nan)
                data_vars[f'{v}_{q}_ensspread'] = (('lat', 'lon'), spread, {'units': self._units(q),
                                                   'description': 'std across ensemble members 1..N'})
            if mean_series[v] is not None:
                n_members = max(self.ens, 1)
                cube_mean = xr.DataArray(mean_series[v] / n_members, dims=('time', 'lat', 'lon'))
                fit_mean = self._fit_cube(cube_mean, tfrac)
                for q, arr in fit_mean.items():
                    data_vars[f'{v}_{q}_ensmean'] = (('lat', 'lon'), arr, {'units': self._units(q),
                                                     'description': 'fit of the ensemble-mean series (members 1..N)'})

        out = xr.Dataset(data_vars, coords={'ens': np.arange(self.ens + 1), 'lat': lat, 'lon': lon},
                         attrs={'case': self.case, 'stage': stage.name, 'period': f'{self.date_begin} to {self.date_end}',
                                'components': ', '.join(c.name for c in self.components),
                                'phase_convention': 'calendar phase phi_cal [deg]: y = A cos(2 pi k f + phi_cal), '
                                                    'f = fraction of the year from 1 January; peak_doy = 1 + 365.25 * ((-phi_cal/360) mod 1) / k',
                                'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        if save:
            fn = self.output_path(self.res_dir, self.case, stage)
            out.to_netcdf(fn, encoding={k: {'zlib': True, 'complevel': 4} for k in out.data_vars})
            print(f"  -> {fn}")
        return out

    @staticmethod
    def output_path(res_dir, case, stage) -> Path:
        """stage: a Stage member (OL, DA) or a plain name such as 'GRACE'"""
        return Path(res_dir) / case / f'Harmonic_{getattr(stage, "name", stage)}.nc'

    @staticmethod
    def load(res_dir, case, stage) -> xr.Dataset:
        """load a previously saved Harmonic_<stage>.nc (for plotting); stage as in output_path"""
        with xr.open_dataset(HarmonicMapAnalysis.output_path(res_dir, case, stage)) as ds:
            return ds.load()

    # ------------------------------------------------------------------ GRACE
    def run_GRACE(self, grace_dir, basin: str, basin_mask_path, land_mask_path=None, save=True) -> xr.Dataset:
        """
        Same harmonic analysis for the gridded GRACE TWS prepared by
        prepare_GRACE_mascon.grid_TWS: <grace_dir>/<basin>_gridded_signal.hdf5 holds 'tws'
        (n_months, n_cells) in mm on the 0.5-degree model grid, the cells being the basin mask
        intersected with GRACE's 0.5-degree land mask (row-major order), and 'time_epoch'
        ('YYYY-MM-15'). The cells are put back on the grid, cropped to the basin bounding box
        (same box as the OL/DA yearly files) and every cell's monthly series is fitted.

        Output: Res/<case>/Harmonic_GRACE.nc with tws_<quantity> on (lat, lon). GRACE values are
        anomalies, so 'bias' is not comparable with the model; trend, amplitudes, phases and
        peak days are.

        land_mask_path : GlobalLandMaskForGRACE.hdf5 (group 'resolution_05'). Needed only when
                         the land mask removed cells from the basin; if the cell count already
                         matches the bare basin mask it is not used.
        """
        import pandas as pd
        fn = Path(grace_dir) / f'{basin}_gridded_signal.hdf5'
        with h5py.File(fn, 'r') as f:
            tws = f['tws'][:].astype(float)                                  # (n_months, n_cells)
            epochs = [t if len(t.split('-')) == 3 else t + '-15' for t in f['time_epoch'][:].astype(str)]
        n_months, n_cells = tws.shape

        with h5py.File(basin_mask_path, 'r') as f:
            basin_2d = f['basin'][:].astype(bool)                            # (360, 720), lat 90 -> -90
        mask = basin_2d
        if mask.sum() != n_cells:
            if land_mask_path is None or not Path(land_mask_path).exists():
                raise ValueError(f"{fn.name} has {n_cells} cells but the basin mask has {basin_2d.sum()}; "
                                 f"the GRACE land mask is needed to reproduce the cell selection (land_mask_path)")
            with h5py.File(land_mask_path, 'r') as f:
                land_05 = np.flipud(f['resolution_05']['mask'][:]).astype(bool)
            mask = basin_2d & land_05
            assert mask.sum() == n_cells, f"cell count mismatch: file {n_cells}, basin&land mask {mask.sum()}"

        # cells back onto the global grid, then crop to the basin bounding box
        res = 0.5; err = res / 10
        lat_g = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
        lon_g = np.arange(-180 + res / 2, 180 - res / 2 + err, res)
        full = np.full((n_months, lat_g.size, lon_g.size), np.nan)
        full[:, mask] = tws
        rows, cols = np.where(basin_2d)
        r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
        cube = xr.DataArray(full[:, r0:r1, c0:c1], dims=('time', 'lat', 'lon'),
                            coords={'time': pd.to_datetime(epochs), 'lat': lat_g[r0:r1], 'lon': lon_g[c0:c1]})
        cube = cube.sel(time=slice(self.date_begin, self.date_end))
        tfrac = self._year_fraction(cube['time'].values)
        print(f"Harmonic map analysis | GRACE | {basin} | {len(tfrac)} months in {self.date_begin} to {self.date_end} | "
              f"{int(mask.sum())} cells | {', '.join(c.name for c in self.components)}")

        maps = self._fit_cube(cube, tfrac)
        var = WaterGap_storage_variables.tws.name
        data_vars = {f'{var}_{q}': (('lat', 'lon'), arr, {'units': self._units(q)}) for q, arr in maps.items()}
        out = xr.Dataset(data_vars, coords={'lat': cube['lat'].values, 'lon': cube['lon'].values},
                         attrs={'case': self.case, 'stage': 'GRACE', 'source': str(fn),
                                'period': f'{self.date_begin} to {self.date_end}', 'n_months': int(len(tfrac)),
                                'components': ', '.join(c.name for c in self.components),
                                'note': 'GRACE TWS anomalies: bias not comparable with the model; trend/amplitude/phase are',
                                'phase_convention': 'calendar phase phi_cal [deg]: y = A cos(2 pi k f + phi_cal), '
                                                    'f = fraction of the year from 1 January; peak_doy = 1 + 365.25 * ((-phi_cal/360) mod 1) / k',
                                'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        if save:
            out_fn = self.output_path(self.res_dir, self.case, 'GRACE')
            out.to_netcdf(out_fn, encoding={k: {'zlib': True, 'complevel': 4} for k in out.data_vars})
            print(f"  -> {out_fn}")
        return out



def demo1():
    # basin_shp = basin_shp_process(save_dir='/media/user/My Book/Fan/WaterGap/Basin/mask',
    #                               basin_name='Brahmaputra', res=0.5)
    # basin_shp.shp_to_mask(shp_path='/media/user/My Book/Fan/WaterGap/Basin/shp/Brahmaputra/Brahmaputra.shp', issave=True)

    # analysis = BasinAverageAnalysis_realtime(basin_mask_path='/media/user/My Book/Fan/WaterGap/Basin/mask/Brahmaputra_res_0.5.h5',
    #                                          state_dir='/media/user/My Book/Fan/WaterGap/Res/test', stage=Stage.OL, ens_id=0)
    #
    # analysis.load_nc_files(start_date="2002-01-01", end_date="2005-04-30").select_variables(variable_list=['tws']).get_basin_average()
    pass


def demo2():
    from mpi4py import MPI
    """Parallel execution using MPI. Each rank will have its own log file."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    analysis = BasinAverageAnalysis(
        basin_mask_path='/media/user/My Book/Fan/WaterGap/Basin/mask/Brahmaputra_res_0.5.h5',
        state_dir='/media/user/My Book/Fan/WaterGap/Res/test', stage=Stage.OL, ens_id=rank)

    analysis.load_nc_files(start_date="2002-01-01", end_date="2005-04-30").select_variables(
        variable_list=['tws']).get_basin_average()


if __name__ == '__main__':
    demo2()
