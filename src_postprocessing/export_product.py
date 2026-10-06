"""
Export a finished PyGLDA v2 case as a public data product. This is not part of the normal workflow: it is called by
hand, only when the results are distributed to others (see demo1 / demo2 below).

Everything is read from what the pipeline has already written (paths from <setting_dir>/DA_setting.json):
    Res/<case>/<OL|DA>/Ens_k/daily_output_<year>.nc   yearly basin-box files of every member (collect step)
    Res/<case>/Res_<OL|DA>.h5                         basin / sub-basin daily series (post-processing)
    <obs.dir>/<basin>_obs_GRACE.hdf5                  the GRACE observations assimilated (get_GRACE_obs)
    <basic.basin_mask>                                basin and sub-basin masks (0.5 deg, global)

Product: <out_dir>/PyGLDA-v2_<case>_<version>/
    gridded/     ..._daily_<year>.nc : for every variable <v> and stage OL / DA: <v>_<stage>_mean, <v>_<stage>_spread,
                 daily, 0.5 deg, NaN outside the basin
    gridded/     ..._GRACE_monthly.nc : the GRACE/GRACE-FO TWS on the same grid (monthly, NaN outside the basin), as
                 prepared for the assimilation (<obs.GRACE.preprocess_res>/<basin>_gridded_signal.hdf5) and as an
                 anomaly on the same baseline as the model tws_anomaly
    timeseries/  ..._basin_timeseries.nc : storages averaged over the basin and every sub-basin (cos-lat weighted),
                 daily mean and spread; the GRACE TWS assimilated and its standard deviation per epoch
    ancillary/   ..._ancillary.nc : cell area, basin mask, sub-basin id, per-cell TWS trend (OL, DA), cell_flag
    README.md    variables, units, method, known limitations, provenance

Mean and spread = mean and standard deviation (ddof=1) over the perturbed members 1..N; member 0 (unperturbed run)
is not part of the ensemble. Storages are in mm water equivalent per unit land + surface-water area of the cell, as
WaterGAP writes them. Discharge, recharge and runoff are experimental and only exported if the run wrote them
(WaterGAP outputs dis, qr = total_groundwater_recharge, qtot = total_runoff in Config_ReWaterGAP.json).
"""
import sys
import json
import warnings
import datetime as dt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # repo root, for src_DA
import numpy as np
import pandas as pd
import xarray as xr
import h5py
from src_DA.EnumDA import Stage


class product_export:
    """
    Collect the final results of a case (OL and DA) into a distributable data product.
    """

    PRODUCT = 'PyGLDA-v2'

    '''product variable: (WaterGAP output variables summed, unit factor, units, long name, kind)'''
    VARIABLES = {
        'tws':           (('tws',), 1.0, 'mm', 'terrestrial water storage', 'storage'),
        'groundwater':   (('groundwstor',), 1.0, 'mm', 'groundwater storage', 'storage'),
        'soil_moisture': (('soilmoist',), 1.0, 'mm', 'soil moisture', 'storage'),
        'swe':           (('swe',), 1.0, 'mm', 'snow water equivalent', 'storage'),
        'surface_water': (('riverstor', 'locallakestor', 'localwetlandstor', 'globallakestor', 'globalwetlandstor',
                           'reservoirstor'), 1.0, 'mm', 'surface water storage (rivers, lakes, wetlands, reservoirs)',
                          'storage'),
        'canopy':        (('canopystor',), 1.0, 'mm', 'canopy storage', 'storage'),
        'discharge':     (('dis',), 1.0, 'm3 s-1', 'river discharge (experimental)', 'flux'),
        'recharge':      (('qr',), 86400.0, 'mm day-1', 'total groundwater recharge (experimental)', 'flux'),
        'runoff':        (('qtot',), 86400.0, 'mm day-1', 'total runoff (experimental)', 'flux'),
    }

    TREND_FLAG = 20.0        # cell_flag: |TWS trend| above this [mm/yr] in OL or DA (same rule as da_evaluation)
    COMPLEVEL = 4

    def __init__(self, setting_dir: str, version='v1.0'):
        from src_DA.configure_DA import config_DA

        self.__setting_dir = Path(setting_dir)
        self.__version = version
        self.__raw = json.load(open(self.__setting_dir / 'DA_setting.json', 'r'))

        configDA = config_DA.loadjson(self.__setting_dir / 'DA_setting.json').process()
        self.__res = Path(configDA.basic.res_permanent)
        self.__case = configDA.basic.case
        self.__basin = configDA.basic.basin
        self.__ens = int(configDA.basic.ensemble)
        self.__begin, self.__end = configDA.basic.fromdate, configDA.basic.todate
        self.__mask_file = Path(configDA.basic.basin_mask)
        self.__obs_dir = Path(configDA.obs.dir)
        self.__grace_dir = Path(configDA.obs.GRACE['preprocess_res'])
        self.__grace_kind = configDA.obs.GRACE.get('kind', 'GRACE')

        self.__days = pd.date_range(self.__begin, self.__end, freq='D')
        self.__years = sorted(self.__days.year.unique())
        self.__tag = '%s_%s_%s' % (self.PRODUCT, self.__case, self.__version)

        '''defaults, changed with the configure_* methods'''
        self.__out_root = self.__res / self.__case / 'product'
        self.__baseline = ('2004-01-01', '2009-12-31')
        pass

    def configure_output(self, out_dir=None):
        """out_dir: folder in which the product folder is created (default Res/<case>/product)"""
        if out_dir is not None:
            self.__out_root = Path(out_dir)
        return self

    def configure_baseline(self, begin='2004-01-01', end='2009-12-31'):
        """period whose mean is removed for tws_anomaly (GRACE convention 2004.0-2010.0)"""
        self.__baseline = (begin, end)
        return self

    def run(self):
        self.__root = self.__out_root / self.__tag
        for d in ['gridded', 'timeseries', 'ancillary']:
            (self.__root / d).mkdir(parents=True, exist_ok=True)
        print('Exporting %s | %s to %s | members 1..%d | -> %s' % (self.__tag, self.__begin, self.__end, self.__ens,
                                                                  self.__root))

        '''which product variables the run has written (both stages)'''
        self.__find_variables()

        '''GRACE: epochs, values, sigma (for the time series)'''
        self.__load_GRACE()

        '''pass 1: TWS baseline of each stage'''
        self.__tws_baseline()

        '''pass 2: gridded daily mean and spread, one file per year'''
        self.__gridded()

        '''ancillary fields, gridded GRACE, basin time series, README'''
        self.__ancillary()
        self.__grace_gridded()
        self.__timeseries()
        self.__readme()

        print('Done: %s' % self.__root)
        return self

    # ------------------------------------------------------------------------------------------------ reading
    def __member_file(self, stage: Stage, ens_id, year):
        return self.__res / self.__case / stage.name / ('Ens_%s' % ens_id) / ('daily_output_%s.nc' % year)

    def __find_variables(self):
        names = None
        for stage in [Stage.OL, Stage.DA]:
            with xr.open_dataset(self.__member_file(stage, 1, self.__years[0])) as ds:
                have = set(ds.data_vars)
            names = have if names is None else names & have

        self.__avail, self.__missing = {}, {}
        for v, (src, *_rest) in self.VARIABLES.items():
            present = tuple(x for x in src if x in names)
            if v == 'surface_water' and present:
                self.__avail[v] = present
            elif len(present) == len(src):
                self.__avail[v] = src
            else:
                self.__missing[v] = [x for x in src if x not in names]
        print('  variables: %s' % ', '.join(self.__avail))
        if self.__missing:
            print('  not written by the run (left out): %s' %
                  ', '.join('%s (%s)' % (v, '+'.join(m)) for v, m in self.__missing.items()))
        pass

    def __read_year(self, stage: Stage, ens_id, year, variables, days):
        """one member, one year: {product variable: (time, lat, lon)}"""
        fn = self.__member_file(stage, ens_id, year)
        if not fn.exists():
            raise FileNotFoundError('missing %s - run collect_and_statistics(Stage.%s) first' % (fn, stage.name))
        need = sorted({x for v in variables for x in self.__avail[v]})
        with xr.open_dataset(fn) as ds:
            ds = ds[need].sel(time=slice(days[0], days[-1])).load()
        out = {}
        for v in variables:
            out[v] = sum(ds[x].values.astype(np.float64) for x in self.__avail[v]) * self.VARIABLES[v][1]
        self.__lat, self.__lon = ds['lat'].values, ds['lon'].values
        return out, ds['time'].values

    def __load_GRACE(self):
        self.__obs_file = self.__obs_dir / ('%s_obs_GRACE.hdf5' % self.__basin)
        with h5py.File(self.__obs_file, 'r') as f:
            self.__epochs = [str(t)[:7] for t in f['time_epoch'][:].astype(str)]
            val = np.asarray(f['ens_0'][:], float)
            cov = np.asarray(f['cov'][:], float) if 'cov' in f else None
            area = np.asarray(f['sub_basin_area'][:], float) if 'sub_basin_area' in f else None
        if val.ndim == 1:
            val = val[:, None]
        if cov is None:
            cov = np.full((len(self.__epochs), val.shape[1], val.shape[1]), np.nan)
        elif cov.ndim == 1:
            cov = cov[:, None, None]
        if area is None:
            area = np.ones(val.shape[1])
        self.__gr_val, self.__gr_cov, self.__gr_area = val, cov, area
        self.__gr_sig = np.sqrt(np.clip(np.einsum('kii->ki', cov), 0, None))

        print('  GRACE: %d epochs' % len(self.__epochs))
        pass

    # ------------------------------------------------------------------------------------------------ writing
    def __tws_baseline(self):
        b0, b1 = pd.Timestamp(self.__baseline[0]), pd.Timestamp(self.__baseline[1])
        if b0 < self.__days[0] or b1 > self.__days[-1]:
            b0, b1 = self.__days[0], self.__days[-1]
            print('  baseline outside the run period -> the whole period is used')
        self.__baseline_used = (b0, b1)

        self.__base = {}
        for stage in [Stage.OL, Stage.DA]:
            acc, n = None, 0
            for year in [y for y in self.__years if b0.year <= y <= b1.year]:
                days = self.__days[(self.__days.year == year) & (self.__days >= b0) & (self.__days <= b1)]
                for ens_id in range(1, self.__ens + 1):
                    d, _ = self.__read_year(stage, ens_id, year, ['tws'], days)
                    s = d['tws'].sum(0)
                    acc = s if acc is None else acc + s
                    n += d['tws'].shape[0]
            self.__base[stage] = acc / n
        print('  TWS anomaly baseline: %s to %s' % (b0.date(), b1.date()))
        pass

    def __gridded(self):
        N = self.__ens
        self.__tws_mean = {Stage.OL: [], Stage.DA: []}
        for year in self.__years:
            days = self.__days[self.__days.year == year]
            data_vars = {}
            for stage in [Stage.OL, Stage.DA]:
                '''sum and sum of squares over the members'''
                S1, S2 = {}, {}
                for ens_id in range(1, N + 1):
                    d, time = self.__read_year(stage, ens_id, year, list(self.__avail), days)
                    for v, a in d.items():
                        S1[v] = a if v not in S1 else S1[v] + a
                        S2[v] = a * a if v not in S2 else S2[v] + a * a

                for v in self.__avail:
                    mean = S1[v] / N
                    spread = np.sqrt(np.clip((S2[v] - N * mean ** 2) / (N - 1), 0, None)) if N > 1 else \
                        np.full_like(mean, np.nan)
                    units, long = self.VARIABLES[v][2], self.VARIABLES[v][3]
                    self.__add(data_vars, v, stage, mean, spread, units, long)
                    if v == 'tws':
                        b0, b1 = self.__baseline_used
                        self.__add(data_vars, 'tws_anomaly', stage, mean - self.__base[stage][None], spread, 'mm',
                                   'TWS anomaly (minus the %s mean of %s to %s)' % (stage.name, b0.date(), b1.date()))
                        self.__tws_mean[stage].append(mean)

            ds = xr.Dataset(data_vars, coords=self.__coords(time),
                            attrs=self.__attrs('daily gridded ensemble mean and spread, %s' % year))
            encoding = {v: {'zlib': True, 'complevel': self.COMPLEVEL, '_FillValue': np.float32(np.nan)}
                        for v in ds.data_vars if ds[v].dtype == np.float32}
            fn = self.__root / 'gridded' / ('%s_daily_%s.nc' % (self.__tag, year))
            ds.to_netcdf(fn, encoding=encoding)
            print('  %s -> %s (%.1f MB)' % (year, fn.name, fn.stat().st_size / 1e6))
        pass

    def __add(self, data_vars, v, stage: Stage, mean, spread, units, long):
        name = 'open loop' if stage == Stage.OL else 'data assimilation'
        data_vars['%s_%s_mean' % (v, stage.name)] = (('time', 'lat', 'lon'), mean.astype(np.float32),
                                                    dict(units=units, long_name='%s, %s, ensemble mean' % (long, name)))
        data_vars['%s_%s_spread' % (v, stage.name)] = (('time', 'lat', 'lon'), spread.astype(np.float32),
                                                      dict(units=units, long_name='%s, %s, ensemble spread (standard '
                                                                                  'deviation)' % (long, name)))
        pass

    def __coords(self, time=None):
        c = dict(lat=('lat', self.__lat, dict(units='degrees_north')),
                 lon=('lon', self.__lon, dict(units='degrees_east')))
        if time is not None:
            c['time'] = pd.DatetimeIndex(time)
        return c

    def __ancillary(self):
        '''masks on the product grid (global 0.5-degree masks cropped to the box)'''
        res = 0.5
        glat = np.arange(90 - res / 2, -90, -res)
        glon = np.arange(-180 + res / 2, 180, res)
        iy = np.array([int(np.argmin(np.abs(glat - y))) for y in self.__lat])
        ix = np.array([int(np.argmin(np.abs(glon - x))) for x in self.__lon])
        with h5py.File(self.__mask_file, 'r') as f:
            basin = f['basin'][:].astype(bool)[np.ix_(iy, ix)]
            sub = np.zeros(basin.shape, dtype=np.int16)
            subs = sorted([k for k in f.keys() if k.startswith('sub_basin_')], key=lambda s: int(s.split('_')[-1]))
            for k in subs:
                sub[f[k][:].astype(bool)[np.ix_(iy, ix)]] = int(k.split('_')[-1])

        '''cell area [km2]'''
        band = np.abs(np.sin(np.deg2rad(self.__lat + res / 2)) - np.sin(np.deg2rad(self.__lat - res / 2)))
        area = (6371.0 ** 2 * np.deg2rad(res) * band)[:, None] * np.ones((1, len(self.__lon)))

        '''linear TWS trend of the ensemble mean per cell, and the flag'''
        t = self.__days
        tdec = np.asarray(t.year + (t.dayofyear - 0.5) / np.where(t.is_leap_year, 366, 365), dtype=float)
        A = np.vstack([tdec - tdec.mean(), np.ones(len(tdec))]).T
        trend = {}
        for stage in [Stage.OL, Stage.DA]:
            Y = np.concatenate(self.__tws_mean[stage], 0).reshape(len(t), -1)
            ok = np.isfinite(Y).all(0)
            tr = np.full(Y.shape[1], np.nan)
            if ok.any():
                tr[ok] = np.linalg.lstsq(A, Y[:, ok], rcond=None)[0][0]
            trend[stage] = tr.reshape(len(self.__lat), len(self.__lon))
        with np.errstate(invalid='ignore'):
            flag = ((np.abs(trend[Stage.OL]) > self.TREND_FLAG) | (np.abs(trend[Stage.DA]) > self.TREND_FLAG))
        self.__n_flag = int(flag.sum())

        ds = xr.Dataset(dict(
            cell_area=(('lat', 'lon'), area.astype(np.float32), dict(units='km2', long_name='cell area')),
            basin_mask=(('lat', 'lon'), basin.astype(np.int8), dict(long_name='1 inside the basin')),
            sub_basin_id=(('lat', 'lon'), sub, dict(long_name='sub-basin number of the GRACE observations '
                                                              '(0 outside)')),
            tws_trend_OL=(('lat', 'lon'), trend[Stage.OL].astype(np.float32),
                          dict(units='mm yr-1', long_name='linear trend of the open-loop ensemble-mean TWS')),
            tws_trend_DA=(('lat', 'lon'), trend[Stage.DA].astype(np.float32),
                          dict(units='mm yr-1', long_name='linear trend of the data-assimilation ensemble-mean TWS')),
            cell_flag=(('lat', 'lon'), flag.astype(np.int8),
                       dict(long_name='1: |TWS trend| > %g mm/yr in OL or DA - a storage drift in the model (e.g. a '
                                      'lake or reservoir) that the assimilation cannot correct' % self.TREND_FLAG))),
            coords=self.__coords(), attrs=self.__attrs('ancillary fields'))
        ds.to_netcdf(self.__root / 'ancillary' / ('%s_ancillary.nc' % self.__tag))
        print('  ancillary: %d sub-basins, %d flagged cells' % (len(subs), self.__n_flag))
        pass

    def __grace_gridded(self):
        """
        GRACE/GRACE-FO TWS on the product grid: <grace_dir>/<basin>_gridded_signal.hdf5 holds 'tws' (n_months,
        n_cells) in mm, the cells being the basin mask intersected with GRACE's 0.5-degree land mask (row-major), and
        'time_epoch' (prepare_GRACE_mascon.grid_TWS; the same file as HarmonicMapAnalysis.run_GRACE)
        """
        fn = self.__grace_dir / ('%s_gridded_signal.hdf5' % self.__basin)
        if not fn.exists():
            print('  [warn] %s missing - gridded GRACE skipped (run get_GRACE_obs first)' % fn)
            return
        with h5py.File(fn, 'r') as f:
            tws = f['tws'][:].astype(float)
            epochs = [t if len(t.split('-')) == 3 else t + '-15' for t in f['time_epoch'][:].astype(str)]
        n_months, n_cells = tws.shape

        '''the cell selection: basin mask, and if needed GRACE's land mask'''
        with h5py.File(self.__mask_file, 'r') as f:
            basin_2d = f['basin'][:].astype(bool)
        mask = basin_2d
        if mask.sum() != n_cells:
            land_file = self.__grace_dir.parent / 'global_mask' / 'GlobalLandMaskForGRACE.hdf5'
            if not land_file.exists():
                print('  [warn] %s has %d cells, the basin mask %d, and %s is missing - gridded GRACE skipped'
                      % (fn.name, n_cells, int(basin_2d.sum()), land_file))
                return
            with h5py.File(land_file, 'r') as f:
                mask = basin_2d & np.flipud(f['resolution_05']['mask'][:]).astype(bool)
            if mask.sum() != n_cells:
                print('  [warn] %s: cell count %d does not match the basin & land mask (%d) - gridded GRACE skipped'
                      % (fn.name, n_cells, int(mask.sum())))
                return

        '''cells back onto the global grid, then the product box'''
        res = 0.5
        glat = np.arange(90 - res / 2, -90, -res)
        glon = np.arange(-180 + res / 2, 180, res)
        full = np.full((n_months, glat.size, glon.size), np.nan)
        full[:, mask] = tws
        iy = np.array([int(np.argmin(np.abs(glat - y))) for y in self.__lat])
        ix = np.array([int(np.argmin(np.abs(glon - x))) for x in self.__lon])
        cube = full[:, iy][:, :, ix]
        time = pd.to_datetime(epochs)
        sel = (time >= self.__days[0]) & (time <= self.__days[-1])
        cube, time = cube[sel], time[sel]

        '''anomaly on the baseline of the model tws_anomaly'''
        b0, b1 = self.__baseline_used
        inb = (time >= b0) & (time <= b1)
        with warnings.catch_warnings():                          # all-NaN cells outside the basin
            warnings.simplefilter('ignore', category=RuntimeWarning)
            base = np.nanmean(cube[inb], 0) if inb.any() else np.nanmean(cube, 0)

        ds = xr.Dataset(dict(
            tws=(('time', 'lat', 'lon'), cube.astype(np.float32),
                 dict(units='mm', long_name='GRACE/GRACE-FO terrestrial water storage anomaly, as prepared for the '
                                            'assimilation (%s)' % self.__grace_kind)),
            tws_anomaly=(('time', 'lat', 'lon'), (cube - base[None]).astype(np.float32),
                         dict(units='mm', long_name='GRACE/GRACE-FO TWS minus its mean over %s to %s (the baseline of '
                                                    'the model tws_anomaly)' % (b0.date(), b1.date())))),
            coords=self.__coords(time), attrs=self.__attrs('GRACE/GRACE-FO TWS on the model grid, monthly'))
        ds.attrs['source'] = str(fn.name)
        ds.attrs['note'] = ('monthly solutions, time = the epoch of each solution (mid-month); months without a '
                            'solution (e.g. the GRACE/GRACE-FO gap) are absent; GRACE has no information below its '
                            'effective resolution, the 0.5-degree values are the gridded product')
        out = self.__root / 'gridded' / ('%s_GRACE_monthly.nc' % self.__tag)
        ds.to_netcdf(out, encoding={v: {'zlib': True, 'complevel': self.COMPLEVEL, '_FillValue': np.float32(np.nan)}
                                    for v in ds.data_vars})
        print('  GRACE gridded: %d months -> %s' % (len(time), out.name))
        pass

    def __timeseries(self):
        N = self.__ens
        data_vars, units = {}, None
        storages = [v for v in self.__avail if self.VARIABLES[v][4] == 'storage']
        for stage in [Stage.OL, Stage.DA]:
            fn = self.__res / self.__case / ('Res_%s.h5' % stage.name)
            if not fn.exists():
                print('  [warn] %s missing - time series skipped (run post_processing first)' % fn)
                return
            with h5py.File(fn, 'r') as f:
                units = sorted([u for u in f.keys() if u != 'time'],
                               key=lambda s: (s != 'basin', int(s.split('_')[-1]) if s.split('_')[-1].isdigit() else 0))
                if len(f['time'][:]) != len(self.__days):
                    raise ValueError('%s has %d days, the settings period %d' % (fn, len(f['time'][:]), len(self.__days)))
                for v in storages:
                    src = [x for x in self.__avail[v] if all(x in f[u] for u in units)]
                    if not src:
                        continue
                    '''(unit, member, time)'''
                    M = np.stack([np.stack([sum(f[u][x][str(k)][:] for x in src) for k in range(1, N + 1)])
                                  for u in units])
                    name = 'open loop' if stage == Stage.OL else 'data assimilation'
                    data_vars['%s_%s_mean' % (v, stage.name)] = (
                        ('unit', 'time'), M.mean(1).astype(np.float32),
                        dict(units='mm', long_name='%s, %s, ensemble mean' % (self.VARIABLES[v][3], name)))
                    data_vars['%s_%s_spread' % (v, stage.name)] = (
                        ('unit', 'time'), (M.std(1, ddof=1) if N > 1 else np.full(M[:, 0].shape, np.nan)).astype(np.float32),
                        dict(units='mm', long_name='%s, %s, ensemble spread' % (self.VARIABLES[v][3], name)))

        '''GRACE as assimilated: sub-basins, and the basin as the area-weighted mean (with its error)'''
        w = self.__gr_area / self.__gr_area.sum()
        nsub = self.__gr_val.shape[1]
        G = np.full((len(units), len(self.__epochs)), np.nan)
        Gs = np.full_like(G, np.nan)
        for i, u in enumerate(units):
            if u == 'basin':
                G[i] = self.__gr_val @ w
                Gs[i] = np.sqrt(np.clip(np.einsum('i,kij,j->k', w, self.__gr_cov, w), 0, None))
            elif u.startswith('sub_basin_') and int(u.split('_')[-1]) <= nsub:
                G[i] = self.__gr_val[:, int(u.split('_')[-1]) - 1]
                Gs[i] = self.__gr_sig[:, int(u.split('_')[-1]) - 1]
        data_vars['grace_tws'] = (('unit', 'grace_epoch'), G.astype(np.float32),
                                  dict(units='mm', long_name='GRACE/GRACE-FO TWS as assimilated (unperturbed values of '
                                                             'the observation file; basin = area-weighted mean of the '
                                                             'sub-basins)'))
        data_vars['grace_tws_sigma'] = (('unit', 'grace_epoch'), Gs.astype(np.float32),
                                        dict(units='mm', long_name='standard deviation of the assimilated GRACE TWS '
                                                                   '(sub-basin errors are correlated; the basin value '
                                                                   'includes the correlations)'))

        ds = xr.Dataset(data_vars, coords=dict(unit=('unit', np.array(units, dtype=object)), time=self.__days,
                                               grace_epoch=('grace_epoch', np.array(self.__epochs, dtype=object))),
                        attrs=self.__attrs('basin and sub-basin daily time series (cos-latitude weighted means)'))
        fn = self.__root / 'timeseries' / ('%s_basin_timeseries.nc' % self.__tag)
        ds.to_netcdf(fn, encoding={v: {'zlib': True, 'complevel': self.COMPLEVEL} for v in ds.data_vars})
        print('  time series: %d units -> %s' % (len(units), fn.name))
        pass

    def __attrs(self, what):
        m = self.__raw.get('method', {})
        return dict(title='%s %s: %s, %s' % (self.PRODUCT, self.__version, self.__basin, what),
                    product_version=self.__version, case=self.__case, basin=self.__basin,
                    period='%s to %s' % (self.__begin, self.__end), ensemble_size=self.__ens,
                    ensemble_statistics='mean and standard deviation (ddof=1) over the perturbed members 1..N; '
                                        'member 0 (unperturbed run) excluded',
                    model='WaterGAP 2.2e (ReWaterGAP) with the PyGLDA modifications listed in the PyGLDA v2 README',
                    assimilation='PyGLDA v2, %s; GRACE/GRACE-FO TWS, observation errors from DDK3 covariance samples'
                                 % m.get('fusion_method', 'EnKF'),
                    units_note='storages in mm water equivalent per unit land + surface-water area of the cell',
                    institution='Geodesy, Aalborg University (AAU)', creator_name='Fan Yang',
                    creator_email='fany@plan.aau.dk',
                    Conventions='CF-1.8', date_created=dt.datetime.now().strftime('%Y-%m-%d'))

    def __readme(self):
        m = self.__raw.get('method', {})
        b0, b1 = self.__baseline_used
        N = self.__ens
        rows = ['| `%s` | %s | %s | %s |' % (v, self.VARIABLES[v][3], self.VARIABLES[v][2],
                                            ' + '.join('`%s`' % x for x in self.__avail[v])) for v in self.__avail]
        rows.append('| `tws_anomaly` | TWS minus its mean over %s to %s (per stage) | mm | `tws` |' % (b0.date(), b1.date()))
        layer = ', '.join(k for k, v in self.__raw.get('model', {}).get('layer', {}).items() if v)

        L = ['# %s' % self.__tag, '',
             'Daily terrestrial water storage and its components for the %s basin, %s to %s, from the assimilation of '
             'GRACE/GRACE-FO terrestrial water storage into the WaterGAP 2.2e global hydrological model with PyGLDA v2 '
             '(ensemble Kalman filter, %d members).' % (self.__basin, self.__begin, self.__end, N), '',
             '## Files', '',
             '- `gridded/%s_daily_<year>.nc`: daily 0.5° fields, one file per year, NaN outside the basin. For every '
             'variable `<v>`: `<v>_OL_mean`, `<v>_OL_spread` (open loop, without assimilation) and `<v>_DA_mean`, '
             '`<v>_DA_spread` (with assimilation).' % self.__tag,
             '- `gridded/%s_GRACE_monthly.nc`: the GRACE/GRACE-FO TWS on the same grid, monthly (one field per '
             'solution), as prepared for the assimilation (`tws`) and minus its mean over the baseline of the model '
             '`tws_anomaly` (`tws_anomaly`), for direct comparison.' % self.__tag,
             '- `timeseries/%s_basin_timeseries.nc`: the storages averaged over the basin and every sub-basin '
             '(cos-latitude weighted), daily; the GRACE TWS assimilated and its standard deviation per epoch.'
             % self.__tag,
             '- `ancillary/%s_ancillary.nc`: cell area, basin mask, sub-basin id, TWS trend per cell (OL, DA), '
             '`cell_flag`.' % self.__tag,
             '',
             '## Variables', '',
             '| name | description | units | WaterGAP source |', '|---|---|---|---|'] + rows + ['',
             'Ensemble mean and ensemble spread (standard deviation, ddof = 1) over the perturbed members 1–%d; '
             'member 0, the unperturbed run, is not part of the ensemble. Storages are in mm water equivalent per unit '
             'land + surface-water area of the cell, as written by WaterGAP. Discharge, recharge and runoff are '
             'experimental: they respond to the assimilated storages through the model but have not yet been '
             'evaluated against observations.' % N, '']
        if self.__missing:
            L += ['Not in this release (not written by the run): %s.' % ', '.join(self.__missing), '']
        L += ['## Method', '',
              '- Model: WaterGAP 2.2e (ReWaterGAP, Goethe University Frankfurt) with the PyGLDA modifications listed '
              'in the PyGLDA v2 README; forcing GSWP3-W5E5.',
              '- Ensemble: %d members from perturbed forcing and parameters.' % N,
              '- Observations: GRACE/GRACE-FO TWS per sub-basin, errors from DDK3 covariance samples (`%s`).'
              % self.__obs_file.name,
              '- Filter: %s; localization %s; inflation %s; partition %s; storages updated: %s.'
              % (m.get('fusion_method'), (m.get('localization') or {}).get('kind'),
                 (m.get('inflation') or {}).get('kind'), (m.get('increment_partition') or {}).get('kind'), layer), '',
              '## Known limitations', '',
              '- Months without a GRACE observation (e.g. the GRACE/GRACE-FO gap 2017-07 to 2018-05; the epochs are '
              'listed in the time-series file): the DA runs free there and its spread grows.',
              '- `cell_flag = 1` (%d cells): TWS trend above %g mm/yr in a single cell, from a storage drift in the '
              'model (e.g. a lake or reservoir) that the assimilation cannot correct.' % (self.__n_flag, self.TREND_FLAG),
              '- GRACE has no information below the sub-basin scale; patterns inside a sub-basin come from the model.',
              '- The model has a systematic seasonal bias (amplitude too small, peak too early); the assimilation '
              'corrects it every month, so the corrections are largest in spring and autumn.', '',
              '## Contact', '',
              'Fan Yang, Geodesy, Aalborg University, fany@plan.aau.dk.', '',
              'Created %s with `src_postprocessing/export_product.py`.' % dt.datetime.now().strftime('%Y-%m-%d')]
        (self.__root / 'README.md').write_text('\n'.join(L) + '\n')
        pass


def demo1():
    """local workstation: the 4-member Danube case"""
    pe = product_export(setting_dir=str(Path(__file__).resolve().parent.parent / 'settings' / 'demo_2'), version='v0.1')
    pe.run()
    pass


def demo2():
    """UCloud: the 30-member Danube case"""
    pe = product_export(setting_dir=str(Path(__file__).resolve().parent.parent / 'settings' / 'demo_Danube'),
                        version='v1.0')
    pe.configure_output(out_dir=None)
    pe.configure_baseline(begin='2004-01-01', end='2009-12-31')
    pe.run()
    pass


if __name__ == '__main__':
    demo1()
