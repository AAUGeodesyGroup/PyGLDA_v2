"""
Extraction of OS LISFLOOD (GFZ run 'lisfHR3a', L. Jensen, CC BY) simulated TWS for PyGLDA, written in the
same format as the GRACE observation pre-processing (src_OBS.prepare_GRACE*), without uncertainty:

    <out_dir>/<basin>_lisflood_signal.hdf5          sub_basin_k (mm), time_epoch, duration, sub_basin_area
    <out_dir>/<basin>_lisflood_gridded_signal.hdf5  tws (n_epoch, n_cell) mm, time_epoch, duration, sub_basin_area
(the 'suffix' argument, default '_lisflood', is inserted after the basin name; e.g. suffix='_lisflood_daily')

so that LISFLOOD TWS can be used as pseudo-observations (twin experiments) or compared with GRACE by the
same post-processing code. Windows are configurable: calendar-monthly (epoch = 15th, as for the mascons),
regular N-day windows, or the exact windows of an existing observation record (e.g. the TUD 5-daily
'duration' strings) so that model and GRACE are averaged over identical periods.

Input (folder Extra/lisfHR3a/daily_05deg): one netCDF per variable and year, daily, global 0.5 deg,
lat -89.75 -> 89.75 (south up; PyGLDA masks are north up -> flipped here), lon -179.75 -> 179.75,
storages in metres of water (converted to mm), discharge in m3/s. Interception ('cum') is zero in this run.

Developer: Fan Yang (fany@plan.aau.dk), Geodesy Group, Aalborg University. 2026-09.
"""
import numpy as np
import h5py
import netCDF4 as nc
import geopandas as gpd
from pathlib import Path
from datetime import datetime, timedelta
import calendar
from tqdm import tqdm

VARIABLES = {  # name : (file tag, netCDF variable, factor -> mm)
    'tws':    ('tws_total',  'TWS',       1000.0),
    'soil':   ('tws_soil',   'Soilstor',  1000.0),
    'ground': ('tws_ground', 'GWstor',    1000.0),
    'river':  ('tws_river',  'Riverstor', 1000.0),
    'lake':   ('tws_lake',   'Lakestor',  1000.0),
    'snow':   ('tws_snow',   'Snowstor',  1000.0),
    'cum':    ('tws_cum',    'Cumstor',   1000.0),
    'dis':    ('dis',        'dis',       1.0),     # m3/s, kept
}


class LISFLOOD_HR3a:
    """Basin / sub-basin / gridded LISFLOOD TWS in GRACE-observation format."""

    def __init__(self, basin_name='Amazon', shp_path='../data/basin/shp/Amazon/Amazon.shp', mask_path=None,
                 dir_in='/media/user/My Book/Fan/PyGLDA_v2_external_data/Extra/lisfHR3a/daily_05deg',
                 land_mask_path=None, prefix='lisfHR3a_'):
        """
        mask_path: PyGLDA mask '<basin>_res_0.5.h5' (north up); land_mask_path: GlobalLandMaskForGRACE.hdf5
        (optional) - cells outside the GRACE land mask are dropped, as in the GRACE pre-processing, so that
        model and observation means are formed over the same cells.
        """
        self.basin_name = basin_name
        self.dir_in = Path(dir_in)
        self.prefix = prefix
        self._sub_basin_area = gpd.read_file(shp_path).area.values     # same (planar) convention as prepare_GRACE
        mf = h5py.File(mask_path, 'r')
        land = np.ones((360, 720), dtype=bool)
        if land_mask_path is not None:
            land = np.flipud(h5py.File(land_mask_path, 'r')['resolution_05']['mask'][:]) > 0
        self.nsub = len([k for k in mf.keys() if k.startswith('sub_basin')])
        self.sub = {i: mf['sub_basin_%d' % i][:].astype(bool) & land for i in range(1, self.nsub + 1)}
        self.basin = mf['basin'][:].astype(bool) & land
        ii = np.where(self.basin.any(1))[0]; jj = np.where(self.basin.any(0))[0]
        self.r0, self.r1, self.c0, self.c1 = ii.min(), ii.max() + 1, jj.min(), jj.max() + 1
        self.d0, self.d1 = 360 - self.r1, 360 - self.r0            # rows in the south-up data
        lat = np.arange(90 - 0.25, -90, -0.5)[self.r0:self.r1]
        self.w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, self.c1 - self.c0))
        self.bm = self.basin[self.r0:self.r1, self.c0:self.c1]
        self.subm = {i: self.sub[i][self.r0:self.r1, self.c0:self.c1] for i in self.sub}
        self._cache = {}
        pass

    # ------------------------------------------------------------------ windows
    @staticmethod
    def make_windows(day_begin, day_end, windows='monthly'):
        """
        -> (time_epoch [str], duration [str 'first_last'], list of (first, last) datetimes)
        windows: 'monthly' | 'daily' | N (int, regular N-day windows from day_begin) |
                 list of 'YYYY-MM-DD_YYYY-MM-DD' strings (e.g. an observation record's 'duration')
        """
        b = datetime.strptime(day_begin, '%Y-%m-%d'); e = datetime.strptime(day_end, '%Y-%m-%d')
        win = []
        if isinstance(windows, str) and windows == 'monthly':
            d = datetime(b.year, b.month, 1)
            while d <= e:
                last = datetime(d.year, d.month, calendar.monthrange(d.year, d.month)[1])
                if d >= b and last <= e:
                    win.append((d, last, datetime(d.year, d.month, 15)))
                d = last + timedelta(days=1)
        elif isinstance(windows, str) and windows == 'daily':
            d = b
            while d <= e:
                win.append((d, d, d)); d += timedelta(days=1)
        elif isinstance(windows, (int, np.integer)):
            d = b
            while d + timedelta(days=int(windows) - 1) <= e:
                last = d + timedelta(days=int(windows) - 1)
                win.append((d, last, d + timedelta(days=(int(windows) - 1) // 2))); d = last + timedelta(days=1)
        else:
            for s in windows:
                f, l = [datetime.strptime(x, '%Y-%m-%d') for x in str(s).split('_')]
                if f >= b and l <= e:
                    win.append((f, l, f + timedelta(days=(l - f).days // 2)))
        time_epoch = [c.strftime('%Y-%m-%d') for _, _, c in win]
        duration = ['%s_%s' % (f.strftime('%Y-%m-%d'), l.strftime('%Y-%m-%d')) for f, l, _ in win]
        return time_epoch, duration, [(f, l) for f, l, _ in win]

    # ------------------------------------------------------------------ reading
    def _read_year(self, var, year):
        """daily box (n_day, nlat, nlon) in mask orientation, mm; cached per (var, year)"""
        key = (var, year)
        if key in self._cache:
            return self._cache[key]
        tag, ncvar, fac = VARIABLES[var]
        fn = self.dir_in / ('%s%s_05deg_%d.nc' % (self.prefix, tag, year))
        with nc.Dataset(fn) as ds:
            data = np.ma.filled(ds.variables[ncvar][:, self.d0:self.d1, self.c0:self.c1].astype(np.float64), np.nan)[:, ::-1, :]
            t = nc.num2date(ds.variables['time'][:], ds.variables['time'].units, only_use_cftime_datetimes=False)
        data[~np.isfinite(data)] = np.nan
        data *= fac
        days = np.array([datetime(x.year, x.month, x.day) for x in t])
        self._cache = {key: (data, days)}                          # keep only the current year
        return self._cache[key]

    def _window_mean_box(self, var, first, last):
        """mean over the days first..last (inclusive) of the box field"""
        parts = []
        for year in range(first.year, last.year + 1):
            data, days = self._read_year(var, year)
            sel = (days >= first) & (days <= last)
            if sel.any():
                parts.append(data[sel])
        return np.concatenate(parts).mean(0)

    def _mean(self, field, m):
        a = field[m]; b = self.w[m]; ok = np.isfinite(a)
        return np.nansum(a * b) / np.sum(b * ok)

    # ------------------------------------------------------------------ products (GRACE-observation format)
    def basin_TWS(self, day_begin='2007-01-01', day_end='2020-12-31', windows='monthly', var='tws', dir_out='.',
                  suffix='_lisflood'):
        """cosine-weighted sub-basin means of the window-averaged field -> <basin><suffix>_signal.hdf5"""
        print('\nStart to extract LISFLOOD %s to obtain basin-wise TWS over places of interest...' % var)
        time_epoch, duration, win = self.make_windows(day_begin, day_end, windows)
        TWS = {i: np.full(len(win), np.nan) for i in range(1, self.nsub + 1)}
        for t, (f, l) in enumerate(tqdm(win, desc='Processing windows')):
            field = self._window_mean_box(var, f, l)
            for i in range(1, self.nsub + 1):
                TWS[i][t] = self._mean(field, self.subm[i])
        out = Path(dir_out); out.mkdir(parents=True, exist_ok=True)
        with h5py.File(out / ('%s%s_signal.hdf5' % (self.basin_name, suffix)), 'w') as hm:
            for i in range(1, self.nsub + 1):
                hm.create_dataset('sub_basin_%d' % i, data=TWS[i])
            dt = h5py.special_dtype(vlen=str)
            hm.create_dataset('time_epoch', data=time_epoch, dtype=dt)
            hm.create_dataset('duration', data=duration, dtype=dt)
            hm.create_dataset('sub_basin_area', data=self._sub_basin_area)
            hm.attrs['source'] = 'OS LISFLOOD v4.3.1 (GFZ lisfHR3a, L. Jensen), variable %s, mm' % var
        print('Finished: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        pass

    def grid_TWS(self, day_begin='2007-01-01', day_end='2020-12-31', windows='monthly', var='tws', dir_out='.',
                 suffix='_lisflood'):
        """window-averaged field on the basin cells -> <basin><suffix>_gridded_signal.hdf5 (tws: n_epoch x n_cell)"""
        print('\nStart to extract LISFLOOD %s to obtain grid-wise TWS over places of interest...' % var)
        time_epoch, duration, win = self.make_windows(day_begin, day_end, windows)
        TWS = np.full((len(win), int(self.bm.sum())), np.nan)
        for t, (f, l) in enumerate(tqdm(win, desc='Processing windows')):
            TWS[t] = self._window_mean_box(var, f, l)[self.bm]
        out = Path(dir_out); out.mkdir(parents=True, exist_ok=True)
        with h5py.File(out / ('%s%s_gridded_signal.hdf5' % (self.basin_name, suffix)), 'w') as hm:
            hm.create_dataset('tws', data=TWS)
            dt = h5py.special_dtype(vlen=str)
            hm.create_dataset('time_epoch', data=time_epoch, dtype=dt)
            hm.create_dataset('duration', data=duration, dtype=dt)
            hm.create_dataset('sub_basin_area', data=self._sub_basin_area)
            hm.attrs['source'] = 'OS LISFLOOD v4.3.1 (GFZ lisfHR3a, L. Jensen), variable %s, mm' % var
        print('Finished: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        pass

    def components_TWS(self, day_begin='2007-01-01', day_end='2020-12-31', windows='monthly', dir_out='.',
                       variables=('tws', 'soil', 'ground', 'river', 'lake', 'snow', 'dis'), suffix='_lisflood'):
        """all compartments in one file: <basin><suffix>_components.hdf5 with <var>/sub_basin_k and <var>/basin"""
        time_epoch, duration, win = self.make_windows(day_begin, day_end, windows)
        out = Path(dir_out); out.mkdir(parents=True, exist_ok=True)
        with h5py.File(out / ('%s%s_components.hdf5' % (self.basin_name, suffix)), 'w') as hm:
            dt = h5py.special_dtype(vlen=str)
            hm.create_dataset('time_epoch', data=time_epoch, dtype=dt)
            hm.create_dataset('duration', data=duration, dtype=dt)
            hm.create_dataset('sub_basin_area', data=self._sub_basin_area)
            for var in variables:
                print('\nLISFLOOD %s ...' % var)
                g = hm.create_group(var); g.attrs['units'] = 'm3/s' if var == 'dis' else 'mm'
                res = {i: np.full(len(win), np.nan) for i in range(0, self.nsub + 1)}
                for t, (f, l) in enumerate(tqdm(win, desc='Processing windows')):
                    field = self._window_mean_box(var, f, l)
                    res[0][t] = self._mean(field, self.bm)
                    for i in range(1, self.nsub + 1):
                        res[i][t] = self._mean(field, self.subm[i])
                g.create_dataset('basin', data=res[0])
                for i in range(1, self.nsub + 1):
                    g.create_dataset('sub_basin_%d' % i, data=res[i])
        print('Finished: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        pass


def demo_Amazon():
    base = Path('/media/user/My Book/Fan/PyGLDA_v2_external_data')
    lf = LISFLOOD_HR3a(basin_name='Amazon', shp_path=str(base / 'Basin/shp/Amazon/Amazon.shp'),
                       mask_path=str(base / 'Basin/mask/Amazon/Amazon_res_0.5.h5'),
                       dir_in=str(base / 'Extra/lisfHR3a/daily_05deg'),
                       land_mask_path=str(base / 'GRACE/global_mask/GlobalLandMaskForGRACE.hdf5'))
    out = str(base / 'Extra/res_file')
    '''monthly, like the CSR mascons'''
    lf.basin_TWS(day_begin='2007-01-01', day_end='2020-12-31', windows='daily', dir_out=out)
    lf.grid_TWS(day_begin='2007-01-01', day_end='2020-12-31', windows='daily', dir_out=out)
    '''daily'''
    # lf.basin_TWS(day_begin='2007-01-01', day_end='2020-12-31', windows='daily', dir_out=out, suffix='_lisflood_daily')
    '''on the TUD 5-daily windows: pass the 'duration' strings of the TUD signal file'''
    # dur = h5py.File(base / 'GRACE/output_TUD/Amazon_signal.hdf5')['duration'][:].astype(str)
    # lf.basin_TWS(day_begin='2007-01-01', day_end='2016-04-30', windows=list(dur), dir_out=out, suffix='_lisflood_TUDwindows')
    '''all compartments, monthly'''
    # lf.components_TWS(day_begin='2007-01-01', day_end='2020-12-31', windows='monthly', dir_out=out)
    pass


if __name__ == '__main__':
    demo_Amazon()
