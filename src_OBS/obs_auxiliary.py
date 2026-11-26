"""
Time reference of the observation products: for every epoch the day it stands for ('time_epoch', 'YYYY-MM-DD') and
its averaging window ('duration', 'YYYY-MM-DD_YYYY-MM-DD', both days inclusive).

    aux_GRACE_mascon_monthly   CSR mascons (one NetCDF)                         obs.GRACE.kind "Mascon_monthly"
    aux_GRACE_SH_monthly       monthly gridded product in one CF NetCDF          obs.GRACE.kind "SH_monthly"
    aux_TUD_5daily             TU Delft 5-daily hybrid product                   obs.GRACE.kind "TUD_5daily"
    aux_ESAsing_5daily, aux_ESM3_5daily   ESA-SING simulated 5-daily solutions (prepare_ESA_SING)
"""
import numpy as np
import h5py
import netCDF4 as nc
from src_auxiliary.GeoMathKit import GeoMathKit
from pathlib import Path
from datetime import datetime, timedelta
import calendar
import os


def _window(first, last):
    """'YYYY-MM-DD_YYYY-MM-DD' of two dates (inclusive)"""
    return first.strftime('%Y-%m-%d') + '_' + last.strftime('%Y-%m-%d')


def _month_window(year, month):
    """calendar month as a window, e.g. '2002-04-01_2002-04-30'"""
    return '%04d-%02d-01_%04d-%02d-%02d' % (year, month, year, month, calendar.monthrange(year, month)[1])


def gridded_nc_file(dir_in, filename=None):
    """path of the gridded GRACE NetCDF in dir_in: the given file name, or the only *.nc file there"""
    d = Path(dir_in)
    if filename:
        return d / filename
    fns = sorted(d.glob('*.nc'))
    if len(fns) != 1:
        raise ValueError('gridded GRACE product: give obs.GRACE.nc_file, %d *.nc files in %s' % (len(fns), d))
    return fns[0]


def sh_monthly_epochs(ds):
    """
    The solutions of a monthly gridded product (one CF NetCDF, obs.GRACE.kind = "SH_monthly") as
    [(time index, 'YYYY-MM', first day, last day)], in file order.
    The month of a solution is the month of the MIDPOINT of its time_bounds when the file has them (CF: upper bound
    exclusive), else the month of its time value. Products that stamp a solution with the mean day of its data
    (e.g. C3S/GFZ: the 11/2011 solution at 2011-10-31, the 05/2015 one at 2015-04-27) are labelled by their window,
    not by the stamp. Without bounds the window is the calendar month.
    """
    cal = getattr(ds['time'], 'calendar', 'standard')
    t = nc.num2date(ds['time'][:], ds['time'].units, cal)
    bname = getattr(ds['time'], 'bounds', None) or 'time_bounds'
    tb = nc.num2date(ds[bname][:], ds['time'].units, cal) if bname in ds.variables else None
    out = []
    for i, d in enumerate(t):
        if tb is None:
            first = datetime(d.year, d.month, 1)
            last = datetime(d.year, d.month, calendar.monthrange(d.year, d.month)[1])
            mid = first
        else:
            first = datetime(tb[i][0].year, tb[i][0].month, tb[i][0].day)
            end = datetime(tb[i][1].year, tb[i][1].month, tb[i][1].day)      # exclusive
            last = max(first, end - timedelta(days=1))
            mid = first + (end - first) / 2
        out.append((i, '%04d-%02d' % (mid.year, mid.month), first, last))
    return out


class obs_auxiliary:
    """
    1. provide a list of observation's time reference: which days the obs stand for.
    2. set layer attribution
    3. more..
    """

    def __init__(self):
        self._layer = None
        self._TimeRef = {
            'duration': None,
            'time_epoch': None
        }

    def getTimeReference(self):
        return self._TimeRef

    def setTimeReference(self):
        return self

    def _set(self, time_epoch, duration):
        """store the time reference; returns self (end of every setTimeReference)"""
        self._TimeRef['time_epoch'] = time_epoch
        self._TimeRef['duration'] = duration
        return self

    @staticmethod
    def _need(dir_in, who):
        if dir_in is None:
            raise ValueError('%s.setTimeReference: dir_in must be given' % who)

    def selectSubTimePeriod(self, day_begin='2003-09-01', day_end='2006-12-31'):
        timeRef = self.getTimeReference()
        ss = np.array(timeRef['duration'])
        assert day_end > ss[0] and day_begin < ss[-1]

        a = np.where(ss > day_begin)[0][0]
        b = np.where(ss < day_end)[0][-1]

        if day_end >= timeRef['duration'][b].split('_')[1]:
            b += 1

        x = {}
        x['time_epoch'] = timeRef['time_epoch'][a:b].copy()
        x['duration'] = timeRef['duration'][a:b].copy()
        return x, a, b

    def save_H5(self, basin_name: str, dir_out=None):
        """save it into h5"""
        if dir_out is None:
            raise ValueError('save_H5: dir_out (output directory) must be given')
        hm = h5py.File(Path(dir_out) / ('%s_aux.hdf5' % basin_name), 'w')
        dt = h5py.special_dtype(vlen=str)
        hm.create_dataset('time_epoch', data=self._TimeRef['time_epoch'], dtype=dt)
        hm.create_dataset('duration', data=self._TimeRef['duration'], dtype=dt)
        hm.create_dataset('layer', data=self._layer, dtype=dt)
        hm.close()


# ====================================================================== monthly products
class aux_GRACE_mascon_monthly(obs_auxiliary):

    def setTimeReference(self, month_begin='2002-04', month_end='2023-05', dir_in=None):
        """dir_in: directory holding CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc"""
        self._need(dir_in, 'aux_GRACE_mascon_monthly')
        csr = nc.Dataset(Path(dir_in) / 'CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc')
        initial_day = datetime.strptime(csr.time_epoch.split('T')[0], '%Y-%m-%d')
        available = {(initial_day + timedelta(days=int(dd))).strftime('%Y-%m')
                     for dd in np.array(csr['time'][:].astype(int))}
        csr.close()
        '''CSR labels two solutions with the month before (2011-11 and 2015-04 fall on the previous month): the
        2011-11 solution is listed under 2011-10, so 2011-11 is added here (the data index is handled in
        prepare_GRACE_mascon)'''
        available.add('2011-11')

        time_epoch, duration = [], []
        for month in GeoMathKit.monthListByMonth(begin=month_begin, end=month_end):
            m = month.strftime('%Y-%m')
            if m in available:
                time_epoch.append(m + '-15')
                duration.append(_month_window(month.year, month.month))
        return self._set(time_epoch, duration)


class aux_GRACE_SH_monthly(obs_auxiliary):
    """time reference of a monthly gridded GRACE product (e.g. a spherical-harmonic solution gridded with SaGEA) in one
    CF NetCDF (obs.GRACE.kind = "SH_monthly"; read by prepare_GRACE.GRACE_preparation):
    month of a solution and its window from sh_monthly_epochs (time_bounds if present, else the calendar month of the
    time value); epoch = the 15th of that month; a month that occurs twice is used once"""

    def setTimeReference(self, month_begin='2002-04', month_end='2023-05', dir_in=None, filename=None):
        self._need(dir_in, 'aux_GRACE_SH_monthly')
        ds = nc.Dataset(gridded_nc_file(dir_in, filename), 'r')
        sol = sh_monthly_epochs(ds)
        ds.close()

        time_epoch, duration, seen = [], [], set()
        for i, m, first, last in sol:
            if not (month_begin <= m <= month_end) or m in seen:
                continue
            seen.add(m)
            time_epoch.append(m + '-15')
            duration.append(_window(first, last))
        return self._set(time_epoch, duration)


# ====================================================================== 5-daily products
class _aux_5daily_files(obs_auxiliary):
    """5-daily solutions stored one file per solution in one subfolder per year; the file name carries the first
    day and the day after the last day of the window ('YYYYMMDD'); epoch = first day + 2"""
    years = ()

    @staticmethod
    def _dates(fn):
        """(first day, day after the last day) as 'YYYYMMDD' strings from a file name"""
        raise NotImplementedError

    def setTimeReference(self, day_begin='1990-04-01', day_end='2008-05-01', dir_in=None):
        self._need(dir_in, type(self).__name__)
        day_begin = datetime.strptime(day_begin, '%Y-%m-%d')
        day_end = datetime.strptime(day_end, '%Y-%m-%d')
        time_epoch, duration = [], []
        for year in self.years:
            for ff in sorted(os.listdir(Path(dir_in) / ('%s' % year))):
                a, b = self._dates(ff)
                first = datetime.strptime(a, '%Y%m%d')
                last = datetime.strptime(b, '%Y%m%d') - timedelta(days=1)
                if first >= day_begin and last <= day_end:
                    duration.append(_window(first, last))
                    time_epoch.append((first + timedelta(days=2)).strftime('%Y-%m-%d'))
        return self._set(time_epoch, duration)


class aux_ESAsing_5daily(_aux_5daily_files):
    """dir_in: folder of the ESA SING 5-daily files, one subfolder per year (1995-2006)"""
    years = range(1995, 2007)

    @staticmethod
    def _dates(fn):
        return fn.split('_')[4].split('-')


class aux_ESM3_5daily(_aux_5daily_files):
    """dir_in: folder of the ESM 3.0 5-daily files, one subfolder per year (2007-2020)"""
    years = range(2007, 2021)

    @staticmethod
    def _dates(fn):
        x = fn.split('.')[0].split('_')
        return x[2], x[3]


class aux_TUD_5daily(obs_auxiliary):
    """
    Time reference for the TU Delft 5-daily (to weekly) hybrid GRACE L3 product.
    The netCDF 'time' axis holds integer days since 2002-04-07T12:00 at irregular 4-7 day spacing and
    marks the CENTRE of each solution; the product carries no window bounds. The averaging window of
    epoch i is therefore reconstructed symmetrically from the spacing to its neighbours:
        half_i = min(gap_prev - 1, gap_next - 1) // 2  (whole days, capped at max_half_window_days)
        duration_i = [t_i - half_i, t_i + half_i]   (inclusive, 'YYYY-MM-DD_YYYY-MM-DD')
    which guarantees non-overlapping windows (required by EnKF.helper_resolve_time); days in between
    windows simply receive no observation. getDataIndex() returns, for every kept epoch, its index on the
    netCDF time axis so that the preparation class reads exactly these records (selectSubTimePeriod returns
    the positions a, b, so getDataIndex()[a:b] stays aligned).
    """

    def __init__(self):
        super().__init__()
        self._data_index = None
        self._time_days = None

    def setTimeReference(self, day_begin='2002-01-01', day_end='2030-01-01', dir_in=None,
                         filename='TUD-L3-5dayEWH-GRACEv2-Hybrid-2002_2017-0.5x0.5.nc', max_half_window_days=3):
        self._need(dir_in, 'aux_TUD_5daily')
        day_begin = datetime.strptime(day_begin, '%Y-%m-%d').date()
        day_end = datetime.strptime(day_end, '%Y-%m-%d').date()

        ds = nc.Dataset(Path(dir_in) / filename)
        tv = ds.variables['time']
        ref = datetime.strptime(tv.units.split('since')[1].strip().replace('T', ' ')[:19], '%Y-%m-%d %H:%M:%S')
        days = np.array(tv[:]).astype(float)                # 'days since 2002-04-07T12:00:00'
        ds.close()

        gaps = np.diff(days)
        time_epoch, duration, index = [], [], []
        for i, d in enumerate(days):
            c = ref + timedelta(days=float(d))
            gp = gaps[i - 1] if i > 0 else gaps[0]
            gn = gaps[i] if i < len(gaps) else gaps[-1]
            half = max(0, min(int(min(gp - 1, gn - 1) // 2), int(max_half_window_days)))   # windows never overlap
            first = (c - timedelta(days=half)).date()
            last = (c + timedelta(days=half)).date()
            if first < day_begin or last > day_end:
                continue
            time_epoch.append(c.strftime('%Y-%m-%d'))
            duration.append(_window(first, last))
            index.append(i)

        self._data_index = np.array(index, dtype=int)
        self._time_days = days
        return self._set(time_epoch, duration)

    def getDataIndex(self):
        """index of every kept epoch on the netCDF time axis (aligned with getTimeReference())"""
        return self._data_index
