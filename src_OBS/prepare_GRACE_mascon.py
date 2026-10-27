
import numpy as np
import h5py
import netCDF4 as nc
from src_OBS.prepare_GRACE import GRACE_preparation
from src_auxiliary.GeoMathKit import GeoMathKit
import os
from pathlib import Path
from datetime import datetime, timedelta
from src_OBS.obs_auxiliary import aux_GRACE_mascon_monthly
from tqdm import tqdm


class GRACE_CSR_mascon(GRACE_preparation):

    def __init__(self, basin_name='MDB', shp_path='../data/basin/shp/MDB_4_shapefiles/MDB_4_subbasins.shp'):
        super().__init__(basin_name=basin_name, shp_path=shp_path)
        self.resample = 'subsample'
        self.earthquake_correction = None
        self._half_degree_cache = None
        pass

    def set_options(self, resample='subsample', earthquake_correction=None):
        """
        resample : how the 0.25-degree mascon grid is brought to the 0.5-degree mask grid.
            "subsample"  every second row / column ([0::2, 0::2]; default): the 0.5-degree
                         value is ONE 0.25-degree cell, offset by 0.125 degree from the 0.5-degree cell centre;
            "block_mean" mean of the four 0.25-degree cells of every 0.5-degree cell (centres aligned with the mask grid).
        earthquake_correction : None (off) or the settings of src_OBS.earthquake_correction.fit_and_correct, e.g.
            {"cap_km": 1000, "taper_km": 200, "events": [...], "significance": {"n_placebo": 16, "z_lo": 3, "z_hi": 5}}:
            co- and post-seismic signals of great earthquakes are removed from the gridded mascons (fitted on the whole
            record of the file) before the unit averaging.
        Both are read from DA_setting.json -> obs.GRACE ("resample", "earthquake_correction") by GDA / RDA.get_GRACE_obs.
        """
        r = str(resample or 'subsample').lower()
        if r not in ('subsample', 'block_mean'):
            raise ValueError('GRACE mascon resample must be "subsample" or "block_mean", got %s' % resample)
        self.resample = r
        self.earthquake_correction = dict(earthquake_correction) if earthquake_correction else None
        self._half_degree_cache = None
        return self

    def _mascon_half_degree(self, csr):
        """mascon lwe_thickness [cm] on the 0.5-degree grid in the file's orientation (latitude ascending from the
        south, longitude 0..360), resampled and optionally earthquake-corrected (see set_options); cached"""
        if self._half_degree_cache is not None:
            return self._half_degree_cache
        raw = np.array(csr['lwe_thickness'])
        resample = getattr(self, 'resample', 'subsample')
        if resample == 'block_mean':
            n, ny, nx = raw.shape
            data = raw.reshape(n, ny // 2, 2, nx // 2, 2).mean(axis=(2, 4))
            lat = np.asarray(csr['lat'][:], dtype=float).reshape(-1, 2).mean(1)
            lon = np.asarray(csr['lon'][:], dtype=float).reshape(-1, 2).mean(1)
        else:
            data = raw[:, 0::2, 0::2]
            lat = np.asarray(csr['lat'][:], dtype=float)[0::2]
            lon = np.asarray(csr['lon'][:], dtype=float)[0::2]
        eq = getattr(self, 'earthquake_correction', None)
        if eq:
            from src_OBS.earthquake_correction import fit_and_correct
            day0 = datetime.strptime(csr.time_epoch.split('T')[0], '%Y-%m-%d')
            t = []
            for dd in np.asarray(csr['time'][:], dtype=float):
                d = day0 + timedelta(days=float(dd))
                y0, y1 = datetime(d.year, 1, 1), datetime(d.year + 1, 1, 1)
                t.append(d.year + (d - y0).total_seconds() / (y1 - y0).total_seconds())
            LAT, LON = np.meshgrid(lat, np.where(lon > 180, lon - 360, lon), indexing='ij')
            kw = {k: eq[k] for k in ('cap_km', 'taper_km', 'events', 'significance') if k in eq}
            data, corr, info = fit_and_correct(data, np.array(t), LAT, LON, **kw)
            self.earthquake_info = info
        self._half_degree_cache = data
        return data

    def basin_TWS(self, month_begin='2002-04', month_end='2002-04',
                  dir_in=None,
                  dir_out=None):
        """
        basin averaged TWS [mm] per mascon month: dir_in holds CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc,
        dir_out receives <basin>_signal.hdf5
        """
        if dir_in is None or dir_out is None:
            raise ValueError('GRACE_CSR_mascon.basin_TWS: dir_in and dir_out must be given')

        '''get a complete time list for all dataset'''
        self._aux = aux_GRACE_mascon_monthly().setTimeReference(month_begin=month_begin, month_end=month_end,
                                                                dir_in=dir_in)

        '''===============handling data mask===================='''
        res = 0.5
        mf = h5py.File(Path(self.save_mask_dir)/('%s_res_%s.h5' % (self.basin_name, res)), 'r')

        basins_num = len(list(mf.keys())) - 1

        mask = {}
        for i in range(1, basins_num + 1):
            key = 'sub_basin_%d' % i
            mask[key] = mf[key][:] * self._05deg_mask

        '''load latitude'''
        err = res / 10
        lat = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
        lon = np.arange(-180 + res / 2, 180 - res / 2 + err, res)
        lon, lat = np.meshgrid(lon, lat)
        '''be aware that Mascon has a latitude from -89.875 to 89.875, and a longitude from 0.125 to 359.875'''

        '''===============handling mascon data========================'''
        filename = 'CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc'
        csr = nc.Dataset(Path(dir_in) / filename)

        initial_day = datetime.strptime(csr.time_epoch.split('T')[0], '%Y-%m-%d')
        timelist = np.array(csr['time'][:].astype(int))

        available_month = {}
        index = 0
        for dd in timelist:
            this_month = initial_day + timedelta(days=int(dd))
            available_month[this_month.strftime('%Y-%m')] = index
            index += 1

        '''particular correction is applied to the mascon monthlist'''
        available_month['2011-10'] -= 1
        available_month['2011-11'] = available_month['2011-10'] + 1
        available_month['2015-04'] -= 1

        mascon_data = self._mascon_half_degree(csr)  # the unit is centimeter (resample / earthquake: set_options)

        '''===============computation========================'''
        monthlist = GeoMathKit.monthListByMonth(begin=month_begin, end=month_end)
        TWS = {}
        print('Start to pre-process GRACE to obtain basin-wise TWS over places of interest...')
        for i in range(1, basins_num + 1):
            TWS['sub_basin_%d' % i] = []
            pass

        for month in tqdm(monthlist, desc="Processing months"):
            this_month_shortname = month.strftime('%Y-%m')
            if this_month_shortname not in available_month.keys():
                continue

            data = mascon_data[available_month[this_month_shortname]].copy()
            '''convert to be compatible with the mask grid definition'''
            data = np.flipud(data)
            tws_one_month = np.roll(data, 360)

            for i in range(1, basins_num + 1):
                key = 'sub_basin_%d' % i
                a = tws_one_month[mask[key].astype(bool)]
                b = lat[mask[key].astype(bool)]
                b = np.cos(np.deg2rad(b))
                '''convert unit from cm to mm'''
                TWS[key].append(np.sum(a * b) / np.sum(b) * 10)
                pass

        '''save it into h5'''
        out_dir = Path(dir_out)
        hm = h5py.File(out_dir / ('%s_signal.hdf5' % self.basin_name), 'w')
        for i in range(1, basins_num + 1):
            key = 'sub_basin_%d' % i
            hm.create_dataset(key, data=np.array(TWS[key]))

        dt = h5py.special_dtype(vlen=str)
        hm.create_dataset('time_epoch', data=self._aux.getTimeReference()['time_epoch'], dtype=dt)
        hm.create_dataset('sub_basin_area', data=self._sub_basin_area)
        hm.close()
        print('Finished: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        pass

    def grid_TWS(self, month_begin='2002-04', month_end='2002-04',
                 dir_in=None,
                 dir_out=None):
        """
        gridded TWS [mm] of the basin cells per mascon month: dir_in holds the CSR mascon netCDF,
        dir_out receives <basin>_gridded_signal.hdf5
        """
        if dir_in is None or dir_out is None:
            raise ValueError('GRACE_CSR_mascon.grid_TWS: dir_in and dir_out must be given')
        '''get a complete time list for all dataset'''
        self._aux = aux_GRACE_mascon_monthly().setTimeReference(month_begin=month_begin, month_end=month_end,
                                                                dir_in=dir_in)
        '''===============handling data mask===================='''
        res = 0.5
        mf = h5py.File(Path(self.save_mask_dir)/('%s_res_%s.h5' % (self.basin_name, res)), 'r')

        basins_num = len(list(mf.keys())) - 1

        mask = mf['basin'][:] * self._05deg_mask

        '''load latitude'''
        err = res / 10
        lat = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
        lon = np.arange(-180 + res / 2, 180 - res / 2 + err, res)
        lon, lat = np.meshgrid(lon, lat)
        '''be aware that Mascon has a latitude from -89.875 to 89.875, and a longitude from 0.125 to 359.875'''

        '''===============handling mascon data========================'''
        filename = 'CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc'
        csr = nc.Dataset(Path(dir_in) / filename)

        initial_day = datetime.strptime(csr.time_epoch.split('T')[0], '%Y-%m-%d')
        timelist = np.array(csr['time'][:].astype(int))

        available_month = {}
        index = 0
        for dd in timelist:
            this_month = initial_day + timedelta(days=int(dd))
            available_month[this_month.strftime('%Y-%m')] = index
            index += 1

        '''particular correction is applied to the mascon monthlist'''
        available_month['2011-10'] -= 1
        available_month['2011-11'] = available_month['2011-10'] + 1
        available_month['2015-04'] -= 1

        mascon_data = self._mascon_half_degree(csr)  # the unit is centimeter (resample / earthquake: set_options)

        '''===============computation========================'''
        monthlist = GeoMathKit.monthListByMonth(begin=month_begin, end=month_end)
        TWS = []
        tqdm.write('\nStart to pre-process GRACE to obtain grid-wise TWS over places of interest...')
        for month in tqdm(monthlist, desc="Processing months"):
            this_month_shortname = month.strftime('%Y-%m')
            if this_month_shortname not in available_month.keys():
                continue

            data = mascon_data[available_month[this_month_shortname]].copy()
            '''convert to be compatible with the mask grid definition, convert unit from cm to mm'''
            data = np.flipud(data) * 10
            tws_one_month = np.roll(data, 360)

            a = tws_one_month[mask.astype(bool)]
            TWS.append(a)

        '''save it into h5'''
        out_dir = Path(dir_out)
        hm = h5py.File(out_dir / ('%s_gridded_signal.hdf5' % self.basin_name), 'w')

        hm.create_dataset(name='tws', data=np.array(TWS))

        dt = h5py.special_dtype(vlen=str)
        hm.create_dataset('time_epoch', data=self._aux.getTimeReference()['time_epoch'], dtype=dt)
        hm.create_dataset('sub_basin_area', data=self._sub_basin_area)
        hm.close()
        print('Finished: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        pass

