
import numpy as np
import h5py
from src_auxiliary.shp2mask import basin_shp_process, mask_05_to_1
from src_auxiliary.GeoMathKit import GeoMathKit
import os
from pathlib import Path
from datetime import datetime
import geopandas as gpd
import netCDF4 as nc
from src_OBS.obs_auxiliary import aux_GRACE_SH_monthly, gridded_nc_file, sh_monthly_epochs
from tqdm import tqdm

class _GriddedNC:
    """
    Reading of a monthly gridded GRACE/GRACE-FO product stored as ONE CF NetCDF (e.g. a spherical-harmonic solution
    processed with SaGEA: ITSG-Grace2018 + ITSG-Grace_operational, DDK3, 0.5 degree; obs.GRACE.kind = "SH_monthly"):
    variables time (CF units), lat, lon and the EWH field `nc_var` (time x lat x lon) in mm, cm or m (attribute
    "units", mm if missing); lat may run either way and lon may be -180..180 or 0..360. time_bounds, if present, give
    the observation windows (obs_auxiliary.aux_GRACE_SH_monthly). With `earthquake_correction` set, co- and
    post-seismic signals are fitted on the whole record of the file and removed (src_OBS.earthquake_correction), as for
    the mascons.
    """
    nc_file = None
    nc_var = 'tws'
    earthquake_correction = None
    _cube = None

    def set_options(self, nc_file=None, nc_var='tws', earthquake_correction=None):
        """nc_file: file name inside dir_in (None: the only *.nc file there); nc_var: EWH variable name;
        earthquake_correction: None (off) or the settings of src_OBS.earthquake_correction.fit_and_correct"""
        self.nc_file = nc_file
        self.nc_var = nc_var or 'tws'
        self.earthquake_correction = dict(earthquake_correction) if earthquake_correction else None
        self._cube = None
        return self

    def _open(self, dir_in):
        return nc.Dataset(gridded_nc_file(dir_in, self.nc_file), 'r')

    @staticmethod
    def _months(ds, month_begin, month_end):
        """[(time index, 'YYYY-MM')] of the file inside [month_begin, month_end]; a repeated month is used once
        (the same months as obs_auxiliary.aux_GRACE_SH_monthly, labelled by sh_monthly_epochs)"""
        out, seen = [], set()
        for i, m, _, _ in sh_monthly_epochs(ds):
            if month_begin <= m <= month_end and m not in seen:
                seen.add(m)
                out.append((i, m))
        return out

    def _field(self, ds, i):
        """EWH of time index i in mm on the 0.5-degree grid, lat -90 -> 90, lon -180 -> 180 (360 x 720)"""
        if self.earthquake_correction:
            return self._corrected(ds)[i]
        return self._raw(ds, i)

    def _raw(self, ds, i):
        v = ds[self.nc_var]
        a = np.ma.filled(v[i, :, :].astype(float), np.nan)
        lat = np.asarray(ds['lat'][:], dtype=float)
        lon = np.asarray(ds['lon'][:], dtype=float)
        if lat[0] > lat[-1]:
            a, lat = a[::-1, :], lat[::-1]
        lon = np.where(lon >= 180, lon - 360, lon)
        o = np.argsort(lon)
        a, lon = a[:, o], lon[o]
        if a.shape != (360, 720) or abs(lat[0] + 89.75) > 1e-3 or abs(lon[0] + 179.75) > 1e-3:
            raise ValueError('gridded GRACE product: expected a 0.5-degree global grid with cell centres '
                             '(-89.75.., -179.75..), got %s, lat0 %.3f, lon0 %.3f' % (a.shape, lat[0], lon[0]))
        unit = getattr(v, 'units', 'mm').strip().lower()
        scale = {'mm': 1.0, 'cm': 10.0, 'm': 1000.0, 'kg m-2': 1.0, 'kg/m2': 1.0, 'kg m**-2': 1.0}.get(unit)
        if scale is None:
            raise ValueError('gridded GRACE product: unknown unit %s of %s' % (unit, self.nc_var))
        return a * scale

    def _corrected(self, ds):
        """the whole record, earthquake-corrected (computed once, cached)"""
        if self._cube is None:
            from src_OBS.earthquake_correction import fit_and_correct
            data = np.array([self._raw(ds, i) for i in range(len(ds['time']))])
            t = []
            for d in nc.num2date(ds['time'][:], ds['time'].units, getattr(ds['time'], 'calendar', 'standard')):
                d = datetime(d.year, d.month, d.day, getattr(d, 'hour', 0))
                y0, y1 = datetime(d.year, 1, 1), datetime(d.year + 1, 1, 1)
                t.append(d.year + (d - y0).total_seconds() / (y1 - y0).total_seconds())
            LAT, LON = np.meshgrid(np.arange(-89.75, 90, 0.5), np.arange(-179.75, 180, 0.5), indexing='ij')
            eq = self.earthquake_correction
            kw = {k: eq[k] for k in ('cap_km', 'taper_km', 'events', 'significance') if k in eq}
            ok = np.isfinite(data)
            data, corr, info = fit_and_correct(np.nan_to_num(data), np.array(t), LAT, LON, **kw)
            data[~ok] = np.nan
            self.earthquake_info = info
            self._cube = data
        return self._cube

    def _grids(self, month_begin, month_end, dir_in):
        """yield ('YYYY-MM', EWH grid flipped to the mask orientation: lat 90 -> -90, lon -180 -> 180)"""
        ds = self._open(dir_in)
        try:
            for i, m in tqdm(self._months(ds, month_begin, month_end), desc='Processing months'):
                yield m, np.flipud(self._field(ds, i))
        finally:
            ds.close()



class GRACE_preparation(_GriddedNC):

    def __init__(self, basin_name='MDB',
                 shp_path='../data/basin/shp/MDB_4_shapefiles/MDB_4_subbasins.shp'):
        self.basin_name = basin_name
        self.__shp_path = shp_path
        pass

    def generate_mask(self, box_mask=None, res_05=True, res_10=True, save_dir='../data/basin/mask'):
        """
        generate and save basin mask for later use:
          0.5 degree (signal)      rasterised from the shapefile (shp2mask.basin_shp_process)
          1 degree   (covariance)  derived from the 0.5-degree mask by majority of the land cells (shp2mask.mask_05_to_1),
                                   so that both resolutions describe the same cells; call
                                   configure_global_land_ocean_mask first so that only land cells count
        """

        '''for signal: 0.5 degree'''
        bs1 = basin_shp_process(res=0.5, basin_name=self.basin_name, save_dir=save_dir).configureBox(box_mask)
        if res_05:
            bs1.shp_to_mask(shp_path=self.__shp_path, issave=True)
        self.save_mask_dir = bs1.save_dir

        '''for cov: 1 degree, from the 0.5-degree mask'''
        if res_10:
            mask_05_to_1(mask05_path=Path(self.save_mask_dir) / ('%s_res_0.5.h5' % self.basin_name),
                         land_05=getattr(self, '_05deg_mask', None))

        '''get the area information of each subbasin'''
        self._sub_basin_area = self._area_from_shp()
        pass

    def use_mask(self, mask_dir, res_10=True):
        """
        use an existing 0.5-degree mask <mask_dir>/<basin>_res_0.5.h5 instead of rasterising the shapefile (e.g. the
        frozen global unit mask of global_shp2mask, which generate_mask could not reproduce: model land, removed units);
        the 1-degree mask is derived from it when missing (res_10). The sub-basin areas come from the shapefile.
        """
        self.save_mask_dir = Path(mask_dir)
        fn1 = self.save_mask_dir / ('%s_res_1.h5' % self.basin_name)
        if res_10 and not fn1.exists():
            mask_05_to_1(mask05_path=self.save_mask_dir / ('%s_res_0.5.h5' % self.basin_name),
                         land_05=getattr(self, '_05deg_mask', None))
        self._sub_basin_area = self._area_from_shp()
        return self

    def _area_from_shp(self):
        """sub-basin areas in the order of the IDs: SUB_AREA [km2] when the shapefile has it, else the raw polygon area"""
        gdf = gpd.read_file(self.__shp_path)
        if 'SUB_AREA' in gdf.columns and 'ID' in gdf.columns:
            return gdf.sort_values('ID').SUB_AREA.values.astype(float)
        return gdf.area.values

    def configure_global_land_ocean_mask(self, fn='../data/GRACE/global_mask/WaterGAPLandMask.hdf5'):
        """
        This is to prevent accounting for the oceanic GRACE into the land observations.
        Land = WaterGAP land (src_auxiliary.watergap_land_mask: WaterGAPLandMask.hdf5), the same cells over which the
        DA averages the model, so that the GRACE and the model unit means are consistent; the former GRACE land mask
        (GlobalLandMaskForGRACE.hdf5) has the same layout and can still be given.
        lat: 90=>-90
        lon:-180=>180
        """
        self._1deg_mask = np.flipud(h5py.File(fn, 'r')['resolution_1']['mask'][:])
        self._05deg_mask = np.flipud(h5py.File(fn, 'r')['resolution_05']['mask'][:])
        return self

    def basin_TWS(self, month_begin='2002-04', month_end='2002-04', dir_in=None, dir_out=None):
        """
        unit-averaged TWS [mm] per month (cos-latitude weights on the 0.5-degree unit mask) of the gridded product
        in dir_in (one NetCDF, see _GriddedNC) -> <dir_out>/<basin>_signal.hdf5
        """
        if dir_in is None or dir_out is None:
            raise ValueError('GRACE_preparation.basin_TWS: dir_in and dir_out must be given')
        self._aux = aux_GRACE_SH_monthly().setTimeReference(month_begin=month_begin, month_end=month_end,
                                                            dir_in=dir_in, filename=self.nc_file)
        res = 0.5
        mf = h5py.File(Path(self.save_mask_dir) / ('%s_res_%s.h5' % (self.basin_name, res)), 'r')
        basins_num = len(list(mf.keys())) - 1
        mask = {'sub_basin_%d' % i: (mf['sub_basin_%d' % i][:] * self._05deg_mask).astype(bool)
                for i in range(1, basins_num + 1)}                                   # lat 90 -> -90
        lat = np.arange(90 - res / 2, -90, -res)
        w = np.cos(np.deg2rad(np.repeat(lat[:, None], 720, axis=1)))

        print('\nStart to pre-process GRACE to obtain basin-wise TWS over places of interest...')
        TWS = {k: [] for k in mask}
        for _, f in self._grids(month_begin, month_end, dir_in):
            for k in TWS:
                a, b = f[mask[k]], w[mask[k]]
                ok = np.isfinite(a)
                TWS[k].append(np.sum(a[ok] * b[ok]) / np.sum(b[ok]))

        hm = h5py.File(Path(dir_out) / ('%s_signal.hdf5' % self.basin_name), 'w')
        for k in TWS:
            hm.create_dataset(k, data=np.array(TWS[k]))
        dt = h5py.special_dtype(vlen=str)
        hm.create_dataset('time_epoch', data=self._aux.getTimeReference()['time_epoch'], dtype=dt)
        hm.create_dataset('sub_basin_area', data=self._sub_basin_area)
        hm.close()
        print('Finished: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    def basin_COV(self, month_begin='2002-04', month_end='2002-04',
                  dir_in=None,
                  dir_out=None,
                  is_diagonal=False):
        """
        COV of subbasins, monthly time-series.
        Be careful to deal with mask and TWS.
        mask: lon: -180-->180, lat: 90--->-90
        tws: lon: -180-->180, lat: -90--->90
        """
        if dir_in is None or dir_out is None:
            raise ValueError('GRACE_preparation.basin_COV: dir_in and dir_out must be given')

        '''load mask'''
        res = 1
        mf = h5py.File(Path(self.save_mask_dir)/('%s_res_%s.h5' % (self.basin_name, res)), 'r')

        basins_num = len(list(mf.keys())) - 1

        mask = {}
        for i in range(1, basins_num + 1):
            key = 'sub_basin_%d' % i
            '''flip upside down to be compatible with the TWS dataset'''
            mask[key] = np.flipud(mf[key][:] * self._1deg_mask)
            assert np.max(mask[key]) > 0.1, 'Basin mask is incompatible with the 1-deg land mask, please revise!'

        '''load latitude'''
        err = res / 10
        lat = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
        lon = np.arange(-180 + res / 2, 180 - res / 2 + err, res)
        lon, lat = np.meshgrid(lon, lat)
        '''flip upside down to be compatible with the TWS dataset'''
        lat = np.flipud(lat)

        '''calculate the COV, monthly'''
        directory = Path(dir_in)
        fns = sorted(os.listdir(directory))  # deterministic: first match in sorted order (e.g. 2009-06_0 before _1)
        monthlist = GeoMathKit.monthListByMonth(begin=month_begin, end=month_end)

        COV = []

        '''acquire time reference: one covariance per observed month, in the order of the observations (EWH). The
        sample folder may hold more months than the product (e.g. sample_DDK3 has 2016-11..2017-06, which ITSG lacks):
        those are skipped. An observed month without samples stops here with the list of such months.'''
        time_epochs_ref = self._aux.getTimeReference()['time_epoch']
        obs_months = [datetime.strptime(e, '%Y-%m-%d').strftime('%Y-%m') for e in time_epochs_ref]
        in_range = {m.strftime('%Y-%m') for m in monthlist}
        files = {}
        for m in obs_months:
            files[m] = next((f for f in fns if m in f), None)      # first match in sorted order
        missing = [m for m in obs_months if m in in_range and files[m] is None]
        if missing:
            raise FileNotFoundError('GRACE_preparation.basin_COV: no error samples in %s for the observed month(s) %s'
                                    % (directory, ', '.join(missing)))

        print('\nStart to pre-process GRACE to obtain COV over places of interest...')
        for fn in tqdm([m for m in obs_months if m in in_range], desc="Processing months"):
            tn = files[fn]
            fn = directory / tn

            tws_one_month = np.load(str(fn))

            vv = []
            for i in range(1, basins_num + 1):
                key = 'sub_basin_%d' % i
                a = tws_one_month[:, mask[key].astype(bool)]
                b = lat[mask[key].astype(bool)]
                b = np.cos(np.deg2rad(b))
                vv.append(np.sum(a * b, 1) / np.sum(b))
                pass

            '''transform into mm'''
            cov = np.cov(np.array(vv) * 1000)
            if is_diagonal:
                cov = np.diag(np.diag(cov))
            COV.append(cov)
            pass

        '''save it into h5'''
        out_dir = Path(dir_out)
        hm = h5py.File(out_dir / ('%s_cov.hdf5' % self.basin_name), 'w')
        hm.create_dataset('data', data=np.array(COV))

        dt = h5py.special_dtype(vlen=str)
        hm.create_dataset('time_epoch', data=self._aux.getTimeReference()['time_epoch'], dtype=dt)

        hm.close()
        print('Finished: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        pass

    def grid_TWS(self, month_begin='2002-04', month_end='2002-04', dir_in=None, dir_out=None):
        """
        gridded TWS [mm] of the cells of the basin mask, per month, of the gridded product in dir_in (one NetCDF)
        -> <dir_out>/<basin>_gridded_signal.hdf5
        """
        if dir_in is None or dir_out is None:
            raise ValueError('GRACE_preparation.grid_TWS: dir_in and dir_out must be given')
        aux = aux_GRACE_SH_monthly().setTimeReference(month_begin=month_begin, month_end=month_end,
                                                      dir_in=dir_in, filename=self.nc_file)
        res = 0.5
        mf = h5py.File(Path(self.save_mask_dir) / ('%s_res_%s.h5' % (self.basin_name, res)), 'r')
        mask = (mf['basin'][:] * self._05deg_mask).astype(bool)                         # lat 90 -> -90
        print('\nStart to pre-process GRACE to obtain gridded TWS over places of interest...')
        TWS = [f[mask] for _, f in self._grids(month_begin, month_end, dir_in)]
        hm = h5py.File(Path(dir_out) / ('%s_gridded_signal.hdf5' % self.basin_name), 'w')
        hm.create_dataset(name='tws', data=np.array(TWS))
        dt = h5py.special_dtype(vlen=str)
        hm.create_dataset('time_epoch', data=aux.getTimeReference()['time_epoch'], dtype=dt)
        hm.close()
        print('Finished: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


class GRACE_global_preparation(_GriddedNC):

    def __init__(self):
        pass

    def configure_global_mask(self, fn=None):
        """
        This is to prevent accounting for the oceanic GRACE into the land observations.
        fn: global land mask file (same layout as WaterGAPLandMask.hdf5 / GlobalLandMaskForGRACE.hdf5)
        """
        if fn is None:
            raise ValueError('GRACE_global_preparation.configure_global_mask: fn must be given')
        self._1deg_mask = np.flipud(h5py.File(fn, 'r')['resolution_1']['mask'][:])
        self._05deg_mask = np.flipud(h5py.File(fn, 'r')['resolution_05']['mask'][:])
        return self

    def configure_global_shp(self, fn='../data/basin/shp/global_shp_new'):
        valid_basin_mask_05 = {}
        valid_basin_mask_10 = {}

        for i in range(1, 300):
            fshp = Path(fn) / ('Tile%s_subbasins.shp' % i)
            if not fshp.is_file():
                continue
            print(fshp)
            valid_basin_mask_05['Tile%s' % i] = basin_shp_process(res=0.5, basin_name='Tile%s' % i).shp_to_mask(
                shp_path=str(fshp), issave=False).mask
            valid_basin_mask_10['Tile%s' % i] = basin_shp_process(res=1, basin_name='Tile%s' % i).shp_to_mask(
                shp_path=str(fshp), issave=False).mask

            for nn in valid_basin_mask_10['Tile%s' % i].keys():
                valid_basin_mask_05['Tile%s' % i][nn] = (
                        valid_basin_mask_05['Tile%s' % i][nn] * self._05deg_mask).astype(bool)
                valid_basin_mask_10['Tile%s' % i][nn] = (
                        valid_basin_mask_10['Tile%s' % i][nn] * self._1deg_mask).astype(bool)
            pass

        self.valid_mask = {
            '05': valid_basin_mask_05,
            '1': valid_basin_mask_10
        }

        return self

    def basin_TWS(self, month_begin='2002-04', month_end='2002-04', dir_in=None, dir_out=None):
        """
        unit-averaged TWS [mm] of every tile, monthly, from the gridded product in dir_in (one NetCDF, see _GriddedNC)
        -> <dir_out>/<tile>_signal.hdf5
        """
        if dir_in is None or dir_out is None:
            raise ValueError('GRACE_global_preparation.basin_TWS: dir_in and dir_out must be given')
        res = 0.5
        mask = self.valid_mask['05']
        lat = np.arange(90 - res / 2, -90, -res)
        area = np.cos(np.deg2rad(np.repeat(lat[:, None], 720, axis=1)))

        print('\nStart to pre-process GRACE to obtain signal over places of interest...')
        TWS = {tile: {} for tile in mask}
        time_epoch = []
        for m, tws_one_month in self._grids(month_begin, month_end, dir_in):
            time_epoch.append(m + '-15')
            for tile, tile_mask in mask.items():
                for i in range(1, len(list(tile_mask.keys()))):
                    a, b = tws_one_month[tile_mask[i]], area[tile_mask[i]]
                    TWS[tile].setdefault('sub_basin_%d' % i, []).append(np.sum(a * b) / np.sum(b))

        for tile, vv in TWS.items():
            hm = h5py.File(Path(dir_out) / ('%s_signal.hdf5' % tile), 'w')
            for key, item in vv.items():
                hm.create_dataset(key, data=np.array(item))
            dt = h5py.special_dtype(vlen=str)
            hm.create_dataset('time_epoch', data=time_epoch, dtype=dt)
            hm.close()
        print('Finished: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    def basin_COV(self, month_begin='2002-04', month_end='2002-04',
                  dir_in=None,
                  dir_out=None):
        if dir_in is None or dir_out is None:
            raise ValueError('GRACE_global_preparation.basin_COV: dir_in and dir_out must be given')

        '''load mask'''
        res = 1
        mask = self.valid_mask['1']

        '''load latitude'''
        err = res / 10
        lat = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
        lon = np.arange(-180 + res / 2, 180 - res / 2 + err, res)
        lon, lat = np.meshgrid(lon, lat)
        area = np.cos(np.deg2rad(lat))

        '''calculate the basin averaged TWS, monthly'''
        directory = Path(dir_in)
        fns = sorted(os.listdir(directory))  # deterministic: first match in sorted order (e.g. 2009-06_0 before _1)
        monthlist = GeoMathKit.monthListByMonth(begin=month_begin, end=month_end)

        print()
        print('Start to pre-process GRACE to obtain signal over places of interest...')

        COV = {}
        for tile, tile_mask in mask.items():
            COV[tile] = []

        time_epoch = []
        for month in tqdm(monthlist, desc="Processing months"):

            '''search for the file'''
            fn = month.strftime('%Y-%m')
            tn = None
            for filename in fns:
                if fn in filename:
                    tn = filename
                    break

            if tn is None:
                continue

            time_epoch.append(fn)
            fn = directory / tn

            '''flip the data to be compatible with the mask'''
            tws_one_month = np.flipud(np.load(str(fn)))

            '''loop for the tile'''
            for tile, tile_mask in mask.items():
                basins_num = len(list(tile_mask.keys())) - 1
                '''loop for the subbasins'''
                vv = []
                for i in range(1, basins_num + 1):
                    a = tws_one_month[:, tile_mask[i]]
                    b = area[tile_mask[i]]
                    vv.append(np.sum(a * b, 1) / np.sum(b))

                '''transform into mm'''
                cov = np.cov(np.array(vv) * 1000)

                COV[tile].append(cov)

            pass

        '''save it into h5'''
        for tile, vv in COV.items():
            out_dir = Path(dir_out)
            hm = h5py.File(out_dir / ('%s_cov.hdf5' % tile), 'w')
            hm.create_dataset('data', data=np.array(vv))
            dt = h5py.special_dtype(vlen=str)
            hm.create_dataset('time_epoch', data=time_epoch, dtype=dt)
            hm.close()

        print('Finished: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        pass

    def grid_TWS(self, month_begin='2002-04', month_end='2002-04', dir_in=None, dir_out=None):
        """
        gridded TWS [mm] of every tile, monthly, from the gridded product in dir_in (one NetCDF)
        -> <dir_out>/<tile>_gridded_signal.hdf5
        """
        if dir_in is None or dir_out is None:
            raise ValueError('GRACE_global_preparation.grid_TWS: dir_in and dir_out must be given')
        mask = self.valid_mask['05']
        print('\nStart to pre-process GRACE to obtain signal over places of interest...')
        TWS = {tile: [] for tile in mask}
        time_epoch = []
        for m, tws_one_month in self._grids(month_begin, month_end, dir_in):
            time_epoch.append(m + '-15')
            for tile, tile_mask in mask.items():
                TWS[tile].append(tws_one_month[tile_mask[0]])

        for tile, vv in TWS.items():
            hm = h5py.File(Path(dir_out) / ('%s_gridded_signal.hdf5' % tile), 'w')
            hm.create_dataset(name='tws', data=np.array(vv))
            dt = h5py.special_dtype(vlen=str)
            hm.create_dataset('time_epoch', data=time_epoch, dtype=dt)
            hm.close()
        print('Finished: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

