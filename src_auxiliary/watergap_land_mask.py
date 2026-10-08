"""
WaterGAP land mask for the GRACE processing (replaces GlobalLandMaskForGRACE.hdf5).

The observation operator of the DA averages the model over the WaterGAP land cells of a unit (shp2mask / global_shp2mask,
0.5 deg); GRACE is averaged over the same cells, so the land mask applied in src_OBS/prepare_GRACE must be the WaterGAP
land, not the GRACE land mask (which contains Greenland/Antarctica and a different coastline).

  0.5 deg : land = finite continental area in Input_data/static_input/watergap_22e_continentalarea.nc (67 420 cells)
  1 deg   : a 1-deg cell is land when at least `n_min` of its four 0.5-deg cells are WaterGAP land (default 1: 'any',
            so that no 0.5-deg land cell is left without a 1-deg cell for the GRACE covariance)

Output: <out_dir>/WaterGAPLandMask.hdf5 with the layout of GlobalLandMaskForGRACE.hdf5 (groups 'resolution_05' and
'resolution_1', datasets lat (-90 -> 90), lon (-180 -> 180), mask (float 0/1, lat ascending)), so that
GRACE_preparation.configure_global_land_ocean_mask(fn=...) reads it unchanged. Attributes record the source and rule.
Written once and kept permanently under GRACE/global_mask of the external data.
"""
import sys
from pathlib import Path
import numpy as np
import h5py
import xarray as xr


def watergap_land_mask(continental_area_nc, out_dir, n_min=1):
    """
    continental_area_nc : watergap_22e_continentalarea.nc (0.5 deg, lat 90 -> -90)
    out_dir             : folder of WaterGAPLandMask.hdf5
    n_min               : minimum number of WaterGAP 0.5-deg land cells (of 4) for a 1-deg land cell
    returns the path of the written file
    """
    '''step-1: 0.5-deg land from the continental area, lat 90 -> -90 as the model'''
    ds = xr.open_dataset(continental_area_nc, decode_times=False)
    ca = ds['continentalarea'].squeeze()
    assert ca.dims == ('lat', 'lon') and ca.shape == (360, 720)
    ca = ca.sortby('lat', ascending=False).sortby('lon')
    land05 = np.isfinite(ca.values)

    '''step-2: 1-deg land: count of 0.5-deg land cells in each 1-deg cell'''
    n_land = land05.reshape(180, 2, 360, 2).sum(axis=(1, 3))
    land1 = n_land >= n_min

    '''step-3: write with the layout of GlobalLandMaskForGRACE.hdf5 (lat ascending)'''
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = out_dir / 'WaterGAPLandMask.hdf5'
    with h5py.File(fn, 'w') as f:
        for name, res, land in (('resolution_05', 0.5, land05), ('resolution_1', 1.0, land1)):
            g = f.create_group(name)
            g.create_dataset('lat', data=np.arange(-90 + res / 2, 90, res))
            g.create_dataset('lon', data=np.arange(-180 + res / 2, 180, res))
            g.create_dataset('mask', data=np.flipud(land).astype(float))
        f.attrs['source'] = 'WaterGAP 2.2e continental area: %s' % Path(continental_area_nc).name
        f.attrs['rule_1deg'] = 'land if >= %d of the four 0.5-deg cells are WaterGAP land' % n_min
        f.attrs['n_land_05'] = int(land05.sum())
        f.attrs['n_land_1'] = int(land1.sum())
    print('%s: %d land cells at 0.5 deg, %d at 1 deg (rule: >= %d of 4)' % (fn, land05.sum(), land1.sum(), n_min))
    return fn


def demo1():
    """write the permanent WaterGAP land mask into GRACE/global_mask of the external data"""
    EXT = Path('/media/user/My Book/Fan/PyGLDA_v2_external_data')
    watergap_land_mask(continental_area_nc=EXT / 'Input_data/static_input/watergap_22e_continentalarea.nc',
                       out_dir=EXT / 'GRACE/global_mask')
    pass


if __name__ == '__main__':
    demo1()
