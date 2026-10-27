"""
Block-mean upscaling of a model state vector to the GRACE grid (e.g. 0.1 deg -> 0.5 deg), and the matching
GRACE basin mask, for comparing a high-resolution model with GRACE on the same grid.

    us = upscaling().configure_model_state(globalmask_fn='<mask_global.h5>', upscaled_res=0.5, modelres=0.1)
    grid = us.get2D_model_state(state_1d)              # 2-D, upscaled_res
    us.configure_GRACE(shp_path='<units.shp>', res=0.5, land_mask_fn='<GlobalLandMaskForGRACE.hdf5>')
    obs2d = us.get2D_GRACE(obs_1d)
    us.extra_mask(shp_path='<units_without_overlap.shp>')   # optional: blank the cells outside this shapefile
"""
import tempfile
import warnings

import h5py
import numpy as np

from src_auxiliary.shp2mask import basin_shp_process


def _shp_mask(shp_path, res):
    """basin mask (all units together, north-up global grid) of a shapefile with column ID = 1..N"""
    bp = basin_shp_process(res=res, basin_name='upscaling', save_dir=tempfile.gettempdir())
    return bp.shp_to_mask(shp_path=str(shp_path)).mask[0]


class upscaling:
    """A simple block mean of model states, to the same resolution as GRACE"""

    def __init__(self):
        self.__extra_mask = None
        self.__gm = None
        self.__scale_factor = None
        self.bshp_mask = None

    def configure_model_state(self, globalmask_fn, upscaled_res=0.5, modelres=0.1, key='mask'):
        """
        globalmask_fn: h5 file with the global land mask of the model (dataset `key`, shape 180/modelres x
        360/modelres, north-up); the model state is a 1-D array over the True cells of this mask.
        """
        with h5py.File(globalmask_fn, 'r') as h:
            gm = h[key][:].astype(bool)
        assert np.shape(gm) == (int(round(180 / modelres)), int(round(360 / modelres))), np.shape(gm)
        self.__scale_factor = int(round(upscaled_res / modelres))
        self.__gm = gm
        return self

    def get2D_model_state(self, model_state):
        """1-D model state (high resolution) -> 2-D block mean at the coarse resolution; NaN where no land"""
        n = self.__scale_factor
        sample = np.full(np.shape(self.__gm), np.nan)
        sample[self.__gm] = model_state
        rows, cols = sample.shape
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)   # all-NaN blocks (ocean) give NaN
            new = np.nanmean(sample.reshape(rows // n, n, cols // n, n), axis=(1, 3))
        if self.__extra_mask is not None:
            new[~self.__extra_mask] = np.nan
        return new

    def configure_GRACE(self, shp_path, res=0.5, land_mask_fn=None):
        """
        Basin mask of the GRACE observations: the cells of the units in shp_path, optionally restricted to the
        GRACE land mask (land_mask_fn: GlobalLandMaskForGRACE.hdf5 with groups resolution_05 / resolution_1, stored
        south-up).
        """
        mf = _shp_mask(shp_path, res)
        if land_mask_fn is not None:
            group = {0.5: 'resolution_05', 1.0: 'resolution_1'}.get(round(float(res), 2))
            if group is None:
                raise ValueError('land mask only for res 0.5 or 1, got %s' % res)
            with h5py.File(land_mask_fn, 'r') as h:
                lm = np.flipud(h[group]['mask'][:])
            mf = mf * lm
        self.bshp_mask = mf.astype(bool)
        return self

    def get2D_GRACE(self, GRACE_obs):
        """1-D observation vector over the basin cells -> 2-D global grid"""
        sample = np.full(self.bshp_mask.shape, np.nan)
        sample[self.bshp_mask] = GRACE_obs
        if self.__extra_mask is not None:
            sample[~self.__extra_mask] = np.nan
        return sample

    def extra_mask(self, shp_path, res=0.5):
        """blank the cells outside shp_path (e.g. the tile without its overlapping margin) in both outputs"""
        self.__extra_mask = self.in_mask(shp_path, res).astype(bool)
        return self

    @staticmethod
    def in_mask(shp_path, res=0.5):
        """mask of the cells of shp_path on the global grid"""
        return _shp_mask(shp_path, res)
