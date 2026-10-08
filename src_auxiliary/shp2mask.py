import sys

sys.path.append('../')

import os
import numpy as np
from pathlib import Path
import h5py
import xarray as xr


class basin_shp_process:
    """
    Rasterise a (sub-)basin shapefile (column ID = 1..N) to the 0.5 or 1 degree grid.
    rule (class default `default_rule`, or argument of shp_to_mask):
      'area'   : (default) a cell belongs to the sub-basin that covers the largest part of it, provided the whole
                 basin covers at least `area_threshold` of the cell (default 0.4: basin area within ~1 %, the
                 per-sub-basin areas within ~4 %, for the Danube); sub-basins never overlap
      'centre' : a cell belongs to the sub-basin whose polygon contains the cell centre (classic rule used for
                 the Amazon/demo_3 masks; about half of the cut boundary cells are dropped)
    Both rules are vectorised with shapely 2 (_cover: one STRtree query per sub-basin, cells entirely inside a
    polygon get cover 1 without clipping); the masks are identical to the earlier cell-by-cell loops (checked for
    the Danube and the Amazon, with and without configureBox). For shapefiles with hundreds of units on the global
    grid use global_shp2mask.py (same rules and file format, plus model land / removed-unit options).
    """
    default_rule = 'area'
    area_threshold = 0.4

    def __init__(self, res, basin_name='MDB', save_dir='../data/basin/mask'):
        self._res = res
        self._basin_name = basin_name
        self.save_dir = Path(save_dir) / basin_name

        self.save_dir.mkdir(parents=False, exist_ok=True)

        self.collect_mask = None
        self.NaN_mask = None

        self.box_mask = None
        pass

    def configureBox(self, box: dir):
        """
        box = {
        "lat": [
            -9.9,
            -43.8
        ],
        "lon": [
            112.4,
            154.3
        ]}
        """
        if box is None:
            return self

        res = self._res
        err = res / 10
        lat = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
        lon = np.arange(-180 + res / 2, 180 - res / 2 + err, res)

        lati = [np.argmin(np.fabs(lat - max(box['lat']))),
                np.argmin(np.fabs(lat - min(box['lat'])))]
        loni = [np.argmin(np.fabs(lon - min(box['lon']))),
                np.argmin(np.fabs(lon - max(box['lon'])))]
        id = [lati[0], lati[1] + 1, loni[0], loni[1] + 1]

        lon, lat = np.meshgrid(lon, lat)
        mask = np.zeros(np.shape(lon))

        mask[id[0]:id[1], id[2]:id[3]] = 1

        self.box_mask = mask

        return self

    def shp_to_mask(self, shp_path='../data/basin/shp/MDB_4_shapefiles/MDB_4_subbasins.shp', issave=False, rule=None):
        import geopandas as gpd
        rule = (rule or self.default_rule).lower()
        assert rule in ('centre', 'center', 'area'), "rule must be 'centre' or 'area'"
        gdf = gpd.read_file(shp_path)
        if rule == 'area':
            mask_basin = self._masks_by_area(gdf)
        else:
            mask_basin = self._masks_by_centre(gdf)

        mask_all = np.zeros(mask_basin[1].shape, dtype=bool)
        for id in mask_basin:
            mask_all |= mask_basin[id]
        if issave:
            h5fn = str(self.save_dir / ('%s_res_%s.h5' % (self._basin_name, self._res)))
            with h5py.File(h5fn, 'w') as hf:
                for id in mask_basin:
                    hf.create_dataset('sub_basin_%s' % id, data=mask_basin[id].astype(int))
                hf.create_dataset('basin', data=mask_all.astype(int))
                hf.attrs['rule'] = rule
        mask_basin[0] = mask_all
        self.mask = mask_basin
        return self

    def _grid(self):
        res = self._res
        err = res / 10
        lat = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
        lon = np.arange(-180 + res / 2, 180 - res / 2 + err, res)
        return lat, lon

    def _masks_by_centre(self, gdf):
        """classic rule: the cell centre is inside (or on the boundary of) the sub-basin polygon"""
        import shapely
        lat1, lon1 = self._grid()
        lon, lat = np.meshgrid(lon1, lat1)
        mask_basin = {}
        for id in np.arange(gdf.ID.size) + 1:
            bd1 = gdf[gdf.ID == id]
            bb = bd1.total_bounds
            crop_box = {"lat": [bb[3], bb[1]], "lon": [bb[0], bb[2]]}
            sub_lat, sub_lon, box_mask_id = self._crop_box(crop_box=crop_box)
            inside = shapely.intersects_xy(bd1.geometry.item(), sub_lon, sub_lat)     # contains or touches
            mask_gl = np.full_like(lat, fill_value=0, dtype=bool)
            mask_gl[box_mask_id[0]:box_mask_id[1], box_mask_id[2]:box_mask_id[3]] = inside
            if self.box_mask is not None:
                mask_gl = (mask_gl * self.box_mask).astype(bool)
            mask_basin[int(bd1.ID.values[0])] = mask_gl
        return mask_basin

    def _cover(self, gdf):
        """
        cover of the cells in the bounding box of gdf by every sub-basin polygon (intersection area / cell area, in
        degrees), computed for all cells at once (shapely 2 vectorised; one STRtree query per polygon instead of a
        Python loop over cells x sub-basins). Returns rows, cols (grid indices of the cells), the cell boxes and
        their tree, ids, best (largest cover), win (its sub-basin ID; first in the shapefile order on ties) and total
        """
        import shapely
        res = self._res
        lat1, lon1 = self._grid()
        bb = gdf.total_bounds
        rows = np.where((lat1 + res / 2 > bb[1]) & (lat1 - res / 2 < bb[3]))[0]
        cols = np.where((lon1 + res / 2 > bb[0]) & (lon1 - res / 2 < bb[2]))[0]
        R, C = np.meshgrid(rows, cols, indexing='ij')
        R, C = R.ravel(), C.ravel()
        cells = shapely.box(lon1[C] - res / 2, lat1[R] - res / 2, lon1[C] + res / 2, lat1[R] + res / 2)
        tree = shapely.STRtree(cells)
        ids = [int(i) for i in gdf.ID.values]
        best, win, total = np.zeros(len(R)), np.zeros(len(R), dtype=np.int64), np.zeros(len(R))
        for id, g in zip(ids, gdf.geometry.values):
            shapely.prepare(g)
            idx = tree.query(g, predicate='intersects')
            full = shapely.contains(g, cells[idx])                    # cells entirely inside: cover 1 (no clipping)
            a = np.ones(len(idx))
            a[~full] = shapely.area(shapely.intersection(cells[idx[~full]], g)) / res ** 2
            total[idx] += a
            m = a > best[idx]
            best[idx[m]], win[idx[m]] = a[m], id
        return R, C, cells, tree, ids, best, win, total

    def _masks_by_area(self, gdf):
        """area rule: coverage of every cell by every sub-basin polygon; the cell goes to the sub-basin with the
        largest coverage if the basin as a whole covers >= area_threshold of the cell"""
        lat1, lon1 = self._grid()
        R, C, cells, tree, ids, best, win, total = self._cover(gdf)
        inside = total >= self.area_threshold               # coverage by the whole basin (sub-basins do not overlap)
        mask_basin = {}
        for id in ids:
            m = np.zeros((len(lat1), len(lon1)), dtype=bool)
            sel = inside & (win == id)
            m[R[sel], C[sel]] = True
            if self.box_mask is not None:
                m = (m * self.box_mask).astype(bool)
            mask_basin[id] = m
        return mask_basin

    def plot_mask(self, shp_path=None, fn=None, title=None, allow_pop_up=True, dpi=200):
        """
        Matplotlib map of the rasterised mask (call after shp_to_mask): one colour per sub-basin on the model
        grid with the cell edges drawn, the polygons of the shapefile in black, an ID badge per sub-basin and a
        legend (ID, name, cells, area of the mask) to the right of the map.
        fn: output name without extension (.png and .pdf are written); None = no file.
        """
        import matplotlib
        if not allow_pop_up:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
        from matplotlib.patches import Patch
        import geopandas as gpd

        plt.rcParams.update({'font.size': 12, 'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
                             'axes.linewidth': 1.8, 'xtick.major.width': 1.6, 'ytick.major.width': 1.6})
        res = self._res
        lat1, lon1 = self._grid()
        ids = sorted(k for k in self.mask if k != 0)
        lab = np.zeros(self.mask[0].shape, int)
        for k in ids:
            lab[self.mask[k]] = k
        r = np.where(self.mask[0].any(1))[0]; c = np.where(self.mask[0].any(0))[0]
        r0, r1 = max(r[0] - 2, 0), min(r[-1] + 3, len(lat1)); c0, c1 = max(c[0] - 2, 0), min(c[-1] + 3, len(lon1))
        X = np.append(lon1[c0:c1] - res / 2, lon1[c1 - 1] + res / 2)          # cell edges
        Y = np.append(lat1[r0:r1] + res / 2, lat1[r1 - 1] - res / 2)
        wcell = (111.2 * res) ** 2 * np.cos(np.deg2rad(lat1))[:, None]          # km2 per cell
        colors = ['steelblue', 'seagreen', 'plum', 'pink', 'khaki', 'turquoise', 'salmon', 'gold', 'orchid',
                  'lightcoral', 'palegreen', 'skyblue', 'tan', 'lightpink', 'yellowgreen', 'thistle', 'wheat', 'aquamarine']
        cols = [colors[(k - 1) % len(colors)] for k in ids]
        cmap = ListedColormap(['white'] + cols); norm = BoundaryNorm(np.arange(-0.5, len(ids) + 1.5, 1), len(ids) + 1)
        gdf = gpd.read_file(shp_path) if shp_path is not None else None
        names = {int(rw.ID): str(rw.NAME).split(' (')[0] for _, rw in gdf.iterrows()} if gdf is not None and 'NAME' in gdf else {}

        fig, ax = plt.subplots(figsize=(16, 7.5))
        ax.pcolormesh(X, Y, lab[r0:r1, c0:c1], cmap=cmap, norm=norm, edgecolors='0.75', linewidth=0.4, shading='flat')
        if gdf is not None:
            gdf.boundary.plot(ax=ax, color='k', lw=1.5)
            for _, rw in gdf.iterrows():
                p = rw.geometry.representative_point()
                ax.text(p.x, p.y, str(rw.ID), ha='center', va='center', fontsize=14, fontweight='bold',
                        bbox=dict(boxstyle='circle,pad=0.3', fc='w', ec='k', lw=1.2))
        handles = [Patch(fc=cols[i], ec='k', label='%d  %s\n     %d cells, %.0f x10$^3$ km$^2$'
                         % (k, names.get(k, 'sub-basin %d' % k), self.mask[k].sum(), (self.mask[k] * wcell).sum() / 1e3))
                   for i, k in enumerate(ids)]
        handles.append(Patch(fc='none', ec='none', label='total: %d cells, %.0f x10$^3$ km$^2$'
                             % (self.mask[0].sum(), (self.mask[0] * wcell).sum() / 1e3)))
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=10, frameon=False,
                  title='sub-basins (ID, cells, mask area)', title_fontsize=11, labelspacing=1.0)
        lat0 = 0.5 * (Y[0] + Y[-1])
        ax.set_xticks(np.arange(np.ceil(X[0]), X[-1] + 0.01, 2 if X[-1] - X[0] > 15 else 1))
        ax.set_yticks(np.arange(np.ceil(Y[-1]), Y[0] + 0.01, 1 if Y[0] - Y[-1] < 15 else 2))
        ax.set_xlim(X[0], X[-1]); ax.set_ylim(Y[-1], Y[0]); ax.set_aspect(1 / np.cos(np.deg2rad(lat0)))
        ax.set_xlabel('longitude [deg]'); ax.set_ylabel('latitude [deg]')
        ax.set_title(title or '%s %s-degree model grid: %d cells in %d sub-basins (rule %s, threshold %.2f)\nblack: shapefile polygons'
                     % (self._basin_name, res, int(self.mask[0].sum()), len(ids), self.default_rule, self.area_threshold))
        fig.tight_layout()
        if fn is not None:
            Path(fn).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(fn) + '.png', dpi=dpi, bbox_inches='tight')
            fig.savefig(str(fn) + '.pdf', bbox_inches='tight')
            print('figure', str(fn) + '.png/.pdf')
        if allow_pop_up:
            plt.show()
        return fig

    def mask_to_vec(self, external_mask=None,
                    model_mask_global='/media/user/My Book/Fan/W3RA_data/crop_input/test/mask/mask_global.h5'):

        basin_name = self._basin_name
        res = self._res

        err = res / 10
        lat = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
        lon = np.arange(-180 + res / 2, 180 - res / 2 + err, res)

        lon, lat = np.meshgrid(lon, lat)

        if external_mask is None:
            this_mask = self.mask
        else:
            this_mask = external_mask

        """
        get 1-d vector of the mask of interest
        """

        ff = h5py.File(model_mask_global, 'r')
        model_mask = ff['mask'][:]

        mask_2D = {}
        '''overlap the basin masks'''
        for k, v in this_mask.items():
            mask_2D['basin_%s' % k] = (v * model_mask).astype(bool)

        '''mask 1D inside the land'''
        mask_1D = {}
        # mask_1D['land'] = model_mask[model_mask.astype(bool)]
        for k, v in mask_2D.items():
            mask_1D[k] = (v.astype(int))[model_mask.astype(bool)]
            mask_1D[k] = mask_1D[k].astype(bool)

        self.collect_mask = (mask_2D, mask_1D, lat, lon, res)
        return self

    def mask_to_vec_for_WaterGap(self, external_mask):


        pass

    def mask_nan(self, sample='/media/user/My Book/Fan/W3RA_data/states_sample/state.h5'):
        """
        this is done for masking out the NaN values in states. And to do this work, one sample state file is necessary.
        NaN likely exists because of the mismatch between forcing field and model parameters/mask.
        """

        h5_fn = h5py.File(Path(sample), 'r')
        sample = h5_fn['FreeWater'][0, :]

        NaN_mask = (1 - np.isnan(sample)).astype(bool)

        mask_2D, mask_1D, lat, lon, res = self.collect_mask

        for k, v in mask_2D.items():
            mask_2D[k][mask_2D[k]] = NaN_mask[mask_1D[k]]
            pass

        for k, v in mask_1D.items():
            mask_1D[k] = mask_1D[k] * NaN_mask

        self.collect_mask = (mask_2D, mask_1D, lat, lon, res)

        return self

    def _crop_box(self, crop_box: dir):
        res = self._res
        err = res / 10
        lat = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
        lon = np.arange(-180 + res / 2, 180 - res / 2 + err, res)

        lati = [np.argmin(np.fabs(lat - max(crop_box['lat']))),
                np.argmin(np.fabs(lat - min(crop_box['lat'])))]
        loni = [np.argmin(np.fabs(lon - min(crop_box['lon']))),
                np.argmin(np.fabs(lon - max(crop_box['lon'])))]
        box_mask_id = [lati[0], lati[1] + 1, loni[0], loni[1] + 1]

        '''to extend the box a little bit to avoid potential missing point'''
        box_mask_id = [lati[0]-2, lati[1] + 3, loni[0]-2, loni[1] + 3]

        lon, lat = np.meshgrid(lon, lat)
        # box_mask = np.zeros(np.shape(lon))
        #
        # box_mask[box_mask_id[0]:box_mask_id[1], box_mask_id[2]:box_mask_id[3]] = 1

        lon = lon[box_mask_id[0]:box_mask_id[1], box_mask_id[2]:box_mask_id[3]]
        lat = lat[box_mask_id[0]:box_mask_id[1], box_mask_id[2]:box_mask_id[3]]

        return lat, lon, box_mask_id


_MASK_CACHE = {}


def load_mask(mask_path: dir):
    """
    Basin mask of the DA (WaterGAP grid, 0.5 degree) -> (box_crop, local_mask)
      box_crop   : lat/lon edges of the regional box (the basin bounding box; the whole grid for a global unit
                   mask, attribute extent = 'global')
      local_mask : 'basin', 'sub_basin_<k>' (bool, one value per basin cell in row-major order of the global grid),
                   'unit_id' when the file has it, 'lat' of the basin cells, 'basin_num', 'unit_label' (index of
                   the sub-basin of every basin cell, 0-based, -1 = none), 'basin_2d' (the box, int),
                   'global_2d' (the global grid, int)
    The datasets are read one at a time (the global file has 773 full-grid int64 datasets: 1.6 GB when read at
    once), the sub-basin vectors are kept as bool (772 x 55 461 cells: 43 MB instead of 343 MB), and the result
    of the last file is cached: the DA constructs ExtractStates, the design matrices and the threshold from the
    same file on every rank (until 8 Oct 2026 six reads of ~7 s and four copies per rank). Callers never modify
    the dictionary.
    """
    mask_path = str(mask_path)
    key = (mask_path, os.path.getmtime(mask_path))
    if key in _MASK_CACHE:
        return _MASK_CACHE[key]

    res = 0.5
    err = res / 10
    lat_coords = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
    lon_coords = np.arange(-180 + res / 2, 180 - res / 2 + err, res)

    local_mask = {}
    with h5py.File(name=mask_path, mode='r') as f:
        basin = f['basin'][()]
        basin_bool = basin.astype(bool)
        is_global = f.attrs.get('extent') == 'global'
        '''one dataset at a time: reduced to the basin cells immediately'''
        keys = sorted((k for k in f.keys() if k.startswith('sub_basin_')), key=lambda k: int(k.split('_')[-1]))
        local_mask['basin'] = basin[basin_bool]
        for k in keys:
            local_mask[k] = f[k][()][basin_bool].astype(bool)
        if 'unit_id' in f:
            local_mask['unit_id'] = f['unit_id'][()][basin_bool]

    basin_mask = xr.DataArray(basin_bool, coords={"lat": lat_coords, "lon": lon_coords}, dims=["lat", "lon"],
                              name="region_mask")
    mask_bbox = basin_mask.where(basin_mask, drop=True)
    # Extract the exact min/max coordinates of the target region
    lat_min, lat_max = float(mask_bbox.lat.min()), float(mask_bbox.lat.max())
    lon_min, lon_max = float(mask_bbox.lon.min()), float(mask_bbox.lon.max())

    '''a global unit mask (attribute extent = 'global', written by global_shp2mask) is not cropped at all: the DA
    output then has the full grid of the open loop (regional masks carry no such attribute and are unaffected)'''
    if is_global:
        lat_min, lat_max = float(lat_coords.min()), float(lat_coords.max())
        lon_min, lon_max = float(lon_coords.min()), float(lon_coords.max())

    box_crop = {'lat_min': lat_min,
                'lat_max': lat_max,
                'lon_min': lon_min,
                'lon_max': lon_max
                }  # with this it is possible to crop the global nc file within a small box to reduce the datasize.

    lon_mesh_global, lat_mesh_global = np.meshgrid(lon_coords, lat_coords)
    local_mask['lat'] = lat_mesh_global[basin_bool]
    local_mask['basin_num'] = len(keys)

    '''sub-basin of every basin cell (the last one wins should a cell be in two)'''
    label = np.full(int(basin_bool.sum()), -1, dtype=np.int32)
    for k in keys:
        label[local_mask[k]] = int(k.split('_')[-1]) - 1
    local_mask['unit_label'] = label

    '''regional box 2D --> basin grid 1D: every cell between the box edges, as in the cropped daily files of the DA.
    (where(drop=True) would also remove an empty row or column INSIDE the box, e.g. an ocean column between two land
    masses of a global mask, and the array would no longer fit the daily file; identical for a compact basin)'''
    local_mask['basin_2d'] = basin_mask.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max)).values.astype(int)

    local_mask['global_2d'] = basin                   # global 2D ---> basin grid 1D
    _MASK_CACHE.clear()
    _MASK_CACHE[key] = (box_crop, local_mask)
    return box_crop, local_mask


def unit_weights(mask_path, lat, lon, include_basin=True):
    """
    Sparse averaging matrix of a basin mask on the grid (lat, lon) of a 0.5-degree file: one row per key ('basin' first
    if include_basin, then 'sub_basin_1', 'sub_basin_2', ...), cos(lat) weights of the key's cells normalised to 1, so
    that  W @ field.reshape(-1)  gives the area-weighted mean of every key in one product (W @ field2d.reshape(T, -1).T
    for a time series). The file grid may be the global grid or a box cropped from it (the regional daily output); a key
    without a cell on the grid gets an all-zero row. A NaN cell inside a key gives NaN for that key, as a plain mean would.
    Used by statistical_analysis.BasinAverageAnalysis (regional) and Global_DA.collect_and_statistics (global).
    returns W (scipy.sparse.csr_matrix, n_key x n_lat*n_lon), keys (list)
    """
    from scipy import sparse
    res = 0.5
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    i = np.rint((90 - res / 2 - lat) / res).astype(int)                 # rows of the global grid
    j = np.rint((lon - (-180 + res / 2)) / res).astype(int)             # columns of the global grid
    rows, cols, vals = [], [], []
    with h5py.File(name=mask_path, mode='r') as f:
        ids = sorted(int(k.split('_')[-1]) for k in f.keys() if k.startswith('sub_basin_'))
        keys = (['basin'] if include_basin else []) + ['sub_basin_%d' % k for k in ids]
        for r, key in enumerate(keys):
            m = f[key][()][np.ix_(i, j)].astype(bool)                   # the key on the grid of the file
            ii, jj = np.nonzero(m)
            if len(ii) == 0:
                continue
            w = np.cos(np.deg2rad(lat[ii]))
            rows.append(np.full(len(ii), r))
            cols.append(ii * len(lon) + jj)
            vals.append(w / w.sum())
    if rows:
        rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    W = sparse.csr_matrix((vals, (rows, cols)), shape=(len(keys), len(lat) * len(lon)))
    return W, keys


def mask_05_to_1(mask05_path, out_path=None, land_05=None, verbose=True):
    """
    1-degree mask for the GRACE covariance (src_OBS.prepare_GRACE.basin_COV) derived from the 0.5-degree DA mask, so that
    both resolutions describe the same cells (instead of rasterising the shapefile a second time at 1 degree):
      - a 1-degree cell belongs to the sub-basin that owns most of its four 0.5-degree cells (land cells only when
        `land_05` [360 x 720, lat 90 -> -90, bool] is given, e.g. the WaterGAP land of GRACE/global_mask/
        WaterGAPLandMask.hdf5); a tie goes to the sub-basin with the smaller 0.5-degree area
      - a sub-basin that wins no 1-degree cell keeps the cell where it has most 0.5-degree cells (that cell is then in
        two sub-basins; reported)
    Output <out_path> (default: <basin>_res_1.h5 next to the input) in the shp2mask layout: 'basin', 'sub_basin_<k>'
    (int, 180 x 360, lat 90 -> -90), attributes of the 0.5-degree file plus 'derived_from'.
    returns out_path
    """
    mask05_path = Path(mask05_path)
    out_path = Path(out_path) if out_path is not None else mask05_path.with_name(
        mask05_path.name.replace('_res_0.5.h5', '_res_1.h5'))
    with h5py.File(mask05_path, 'r') as f:
        ids = sorted(int(k.split('_')[-1]) for k in f.keys() if k.startswith('sub_basin_'))
        attrs = dict(f.attrs)
        '''step-1: count of 0.5-degree (land) cells of every sub-basin in every 1-degree cell'''
        count = np.zeros((len(ids), 180, 360), dtype=np.int16)
        area05 = np.zeros(len(ids), dtype=np.int64)
        for r, k in enumerate(ids):
            m = f['sub_basin_%d' % k][()].astype(bool)
            if land_05 is not None:
                m &= np.asarray(land_05, dtype=bool)
            area05[r] = m.sum()
            count[r] = m.reshape(180, 2, 360, 2).sum(axis=(1, 3))

    '''step-2: majority rule, ties to the smaller sub-basin (stable sort on area, argmax takes the first maximum)'''
    order = np.argsort(area05, kind='stable')
    best = np.argmax(count[order], axis=0)                                  # index into `order`
    owner = np.where(count.sum(axis=0) > 0, order[best], -1)                # (180, 360), -1 = no cell
    masks1 = {k: owner == r for r, k in enumerate(ids)}

    '''step-3: a sub-basin without a 1-degree cell keeps its best cell'''
    rescued = []
    for r, k in enumerate(ids):
        if not masks1[k].any() and area05[r] > 0:
            i, j = np.unravel_index(np.argmax(count[r]), count[r].shape)
            masks1[k][i, j] = True
            rescued.append(k)

    '''step-4: write in the shp2mask layout'''
    basin1 = np.zeros((180, 360), dtype=bool)
    for k in ids:
        basin1 |= masks1[k]
    with h5py.File(out_path, 'w') as hf:
        for k in ids:
            hf.create_dataset('sub_basin_%d' % k, data=masks1[k].astype(int), compression='gzip')
        hf.create_dataset('basin', data=basin1.astype(int), compression='gzip')
        for a, v in attrs.items():
            hf.attrs[a] = v
        hf.attrs['derived_from'] = '%s, majority of the 0.5-degree%s cells, ties to the smaller sub-basin' % (
            mask05_path.name, ' land' if land_05 is not None else '')
        hf.attrs['n_empty_rescued'] = len(rescued)
    if verbose:
        print('written %s: %d sub-basins, %d cells at 1 degree from %d at 0.5 degree%s'
              % (out_path, len(ids), basin1.sum(), area05.sum(),
                 '; %d sub-basin(s) without a majority cell kept their best cell: %s' % (len(rescued), rescued)
                 if rescued else ''))
    return out_path


def demo_Danube():
    """rasterise the HydroBASINS Danube shapefile with the default rule and plot the grid"""
    base = Path('/media/user/My Book/Fan/PyGLDA_v2_external_data')
    shp = base / 'Basin/shp/Danube/Danube.shp'
    bs = basin_shp_process(res=0.5, basin_name='Danube', save_dir=base / 'Basin/mask').shp_to_mask(shp_path=str(shp), issave=False)
    bs.plot_mask(shp_path=str(shp), fn=base / 'Extra/fig/Danube_grid_0.5', title='Danube 0.5-degree model grid')


def demo1():
    basin_shp = basin_shp_process(save_dir='/media/user/My Book/Fan/WaterGap/Basin/mask',
                                  basin_name='Brahmaputra', res=0.5)
    basin_shp.shp_to_mask(shp_path='/media/user/My Book/Fan/WaterGap/Basin/shp/Brahmaputra/Brahmaputra.shp', issave=True)

    pass

def demo2():
    box_crop, local_mask = load_mask(mask_path='/media/user/My Book/Fan/WaterGap/Basin/mask/Brahmaputra/Brahmaputra_res_0.5.h5')
    pass

if __name__ == '__main__':
    # demo1()
    # demo2()
    demo_Danube()
