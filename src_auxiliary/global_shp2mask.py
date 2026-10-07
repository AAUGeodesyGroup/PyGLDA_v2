"""
Rasterise a (global) sub-basin shapefile to the model grid - the same rules, output and reader as shp2mask.py, for
shapefiles with hundreds of units (e.g. Basin/shp/Global/GlobalBasins.shp from HydroShed_global.py).

global_shp_process extends shp2mask.basin_shp_process (same rules via shp2mask._cover, same file format, same
reader) with the options a global run needs:
  rule 'area'   : (default) a cell belongs to the sub-basin that covers the largest part of it, provided the whole
                  basin covers at least area_threshold of the cell (default 0.4, as shp2mask)
  rule 'centre' : a cell belongs to the sub-basin whose polygon contains (or touches) the cell centre
  output        : <save_dir>/<basin_name>/<basin_name>_res_<res>.h5 with 'basin' and 'sub_basin_<ID>' (int), attrs
                  'rule'; gzip-compressed (600+ layers of mostly zeros); read with shp2mask.load_mask
  mask_to_vec   : inherited from shp2mask
demo1 checks that both codes give identical masks for the Danube.

Global options (both off by default, so the result equals shp2mask):
  configure_model_mask : keep only the cells of the model land (WaterGAP: finite continental area)
  configure_exclude    : a cell whose largest cover is a removed unit (GlobalBasins_removed.shp: islands, glaciers,
                         lakes) stays outside every sub-basin, instead of going to the neighbouring unit
  write_label          : also write 'unit_id' [lat, lon] (sub-basin ID, 0 = none); off by default because readers that
                         loop over all keys of the file expect only 'basin' and 'sub_basin_<ID>'
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_auxiliary.shp2mask import basin_shp_process, load_mask


class global_shp_process(basin_shp_process):
    default_rule = 'area'
    area_threshold = 0.4

    def __init__(self, res=0.5, basin_name='GlobalBasins', save_dir='../data/basin/mask'):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        super().__init__(res=res, basin_name=basin_name, save_dir=save_dir)
        self.model_mask = None
        self.exclude_shp = None
        self.write_label = False
        self.label = None
        pass

    # ------------------------------------------------------------------ configuration
    def configure_threshold(self, area_threshold=0.4):
        """'area' rule: minimum cover of a cell by the whole basin (0.4 = shp2mask; > 0 for any touched cell)"""
        self.area_threshold = area_threshold
        return self

    def configure_model_mask(self, continental_area_nc=None, mask=None):
        """
        restrict the sub-basins to the model land: mask[lat, lon] (bool), or the WaterGAP continental area file
        (Input_data/static_input/watergap_22e_continentalarea.nc; land = finite values)
        """
        if mask is None:
            import xarray as xr
            with xr.open_dataset(continental_area_nc, decode_times=False) as d:
                v = list(d.data_vars)[0]
                mask = np.isfinite(np.asarray(d[v]).squeeze())
        lat, lon = self._grid()
        assert mask.shape == (len(lat), len(lon)), 'model mask %s does not match the %s-degree grid' % (mask.shape,
                                                                                                      self._res)
        self.model_mask = np.asarray(mask, dtype=bool)
        return self

    def configure_exclude(self, shp_path):
        """polygons of removed units: cells mostly covered by them are left out (see module doc)"""
        self.exclude_shp = shp_path
        return self

    def configure_label(self, write_label=True):
        self.write_label = write_label
        return self

    # ------------------------------------------------------------------ rasterise
    def shp_to_mask(self, shp_path, issave=False, rule=None):
        import h5py
        import geopandas as gpd
        rule = (rule or self.default_rule).lower()
        assert rule in ('centre', 'center', 'area'), "rule must be 'centre' or 'area'"
        gdf = gpd.read_file(shp_path)

        '''step-1: sub-basin masks with the shp2mask rules (vectorised)'''
        if rule == 'area':
            mask_basin = self._masks_by_area(gdf)
        else:
            mask_basin = self._masks_by_centre(gdf)

        '''step-2: optional model land'''
        if self.model_mask is not None:
            for k in mask_basin:
                mask_basin[k] = mask_basin[k] & self.model_mask

        mask_all = np.zeros(next(iter(mask_basin.values())).shape, dtype=bool)
        label = np.zeros(mask_all.shape, dtype=np.int32)
        for k in mask_basin:
            mask_all |= mask_basin[k]
            label[mask_basin[k]] = k
        self.label = label

        '''step-3: save in the shp2mask format'''
        if issave:
            h5fn = str(self.save_dir / ('%s_res_%s.h5' % (self._basin_name, self._res)))
            with h5py.File(h5fn, 'w') as hf:
                for k in mask_basin:
                    hf.create_dataset('sub_basin_%s' % k, data=mask_basin[k].astype(int), compression='gzip')
                hf.create_dataset('basin', data=mask_all.astype(int), compression='gzip')
                if self.write_label:
                    hf.create_dataset('unit_id', data=label, compression='gzip')
                hf.attrs['rule'] = rule
                if rule == 'area':
                    hf.attrs['area_threshold'] = self.area_threshold
                hf.attrs['model_mask'] = int(self.model_mask is not None)
                hf.attrs['exclude'] = str(self.exclude_shp) if self.exclude_shp is not None else ''
            print('written %s: %d sub-basins, %d cells (rule %s%s%s%s)'
                  % (h5fn, len(mask_basin), mask_all.sum(), rule,
                     ', threshold %.2f' % self.area_threshold if rule == 'area' else '',
                     ', model land' if self.model_mask is not None else '',
                     ', removed units excluded' if self.exclude_shp is not None else ''))
        mask_basin[0] = mask_all
        self.mask = mask_basin
        return self

    def _masks_by_area(self, gdf):
        """
        shp2mask area rule (shp2mask._cover: the cell goes to the sub-basin with the largest cover if the basin
        covers >= area_threshold of it); with configure_exclude, a cell whose largest cover is a removed unit stays out
        """
        import shapely
        lat1, lon1 = self._grid()
        R, C, cells, tree, ids, best, win, total = self._cover(gdf)
        keep = total >= self.area_threshold

        if self.exclude_shp is not None:
            import geopandas as gpd
            ex = gpd.read_file(self.exclude_shp)
            ex_best = np.zeros(len(R))
            for g in ex.geometry.values:
                if g is None or g.is_empty:
                    continue
                g = shapely.make_valid(g)
                idx = tree.query(g, predicate='intersects')
                ex_best[idx] = np.maximum(ex_best[idx], shapely.area(shapely.intersection(cells[idx], g)) / self._res ** 2)
            dropped = keep & (ex_best > best)
            keep &= ~dropped
            print('  %d cells left out: mostly covered by removed units (%s)' % (dropped.sum(), Path(self.exclude_shp).name))

        mask_basin = {}
        for id in ids:
            m = np.zeros((len(lat1), len(lon1)), dtype=bool)
            sel = keep & (win == id)
            m[R[sel], C[sel]] = True
            if self.box_mask is not None:
                m = (m * self.box_mask).astype(bool)
            mask_basin[id] = m
        return mask_basin

    def _masks_by_centre(self, gdf):
        """shp2mask centre rule (cell centre inside or on the boundary of the polygon), without the +-2 cell crop box
        of shp2mask, which wraps at the date line for global units"""
        import shapely
        lat1, lon1 = self._grid()
        lon, lat = np.meshgrid(lon1, lat1)
        mask_basin = {}
        for k, (id, g) in enumerate(zip(gdf.ID.values, gdf.geometry.values)):
            x0, y0, x1, y1 = g.bounds
            m = (lon >= x0 - self._res) & (lon <= x1 + self._res) & (lat >= y0 - self._res) & (lat <= y1 + self._res)
            inside = shapely.intersects_xy(g, lon[m], lat[m])          # contains or touches
            mg = np.zeros(lon.shape, dtype=bool)
            mg.ravel()[np.flatnonzero(m.ravel())[inside]] = True
            if self.box_mask is not None:
                mg = (mg * self.box_mask).astype(bool)
            mask_basin[int(id)] = mg
        return mask_basin

    # ------------------------------------------------------------------ figure
    def plot_mask(self, shp_path=None, fn=None, title=None, allow_pop_up=True, dpi=200):
        """global map of the mask: one colour per sub-basin, grey = model land outside the sub-basins (with
        configure_model_mask); for a regional shapefile the shp2mask figure is used"""
        if len(self.mask) - 1 <= 30:
            return super().plot_mask(shp_path=shp_path, fn=fn, title=title, allow_pop_up=allow_pop_up, dpi=dpi)
        import matplotlib
        if not allow_pop_up:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        n = int(self.label.max())
        pal = np.delete(plt.cm.tab20(np.arange(20)), [14, 15], axis=0)       # no greys
        cols = pal[np.random.default_rng(3).permutation(n) % len(pal)]
        cmap = ListedColormap(np.vstack([[0.82, 0.84, 0.84, 1.0], cols]))
        land = self.model_mask if self.model_mask is not None else self.label > 0
        img = np.where(land, self.label, np.nan).astype(float)
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.imshow(img, extent=(-180, 180, -90, 90), cmap=cmap, vmin=-0.5, vmax=n + 0.5, interpolation='nearest')
        if shp_path is not None:
            import geopandas as gpd
            gpd.read_file(shp_path).boundary.plot(ax=ax, color='k', linewidth=0.25)
        ax.set_xlim(-180, 180); ax.set_ylim(-60, 84)
        ax.set_title(title or '%s: %d sub-basins, %d cells (rule %s, threshold %.2f)%s'
                     % (self._basin_name, n, (self.label > 0).sum(), self.default_rule, self.area_threshold,
                        '; grey: model land not in a sub-basin' if self.model_mask is not None else ''))
        if fn is not None:
            Path(fn).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(fn) + '.png', dpi=dpi, bbox_inches='tight')
            print('figure', str(fn) + '.png')
        if allow_pop_up:
            plt.show()
        return fig


# ------------------------------------------------------------------ demos
EXT = Path('/media/user/My Book/Fan/PyGLDA_v2_external_data')


def demo1():
    """consistency check: the Danube mask of global_shp2mask must equal the one of shp2mask (both rules)"""
    shp = EXT / 'Basin/shp/Danube/Danube.shp'
    tmp = EXT / 'Extra/test_global_shp2mask'                  # scratch (nothing is saved), delete afterwards
    tmp.mkdir(parents=True, exist_ok=True)
    for rule in ('area', 'centre'):
        a = basin_shp_process(res=0.5, basin_name='Danube_ref', save_dir=tmp).shp_to_mask(str(shp), rule=rule)
        b = global_shp_process(res=0.5, basin_name='Danube_new', save_dir=tmp).shp_to_mask(str(shp), rule=rule)
        same = all(np.array_equal(a.mask[k], b.mask[k]) for k in a.mask)
        print('Danube, rule %-6s: %d cells, identical: %s' % (rule, a.mask[0].sum(), same))
    pass


def demo2():
    """global DA mask: GlobalBasins.shp on the WaterGAP land, removed units excluded -> Basin/mask/GlobalBasins/"""
    gp = global_shp_process(res=0.5, basin_name='GlobalBasins', save_dir=EXT / 'Basin/mask')
    gp.configure_threshold(0.4)
    gp.configure_model_mask(continental_area_nc=EXT / 'Input_data/static_input/watergap_22e_continentalarea.nc')
    gp.configure_exclude(EXT / 'Basin/shp/Global/GlobalBasins_removed.shp')
    gp.shp_to_mask(shp_path=EXT / 'Basin/shp/Global/GlobalBasins.shp', issave=True)
    gp.plot_mask(shp_path=EXT / 'Basin/shp/Global/GlobalBasins.shp', fn=EXT / 'Basin/mask/GlobalBasins/GlobalBasins_mask',
                 allow_pop_up=False)
    box_crop, local_mask = load_mask(EXT / 'Basin/mask/GlobalBasins/GlobalBasins_res_0.5.h5')
    print('load_mask: %d sub-basins, %d cells' % (local_mask['basin_num'], local_mask['basin'].sum()))
    pass


if __name__ == '__main__':
    demo1()
    demo2()
