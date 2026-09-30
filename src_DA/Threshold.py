from src_DA.configure_DA import config_DA
import numpy as np
from pathlib import Path
import xarray as xr
import json
from src_auxiliary.shp2mask import load_mask


class model_state_threshold:
    """
    Physical bounds applied to the WaterGAP storages after the EnKF analysis step (regional 2-D fields, mm).

    Rules (only for the storages that are part of the DA state, configDA.model.layer == True):
      soilmoist          0 <= S <= smax (maximum soil water content, static, from <Auxiliary_dir>/smax.nc)
      groundwstor        open-loop envelope (see riverstor); WaterGAP groundwater may go into deficit, so no fixed
                         bound, but the analysis is kept within env_low*min .. env_high*max of the open loop, with
                         the bounds widened symmetrically by the OL range where min and max have different signs
      riverstor          positive floor (a storage of exactly zero makes the Manning velocity divide by zero
                         in river_routing) and, if available, a per-cell envelope learned from the open loop:
                             env_low  * min_OL(cell)  <=  S  <=  env_high * max_OL(cell)
                         read from <Auxiliary_dir>/state_envelope_<basin>.nc (see src_DA.state_envelope);
                         without the file only the floor is applied
      swe                0 <= S <= 1000 mm (WaterGAP stops snow accumulation at 1000 mm per sub-grid band)
      canopystor, reservoirstor, localwetlandstor, globalwetlandstor      S >= 0
      locallakestor, globallakestor                                       no bound (lakes may be negative)
    Capacities of lakes, wetlands and reservoirs exist inside WaterGAP (lateral water balance: max_*_storage,
    glores_capacity) but depend on perturbed parameters and are not needed while these stores are not updated.

    Every clip is recorded: number of cells clipped, water added / removed (mm summed over cells), per storage
    and in total, so that the thresholds can be checked to act as a rare safety net and the clipped water can be
    reported as a term of the water balance. summary() returns the statistics, save_log() writes them to JSON.
    """

    ENVELOPE_VARS = ('riverstor', 'groundwstor')     # storages bounded by the open-loop envelope

    def __init__(self, configDA: config_DA, env_low: float = 0.5, env_high: float = 1.5, river_floor: float = 1e-3,
                 snow_max: float = 1000.0):
        self.layers = [key for key, value in configDA.model.layer.items() if value is True]
        self.env_low, self.env_high, self.river_floor, self.snow_max = env_low, env_high, river_floor, snow_max

        box_crop, _ = load_mask(mask_path=configDA.basic.basin_mask)
        self._lat_slice = slice(box_crop['lat_max'], box_crop['lat_min'])   # descending latitude
        self._lon_slice = slice(box_crop['lon_min'], box_crop['lon_max'])

        '''maximum soil water content'''
        ds = xr.open_dataset(Path(configDA.basic.Auxiliary_dir) / 'smax.nc')
        self.smax = ds.sel(lon=self._lon_slice, lat=self._lat_slice)['smax'].values
        ds.close()

        '''open-loop envelope of river storage (optional)'''
        self.envelope = {}
        env_fn = Path(configDA.basic.Auxiliary_dir) / ('state_envelope_%s.nc' % configDA.basic.basin)
        if env_fn.exists():
            env = xr.open_dataset(env_fn).sel(lon=self._lon_slice, lat=self._lat_slice)
            for var in [v for v in self.layers if v in self.ENVELOPE_VARS]:
                if ('%s_min' % var in env) and ('%s_max' % var in env):
                    lo = env['%s_min' % var].values.astype(float)
                    hi = env['%s_max' % var].values.astype(float)
                    # outside the basin (NaN) -> no bound. For positive storages the bounds are env_low*min and
                    # env_high*max; for signed storages (groundwater) the OL range is widened on both sides by
                    # (env_high-1)*range so that a negative minimum does not shrink the interval
                    rng = hi - lo
                    signed = np.nanmin(lo) < 0
                    if signed:
                        lo_b, hi_b = lo - (self.env_high - 1.0) * rng, hi + (self.env_high - 1.0) * rng
                    else:
                        lo_b, hi_b = self.env_low * lo, self.env_high * hi
                    lo = np.where(np.isfinite(lo), lo_b, -np.inf)
                    hi = np.where(np.isfinite(hi), hi_b, np.inf)
                    self.envelope[var] = (lo, hi)
            env.close()
        elif 'riverstor' in self.layers:
            print('model_state_threshold: %s not found -> riverstor bounded by the positive floor only' % env_fn)

        self.non_negative_vars = ['canopystor', 'reservoirstor', 'localwetlandstor', 'globalwetlandstor']
        self.unbounded_vars = ['locallakestor', 'globallakestor']

        '''clipping statistics'''
        self.stats = {var: dict(n_calls=0, n_cells_checked=0, n_cells_clipped=0, water_added_mm=0.0,
                                water_removed_mm=0.0, max_abs_change_mm=0.0) for var in self.layers}
        self.n_updates = 0
        pass

    # ------------------------------------------------------------------ bounds per storage
    def _bounds(self, var, x):
        """return (lower, upper) arrays or scalars for variable var, or None if unbounded"""
        if var == 'soilmoist':
            return 0.0, self.smax
        if var == 'swe':
            return 0.0, self.snow_max
        if var == 'riverstor':
            if var in self.envelope:
                lo, hi = self.envelope[var]
                return np.maximum(lo, self.river_floor), hi
            return self.river_floor, np.inf
        if var in self.envelope:                        # groundwater (signed storage)
            return self.envelope[var]
        if var in self.non_negative_vars:
            return 0.0, np.inf
        return None                                     # lakes, groundwater without envelope: unbounded

    # ------------------------------------------------------------------ application
    def threshold(self, state: dict):
        """clip the (regional 2-D) analysis fields in place and record what was clipped"""
        self.n_updates += 1
        for var in self.layers:
            if var not in state:
                continue
            b = self._bounds(var, state[var])
            st = self.stats[var]
            st['n_calls'] += 1
            if b is None:
                continue
            lo, hi = b
            old = state[var]
            new = np.clip(old, lo, hi)
            diff = new - old
            finite = np.isfinite(diff)
            changed = finite & (diff != 0)
            st['n_cells_checked'] += int(finite.sum())
            st['n_cells_clipped'] += int(changed.sum())
            st['water_added_mm'] += float(diff[changed & (diff > 0)].sum())
            st['water_removed_mm'] += float(-diff[changed & (diff < 0)].sum())
            if changed.any():
                st['max_abs_change_mm'] = max(st['max_abs_change_mm'], float(np.abs(diff[changed]).max()))
            state[var] = new
        return state

    # ------------------------------------------------------------------ reporting
    def summary(self):
        out = {}
        for var, st in self.stats.items():
            frac = st['n_cells_clipped'] / st['n_cells_checked'] if st['n_cells_checked'] else 0.0
            out[var] = dict(st, fraction_clipped=frac, bound='none' if self._bounds(var, None) is None else
                            ('envelope' if var in self.envelope else 'fixed'))   # envelope only holds ENVELOPE_VARS
        out['_settings'] = dict(env_low=self.env_low, env_high=self.env_high, river_floor=self.river_floor,
                                snow_max=self.snow_max, n_threshold_calls=self.n_updates)
        return out

    def print_summary(self):
        print('\nThreshold / clipping summary (%d calls):' % self.n_updates)
        for var, st in self.summary().items():
            if var.startswith('_'):
                continue
            print('  %-18s bound=%-8s clipped %8d of %10d cell-days (%.3f %%), water added %10.1f mm, removed %10.1f mm, '
                  'max |change| %.1f mm' % (var, st['bound'], st['n_cells_clipped'], st['n_cells_checked'],
                                           100 * st['fraction_clipped'], st['water_added_mm'], st['water_removed_mm'],
                                           st['max_abs_change_mm']))

    def save_log(self, fn):
        Path(fn).parent.mkdir(parents=True, exist_ok=True)
        json.dump(self.summary(), open(fn, 'w'), indent=1)


def demo1():
    configDA = config_DA.loadjson('/media/user/My Book/Fan/PyGLDA_v2/settings/demo_3/DA_setting.json').process()
    mst = model_state_threshold(configDA=configDA)
    mst.print_summary()
    pass


if __name__ == '__main__':
    demo1()
