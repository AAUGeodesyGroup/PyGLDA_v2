from src_DA.configure_DA import config_DA
import numpy as np
from pathlib import Path
import xarray as xr
import json
import csv
from src_auxiliary.shp2mask import load_mask


class model_state_threshold:
    """
    Physical bounds applied to the WaterGAP storages after the EnKF analysis step (regional 2-D fields, mm).

    Rules (only for the storages that are part of the DA state, configDA.model.layer == True):
      soilmoist          0 <= S <= smax x land/continental fraction (smax from <Auxiliary_dir>/smax.nc; the fraction
                         is the member's largest of the window, set_soil_scale)
      groundwstor        no bound by default. The open-loop envelope (min - (env_high-1)*range .. max + (env_high-1)*range)
                         was a safety net against metre-scale blow-ups of the 4-member standard EnKF partition
                         (Amazon run 3); with the non-negative partition and sub-basin-coherent additive inflation it
                         clipped real signal instead (Danube: 2006 spring peak, 2007/2011-12 droughts).
                         gw_lower=True / gw_upper=True restore the lower / upper edge
      riverstor          positive floor (a storage of exactly zero makes the Manning velocity divide by zero
                         in river_routing) and, if available, a per-cell envelope learned from the open loop:
                             env_low  * min_OL(cell)  <=  S  <=  env_high * max_OL(cell)
                         read from <Auxiliary_dir>/state_envelope_<basin>.nc (see src_DA.state_envelope);
                         without the file only the floor is applied
      swe                S >= 0 (no fixed upper limit: the filter limits snow relative to each member's forecast,
                         see src_DA.bounds; snow_max sets an absolute safety limit if wanted, default none)
      canopystor, reservoirstor, localwetlandstor, globalwetlandstor      S >= 0
      locallakestor, globallakestor                                       no bound (lakes may be negative)
    Capacities of lakes, wetlands and reservoirs exist inside WaterGAP (lateral water balance: max_*_storage,
    glores_capacity) but depend on perturbed parameters and are not needed while these stores are not updated.

    Every clip is recorded: number of cells clipped, water added / removed (mm summed over cells), per storage
    and in total, so that the thresholds can be checked to act as a rare safety net and the clipped water can be
    reported as a term of the water balance. summary() returns the statistics, save_log() writes them to JSON.

    When threshold() is given the date of the day, the clips are additionally recorded per month, per sub-basin
    and separately for the lower and the upper bound (e.g. soil at 0 vs soil at smax), save_monthly_csv() writes
    them as a table (var, month, sub_basin, n_checked, n_lower, n_upper, water_added_mm, water_removed_mm).
    """

    ENVELOPE_VARS = ('riverstor', 'groundwstor')     # storages bounded by the open-loop envelope

    def __init__(self, configDA: config_DA, env_low: float = 0.5, env_high: float = 1.5, river_floor: float = 1e-3,
                 snow_max: float = None, gw_lower: bool = False, gw_upper: bool = False):
        self.layers = [key for key, value in configDA.model.layer.items() if value is True]
        self.env_low, self.env_high, self.river_floor, self.snow_max = env_low, env_high, river_floor, snow_max
        self.gw_lower = gw_lower            # lower edge of the open-loop envelope for groundwater (default off)
        self.gw_upper = gw_upper            # upper edge of the open-loop envelope for groundwater (default off)

        box_crop, _ = load_mask(mask_path=configDA.basic.basin_mask)
        self._lat_slice = slice(box_crop['lat_max'], box_crop['lat_min'])   # descending latitude
        self._lon_slice = slice(box_crop['lon_min'], box_crop['lon_max'])
        self.sub_labels = self._sub_basin_labels(configDA.basic.basin_mask, box_crop)

        '''maximum soil water content (per land area, WaterGAP output smax.nc) and the factor that converts it to the
        units of the daily files (per continental area), set for every window by set_soil_scale'''
        ds = xr.open_dataset(Path(configDA.basic.Auxiliary_dir) / 'smax.nc')
        self.smax = ds.sel(lon=self._lon_slice, lat=self._lat_slice)['smax'].values
        ds.close()
        self.soil_scale = None

        '''open-loop envelope of river storage (optional)'''
        self.envelope = {}
        env_fn = Path(configDA.basic.Auxiliary_dir) / ('state_envelope_%s.nc' % configDA.basic.basin)
        if env_fn.exists():
            env = xr.open_dataset(env_fn).sel(lon=self._lon_slice, lat=self._lat_slice)
            for var in [v for v in self.layers if v in self.ENVELOPE_VARS]:
                if var == 'groundwstor' and not (self.gw_lower or self.gw_upper):
                    continue                                # groundwater unbounded (default)
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
                    if var == 'groundwstor' and not self.gw_lower:
                        lo = np.full_like(lo, -np.inf)       # follow GRACE droughts / declines beyond the OL range
                    if var == 'groundwstor' and not self.gw_upper:
                        hi = np.full_like(hi, np.inf)        # follow GRACE wet extremes beyond the OL range
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
        '''monthly statistics: monthly[var][YYYY-MM][sub_basin] = [n_checked, n_lower, n_upper, added, removed]'''
        self.monthly = {var: {} for var in self.layers}
        pass

    @staticmethod
    def _sub_basin_labels(mask_path, box_crop, res=0.5):
        """2-D integer field on the regional box: k for cells of sub_basin_k, 0 outside the basin (from the cached
        load_mask result: global_2d and unit_label; reading the 772 datasets of the global mask again was a 1.6 GB
        transient per rank until 8 Oct 2026)"""
        _, lm = load_mask(mask_path=mask_path)
        i0 = int(round((90 - res / 2 - box_crop['lat_max']) / res))
        i1 = int(round((90 - res / 2 - box_crop['lat_min']) / res)) + 1
        j0 = int(round((box_crop['lon_min'] + 180 - res / 2) / res))
        j1 = int(round((box_crop['lon_max'] + 180 - res / 2) / res)) + 1
        lab_global = np.zeros(lm['global_2d'].shape, dtype=int)
        lab_global[lm['global_2d'].astype(bool)] = lm['unit_label'] + 1          # 0 = cell of no sub-basin
        return lab_global[i0:i1, j0:j1]

    def set_soil_scale(self, scale):
        """
        Land / continental fraction of every cell of the regional box (2-D, the member's largest value of the current
        window; None = 1). WaterGAP writes soil per continental area (soil x land fraction / continental fraction,
        fan_waterbalance_vertical_init), smax.nc is per land area: the capacity of the daily soil is smax x scale.
        With the unscaled smax (until 8 Oct 2026) the analysis could fill the soil of lake / wetland cells above
        capacity and WaterGAP released the excess as runoff the next day. The largest value of the window is used
        because the land fraction changes daily with the lakes and wetlands: no day's own soil is then above it.
        """
        self.soil_scale = None if scale is None else np.asarray(scale, dtype=float)

    # ------------------------------------------------------------------ bounds per storage
    def _bounds(self, var, x):
        """return (lower, upper) arrays or scalars for variable var, or None if unbounded"""
        if var == 'soilmoist':
            if self.soil_scale is not None and np.shape(self.soil_scale) == np.shape(self.smax):
                return 0.0, self.smax * self.soil_scale
            return 0.0, self.smax
        if var == 'swe':
            return 0.0, (np.inf if self.snow_max is None else self.snow_max)
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
    def threshold(self, state: dict, date=None):
        """clip the (regional 2-D) analysis fields in place and record what was clipped;
        date ('YYYY-MM-DD' or datetime) enables the monthly / sub-basin / lower-vs-upper record"""
        month = None if date is None else str(date)[:7]
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
            if month is not None:
                self._record_month(var, month, old, diff, finite)
            state[var] = new
        return state

    def _record_month(self, var, month, old, diff, finite):
        """per sub-basin: cells checked, raised to the lower bound, cut at the upper bound, water added, removed;
        one bincount per quantity instead of a loop over the sub-basins (772 passes over the grid per storage and
        day for the global case until 8 Oct 2026)"""
        lower = finite & (diff > 0)                      # raised to the lower bound
        upper = finite & (diff < 0)                      # cut at the upper bound
        lab = self.sub_labels if self.sub_labels.shape == old.shape else np.zeros(old.shape, dtype=int)
        rec = self.monthly[var].setdefault(month, {})
        n = int(lab.max()) + 1
        n_checked = np.bincount(lab[finite], minlength=n)
        n_lower = np.bincount(lab[lower], minlength=n)
        n_upper = np.bincount(lab[upper], minlength=n)
        added = np.bincount(lab[lower], weights=diff[lower], minlength=n)
        removed = np.bincount(lab[upper], weights=-diff[upper], minlength=n)
        for k in np.flatnonzero(n_checked):
            if k == 0:
                continue
            r = rec.setdefault(int(k), [0, 0, 0, 0.0, 0.0])
            r[0] += int(n_checked[k])
            r[1] += int(n_lower[k])
            r[2] += int(n_upper[k])
            r[3] += float(added[k])
            r[4] += float(removed[k])

    def monthly_rows(self):
        rows = []
        for var, months in self.monthly.items():
            for month in sorted(months):
                for k in sorted(months[month]):
                    n, nl, nu, wa, wr = months[month][k]
                    rows.append(dict(var=var, month=month, sub_basin=k, n_checked=n, n_lower=nl, n_upper=nu,
                                     frac_lower=nl / n if n else 0.0, frac_upper=nu / n if n else 0.0,
                                     water_added_mm=wa, water_removed_mm=wr))
        return rows

    def save_monthly_csv(self, fn):
        rows = self.monthly_rows()
        if not rows:
            return
        Path(fn).parent.mkdir(parents=True, exist_ok=True)
        with open(fn, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # ------------------------------------------------------------------ reporting
    def summary(self):
        out = {}
        for var, st in self.stats.items():
            frac = st['n_cells_clipped'] / st['n_cells_checked'] if st['n_cells_checked'] else 0.0
            out[var] = dict(st, fraction_clipped=frac, bound='none' if self._bounds(var, None) is None else
                            ('envelope' if var in self.envelope else 'fixed'))   # envelope only holds ENVELOPE_VARS
        out['_settings'] = dict(env_low=self.env_low, env_high=self.env_high, river_floor=self.river_floor,
                                gw_lower=self.gw_lower, gw_upper=self.gw_upper,
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
        rows = self.monthly_rows()
        for var in self.layers:
            for side in ('upper', 'lower'):
                r = sorted([x for x in rows if x['var'] == var and x['n_' + side] > 0],
                           key=lambda x: -x['frac_' + side])[:5]
                if r:
                    print('    %s %s-bound clips, top months: %s' % (var, side, ', '.join(
                        '%s sb%d %.1f%%' % (x['month'], x['sub_basin'], 100 * x['frac_' + side]) for x in r)))

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
