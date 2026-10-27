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
      groundwstor        no bound (signed storage; follows GRACE beyond the open-loop range)
      riverstor          positive floor (a storage of exactly zero makes the Manning velocity divide by zero
                         in river_routing) and, if available, a per-cell envelope learned from the open loop:
                             env_low  * min_OL(cell)  <=  S  <=  env_high * max_OL(cell)
                         read from <Auxiliary_dir>/state_envelope_<basin>.nc (see src_DA.state_envelope);
                         without the file only the floor is applied
      swe                S >= 0 (no fixed upper limit: the filter limits snow relative to each member's forecast,
                         see src_DA.bounds). With "envelope_factor": f, "envelope_offset_mm": o in "snow_bounds":
                             S <= f x max_OL(cell, calendar month of the day) + o
                         from swe_max_month of <Auxiliary_dir>/state_envelope_<basin>.nc (the open loop's largest snow
                         of that month over all years and members). The filter applies the same cap inside the
                         bound-aware partition (EnKF_localized), so this clip is a safety net; like the partition it
                         never removes snow the member already had in its forecast of that day (set_snow_forecast):
                         the bound is max(cap, forecast). Without the file only S >= 0 is applied (warning)
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

    def __init__(self, configDA: config_DA, env_low: float = 0.5, env_high: float = 1.5, river_floor: float = 1e-3):
        self.layers = [key for key, value in configDA.model.layer.items() if value is True]
        self.env_low, self.env_high, self.river_floor = env_low, env_high, river_floor

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
        '''snow cap from the monthly open-loop maximum (envelope_factor / envelope_offset_mm of "snow_bounds",
        src_DA.configure_DA.snow_bounds; envelope_factor missing / null = off)'''
        from src_DA.configure_DA import snow_bounds
        method = getattr(configDA, 'method', None)
        sb = snow_bounds(method if method is not None else {})
        self.snow_cap = None                            # (factor, offset_mm) when on
        self.snow_forecast = None                       # forecast snow of the day being clipped (set_snow_forecast)
        self.snow_env_month = None                      # (12, lat, lon) of the box, mm; NaN = no cap
        if sb['envelope_factor'] is not None and 'swe' in self.layers:
            self.snow_cap = (sb['envelope_factor'], sb['envelope_offset_mm'])
        env_fn = Path(configDA.basic.Auxiliary_dir) / ('state_envelope_%s.nc' % configDA.basic.basin)
        if env_fn.exists():
            env = xr.open_dataset(env_fn).sel(lon=self._lon_slice, lat=self._lat_slice)
            if self.snow_cap is not None and 'swe_max_month' in env:
                self.snow_env_month = env['swe_max_month'].transpose('month', 'lat', 'lon').values.astype(float)
            if 'riverstor' in self.layers and 'riverstor_min' in env and 'riverstor_max' in env:
                lo = env['riverstor_min'].values.astype(float)
                hi = env['riverstor_max'].values.astype(float)
                # outside the basin (NaN) -> no bound
                lo = np.where(np.isfinite(lo), self.env_low * lo, -np.inf)
                hi = np.where(np.isfinite(hi), self.env_high * hi, np.inf)
                self.envelope['riverstor'] = (lo, hi)
            env.close()
        elif 'riverstor' in self.layers:
            print('model_state_threshold: %s not found -> riverstor bounded by the positive floor only' % env_fn)
        if self.snow_cap is not None and self.snow_env_month is None:
            print('model_state_threshold: WARNING the snow envelope cap is on but %s has no swe_max_month -> no snow cap'
                  % env_fn)

        self.non_negative_vars = ['canopystor', 'reservoirstor', 'localwetlandstor', 'globalwetlandstor']
        self.unbounded_vars = ['locallakestor', 'globallakestor']
        self.lake_floor = {}            # {lake storage: (lower, upper) 2-D}, set_lake_floor (lakes in the DA state)

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
        load_mask result: global_2d and unit_label)"""
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
        With the unscaled smax the analysis could fill the soil of lake / wetland cells above capacity, and WaterGAP
        would release the excess as runoff the next day. The largest value of the window is used
        because the land fraction changes daily with the lakes and wetlands: no day's own soil is then above it.
        """
        self.soil_scale = None if scale is None else np.asarray(scale, dtype=float)

    def set_lake_floor(self, grids):
        """lake storages in the DA state: per-cell (lower, upper) on the regional box from WaterGAP's maximum lake
        storage (src_DA.bounds.lake_floor_grids): lower slightly above -max_storage, 0 / 0 where there is no lake"""
        self.lake_floor = dict(grids or {})

    # ------------------------------------------------------------------ bounds per storage
    def set_snow_forecast(self, snow):
        """forecast snow (lat, lon) of the day that is clipped next: the cap never removes snow the member already had
        (consistent with the partition, src_DA.bounds); None = cap as it is"""
        self.snow_forecast = None if snow is None else np.asarray(snow, dtype=float)

    def snow_cap_month(self, month):
        """snow cap (lat, lon) of a calendar month 1..12, mm (inf where the open loop has no value), or None"""
        if self.snow_cap is None or self.snow_env_month is None:
            return None
        fac, off = self.snow_cap
        cap = fac * self.snow_env_month[int(month) - 1] + off
        return np.where(np.isfinite(cap), cap, np.inf)

    def _bounds(self, var, x, month=None):
        """return (lower, upper) arrays or scalars for variable var, or None if unbounded; month (1..12) selects the
        monthly snow cap"""
        if var == 'soilmoist':
            if self.soil_scale is not None and np.shape(self.soil_scale) == np.shape(self.smax):
                return 0.0, self.smax * self.soil_scale
            return 0.0, self.smax
        if var == 'swe':
            cap = None if month is None else self.snow_cap_month(month)
            if cap is not None and self.snow_forecast is not None and np.shape(self.snow_forecast) == np.shape(cap):
                cap = np.where(np.isfinite(self.snow_forecast), np.maximum(cap, self.snow_forecast), cap)
            return 0.0, (np.inf if cap is None else cap)
        if var == 'riverstor':
            if var in self.envelope:
                lo, hi = self.envelope[var]
                return np.maximum(lo, self.river_floor), hi
            return self.river_floor, np.inf
        if var in self.non_negative_vars:
            return 0.0, np.inf
        if var in self.lake_floor:                      # lakes in the DA state: WaterGAP's floor (set_lake_floor)
            lo, hi = self.lake_floor[var]
            if np.shape(lo) == np.shape(x):
                return lo, hi
        return None                                     # groundwater, lakes: unbounded

    # ------------------------------------------------------------------ application
    def threshold(self, state: dict, date=None):
        """clip the (regional 2-D) analysis fields in place and record what was clipped;
        date ('YYYY-MM-DD' or datetime) enables the monthly / sub-basin / lower-vs-upper record"""
        month = None if date is None else str(date)[:7]
        self.n_updates += 1
        for var in self.layers:
            if var not in state:
                continue
            b = self._bounds(var, state[var], month=None if month is None else int(month[5:7]))
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
        one bincount per quantity instead of a loop over the sub-basins"""
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
                            ('envelope' if var in self.envelope else 'fixed'))
        out['_settings'] = dict(env_low=self.env_low, env_high=self.env_high, river_floor=self.river_floor,
                                n_threshold_calls=self.n_updates,
                                snow_envelope=None if self.snow_cap is None else
                                dict(factor=self.snow_cap[0], offset_mm=self.snow_cap[1],
                                     monthly_max_loaded=self.snow_env_month is not None))
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
