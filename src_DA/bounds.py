"""
Physical bounds of the state elements as seen by the filter (src_DA.EnKF_localized).

Static bounds per storage (mm):
    snow, soil, canopy, local/global wetlands, reservoirs  >= 0
    river                                                  >= 1e-3 (a zero storage breaks the Manning velocity)
    groundwater, lakes                                     unbounded
Relative upper bounds (StateBounds(..., relative_upper={var: (factor, offset_mm)})), per member and window:
    snow                                                   <= factor x the member's own forecast + offset_mm
                                                              (default 2 x forecast + 20 mm, "upper_factor" /
                                                              "upper_offset_mm" of "snow_bounds"): the analysis may at most double a
                                                              snow pack and add a few cm where there is little snow.
    Snow errors are mostly relative (precipitation under-catch, melt timing), so a fixed number is not meaningful.
    WaterGAP itself has no cell-mean limit; it only stops accumulating in a 100-band sub-grid level once that level
    holds 1000 mm.
Relative lower bounds (StateBounds(..., relative_lower={var: factor})), per member and window:
    snow                                                   >= factor x the member's own forecast
                                                              (default 0.5, "lower_factor" of "snow_bounds"):
                                                              one update removes at most half of a snow pack. GRACE
                                                              has no information below the sub-basin scale, and the
                                                              partition would otherwise empty single cells.
                                                              The removal the bound refuses goes to the other cells
                                                              and storages of the sub-basin (bound-aware partition).
Per-cell upper bounds from fields (StateBounds(..., upper={var: values per cell})):
    soil                                                   <= smax (maximum soil water content, the same
                                                              <Auxiliary_dir>/smax.nc as src_DA.Threshold), passed
                                                              by EnKF_localized when soil_upper_bound is on.
    With it the bound-aware partition moves the water that soil cannot hold to the other storages of the
    sub-basin instead of Threshold deleting it after the update.
Per-cell lower bounds from fields (StateBounds(..., lower={var: values per cell})):
    local / global lakes                                   >= -max_storage (WaterGAP's lake floor; the lake balance
                                                              divides by zero for a storage below it),
                                                              and lb = ub = 0 where the cell has no lake
                                                              (max_storage = 0), passed by EnKF_localized when lakes
                                                              are in the DA state (lake_floor_fields).
    For these rows the window-aware LB is never above the forecast (a storage already below its floor is not forced
    to gain water by the bound).
Absolute upper bounds per window (set_window(..., upper_window={var: values per cell})):
    snow                                                   <= f x max_OL(cell, month of the window) + offset
                                                              ("envelope_*" of "snow_bounds", EnKF_localized): the
                                                              open loop's largest snow of that calendar month over
                                                              all years and members. Stops the ratchet of the relative
                                                              bound in cells whose pack never melts. Window-aware like the
                                                              others: every day of the window stays below it.
Window-aware bounds: the increment of a window is added equally to every day of the window and the threshold
(src_DA.Threshold) is then applied day by day. With the daily minimum / maximum of each member over the window
(passed by EnKF.run_mpi), "every day stays within [lb, ub]" means for the window-mean analysis Xa
        Xa >= lb + (x_mean - x_min) = LB          Xa <= ub - (x_max - x_mean) = UB
Without the window extremes LB = lb and UB = ub. For a relative lower bound every day must keep at least factor x
its own forecast; with the increment equal on all days this holds when Xa >= x_mean - (1-factor) x_min (x_min = the
member's lowest day in the window): LB = max(LB, x_mean - (1-factor) x_min), which is never above the forecast. For a relative bound ub = factor * x_max + offset (the member's
highest day in the window), so every day of the window stays below it: UB = x_mean + (factor-1) x_max + offset.
UB is never below the forecast itself (UB >= Xf), so a storage that already sits at its upper bound gets no
positive increment but is never forced to give water away by the bound.
"""
import numpy as np

LOWER = {'swe': 0.0, 'soilmoist': 0.0, 'canopystor': 0.0, 'localwetlandstor': 0.0, 'globalwetlandstor': 0.0,
         'reservoirstor': 0.0, 'riverstor': 1e-3}
UPPER = {}


class StateBounds:

    def __init__(self, names, n_cell, upper=None, relative_upper=None, relative_lower=None, lower=None):
        """
        names  : storages of the state vector (cell-major layout: element i = cell * len(names) + storage)
        n_cell : number of cells
        upper  : optional {storage: per-cell upper bound (n_cell,)}, e.g. {'soilmoist': smax}; NaN = no bound.
                 Combined with the static bound by taking the smaller one.
        relative_upper : optional {storage: (factor, offset_mm)}: per member ub = factor * forecast + offset_mm
                 (forecast = the member's highest day of the window when the window extremes are known)
        relative_lower : optional {storage: factor}, 0 <= factor < 1: per member lb = factor * forecast, i.e. one
                 update removes at most (1 - factor) of the storage (forecast = the member's lowest day of the
                 window when the window extremes are known)
        """
        self.names = list(names)
        nv = len(self.names)
        self.lb = np.tile(np.array([LOWER.get(v, -np.inf) for v in self.names]), n_cell)
        self.ub = np.tile(np.array([UPPER.get(v, np.inf) for v in self.names]), n_cell)
        self.upper_fields = {}
        for var, values in (upper or {}).items():
            if var not in self.names:
                continue
            a = np.asarray(values, dtype=float).ravel()
            if a.size != n_cell:
                raise ValueError('StateBounds: upper bound of %s has %d values for %d cells' % (var, a.size, n_cell))
            a = np.where(np.isfinite(a), a, np.inf)
            i = self.names.index(var)
            self.ub[i::nv] = np.minimum(self.ub[i::nv], a)
            fin = np.isfinite(a)
            self.upper_fields[var] = dict(n_cells_bounded=int(fin.sum()),
                                          min_mm=float(a[fin].min()) if fin.any() else None,
                                          median_mm=float(np.median(a[fin])) if fin.any() else None,
                                          max_mm=float(a[fin].max()) if fin.any() else None)
        self.lower_fields = {}
        self._lower_rows = []
        for var, values in (lower or {}).items():
            if var not in self.names:
                continue
            a = np.asarray(values, dtype=float).ravel()
            if a.size != n_cell:
                raise ValueError('StateBounds: lower bound of %s has %d values for %d cells' % (var, a.size, n_cell))
            a = np.where(np.isfinite(a), a, -np.inf)
            i = self.names.index(var)
            self.lb[i::nv] = np.maximum(self.lb[i::nv], a)
            self._lower_rows.append(np.arange(i, nv * n_cell, nv))
            fin = np.isfinite(a)
            self.lower_fields[var] = dict(n_cells_bounded=int(fin.sum()), n_cells_zero=int((a[fin] == 0).sum()),
                                          min_mm=float(a[fin].min()) if fin.any() else None,
                                          median_mm=float(np.median(a[fin])) if fin.any() else None)
        self._lower_rows = np.concatenate(self._lower_rows) if self._lower_rows else None
        self.relative_upper = {}
        for var, (fac, off) in (relative_upper or {}).items():
            if var in self.names:
                if float(fac) < 1.0 or float(off) < 0.0:
                    raise ValueError('relative upper bound of %s needs factor >= 1 and offset >= 0, got %s'
                                     % (var, (fac, off)))
                self.relative_upper[var] = (float(fac), float(off))
        self.relative_lower = {}
        for var, fac in (relative_lower or {}).items():
            if var in self.names:
                if not 0.0 <= float(fac) < 1.0:
                    raise ValueError('relative lower bound of %s needs 0 <= factor < 1, got %s' % (var, fac))
                self.relative_lower[var] = float(fac)
        self._rel_rows = {v: np.arange(self.names.index(v), nv * n_cell, nv)
                          for v in set(self.relative_upper) | set(self.relative_lower)}
        self.LB = self.UB = None
        self.ub_member = None          # static upper bound of the window per member (n_state x N), set_window
        self.n_window_bounds = 0
        self.n_window_caps = {}

    def is_bounded(self, var):
        """True if the storage has a finite lower bound (snow, soil, river, ...)"""
        i = self.names.index(var)
        return bool(np.isfinite(self.lb[i]))

    def any_finite(self):
        return bool(np.isfinite(self.lb).any() or np.isfinite(self.ub).any())

    def set_window(self, ens_states, ens_min=None, ens_max=None, upper_scale=None, upper_window=None):
        """effective bounds LB, UB (n_state x N) for the window-mean analysis.
        upper_scale: optional {storage: (n_cell x N)} factor on the static upper bound of that storage, per cell and
        member (soil: land / continental fraction, see Threshold.set_soil_scale)
        upper_window: optional {storage: (n_cell,)} absolute upper bound of this window, mm (inf / NaN = none), e.g.
        the monthly snow cap; every day of the window must stay below it"""
        lb, ub = self.lb[:, None], self.ub[:, None]
        if upper_scale:
            nv = len(self.names)
            ub = np.broadcast_to(ub, np.shape(ens_states)).copy()
            for v, sc in upper_scale.items():
                if v not in self.names or sc is None:
                    continue
                sc = np.asarray(sc, dtype=float)
                r = np.arange(self.names.index(v), ub.shape[0], nv)
                if sc.shape != (r.size, ub.shape[1]):
                    raise ValueError('StateBounds: upper_scale of %s has shape %s, expected %s'
                                     % (v, sc.shape, (r.size, ub.shape[1])))
                fin = np.isfinite(ub[r])
                ub[r] = np.where(fin, ub[r] * sc, ub[r])
        '''static upper bound of every member (soil: smax x land fraction x the member's smax_multiplier), before the
        window extremes; the inflation weights the soil noise by the room to it (AdditiveInflation._room)'''
        self.ub_member = np.broadcast_to(ub, np.shape(ens_states)).copy()
        if ens_min is not None and ens_max is not None and \
                np.shape(ens_min) == ens_states.shape == np.shape(ens_max):
            with np.errstate(invalid='ignore'):
                self.LB = lb + (ens_states - ens_min)
                self.UB = ub - (ens_max - ens_states)
            self.n_window_bounds += 1
        else:
            self.LB = np.broadcast_to(lb, ens_states.shape).copy()
            self.UB = np.broadcast_to(ub, ens_states.shape).copy()
        xs = np.asarray(ens_states, dtype=float)
        has_max = ens_max is not None and np.shape(ens_max) == xs.shape
        for v, (fac, off) in self.relative_upper.items():
            r = self._rel_rows[v]
            xmax = np.asarray(ens_max, dtype=float)[r] if has_max else xs[r]
            rel = xs[r] + (fac - 1.0) * np.maximum(xmax, 0.0) + off       # every day <= fac * x_max + off
            self.UB[r] = np.minimum(self.UB[r], rel)
        for v, cap in (upper_window or {}).items():
            if v not in self.names or cap is None:
                continue
            nv = len(self.names)
            r = np.arange(self.names.index(v), xs.shape[0], nv)
            cap = np.asarray(cap, dtype=float).ravel()
            if cap.size != r.size:
                raise ValueError('StateBounds: upper_window of %s has %d values for %d cells' % (v, cap.size, r.size))
            cap = np.where(np.isfinite(cap), cap, np.inf)[:, None]
            xmax = np.asarray(ens_max, dtype=float)[r] if has_max else xs[r]
            with np.errstate(invalid='ignore'):
                capw = cap - (xmax - xs[r])                                    # every day <= cap
            self.UB[r] = np.minimum(self.UB[r], np.where(np.isfinite(capw), capw, np.inf))
            self.n_window_caps[v] = self.n_window_caps.get(v, 0) + 1
        has_min = ens_min is not None and np.shape(ens_min) == xs.shape
        for v, fac in self.relative_lower.items():
            r = self._rel_rows[v]
            xmin = np.asarray(ens_min, dtype=float)[r] if has_min else xs[r]
            rel = xs[r] - (1.0 - fac) * np.maximum(xmin, 0.0)                # every day >= fac * its forecast
            self.LB[r] = np.maximum(self.LB[r], rel)
        fin = np.isfinite(self.UB)
        self.UB[fin] = np.maximum(self.UB[fin], np.asarray(ens_states, dtype=float)[fin])   # never force a removal
        if self._lower_rows is not None:                                   # field lower bounds: never force an addition
            r = self._lower_rows
            self.LB[r] = np.minimum(self.LB[r], xs[r])

    def summary(self):
        return dict(n_window_bounds=self.n_window_bounds, upper_fields=self.upper_fields, lower_fields=self.lower_fields,
                    windows_with_absolute_cap=dict(self.n_window_caps),
                    relative_upper={v: dict(factor=f, offset_mm=o) for v, (f, o) in self.relative_upper.items()},
                    relative_lower={v: dict(factor=f) for v, f in self.relative_lower.items()})


LAKE_FLOOR_ATTRS = {'locallakestor': 'max_loclake_storage', 'globallakestor': 'max_glolake_storage'}
'''the floor is set a little ABOVE WaterGAP's -max_storage: the state goes mm -> km3 when it is written back, and a
value at exactly -max can come out a few 1e-16 below it; WaterGAP's lake balance then has
petgwr_netabs_sw_max = S + max + inflow < 0 with petgwr_netabs_sw = 0 (lake area 0 at the floor) and divides by
zero'''
LAKE_FLOOR_MARGIN = 1e-6


def lake_max_mm(lateral, converter, shape, var):
    """WaterGAP's maximum storage of a lake storage in mm on the DA box (2-D, shape = the box)"""
    km3 = np.asarray(getattr(lateral, LAKE_FLOOR_ATTRS[var]), dtype=float)
    if km3.shape != tuple(shape):
        r0, r1, c0, c1 = converter._r0, converter._r1, converter._c0, converter._c1
        km3 = km3[r0:r1, c0:c1]
    mm = np.asarray(converter.to_disk(var, km3), dtype=float)
    return np.where(np.isfinite(mm), np.maximum(mm, 0.0), 0.0)


def lake_floor_fields(lateral, converter, mask_2d, names):
    """
    Per-cell bounds of the lake storages in the DA state (mm, state-cell order) from WaterGAP's static maximum lake
    storages (km3): lower = -max_storage (WaterGAP's floor), and lower = upper = 0 where max_storage = 0 (no lake).
    lateral   : the member's lateral water balance (attributes max_loclake_storage / max_glolake_storage, km3, grid)
    converter : src_GHM UnitConverter of the DA box (to_disk: km3 -> mm; _r0.._c1 = box in the model grid)
    mask_2d   : basin_2d of the DA box (bool), selects the state cells
    Returns (lower, upper): dicts {storage: (n_cell,)}; empty when no lake storage is in the state.
    """
    lower, upper = {}, {}
    mk = np.asarray(mask_2d).astype(bool)
    for v in LAKE_FLOOR_ATTRS:
        if v not in names:
            continue
        mm = lake_max_mm(lateral, converter, mk.shape, v)[mk]
        lower[v] = -mm * (1.0 - LAKE_FLOOR_MARGIN)
        upper[v] = np.where(mm > 0, np.inf, 0.0)
    return lower, upper


def lake_floor_grids(lateral, converter, shape, names):
    """the same lake bounds as 2-D boxes for src_DA.Threshold (applied to every day written back):
    {storage: (lower, upper)}, lower = -max (1 - margin), lower = upper = 0 where the cell has no lake"""
    out = {}
    for v in LAKE_FLOOR_ATTRS:
        if v in names:
            mm = lake_max_mm(lateral, converter, shape, v)
            out[v] = (-mm * (1.0 - LAKE_FLOOR_MARGIN), np.where(mm > 0, np.inf, 0.0))
    return out
