"""
Physical bounds of the state elements as seen by the filter (src_DA.EnKF_localized).

Static bounds per storage (mm):
    snow, soil, canopy, local/global wetlands, reservoirs  >= 0
    river                                                  >= 1e-3 (a zero storage breaks the Manning velocity)
    snow                                                   <= 1000 (WaterGAP stops accumulation there)
    groundwater, lakes                                     unbounded
Per-cell upper bounds from fields (StateBounds(..., upper={var: values per cell})):
    soil                                                   <= smax (maximum soil water content, the same
                                                              <Auxiliary_dir>/smax.nc as src_DA.Threshold), passed
                                                              by EnKF_localized when soil_upper_bound is on.
    With it the bound-aware partition moves the water that soil cannot hold to the other storages of the
    sub-basin instead of Threshold deleting it after the update.
Window-aware bounds: the increment of a window is added equally to every day of the window and the threshold
(src_DA.Threshold) is then applied day by day. With the daily minimum / maximum of each member over the window
(passed by EnKF.run_mpi), "every day stays within [lb, ub]" means for the window-mean analysis Xa
        Xa >= lb + (x_mean - x_min) = LB          Xa <= ub - (x_max - x_mean) = UB
Without the window extremes LB = lb and UB = ub.
UB is never below the forecast itself (UB >= Xf), so a storage that already sits at its upper bound gets no
positive increment but is never forced to give water away by the bound.
"""
import numpy as np

LOWER = {'swe': 0.0, 'soilmoist': 0.0, 'canopystor': 0.0, 'localwetlandstor': 0.0, 'globalwetlandstor': 0.0,
         'reservoirstor': 0.0, 'riverstor': 1e-3}
UPPER = {'swe': 1000.0}


class StateBounds:

    def __init__(self, names, n_cell, upper=None):
        """
        names  : storages of the state vector (cell-major layout: element i = cell * len(names) + storage)
        n_cell : number of cells
        upper  : optional {storage: per-cell upper bound (n_cell,)}, e.g. {'soilmoist': smax}; NaN = no bound.
                 Combined with the static bound by taking the smaller one.
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
        self.LB = self.UB = None
        self.n_window_bounds = 0

    def is_bounded(self, var):
        """True if the storage has a finite lower bound (snow, soil, river, ...)"""
        i = self.names.index(var)
        return bool(np.isfinite(self.lb[i]))

    def any_finite(self):
        return bool(np.isfinite(self.lb).any() or np.isfinite(self.ub).any())

    def set_window(self, ens_states, ens_min=None, ens_max=None):
        """effective bounds LB, UB (n_state x N) for the window-mean analysis"""
        lb, ub = self.lb[:, None], self.ub[:, None]
        if ens_min is not None and ens_max is not None and \
                np.shape(ens_min) == ens_states.shape == np.shape(ens_max):
            with np.errstate(invalid='ignore'):
                self.LB = lb + (ens_states - ens_min)
                self.UB = ub - (ens_max - ens_states)
            self.n_window_bounds += 1
        else:
            self.LB = np.broadcast_to(lb, ens_states.shape).copy()
            self.UB = np.broadcast_to(ub, ens_states.shape).copy()
        fin = np.isfinite(self.UB)
        self.UB[fin] = np.maximum(self.UB[fin], np.asarray(ens_states, dtype=float)[fin])   # never force a removal

    def summary(self):
        return dict(n_window_bounds=self.n_window_bounds, upper_fields=self.upper_fields)
