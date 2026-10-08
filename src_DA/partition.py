"""
How the Kalman increment of a sub-basin is distributed over the cells and storages of that sub-basin
(src_DA.EnKF_localized).  DA_setting.json, "method" block:

    "increment_partition": {"kind": "enkf"}           increments as given by the (localized) Kalman gain
    "increment_partition": {"kind": "non_negative"}   the sub-basin TWS increment of the Kalman update is kept, but
        distributed with non-negative shares ~ variance x max(correlation with the sub-basin TWS, 0) (fallback:
        variance only), so no storage is emptied to fill another one. Bound-aware: an element is not pushed across
        its (window-aware) bound, its share goes to the other elements of the same sub-basin. Meant for small
        ensembles, whose vertical partition is dominated by sampling noise (Amazon 4-member tug of war).
    "increment_partition": {"kind": "non_negative", "max_weight_ratio": 5}
        caps the weight of every element at 5x the median weight of the same storage among the elements that
        share the sub-basin increment, so that one cell with an extreme ensemble spread cannot absorb the whole
        correction of its sub-basin (Danube: the Vienna cell, whose groundwater drifts by >100 mm/yr in the
        open loop because of the WaterGAP groundwater abstraction, had ~10^4 times the typical weight and received
        single-window increments of 5-10 m). Default None = no cap. A number caps every storage (Danube run 10);
    "increment_partition": {"kind": "non_negative", "max_weight_ratio": {"groundwstor": 5}}
        caps only the listed storages. Recommended: groundwater only. For snow the median weight is ~0 (most cells
        have little or no snow), so a cap on snow removes the weight of the real mountain snow cells (run 10: snow
        correction halved, up to 94 % of a sub-basin's snow share removed).
    "increment_partition": {"kind": "non_negative", "spatial_blend": {"groundwstor": 0.5}}
        mixes the weights of a storage with their sub-basin mean, w' = (1-b) w + b mean(w), keeping the storage's
        total share: GRACE has no information below the sub-basin scale, so the correction of a storage is spread
        over its cells instead of following the cell-to-cell differences of a 4-member ensemble (Danube run 11a: 89 %
        of the groundwater correction of sub-basin 6 in 7 of 76 cells, -150 to -270 mm in 13 years). Meant for
        unbounded storages (groundwater); for snow / soil it would put water into snow-free or full cells.
    "increment_partition": {"kind": "non_negative", "max_cell_factor": 3}
        no element may change by more than 3 x the sub-basin TWS increment of its member (|dX_i| <= f |delta_j|);
        the excess goes to the other elements of the sub-basin like a bound (Danube run 11a: one Alpine snow cell
        +820 mm in a window with a sub-basin increment of a few tens of mm). The limit acts on every storage of every
        cell separately; a dict sets it per storage, e.g. {"swe": 3, "groundwstor": 3} (storages not listed: no limit).
    The three options can be combined; order: cap, blend (weights), then the per-cell limit (with the bounds).
    (a plain string "enkf" / "non_negative" is accepted as well)
"""
import numpy as np
from scipy import sparse


class Partition:
    KIND = None

    def __init__(self):
        self.f = None
        self.stats = dict(n_updates=0)

    def setup(self, f):
        """f: the filter (EnKF_localized), gives access to _DM, _L_state, bounds"""
        self.f = f

    def split(self, dX, A, HA, N, X=None):
        raise NotImplementedError

    def summary(self):
        return dict(self.stats, kind=self.KIND)


class EnKFPartition(Partition):
    KIND = 'enkf'

    def split(self, dX, A, HA, N, X=None):
        self.stats['n_updates'] += 1
        return dX


class NonNegativePartition(Partition):
    KIND = 'non_negative'

    def __init__(self, max_weight_ratio=None, spatial_blend=None, max_cell_factor=None):
        super().__init__()
        self.spatial_blend = {str(v): float(b) for v, b in (spatial_blend or {}).items() if b}
        bad_b = {v: b for v, b in self.spatial_blend.items() if not 0.0 < b <= 1.0}
        if bad_b:
            raise ValueError('spatial_blend values must be in (0, 1], got %s' % bad_b)
        if isinstance(max_cell_factor, dict):
            self.max_cell_factor = {str(v): float(x) for v, x in max_cell_factor.items() if x is not None} or None
            bad_c = {v: x for v, x in (self.max_cell_factor or {}).items() if x < 1.0}
        else:
            self.max_cell_factor = None if max_cell_factor is None else float(max_cell_factor)
            bad_c = {} if self.max_cell_factor is None or self.max_cell_factor >= 1.0 else {'all': self.max_cell_factor}
        if bad_c:
            raise ValueError('max_cell_factor must be >= 1 (or null), got %s' % bad_c)
        if isinstance(max_weight_ratio, dict):
            self.max_weight_ratio = {str(v): float(r) for v, r in max_weight_ratio.items() if r is not None}
            bad = {v: r for v, r in self.max_weight_ratio.items() if r <= 1}
            if not self.max_weight_ratio:
                self.max_weight_ratio = None
        else:
            self.max_weight_ratio = None if max_weight_ratio is None else float(max_weight_ratio)
            bad = {} if self.max_weight_ratio is None or self.max_weight_ratio > 1 else {'all': self.max_weight_ratio}
        if bad:
            raise ValueError('max_weight_ratio must be > 1 (or null for no cap), got %s' % bad)
        self.stats = dict(n_updates=0, n_obs_fallback=0, n_obs_unresolved=0, max_abs_inc_enkf=0.0,
                          max_abs_inc_new=0.0, n_bound_capped=0, water_redistributed_mm=0.0, max_inc_where=None,
                          max_weight_ratio=self.max_weight_ratio, n_weights_capped=0, weight_share_removed_max=0.0,
                          largest_weight_ratio=0.0, largest_weight_ratio_where=None, capped_by_storage={},
                          spatial_blend=self.spatial_blend, max_cell_factor=self.max_cell_factor,
                          n_cell_limited=0, cell_limited_by_storage={}, increment_not_placed_mm_max=0.0)

    def _ratio(self, var):
        """cap for one storage (None = not capped)"""
        if isinstance(self.max_weight_ratio, dict):
            return self.max_weight_ratio.get(var)
        return self.max_weight_ratio

    def setup(self, f):
        super().setup(f)
        for v in self.spatial_blend:
            if v not in f.bounds.names:
                print('non_negative partition: spatial_blend for %s ignored (not in the DA state %s)' % (v, f.bounds.names))
            elif f.bounds.is_bounded(v):
                print('non_negative partition: spatial_blend for the bounded storage %s spreads its correction also '
                      'into cells where it is empty or full' % v)
        if isinstance(self.max_weight_ratio, dict):
            unknown = [v for v in self.max_weight_ratio if v not in f.bounds.names]
            if unknown:
                print('non_negative partition: max_weight_ratio for %s ignored (not in the DA state %s)'
                      % (unknown, f.bounds.names))

    def _cap_weights(self, W, cols=None):
        """
        W (n_state x n_obs, scipy.sparse CSC with the pattern of the localization taper): weights of every element
        in the increment of each observed sub-basin. Per sub-basin and storage, weights above max_weight_ratio x the
        median positive weight are set to that limit. The normalisation below (H w) keeps the sub-basin increment
        unchanged; only its distribution changes.
        """
        f = self.f
        names = f.bounds.names
        for j in (range(W.shape[1]) if cols is None else cols):
            a, b = W.indptr[j], W.indptr[j + 1]
            idx, col = W.indices[a:b], W.data[a:b].copy()                  # elements of the taper column, weights
            tot = col.sum()
            for iv in range(len(names)):
                cap = self._ratio(names[iv])
                if cap is None:
                    continue
                rows = np.where((f._var_idx[idx] == iv) & (col > 0))[0]
                if len(rows) < 3:
                    continue
                med = np.median(col[rows])
                if med <= 0:
                    continue
                ratio = col[rows] / med
                k = int(np.argmax(ratio))
                if ratio[k] > self.stats['largest_weight_ratio']:
                    self.stats['largest_weight_ratio'] = float(ratio[k])
                    self.stats['largest_weight_ratio_where'] = dict(
                        storage=names[iv], cell=int(idx[rows[k]] // len(names)), sub_basin=j + 1,
                        window_end=str(getattr(f, '_today', '')))
                over = ratio > cap
                if over.any():
                    removed = float((col[rows[over]] - cap * med).sum())
                    col[rows[over]] = cap * med
                    self.stats['n_weights_capped'] += int(over.sum())
                    cb = self.stats['capped_by_storage']
                    cb[names[iv]] = cb.get(names[iv], 0) + int(over.sum())
                    if tot > 0:
                        self.stats['weight_share_removed_max'] = max(self.stats['weight_share_removed_max'],
                                                                     removed / tot)
            W.data[a:b] = col
        return W

    def split(self, dX, A, HA, N, X=None):
        """
        Keep the observation-space part of the EnKF update (how much water each sub-basin gains or loses) but
        redistribute it over the cells and storages of that sub-basin with non-negative shares, so that no storage
        is emptied to fill another one.

        dX : EnKF increments (n_state x N);  A, HA : forecast anomalies of states / observation equivalents
        For sub-basin j:   delta_j = (H dX)_j  (per member)
                           w_ij  = L_ij * var_i * max(corr(A_i, HA_j), 0)       (0 for anti-correlated elements)
                           dX_i  = sum_j  w_ij / (H w_.j)_j * delta_j            (so that H dX is unchanged)
        If no element of a sub-basin is positively correlated, the shares fall back to L_ij * var_i.
        Bound-aware (when X is given): an element may not be pushed below its lower bound (snow, soil, canopy,
        wetlands, reservoirs: 0; river: its floor) or above its upper bound (soil smax, snow relative to its forecast); the share of a capped
        element is redistributed to the other elements of the same sub-basin so that the sub-basin increment is
        preserved (a few fixed-point iterations per sub-basin). Without this, negative winter innovations push
        thin snow packs below zero and the threshold adds the water back silently.
        """
        f = self.f
        delta = f._DM(states=dX)                                                # n_obs x N, sub-basin increments
        var = np.sum(A * A, 1) / (N - 1)                                           # n_state
        sd_x = np.sqrt(var)
        sd_y = np.sqrt(np.sum(HA * HA, 1) / (N - 1))                               # n_obs
        '''weights on the pattern of the localization taper (sparse, n_state x n_obs); the dense n_state x n_obs
        arrays corr / W of the global case were 1 GB each (until 8 Oct 2026)'''
        L = f._L_state.tocsc()                                                  # column access: elements of sub-basin j
        rows = np.repeat(np.arange(L.shape[0]), np.diff(f._L_state.indptr))     # row of every non-zero (CSR order)
        pxy = f.tapered_cross_cov(A, HA, N, taper=False)                        # CSR, pattern of L_state, untapered
        with np.errstate(invalid='ignore', divide='ignore'):
            corr = pxy.data / (sd_x[rows] * sd_y[pxy.indices])
        corr = np.nan_to_num(corr)
        W = sparse.csr_matrix((f._L_state.data * var[rows] * np.maximum(corr, 0.0), pxy.indices, pxy.indptr),
                              shape=L.shape).tocsc()                            # explicit zeros kept (same pattern as L)
        if self.max_weight_ratio is not None:
            W = self._cap_weights(W)
        if self.spatial_blend:
            W = self._blend_weights(W, L)
        hw = (f._DM(states=W)).diagonal().copy()                                # (H w_.j)_j
        bad = hw <= 0
        if bad.any():                                                              # fall back to variance-only shares
            for j in np.where(bad)[0]:
                a, b = W.indptr[j], W.indptr[j + 1]
                W.data[a:b] = L.data[a:b] * var[L.indices[a:b]]
            if self.max_weight_ratio is not None:
                W = self._cap_weights(W, cols=np.where(bad)[0])
            if self.spatial_blend:
                W = self._blend_weights(W, L, cols=np.where(bad)[0])
            hw[bad] = (f._DM(states=W)).diagonal()[bad]
        ok = hw > 0
        scale = np.zeros_like(hw)
        scale[ok] = 1.0 / hw[ok]
        S = (W @ sparse.diags(scale)).tocsc()                                      # normalised shares, H S = I (block)
        dX_new = np.asarray(S @ delta)                                             # n_state x N

        if X is not None and (f.bounds.any_finite() or self.max_cell_factor is not None):
            dX_new = self._apply_bounds(dX_new, S, delta, X)

        self.stats['n_updates'] += 1
        self.stats['n_obs_fallback'] += int(bad.sum())
        self.stats['n_obs_unresolved'] += int((~ok).sum())
        self.stats['max_abs_inc_enkf'] = max(self.stats['max_abs_inc_enkf'], float(np.abs(dX).max()))
        self._record_max(dX_new)
        self.stats['max_abs_inc_new'] = max(self.stats['max_abs_inc_new'], float(np.abs(dX_new).max()))
        return dX_new

    def _blend_weights(self, W, L, cols=None):
        """per sub-basin and listed storage: w' = (1-b) w + b mean(w) over the storage's elements in the sub-basin
        (elements with a localization weight > 0); the storage's total weight in the sub-basin is unchanged.
        W, L: sparse CSC with the same pattern (weights, localization taper)"""
        f = self.f
        names = f.bounds.names
        for j in (range(W.shape[1]) if cols is None else cols):
            a, b = W.indptr[j], W.indptr[j + 1]
            idx, lcol = W.indices[a:b], L.data[a:b]
            for v, bl in self.spatial_blend.items():
                if v not in names:
                    continue
                rows = np.where((f._var_idx[idx] == names.index(v)) & (lcol > 0))[0]
                if len(rows) == 0:
                    continue
                w = W.data[a:b][rows]
                if w.sum() <= 0:
                    continue
                W.data[a + rows] = (1.0 - bl) * w + bl * w.mean() * lcol[rows] / max(lcol[rows].mean(), 1e-12)
        return W

    def _apply_bounds(self, dX_new, S, delta, X, n_iter=12):
        """
        S     : normalised shares (n_state x n_obs, sparse CSC), H S = I on the block diagonal; dX_new = S @ delta
        delta : sub-basin increments (n_obs x N);  X : forecast states (n_state x N)
        For every sub-basin j: elements whose increment would cross a bound are fixed at the bound, the remaining
        increment of j is redistributed over the free elements with their shares; repeated until no new crossing.
        With max_cell_factor f every element of j is in addition limited to |dX_i| <= f |delta_j| (per member).
        """
        f = self.f
        cap_lo = f.bounds.LB - X                                 # most negative allowed increment (window-aware)
        cap_hi = f.bounds.UB - X                                 # most positive allowed increment
        out = dX_new.copy()
        n_capped, moved = 0, 0.0
        for j in range(S.shape[1]):
            a, b = S.indptr[j], S.indptr[j + 1]
            pos = S.data[a:b] > 0
            rows, s = S.indices[a:b][pos], S.data[a:b][pos]
            if len(rows) == 0:
                continue
            hj = f.h_row(j, rows)                             # row j of H on these elements (H S = I: sum(hj s) = 1)
            lo_j, hi_j = cap_lo[rows], cap_hi[rows]
            if self.max_cell_factor is not None:
                if isinstance(self.max_cell_factor, dict):
                    fac = np.array([self.max_cell_factor.get(f.bounds.names[i], np.inf) for i in f._var_idx[rows]])
                else:
                    fac = np.full(len(rows), self.max_cell_factor)
                lim = fac[:, None] * np.abs(delta[j])[None, :]                     # (n_E, N)
                lo_c, hi_c = np.maximum(lo_j, -lim), np.minimum(hi_j, lim)
                cell_lim = (out[rows, :] < lo_c - 1e-9) | (out[rows, :] > hi_c + 1e-9)
                bound_lim = (out[rows, :] < lo_j - 1e-9) | (out[rows, :] > hi_j + 1e-9)
                only_cell = cell_lim & ~bound_lim
                if only_cell.any():
                    self.stats['n_cell_limited'] += int(only_cell.sum())
                    cb = self.stats['cell_limited_by_storage']
                    vi = f._var_idx[rows]
                    for iv in np.unique(vi[only_cell.any(1)]):
                        nm = f.bounds.names[iv]
                        cb[nm] = cb.get(nm, 0) + int(only_cell[vi == iv].sum())
                lo_j, hi_j = lo_c, hi_c
            dj = out[rows, :]                                 # current increments of sub-basin j (n_E x N)
            fixed = np.zeros_like(dj, dtype=bool)
            for _ in range(n_iter):
                lo = dj < lo_j - 1e-9
                hi = dj > hi_j + 1e-9
                new_fix = (lo | hi) & ~fixed
                if not new_fix.any():
                    break
                dj = np.where(lo & ~fixed, lo_j, dj)
                dj = np.where(hi & ~fixed, hi_j, dj)
                fixed |= new_fix
                # what the fixed elements contribute to the sub-basin increment, and what is left for the free ones
                rem = delta[j] - hj @ np.where(fixed, dj, 0.0)                   # (N,)
                denom = hj @ np.where(fixed, 0.0, s[:, None])                    # (N,)
                ok = denom > 1e-12
                scale = np.where(ok, rem / np.where(ok, denom, 1.0), 0.0)
                dj = np.where(fixed, dj, s[:, None] * scale[None, :])
            miss = float(np.abs(delta[j] - hj @ dj).max())                       # increment that found no room
            self.stats['increment_not_placed_mm_max'] = max(self.stats['increment_not_placed_mm_max'], miss)
            n_capped += int(fixed.sum())
            moved += float(np.abs(dj - out[rows, :]).sum())
            out[rows, :] = dj
        self.stats['n_bound_capped'] += n_capped
        self.stats['water_redistributed_mm'] += moved
        return out

    def _record_max(self, dX_new):
        """where the largest single-element increment of the run happened (storage, cell, sub-basin, member, date)"""
        f = self.f
        i, n = np.unravel_index(np.argmax(np.abs(dX_new)), dX_new.shape)
        v = float(dX_new[i, n])
        if abs(v) >= self.stats['max_abs_inc_new'] - 1e-9:
            nvar = len(f.bounds.names)
            self.stats['max_inc_where'] = dict(increment_mm=v, storage=f.bounds.names[i % nvar], cell=int(i // nvar),
                                               sub_basin=int(f._owner[i]) + 1, member=int(n) + 1,
                                               window_end=str(getattr(f, '_today', '')))


PARTITIONS = {c.KIND: c for c in (EnKFPartition, NonNegativePartition)}


def make_partition(cfg=None):
    if cfg is None:
        cfg = {'kind': 'enkf'}
    if isinstance(cfg, str):
        cfg = {'kind': cfg}
    cfg = dict(cfg)
    kind = str(cfg.pop('kind', 'enkf')).lower()
    if kind not in PARTITIONS:
        raise ValueError('increment_partition must be one of %s, got %s' % (list(PARTITIONS), kind))
    try:
        return PARTITIONS[kind](**cfg)
    except TypeError as err:
        raise ValueError('increment_partition "%s": unknown setting (%s)' % (kind, err))
