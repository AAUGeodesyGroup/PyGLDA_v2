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
    (a plain string "enkf" / "non_negative" is accepted as well)
"""
import numpy as np


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

    def __init__(self, max_weight_ratio=None):
        super().__init__()
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
                          largest_weight_ratio=0.0, largest_weight_ratio_where=None, capped_by_storage={})

    def _ratio(self, var):
        """cap for one storage (None = not capped)"""
        if isinstance(self.max_weight_ratio, dict):
            return self.max_weight_ratio.get(var)
        return self.max_weight_ratio

    def setup(self, f):
        super().setup(f)
        if isinstance(self.max_weight_ratio, dict):
            unknown = [v for v in self.max_weight_ratio if v not in f.bounds.names]
            if unknown:
                print('non_negative partition: max_weight_ratio for %s ignored (not in the DA state %s)'
                      % (unknown, f.bounds.names))

    def _cap_weights(self, W, cols=None):
        """
        W (n_state x n_obs): weights of every element in the increment of each observed sub-basin. Per sub-basin
        and storage, weights above max_weight_ratio x the median positive weight are set to that limit. The
        normalisation below (H w) keeps the sub-basin increment unchanged; only its distribution changes.
        """
        f = self.f
        names = f.bounds.names
        for j in (range(W.shape[1]) if cols is None else cols):
            col = W[:, j].copy()
            tot = col.sum()
            for iv in range(len(names)):
                cap = self._ratio(names[iv])
                if cap is None:
                    continue
                rows = np.where((f._var_idx == iv) & (col > 0))[0]
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
                        storage=names[iv], cell=int(rows[k] // len(names)), sub_basin=j + 1,
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
            W[:, j] = col
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
        wetlands, reservoirs: 0; river: its floor) or above its upper bound (snow 1000 mm); the share of a capped
        element is redistributed to the other elements of the same sub-basin so that the sub-basin increment is
        preserved (a few fixed-point iterations per sub-basin). Without this, negative winter innovations push
        thin snow packs below zero and the threshold adds the water back silently.
        """
        f = self.f
        delta = f._DM(states=dX)                                                # n_obs x N, sub-basin increments
        var = np.sum(A * A, 1) / (N - 1)                                           # n_state
        sd_x = np.sqrt(var)
        sd_y = np.sqrt(np.sum(HA * HA, 1) / (N - 1))                               # n_obs
        with np.errstate(invalid='ignore', divide='ignore'):
            corr = (A @ HA.T) / (N - 1) / np.outer(sd_x, sd_y)                    # n_state x n_obs
        corr = np.nan_to_num(corr)
        W = f._L_state * var[:, None] * np.maximum(corr, 0.0)                   # n_state x n_obs
        if self.max_weight_ratio is not None:
            W = self._cap_weights(W)
        HW = f._DM(states=W)                                                    # n_obs x n_obs, column j -> (H w_.j)
        hw = np.diag(HW).copy()
        bad = hw <= 0
        if bad.any():                                                              # fall back to variance-only shares
            W[:, bad] = f._L_state[:, bad] * var[:, None]
            if self.max_weight_ratio is not None:
                W = self._cap_weights(W, cols=np.where(bad)[0])
            hw[bad] = np.diag(f._DM(states=W))[bad]
        ok = hw > 0
        scale = np.zeros_like(hw)
        scale[ok] = 1.0 / hw[ok]
        dX_new = (W * scale[None, :]) @ delta                                      # n_state x N

        if X is not None and f.bounds.any_finite():
            dX_new = self._apply_bounds(dX_new, W * scale[None, :], delta, X)

        self.stats['n_updates'] += 1
        self.stats['n_obs_fallback'] += int(bad.sum())
        self.stats['n_obs_unresolved'] += int((~ok).sum())
        self.stats['max_abs_inc_enkf'] = max(self.stats['max_abs_inc_enkf'], float(np.abs(dX).max()))
        self._record_max(dX_new)
        self.stats['max_abs_inc_new'] = max(self.stats['max_abs_inc_new'], float(np.abs(dX_new).max()))
        return dX_new

    def _apply_bounds(self, dX_new, S, delta, X, n_iter=6):
        """
        S     : normalised shares (n_state x n_obs), H S = I on the block diagonal; dX_new = S @ delta
        delta : sub-basin increments (n_obs x N);  X : forecast states (n_state x N)
        For every sub-basin j: elements whose increment would cross a bound are fixed at the bound, the remaining
        increment of j is redistributed over the free elements with their shares; repeated until no new crossing.
        """
        f = self.f
        cap_lo = f.bounds.LB - X                                 # most negative allowed increment (window-aware)
        cap_hi = f.bounds.UB - X                                 # most positive allowed increment
        out = dX_new.copy()
        n_capped, moved = 0, 0.0
        for j in range(S.shape[1]):
            rows = np.where(S[:, j] > 0)[0]
            if len(rows) == 0:
                continue
            s = S[rows, j]
            dj = out[rows, :]                                 # current increments of sub-basin j (n_E x N)
            fixed = np.zeros_like(dj, dtype=bool)
            for _ in range(n_iter):
                lo = dj < cap_lo[rows] - 1e-9
                hi = dj > cap_hi[rows] + 1e-9
                new_fix = (lo | hi) & ~fixed
                if not new_fix.any():
                    break
                dj = np.where(lo & ~fixed, cap_lo[rows], dj)
                dj = np.where(hi & ~fixed, cap_hi[rows], dj)
                fixed |= new_fix
                # what the fixed elements contribute to the sub-basin increment, and what is left for the free ones
                full = np.zeros_like(out); full[rows] = np.where(fixed, dj, 0.0)
                rem = delta[j] - f._DM(states=full)[j]                           # (N,)
                free_s = np.zeros_like(out); free_s[rows] = np.where(fixed, 0.0, s[:, None])
                denom = f._DM(states=free_s)[j]                                  # (N,)
                ok = denom > 1e-12
                scale = np.where(ok, rem / np.where(ok, denom, 1.0), 0.0)
                dj = np.where(fixed, dj, s[:, None] * scale[None, :])
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
