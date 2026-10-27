"""
How the Kalman increment of a sub-basin is distributed over the cells and storages of that sub-basin
(src_DA.EnKF_localized). DA_setting.json, "method" block, "increment_partition":

    {"kind": "enkf"}           increments as given by the (localized) Kalman gain
    {"kind": "non_negative"}   sign-consistent partition: the sub-basin increment delta_j = (H dX)_j of the Kalman
        update is kept, but distributed with non-negative shares ~ variance x max(correlation with the sub-basin TWS, 0)
        (fallback: variance only), so that no storage is emptied to fill another one. Bound-aware: an element is not
        pushed across its (window-aware) bound; its share goes to the other elements of the same sub-basin.

Options of "non_negative" (all default off):
    "max_weight_ratio": {"groundwstor": 5, ...} or one number
        caps the weight of an element at this multiple of the median weight of the same storage among the active
        elements of the sub-basin (active: at least ACTIVE_WEIGHT_FRACTION of the storage's largest weight there), so
        that one cell with an extreme ensemble spread cannot absorb the correction of its whole sub-basin.
    "spatial_blend": {"groundwstor": 0.8, ...}
        mixes the weights of a storage with their mean over the storage's active elements, w' = (1-b) w + b mean(w),
        keeping the storage's total share: GRACE has no information below the sub-basin scale. Meant for unbounded or
        sparse storages (groundwater, river, wetlands).
    "max_cell_factor": {"soilmoist": 2, ...} or one number
        no element may change by more than this multiple of the sub-basin increment of its member
        (|dX_i| <= f |delta_j|); the excess is redistributed like a bound.
    "storage_shares": {"storages": ["soilmoist", "groundwstor"], "mix": 1.0}
        vertical partition by the open loop's seasonal amplitude: the total share of the listed storages in a
        sub-basin increment is kept as the ensemble gives it, but re-split among them in proportion to each storage's
        seasonal amplitude in the open loop of that sub-basin (std of the 12 calendar-month means of the
        ensemble-mean unit series, Res/<case>/Res_OL.h5), s' = (1 - mix) s_ens + mix s_OL. Within a storage the cells
        keep their ensemble weights. GRACE sees only the sum; the ensemble's soil / groundwater covariance follows how
        the inflation noise is split rather than where the model's seasonal error is, so the model's own seasonal
        partition is the better prior. H dX = delta is unchanged.
    "boundary_smoothing": {"storages": ["groundwstor"], "length_km": 50, "n_iter": 10, "alpha": 1.0, "cutoff": 2.0}
        with one observation per sub-basin the increment field is built inside every sub-basin from that sub-basin's
        increment alone and steps at sub-basin boundaries. Mass-preserving (pycnophylactic, Tobler 1979) smoothing of
        the increment of every listed storage and member: n_iter times Y <- G Y (Gaussian kernel of length_km, zero
        beyond cutoff x length_km, cos(lat)-weighted, over the cells of the sub-basins), each pass followed by adding to
        every sub-basin the constant that restores its mean increment of that storage; then Y' = (1-alpha) Y0 +
        alpha Y. The storage totals of every sub-basin, and therefore H dX = delta, are kept exactly. The bound step
        then works additively, so the smoothing survives the bounds. Meant for unbounded storages (groundwater):
        smoothing bounded storages moves water into cells at their limits.
Order: weight cap, blend, normalisation (H S = I), storage shares, boundary smoothing, bounds and per-cell limit.
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
    ACTIVE_WEIGHT_FRACTION = 0.01      # _cap_weights: an element counts as active above 1 % of the storage's largest weight

    def __init__(self, max_weight_ratio=None, spatial_blend=None, max_cell_factor=None, storage_shares=None,
                 boundary_smoothing=None):
        super().__init__()
        bs = dict(boundary_smoothing or {})
        self.boundary_smoothing = None
        if bs:
            unknown_keys = set(bs) - {'length_km', 'n_iter', 'alpha', 'cutoff', 'storages'}
            if unknown_keys:
                raise ValueError('boundary_smoothing: unknown keys %s' % sorted(unknown_keys))
            if not bs.get('storages'):
                raise ValueError('boundary_smoothing needs "storages", e.g. ["groundwstor"]')
            bsd = dict(length_km=float(bs.get('length_km', 50.0)), n_iter=int(bs.get('n_iter', 10)),
                       alpha=float(bs.get('alpha', 1.0)), cutoff=float(bs.get('cutoff', 2.0)),
                       storages=[str(v) for v in bs['storages']])
            if bsd['length_km'] <= 0 or bsd['n_iter'] < 1 or not 0.0 < bsd['alpha'] <= 1.0 or bsd['cutoff'] <= 0:
                raise ValueError('boundary_smoothing needs length_km > 0, n_iter >= 1, 0 < alpha <= 1, cutoff > 0, '
                                 'got %s' % bs)
            self.boundary_smoothing = bsd
        self._bs = None                     # kernel / unit operators of the boundary smoothing (setup)
        '''storage shares: the soil / groundwater split of each sub-basin increment follows the open loop's seasonal
        amplitudes instead of the ensemble spread, see _storage_share_mix'''
        ss = dict(storage_shares or {})
        self.storage_shares = None
        if ss:
            unknown_keys = set(ss) - {'storages', 'mix'}
            if unknown_keys:
                raise ValueError('storage_shares: unknown keys %s' % sorted(unknown_keys))
            mix = float(ss.get('mix', 1.0))
            if not 0.0 < mix <= 1.0:
                raise ValueError('storage_shares mix must be in (0, 1], got %s' % mix)
            sto = [str(v) for v in ss.get('storages', ['soilmoist', 'groundwstor'])]
            if len(sto) < 2:
                raise ValueError('storage_shares needs at least two storages, got %s' % sto)
            self.storage_shares = dict(storages=sto, mix=mix)
        self._ss_idx = []                   # storage indices of the group
        self._ss_table = None               # (n_obs, n_group) open-loop amplitude shares within the group
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
                          max_abs_inc_new=0.0, n_bound_capped=0, bound_capped_by_storage={}, water_redistributed_mm=0.0,
                          max_inc_where=None, max_weight_ratio=self.max_weight_ratio, n_weights_capped=0,
                          weight_share_removed_max=0.0, largest_weight_ratio=0.0, largest_weight_ratio_where=None,
                          capped_by_storage={}, spatial_blend=self.spatial_blend, max_cell_factor=self.max_cell_factor,
                          n_cell_limited=0, cell_limited_by_storage={}, increment_not_placed_mm_max=0.0,
                          storage_shares=self.storage_shares, ss_n=0, ss_share_ens_sum={}, ss_share_new_sum={},
                          ss_n_obs_no_amplitude=0, ss_n_taper_fallback=0,
                          boundary_smoothing=self.boundary_smoothing, bs_n=0, bs_step_before_sum=0.0,
                          bs_step_after_sum=0.0, bs_within_before_sum=0.0, bs_within_after_sum=0.0,
                          bs_moved_mm_sum=0.0)

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
        if self.boundary_smoothing is not None:
            self._setup_boundary_smoothing(f)
        if self.storage_shares is not None:
            self._setup_storage_shares(f)
        if isinstance(self.max_weight_ratio, dict):
            unknown = [v for v in self.max_weight_ratio if v not in f.bounds.names]
            if unknown:
                print('non_negative partition: max_weight_ratio for %s ignored (not in the DA state %s)'
                      % (unknown, f.bounds.names))

    def _cap_weights(self, W, cols=None):
        """
        W (n_state x n_obs, scipy.sparse CSC with the pattern of the localization taper): weights of every element
        in the increment of each observed sub-basin. Per sub-basin and storage, weights above max_weight_ratio x the
        median ACTIVE weight are set to that limit. Active = at least ACTIVE_WEIGHT_FRACTION of the storage's largest
        weight in the sub-basin: for a storage present in every cell (groundwater, soil) this is practically every
        positive weight; for a storage that exists in a few cells only (wetlands, snow in a warm unit) the many
        near-zero weights of the cells without it are left out, so that the median is that of a typical wetland /
        snow cell and the cap is meaningful.
        The normalisation below (H w) keeps the sub-basin increment unchanged; only its distribution changes.
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
                rows = rows[col[rows] >= self.ACTIVE_WEIGHT_FRACTION * col[rows].max()]
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
        '''weights on the pattern of the localization taper (sparse, n_state x n_obs)'''
        Ls = f._L_state                                                         # pattern of the shares (CSR)
        L = Ls.tocsc()                                                          # column access: elements of sub-basin j
        rows = np.repeat(np.arange(L.shape[0]), np.diff(Ls.indptr))             # row of every non-zero (CSR order)
        pxy = f.tapered_cross_cov(A, HA, N, taper=False)                        # CSR, pattern of L_state, untapered
        with np.errstate(invalid='ignore', divide='ignore'):
            corr = pxy.data / (sd_x[rows] * sd_y[pxy.indices])
        corr = np.nan_to_num(corr)
        W = sparse.csr_matrix((Ls.data * var[rows] * np.maximum(corr, 0.0), pxy.indices, pxy.indptr),
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
        if self.storage_shares is not None:
            S = self._storage_share_mix(S, L, X)                                   # also the shares used by the bounds
        dX_new = np.asarray(S @ delta)                                             # n_state x N

        if self._bs is not None:
            dX_new = self._smooth_boundaries(dX_new, S, delta)
        ss_before = np.abs(np.mean(dX_new, 1)) if self.storage_shares is not None else None
        if X is not None and (f.bounds.any_finite() or self.max_cell_factor is not None):
            dX_new = self._apply_bounds(dX_new, S, delta, X)
        if ss_before is not None:
            '''per-storage size of the member-mean increment before / after the bound step (sum |dX| over the elements):
            shows how much of a storage's share the bounds hand to other storages'''
            ss_after = np.abs(np.mean(dX_new, 1))
            b, a = self.stats.setdefault('ss_inc_before_sum', {}), self.stats.setdefault('ss_inc_after_sum', {})
            for iv, v in enumerate(f.bounds.names):
                m = f._var_idx == iv
                b[v] = b.get(v, 0.0) + float(ss_before[m].sum())
                a[v] = a.get(v, 0.0) + float(ss_after[m].sum())

        self.stats['n_updates'] += 1
        self.stats['n_obs_fallback'] += int(bad.sum())
        self.stats['n_obs_unresolved'] += int((~ok).sum())
        self.stats['max_abs_inc_enkf'] = max(self.stats['max_abs_inc_enkf'], float(np.abs(dX).max()))
        self._record_max(dX_new)
        self.stats['max_abs_inc_new'] = max(self.stats['max_abs_inc_new'], float(np.abs(dX_new).max()))
        return dX_new

    def _setup_storage_shares(self, f):
        """open-loop seasonal amplitude shares of the group storages per sub-basin (n_obs x n_group)"""
        from pathlib import Path
        names = list(f.bounds.names)
        sto = [v for v in self.storage_shares['storages'] if v in names]
        unknown = [v for v in self.storage_shares['storages'] if v not in names]
        if unknown:
            print('non_negative partition: storage_shares storages %s not in the DA state %s - ignored' % (unknown, names))
        if len(sto) < 2:
            print('non_negative partition: storage_shares needs two storages of the DA state - switched off')
            self.storage_shares = None
            return
        self.storage_shares['storages'] = sto
        self._ss_idx = [names.index(v) for v in sto]
        b = f.DA_setting.basic
        src = Path(b.res_permanent) / b.case / 'Res_OL.h5'
        n_obs = f._L_state.shape[1]
        amp = seasonal_amplitude_table(src, sto, n_obs)                         # (n_obs, n_group) mm
        tot = amp.sum(1, keepdims=True)
        self._ss_table = np.where(tot > 0, amp / np.where(tot > 0, tot, 1.0), 0.0)
        med = {v: round(float(np.median(self._ss_table[:, k])), 3) for k, v in enumerate(sto)}
        print('non_negative partition: storage shares %s (mix %.2f) from %s; median OL amplitude share %s'
              % (sto, self.storage_shares['mix'], src, med))

    def _storage_share_mix(self, S, L, X=None):
        """
        S (n_state x n_obs, CSC, H S = I). For sub-basin j and group storage v: q_vj = (H S_v)_j, the share of the
        sub-basin increment that goes into v; Q_j = sum_v q_vj (kept). New shares
            q'_vj = (1 - mix) q_vj + mix Q_j s_vj,   s_vj = open-loop seasonal amplitude share of v in j (sum 1)
        realised by scaling the elements of v in column j by q'_vj / q_vj (cells keep their ensemble weights). A
        storage with q_vj = 0 but q'_vj > 0 gets its share along the localization taper of its elements. Columns
        are re-normalised at the end, so H S' = I holds exactly. A sub-basin without open-loop amplitude keeps S.
        The taper fallback of a BOUNDED storage (river, wetlands, soil) is weighted by the member-mean forecast amount
        of each cell (X given), so that e.g. a wetland share is not spread into cells without wetland.
        """
        f = self.f
        m = self.storage_shares['mix']
        S = sparse.csc_matrix(S, copy=True)
        n_obs = S.shape[1]
        G = self._ss_idx
        masks = [(f._var_idx == iv).astype(float) for iv in G]
        q = np.zeros((n_obs, len(G)))
        for k in range(len(G)):
            Sk = sparse.csc_matrix(S.multiply(masks[k][:, None]))
            q[:, k] = f._DM(states=Sk).diagonal()
        Q = q.sum(1)
        T = self._ss_table
        has = (T.sum(1) > 0) & (Q > 1e-12)
        self.stats['ss_n_obs_no_amplitude'] += int((~has).sum())
        target = np.where(has[:, None], (1.0 - m) * q + m * Q[:, None] * T, q)
        col = np.repeat(np.arange(n_obs), np.diff(S.indptr))                  # column of every stored entry
        vi = f._var_idx[S.indices]
        extra = None
        for k, iv in enumerate(G):
            okq = q[:, k] > 1e-12
            fac = np.where(okq, target[:, k] / np.where(okq, q[:, k], 1.0), 1.0)
            sel = vi == iv
            S.data[sel] *= fac[col[sel]]
            need = has & ~okq & (target[:, k] > 1e-12)                         # no ensemble weight: use the taper
            if need.any():
                Lk = sparse.csc_matrix(L.multiply(masks[k][:, None]))
                if X is not None and f.bounds.is_bounded(f.bounds.names[iv]):
                    amt = np.maximum(np.mean(X, 1), 0.0) * masks[k]          # where the storage holds water
                    Lk = sparse.csc_matrix(Lk.multiply(amt[:, None]))
                Lk = Lk @ sparse.diags(need.astype(float))
                hk = f._DM(states=Lk).diagonal()
                okh = hk > 0
                sc = np.where(okh, target[:, k] / np.where(okh, hk, 1.0), 0.0)
                add = Lk @ sparse.diags(sc)
                extra = add if extra is None else extra + add
                self.stats['ss_n_taper_fallback'] += int((need & okh).sum())
        if extra is not None:
            S = sparse.csc_matrix(S + extra)
        hs = f._DM(states=S).diagonal().copy()                                 # re-normalise: H S' = I
        okn = hs > 0
        S = sparse.csc_matrix(S @ sparse.diags(np.where(okn, 1.0 / np.where(okn, hs, 1.0), 0.0)))
        self.stats['ss_n'] += 1
        for k, v in enumerate(self.storage_shares['storages']):
            if has.any():
                e, n = self.stats['ss_share_ens_sum'], self.stats['ss_share_new_sum']
                e[v] = e.get(v, 0.0) + float(np.mean(q[has, k] / Q[has]))
                n[v] = n.get(v, 0.0) + float(np.mean(target[has, k] / Q[has]))
        return S

    def summary(self):
        out = dict(self.stats, kind=self.KIND)
        n = max(out.get('ss_n', 0), 1)
        if self.storage_shares is not None:
            out['ss_mean_share_ensemble'] = {v: round(x / n, 3) for v, x in out.pop('ss_share_ens_sum', {}).items()}
            out['ss_mean_share_used'] = {v: round(x / n, 3) for v, x in out.pop('ss_share_new_sum', {}).items()}
            for key, name in (('ss_inc_before_sum', 'ss_increment_share_before_bounds'),
                              ('ss_inc_after_sum', 'ss_increment_share_after_bounds')):
                d = out.pop(key, {})
                tot = sum(d.values())
                out[name] = {v: round(x / tot, 3) for v, x in d.items()} if tot > 0 else {}
        else:
            for k in [k for k in out if k.startswith('ss_')] + ['storage_shares']:
                out.pop(k, None)
        if self.boundary_smoothing is None:
            for k in [k for k in out if k.startswith('bs_')] + ['boundary_smoothing']:
                out.pop(k, None)
        return out

    def _blend_weights(self, W, L, cols=None):
        """per sub-basin and listed storage: w' = (1-b) w + b mean(w) over the storage's ACTIVE elements in the
        sub-basin (localization weight > 0 and weight >= ACTIVE_WEIGHT_FRACTION x the storage's largest weight there);
        the storage's total weight in the sub-basin is unchanged. Elements below the active threshold keep their
        (near-zero) weight, so that a river / wetland correction is spread along the active channel / wetland cells
        only, not into cells without that storage.
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
                rows = rows[w >= self.ACTIVE_WEIGHT_FRACTION * w.max()]
                w = W.data[a:b][rows]
                W.data[a + rows] = (1.0 - bl) * w + bl * w.mean() * lcol[rows] / max(lcol[rows].mean(), 1e-12)
        return W

    def _setup_boundary_smoothing(self, f):
        """kernel G (n_dom x n_dom, row-normalised, cos(lat)-weighted Gaussian), the unit-mean operator on the domain
        cells, the membership and the 4-neighbour-like boundary / within pairs for the log (see boundary_smoothing)"""
        from scipy.spatial import cKDTree
        from src_DA.localization import GaussianLocalization
        bs = self.boundary_smoothing
        nvar = len(f.bounds.names)
        owner_c = np.asarray(f._owner[::nvar])                                  # sub-basin of every cell (-1: none)
        h_c = np.asarray(f._h_state[::nvar], dtype=float)                       # cell weight in its sub-basin mean
        dom = np.flatnonzero(owner_c >= 0)
        lm = f._sv.DM.local_mask
        if 'lon' in lm:
            lat, lon = np.asarray(lm['lat'], dtype=float), np.asarray(lm['lon'], dtype=float)
        else:
            lat, lon = GaussianLocalization.cell_coordinates(lm)
        if len(lat) != len(owner_c):
            raise ValueError('boundary_smoothing: %d cell coordinates for %d state cells' % (len(lat), len(owner_c)))
        la, lo = np.deg2rad(lat[dom]), np.deg2rad(lon[dom])
        xyz = np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
        R = 6371.0
        rmax = bs['cutoff'] * bs['length_km']
        tree = cKDTree(xyz)
        pairs = tree.query_pairs(2.0 * np.sin(min(rmax / R, np.pi) / 2.0), output_type='ndarray')
        i, j = pairs[:, 0], pairs[:, 1]
        d = 2 * R * np.arcsin(np.clip(np.linalg.norm(xyz[i] - xyz[j], axis=1) / 2, 0, 1))
        w = np.exp(-0.5 * (d / bs['length_km']) ** 2)
        n = len(dom)
        cw = np.cos(la)
        rows = np.concatenate([i, j, np.arange(n)])
        cols = np.concatenate([j, i, np.arange(n)])
        vals = np.concatenate([w * cw[j], w * cw[i], cw])
        G = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
        G = sparse.diags(1.0 / np.asarray(G.sum(axis=1)).ravel()) @ G
        n_obs = f._L_state.shape[1]
        own = owner_c[dom]
        Hc = sparse.csr_matrix((h_c[dom], (own, np.arange(n))), shape=(n_obs, n))
        P = sparse.csr_matrix((np.ones(n), (np.arange(n), own)), shape=(n, n_obs))
        # nearest-neighbour pairs (<= 1.5 cell spacing, ~4-8 neighbours) for the step statistics in the log
        nn = tree.query_pairs(2.0 * np.sin(0.0145 * 1.5 / 2.0), output_type='ndarray')   # 0.0145 rad ~ 0.83 deg
        across = own[nn[:, 0]] != own[nn[:, 1]]
        own_e = np.asarray(f._owner)
        order = np.argsort(own_e, kind='stable')
        starts = np.searchsorted(own_e[order], np.arange(n_obs + 1))
        unit_rows = [order[starts[q]:starts[q + 1]] for q in range(n_obs)]       # all state elements of every unit
        self._bs = dict(dom=dom, G=G.tocsr(), Hc=Hc, P=P, nvar=nvar, nn=nn, across=across, unit_rows=unit_rows)
        print('non_negative partition: boundary_smoothing %s - %d cells, %d kernel pairs, %d neighbour pairs '
              '(%d across unit boundaries)' % (bs, n, len(i), len(nn), int(across.sum())))

    def _smooth_per_storage(self, dX_new, D, T0, n_cell):
        """boundary_smoothing with "storages": every listed storage smoothed separately, its unit means restored"""
        b, bs = self._bs, self.boundary_smoothing
        names = list(self.f.bounds.names)
        D = D.copy()
        dom = b['dom']
        for v in bs['storages']:
            if v not in names:
                continue
            k = names.index(v)
            Y0 = D[dom, k, :]                                                    # (n_dom, N)
            tgt = b['Hc'] @ Y0                                                   # unit-mean increment of storage v
            Y = Y0.copy()
            for _ in range(bs['n_iter']):
                Y = b['G'] @ Y
                Y = Y + b['P'] @ (tgt - b['Hc'] @ Y)
            D[dom, k, :] = (1.0 - bs['alpha']) * Y0 + bs['alpha'] * Y
        out = D.reshape(dX_new.shape)
        T = D[dom].sum(axis=1)
        self._log_steps(T0, T, out)
        return out

    def _log_steps(self, T0, T, out):
        b = self._bs
        nn, ac = b['nn'], b['across']
        st = self.stats
        st['bs_n'] += 1
        if len(nn):
            m0, m1 = T0.mean(axis=1), T.mean(axis=1)
            s0, s1 = np.abs(m0[nn[:, 0]] - m0[nn[:, 1]]), np.abs(m1[nn[:, 0]] - m1[nn[:, 1]])
            if ac.any():
                st['bs_step_before_sum'] += float(np.median(s0[ac]))
                st['bs_step_after_sum'] += float(np.median(s1[ac]))
            if (~ac).any():
                st['bs_within_before_sum'] += float(np.median(s0[~ac]))
                st['bs_within_after_sum'] += float(np.median(s1[~ac]))
        st['bs_moved_mm_sum'] += float(np.abs(T - T0).mean())

    def _smooth_boundaries(self, dX_new, S, delta):
        """mass-preserving smoothing of the listed storages' increments, see boundary_smoothing; returns the new dX"""
        b = self._bs
        nvar, dom = b['nvar'], b['dom']
        n_cell = dX_new.shape[0] // nvar
        D = dX_new.reshape(n_cell, nvar, -1)
        T0 = D[dom].sum(axis=1)                                                  # (n_dom, N) cell TWS increment
        return self._smooth_per_storage(dX_new, D, T0, n_cell)

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
            if self._bs is not None:
                # boundary smoothing can give an increment to elements without a share: bound-check every element
                # of the unit (those keep s = 0, i.e. they take no remainder)
                allr = self._bs['unit_rows'][j]
                sa = np.zeros(len(allr))
                o = np.argsort(rows)
                k = np.searchsorted(rows[o], allr)
                hit = (k < len(rows)) & (rows[o][np.minimum(k, len(rows) - 1)] == allr) if len(rows) else np.zeros(len(allr), bool)
                sa[hit] = s[o][k[hit]]
                rows, s = allr, sa
            if len(rows) == 0:
                continue
            hj = f.h_row(j, rows)                             # row j of H on these elements (H S = I: sum(hj s) = 1)
            lo_j, hi_j = cap_lo[rows], cap_hi[rows]
            ph_lo, ph_hi = lo_j, hi_j                         # physical (window-aware) bounds, for the counter
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
                denom = hj @ np.where(fixed, 0.0, s[:, None])                    # (N,)
                ok = denom > 1e-12
                if self._bs is not None:
                    # free elements keep their smoothed values; the remainder is added on top
                    rem = delta[j] - hj @ dj                                     # (N,)
                    add = np.where(ok, rem / np.where(ok, denom, 1.0), 0.0)
                    dj = np.where(fixed, dj, dj + s[:, None] * add[None, :])
                else:
                    rem = delta[j] - hj @ np.where(fixed, dj, 0.0)               # (N,)
                    scale = np.where(ok, rem / np.where(ok, denom, 1.0), 0.0)
                    dj = np.where(fixed, dj, s[:, None] * scale[None, :])
            miss = float(np.abs(delta[j] - hj @ dj).max())                       # increment that found no room
            self.stats['increment_not_placed_mm_max'] = max(self.stats['increment_not_placed_mm_max'], miss)
            n_capped += int(fixed.sum())
            if fixed.any():                                   # physical bound hits per storage (lower / upper)
                at_lo = fixed & (np.abs(dj - ph_lo) <= 1e-9)
                at_hi = fixed & (np.abs(dj - ph_hi) <= 1e-9) & ~at_lo
                bc = self.stats['bound_capped_by_storage']
                vi = f._var_idx[rows]
                for iv in np.unique(vi[fixed.any(1)]):
                    nm = f.bounds.names[iv]
                    d = bc.setdefault(nm, {'lower': 0, 'upper': 0})
                    d['lower'] += int(at_lo[vi == iv].sum())
                    d['upper'] += int(at_hi[vi == iv].sum())
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


def seasonal_amplitude_table(path, storages, n_sub):
    """
    Seasonal amplitude of the open loop per sub-basin and storage: std over the 12 calendar-month means of the
    ensemble-mean daily unit series (Res_OL.h5: <sub_basin_k>/<storage>/<member>; member '0' left out as in the
    evaluation and in inflation.split_table). Storages missing in the file get 0. Returns (n_sub, n_storage) [mm].
    """
    import h5py, datetime as dt
    from pathlib import Path
    if not Path(path).exists():
        raise FileNotFoundError('storage_shares needs the open-loop basin series %s - run '
                                'collect_and_statistics(Stage.OL) first' % path)
    out = np.zeros((n_sub, len(storages)))
    with h5py.File(path, 'r') as h:
        if 'sub_basin_%d' % n_sub not in h or 'sub_basin_%d' % (n_sub + 1) in h:
            raise ValueError('storage_shares: %s does not have the %d sub-basins of the DA' % (path, n_sub))
        t = h['time'][:]

        def month_of(x):
            y = int(x)
            n = (dt.date(y + 1, 1, 1) - dt.date(y, 1, 1)).days
            d = dt.date(y, 1, 1) + dt.timedelta(days=min(int(round((x - y) * n)), n - 1))
            return d.month
        mon = np.array([month_of(x) for x in t])
        for j in range(n_sub):
            for i, v in enumerate(storages):
                key = 'sub_basin_%d/%s' % (j + 1, v)
                if key not in h:
                    continue
                mem = [m for m in h[key] if m != '0'] or list(h[key])
                x = np.mean([h[key][m][:] for m in mem], axis=0)
                clim = np.array([np.nanmean(x[mon == k]) for k in range(1, 13) if (mon == k).any()])
                out[j, i] = float(np.nanstd(clim)) if len(clim) > 1 else 0.0
    return np.nan_to_num(np.maximum(out, 0.0))


PARTITIONS = {c.KIND: c for c in (EnKFPartition, NonNegativePartition)}


def make_partition(cfg=None):
    cfg = dict(cfg or {'kind': 'enkf'})
    kind = str(cfg.pop('kind', 'enkf')).lower()
    if kind not in PARTITIONS:
        raise ValueError('increment_partition must be one of %s, got %s' % (list(PARTITIONS), kind))
    try:
        return PARTITIONS[kind](**cfg)
    except TypeError as err:
        raise ValueError('increment_partition "%s": unknown setting (%s)' % (kind, err))
