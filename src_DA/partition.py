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
        "residence": {"tau_ref_days": 30} (optional, default off) weights each group storage's amplitude share by how
        long that storage keeps water in the open loop of the sub-basin: s_vj ~ amp_vj * tau_vj / (tau_vj + tau_ref),
        tau_vj = e-folding time (days) of the deseasonalised daily ensemble-mean unit series, -1 / ln(rho_1) from its
        lag-1 autocorrelation (calendar-day climatology removed; clipped to [tau_min_days, tau_max_days], default
        [0.5, 3650]; over a few years long times are biased low, e.g. 400 d read as ~150-250 d, which hardly changes
        the factor of slow stores). Storages that drain within days (small and medium rivers) get a small share, storages that hold
        water for months (soil, wetlands, floodplains, large rivers) keep theirs. Put "riverstor" into "storages" for
        the river to be weighted at all (otherwise its share follows the ensemble weights). Writes
        <res_permanent>/<case>/storage_shares_residence.csv (tau and factor per sub-basin and storage). Motivation:
        the DA's river increments leave the cell within the month and widen the discharge variability (DA reviews
        207 (5), 211, 214).
    "boundary_smoothing": {"storages": ["groundwstor"], "length_km": 50, "n_iter": 10, "alpha": 1.0, "cutoff": 2.0}
        with one observation per sub-basin the increment field is built inside every sub-basin from that sub-basin's
        increment alone and steps at sub-basin boundaries. Mass-preserving (pycnophylactic, Tobler 1979) smoothing of
        the increment of every listed storage and member: n_iter times Y <- G Y (Gaussian kernel of length_km, zero
        beyond cutoff x length_km, cos(lat)-weighted, over the cells of the sub-basins), each pass followed by adding to
        every sub-basin the constant that restores its mean increment of that storage; then Y' = (1-alpha) Y0 +
        alpha Y. The storage totals of every sub-basin, and therefore H dX = delta, are kept exactly. The bound step
        then works additively, so the smoothing survives the bounds. Meant for unbounded storages (groundwater):
        smoothing bounded storages moves water into cells at their limits.
    "bound_overflow": {"exclude": ["groundwstor"], "fallback": false}
        where an element hits a bound, its share of the sub-basin increment normally goes to all free elements of the
        sub-basin in proportion to their shares; unbounded groundwater then collects the seasonal correction that full
        or empty soil and snow cannot take (diagnosis 1 Nov 2026, DA reviews 180/181). With "exclude", the listed
        storages keep their own increment and take no overflow: the overflow goes to the other free elements only.
        If those have no room, the remainder is left unplaced (fallback false, it reappears in the next innovation) or
        given to the excluded storages as before (fallback true).
    "seasonal_split": {"storages_slow": ["groundwstor"], "method": "lowpass", "tau_months": 12}
        keeps the fast (seasonal) part of each sub-basin's ensemble-mean increment out of the slow storages. The
        ensemble-mean increment m_j = mean_n(delta_j) is split into a slow part and a fast part c_j: "lowpass": slow =
        recursive low-pass of m_j over the windows (time constant tau_months, started at the first window's value);
        "climatology": c_j = running calendar-month mean of m_j minus its mean over the months seen (memory_years: the
        running mean turns into an exponential one with weight 1/memory_years per year). Then
            dX = S (delta - c) + S_fast c,   S_fast = S without the slow storages, re-normalised (H S_fast = I),
        so H dX = delta, the member spread of the increment is shared as before, and only the ensemble-mean fast part
        bypasses the slow storages. Sub-basins without non-slow shares keep c_j = 0. State is kept in memory across
        windows (one process per run, like the adaptive inflation). "climatology" only: "save": path of an .npz the
        monthly means and visit counts are written to after every window; "init": such a file from an earlier run
        (same sub-basins) to start from, so that the first year is not used to learn the climatology (its counts are
        raised to memory_years, i.e. the loaded climatology is treated as established). Motivation: at wells the seasonal GRACE innovation
        is anti-correlated with the open loop's groundwater error, its interannual part is positively correlated
        (DA review 186, 1 Nov 2026).
        "fraction" (0..1, default 1): only this share of the fast part bypasses the slow storages, c_j -> fraction * c_j
        (0 = no split, i.e. the standard partition; 1 = full split). The climatology / low-pass state still learns
        from the full m_j. One knob between U2-like (0) and the full split (1) for the configuration choice
        (DA review 194, 2 Nov 2026).
        "interannual" {"tau_months": 36, "fraction": 1}: optional second split of the remainder r_j = m_j - c_j (the
        non-seasonal part, full c_j before "fraction"). A recursive low-pass s_j of r_j over the windows (time constant
        tau_months, neutral start s_j = first r_j) is the slow part that the slow storages keep; b_j = fraction (r_j - s_j),
        the 1-3-year part and noise, also bypasses them: c_j -> c_j + b_j. Motivation: v1.1 gw loses skill at wells in
        the 13-36-month band and in the low-flow test while its trend and > 36-month skill improve, and gw increments
        persist (DA reviews 218 / 221, 4 Nov 2026). With "save" the low-pass state is written with the climatology
        ("ia_lp"), and "init" loads it when present. Absent = previous behaviour, bit-identical.
Order: weight cap, blend, normalisation (H S = I), storage shares, seasonal split, boundary smoothing, bounds and
per-cell limit.
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
                 boundary_smoothing=None, bound_overflow=None, seasonal_split=None):
        super().__init__()
        sp = dict(seasonal_split or {})
        self.seasonal_split = None
        if sp:
            unknown_keys = set(sp) - {'storages_slow', 'method', 'tau_months', 'memory_years', 'init', 'save', 'fraction',
                                      'interannual'}
            if unknown_keys:
                raise ValueError('seasonal_split: unknown keys %s' % sorted(unknown_keys))
            meth = str(sp.get('method', 'lowpass')).lower()
            if meth not in ('lowpass', 'climatology'):
                raise ValueError('seasonal_split method must be "lowpass" or "climatology", got %s' % meth)
            tau, mem = float(sp.get('tau_months', 12.0)), float(sp.get('memory_years', 3.0))
            if tau < 1.0 or mem < 1.0:
                raise ValueError('seasonal_split needs tau_months >= 1 and memory_years >= 1, got %s' % sp)
            if meth != 'climatology' and (sp.get('init') or sp.get('save')):
                raise ValueError('seasonal_split: "init" / "save" are for method "climatology" only')
            frac = float(sp.get('fraction', 1.0))
            if not 0.0 <= frac <= 1.0:
                raise ValueError('seasonal_split fraction must be in [0, 1], got %s' % frac)
            self.seasonal_split = dict(storages_slow=[str(v) for v in sp.get('storages_slow', ['groundwstor'])],
                                       method=meth, tau_months=tau, memory_years=mem,
                                       init=str(sp['init']) if sp.get('init') else None,
                                       save=str(sp['save']) if sp.get('save') else None, fraction=frac)
            ia = sp.get('interannual')
            if ia:
                ia = dict(ia)
                bad_k = set(ia) - {'tau_months', 'fraction'}
                if bad_k:
                    raise ValueError('seasonal_split interannual: unknown keys %s' % sorted(bad_k))
                ia = dict(tau_months=float(ia.get('tau_months', 36.0)), fraction=float(ia.get('fraction', 1.0)))
                if ia['tau_months'] < 2.0 or not 0.0 <= ia['fraction'] <= 1.0:
                    raise ValueError('seasonal_split interannual needs tau_months >= 2 and 0 <= fraction <= 1, got %s'
                                     % ia)
                self.seasonal_split['interannual'] = ia
        self._sp_idx = np.zeros(0, dtype=int)   # slow storage indices (setup)
        self._sp_lp = None                      # lowpass state (n_obs,)
        self._sp_clim = None                    # climatology state (n_obs, 12) and visit counts (12,)
        self._sp_cnt = np.zeros(12)
        self._sp_ia = None                      # interannual low-pass state of the non-seasonal remainder (n_obs,)
        self._fast_offset = None                # (n_obs,) set by the filter for one update: added to the sub-basin
                                                # increment through the fast storages only (innovation_bias "fast_only")
        bo = dict(bound_overflow or {})
        self.bound_overflow = None
        if bo:
            unknown_keys = set(bo) - {'exclude', 'fallback'}
            if unknown_keys:
                raise ValueError('bound_overflow: unknown keys %s' % sorted(unknown_keys))
            excl = [str(v) for v in bo.get('exclude', [])]
            if not excl:
                raise ValueError('bound_overflow needs "exclude", e.g. ["groundwstor"]')
            self.bound_overflow = dict(exclude=excl, fallback=bool(bo.get('fallback', False)))
        self._bo_idx = np.zeros(0, dtype=int)   # storage indices that take no overflow (setup)
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
            unknown_keys = set(ss) - {'storages', 'mix', 'residence'}
            if unknown_keys:
                raise ValueError('storage_shares: unknown keys %s' % sorted(unknown_keys))
            mix = float(ss.get('mix', 1.0))
            if not 0.0 < mix <= 1.0:
                raise ValueError('storage_shares mix must be in (0, 1], got %s' % mix)
            sto = [str(v) for v in ss.get('storages', ['soilmoist', 'groundwstor'])]
            if len(sto) < 2:
                raise ValueError('storage_shares needs at least two storages, got %s' % sto)
            rs = ss.get('residence')
            if rs:
                rs = dict(rs)
                bad_k = set(rs) - {'tau_ref_days', 'tau_min_days', 'tau_max_days'}
                if bad_k:
                    raise ValueError('storage_shares residence: unknown keys %s' % sorted(bad_k))
                rs = dict(tau_ref_days=float(rs.get('tau_ref_days', 30.0)), tau_min_days=float(rs.get('tau_min_days', 0.5)),
                          tau_max_days=float(rs.get('tau_max_days', 3650.0)))
                if not (rs['tau_ref_days'] > 0 and 0 < rs['tau_min_days'] < rs['tau_max_days']):
                    raise ValueError('storage_shares residence needs tau_ref_days > 0 and 0 < tau_min_days < '
                                     'tau_max_days, got %s' % rs)
            self.storage_shares = dict(storages=sto, mix=mix, residence=rs or None)
        self._ss_idx = []                   # storage indices of the group
        self._ss_table = None               # (n_obs, n_group) open-loop amplitude shares within the group
        self._ss_tau = None                 # (n_obs, n_group) open-loop residence times [d] (residence option)
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
                          bs_moved_mm_sum=0.0, bound_overflow=self.bound_overflow, bo_unplaced_mm_sum=0.0,
                          bo_n_fallback=0, seasonal_split=self.seasonal_split, sp_n=0, sp_fast_abs_sum=0.0,
                          sp_mean_abs_sum=0.0, sp_n_obs_no_fast_share=0)
        if self.seasonal_split is not None and self.seasonal_split.get('interannual'):
            self.stats['sp_ia_abs_sum'] = 0.0   # sum |b_j|: the non-seasonal part kept out of the slow storages

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
        if self.seasonal_split is not None:
            names = list(f.bounds.names)
            self._sp_idx = np.array([names.index(v) for v in self.seasonal_split['storages_slow'] if v in names], dtype=int)
            unknown = [v for v in self.seasonal_split['storages_slow'] if v not in names]
            if unknown:
                print('non_negative partition: seasonal_split storages %s not in the DA state %s - ignored' % (unknown, names))
            print('non_negative partition: seasonal part of the increment kept out of %s (%s)'
                  % ([names[i] for i in self._sp_idx], self.seasonal_split))
            if self.seasonal_split['init']:
                self._load_climatology(self.seasonal_split['init'], f._L_state.shape[1])
        if self.bound_overflow is not None:
            names = list(f.bounds.names)
            unknown = [v for v in self.bound_overflow['exclude'] if v not in names]
            if unknown:
                print('non_negative partition: bound_overflow exclude %s not in the DA state %s - ignored' % (unknown, names))
            self._bo_idx = np.array([names.index(v) for v in self.bound_overflow['exclude'] if v in names], dtype=int)
            print('non_negative partition: bound overflow kept out of %s (fallback %s)'
                  % ([names[i] for i in self._bo_idx], self.bound_overflow['fallback']))
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
        off, self._fast_offset = self._fast_offset, None                           # one update only
        if self.seasonal_split is not None and len(self._sp_idx):
            dX_new = self._seasonal_split(S, delta, off)
            if off is not None:
                '''the sub-basin totals now include the fast-only offset; the bounds and smoothing preserve these'''
                delta = delta + off[:, None]
        else:
            if off is not None:
                raise ValueError('partition: a fast-only offset (innovation_bias "fast_only") needs seasonal_split')
            dX_new = np.asarray(S @ delta)                                         # n_state x N

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

    def _fast_part(self, m):
        """ensemble-mean fast (seasonal) part c_j of the sub-basin increments m (n_obs,), updating the state"""
        sp = self.seasonal_split
        if sp['method'] == 'lowpass':
            if self._sp_lp is None:
                self._sp_lp = m.copy()                                          # neutral start: c = 0
            else:
                self._sp_lp = self._sp_lp + (m - self._sp_lp) / sp['tau_months']
            return m - self._sp_lp
        month = int(str(getattr(self.f, '_today', '0000-01'))[5:7] or 1) - 1
        if self._sp_clim is None:
            self._sp_clim = np.zeros((len(m), 12))
        self._sp_cnt[month] += 1
        a = max(1.0 / self._sp_cnt[month], 1.0 / sp['memory_years'])
        self._sp_clim[:, month] += a * (m - self._sp_clim[:, month])
        if sp.get('save'):
            if sp.get('interannual') and self._sp_ia is not None:
                np.savez(sp['save'], clim=self._sp_clim, count=self._sp_cnt, ia_lp=self._sp_ia)
            else:
                np.savez(sp['save'], clim=self._sp_clim, count=self._sp_cnt)
        seen = self._sp_cnt > 0
        return self._sp_clim[:, month] - self._sp_clim[:, seen].mean(1)

    def _load_climatology(self, path, n_obs):
        """start the climatology from an earlier run's saved state (see seasonal_split "init")"""
        z = np.load(path)
        clim, cnt = np.asarray(z['clim'], dtype=float), np.asarray(z['count'], dtype=float)
        if clim.shape != (n_obs, 12) or cnt.shape != (12,):
            raise ValueError('seasonal_split init %s: climatology has shape %s / counts %s, expected (%d, 12) / (12,)'
                             % (path, clim.shape, cnt.shape, n_obs))
        self._sp_clim = clim.copy()
        self._sp_cnt = np.where(cnt > 0, np.maximum(cnt, self.seasonal_split['memory_years']), 0.0)
        if self.seasonal_split.get('interannual') and 'ia_lp' in z.files:
            ia = np.asarray(z['ia_lp'], dtype=float)
            if ia.shape == (n_obs,):
                self._sp_ia = ia.copy()
                print('non_negative partition: seasonal_split interannual low-pass started from %s' % path)
        self.stats['sp_init'] = dict(path=str(path), months_known=int((cnt > 0).sum()))
        print('non_negative partition: seasonal_split climatology started from %s (%d months known)'
              % (path, int((cnt > 0).sum())))

    def _seasonal_split(self, S, delta, offset=None):
        """dX = S (delta - c) + S_fast (c + offset), see seasonal_split; offset (n_obs,) or None: an extra sub-basin
        increment placed in the fast storages only, the same for every member (innovation_bias "fast_only")"""
        f = self.f
        m = np.mean(delta, 1)
        ia = self.seasonal_split.get('interannual')
        if ia:
            cfull = self._fast_part(m)
            c = self.seasonal_split['fraction'] * cfull
            r = m - cfull                                               # non-seasonal remainder
            if self._sp_ia is None:
                self._sp_ia = r.copy()                                  # neutral start: b = 0
            else:
                self._sp_ia = self._sp_ia + (r - self._sp_ia) / ia['tau_months']
            b = ia['fraction'] * (r - self._sp_ia)
            self.stats['sp_ia_abs_sum'] += float(np.abs(b).sum())
            if self.seasonal_split.get('save') and self._sp_clim is not None:
                np.savez(self.seasonal_split['save'], clim=self._sp_clim, count=self._sp_cnt, ia_lp=self._sp_ia)
            c = c + b
        else:
            c = self.seasonal_split['fraction'] * self._fast_part(m)
        slow = np.isin(f._var_idx, self._sp_idx).astype(float)
        Sf = sparse.csc_matrix(S.multiply((1.0 - slow)[:, None]))
        hs = f._DM(states=Sf).diagonal().copy()
        okf = hs > 1e-12
        Sf = sparse.csc_matrix(Sf @ sparse.diags(np.where(okf, 1.0 / np.where(okf, hs, 1.0), 0.0)))
        c = np.where(okf, c, 0.0)
        self.stats['sp_n_obs_no_fast_share'] += int((~okf).sum())
        self.stats['sp_n'] += 1
        self.stats['sp_fast_abs_sum'] += float(np.abs(c).sum())
        self.stats['sp_mean_abs_sum'] += float(np.abs(m).sum())
        if offset is not None:
            o = np.where(okf, np.asarray(offset, dtype=float), 0.0)
            self.stats['fast_offset_sum'] = self.stats.get('fast_offset_sum', 0.0) + float(o.sum())
            self.stats['fast_offset_unplaced_sum'] = (self.stats.get('fast_offset_unplaced_sum', 0.0)
                                                      + float(np.abs(np.asarray(offset) - o).sum()))
            return np.asarray(S @ (delta - c[:, None])) + np.asarray(Sf @ (c + o))[:, None]
        return np.asarray(S @ (delta - c[:, None])) + np.asarray(Sf @ c)[:, None]

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
        rs = self.storage_shares.get('residence')
        if rs:
            tau = residence_time_table(src, sto, n_obs, tau_min=rs['tau_min_days'], tau_max=rs['tau_max_days'])
            fac = tau / (tau + rs['tau_ref_days'])
            amp = amp * fac
            self._ss_tau = tau
            try:
                out = Path(b.res_permanent) / b.case / 'storage_shares_residence.csv'
                with open(out, 'w') as fh:
                    fh.write('sub_basin,' + ','.join('tau_days_%s,factor_%s' % (v, v) for v in sto) + '\n')
                    for j in range(n_obs):
                        fh.write('%d,' % (j + 1) + ','.join('%.3f,%.4f' % (tau[j, k], fac[j, k])
                                                             for k in range(len(sto))) + '\n')
            except OSError as err:
                print('non_negative partition: could not write %s (%s)' % (out, err))
            print('non_negative partition: residence weighting tau_ref %.1f d; median tau (d) %s, median factor %s'
                  % (rs['tau_ref_days'], {v: round(float(np.median(tau[:, k])), 1) for k, v in enumerate(sto)},
                     {v: round(float(np.median(fac[:, k])), 3) for k, v in enumerate(sto)}))
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
        if self.bound_overflow is None:
            for k in [k for k in out if k.startswith('bo_')] + ['bound_overflow']:
                out.pop(k, None)
        if self.seasonal_split is None:
            for k in [k for k in out if k.startswith('sp_')] + ['seasonal_split']:
                out.pop(k, None)
        elif out.get('sp_mean_abs_sum', 0) > 0:
            out['sp_fast_share_of_mean_increment'] = round(out['sp_fast_abs_sum'] / out['sp_mean_abs_sum'], 3)
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
            frozen = np.zeros_like(dj, dtype=bool)            # bound_overflow: elements that keep their increment
            if len(self._bo_idx):
                frozen[np.isin(f._var_idx[rows], self._bo_idx), :] = True
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
                if frozen.any():
                    '''bound_overflow: frozen (excluded) elements keep their increment; the free, non-frozen elements
                    take the rest. Where those have no room: fallback -> old behaviour, else the rest stays unplaced'''
                    held = fixed | frozen
                    denom2 = hj @ np.where(held, 0.0, s[:, None])
                    ok2 = denom2 > 1e-12
                    if self._bs is not None:
                        rem2 = delta[j] - hj @ dj
                        add2 = np.where(ok2, rem2 / np.where(ok2, denom2, 1.0), 0.0)
                        dj2 = np.where(held, dj, dj + s[:, None] * add2[None, :])
                    else:
                        rem2 = delta[j] - hj @ np.where(held, dj, 0.0)
                        scale2 = np.where(ok2, rem2 / np.where(ok2, denom2, 1.0), 0.0)
                        dj2 = np.where(held, dj, s[:, None] * scale2[None, :])
                    if self.bound_overflow['fallback'] and not ok2.all():
                        self.stats['bo_n_fallback'] += int((~ok2).sum())
                        if self._bs is not None:
                            rem = delta[j] - hj @ dj
                            add = np.where(ok, rem / np.where(ok, denom, 1.0), 0.0)
                            dj1 = np.where(fixed, dj, dj + s[:, None] * add[None, :])
                        else:
                            rem = delta[j] - hj @ np.where(fixed, dj, 0.0)
                            scale = np.where(ok, rem / np.where(ok, denom, 1.0), 0.0)
                            dj1 = np.where(fixed, dj, s[:, None] * scale[None, :])
                        dj = np.where(ok2[None, :], dj2, dj1)
                    else:
                        dj = dj2
                    continue
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
            if frozen.any():
                self.stats['bo_unplaced_mm_sum'] += float(np.abs(np.mean(delta[j] - hj @ dj)))
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


def residence_time_table(path, storages, n_sub, tau_min=0.5, tau_max=3650.0):
    """
    E-folding time (days) per sub-basin and storage of the open loop's deseasonalised daily ensemble-mean unit
    series (Res_OL.h5, members as in seasonal_amplitude_table): anomaly = series - its calendar-day mean (day of year
    1..366), tau = -1 / ln(rho_1) with rho_1 its lag-1 autocorrelation, clipped to [tau_min, tau_max]. A storage
    missing in the file or without variance gets tau = 0 (its amplitude is 0 there as well). Returns (n_sub, n_storage).
    """
    import h5py, datetime as dt
    from pathlib import Path
    if not Path(path).exists():
        raise FileNotFoundError('storage_shares residence needs the open-loop basin series %s' % path)
    out = np.zeros((n_sub, len(storages)))
    with h5py.File(path, 'r') as h:
        t = h['time'][:]

        def doy_of(x):
            y = int(x)
            n = (dt.date(y + 1, 1, 1) - dt.date(y, 1, 1)).days
            return min(int(round((x - y) * n)), n - 1) + 1
        doy = np.array([doy_of(x) for x in t])
        for j in range(n_sub):
            for i, v in enumerate(storages):
                key = 'sub_basin_%d/%s' % (j + 1, v)
                if key not in h:
                    continue
                mem = [m for m in h[key] if m != '0'] or list(h[key])
                x = np.mean([h[key][m][:] for m in mem], axis=0).astype(float)
                ok = np.isfinite(x)
                if ok.sum() < 30:
                    continue
                clim = np.zeros(367)
                for d in np.unique(doy[ok]):
                    clim[d] = np.mean(x[ok & (doy == d)])
                a = np.where(ok, x - clim[doy], np.nan)
                a0, a1 = a[:-1], a[1:]
                k = np.isfinite(a0) & np.isfinite(a1)
                if k.sum() < 30 or np.std(a0[k]) == 0 or np.std(a1[k]) == 0:
                    continue
                rho = float(np.corrcoef(a0[k], a1[k])[0, 1])
                if not np.isfinite(rho):
                    continue
                if rho <= np.exp(-1.0 / tau_min):
                    tau = tau_min
                elif rho >= np.exp(-1.0 / tau_max):
                    tau = tau_max
                else:
                    tau = -1.0 / np.log(rho)
                out[j, i] = tau
    return out


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
