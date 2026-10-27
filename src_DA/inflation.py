"""
Spread maintenance ("inflation") of the localized EnKF (src_DA.EnKF_localized): one class per scheme.
DA_setting.json, "method" block, key "inflation":

    {"kind": "none"}
    {"kind": "multiplicative", "factor": 1.6}
        forecast anomalies multiplied by factor before the gain (Anderson & Anderson 1999)
    {"kind": "rtps", "alpha": 0.7}
        relaxation to prior spread (Whitaker & Hamill 2012), element-wise: sigma_a' = alpha sigma_f + (1-alpha) sigma_a
    {"kind": "additive", "sigma": {"groundwstor": 15, "swe": 5}, "seed": 42}
        sub-basin-coherent zero-mean perturbations of selected storages added before the gain (model error); they stay
        in the members and are propagated by WaterGAP
    {"kind": "adaptive_additive", "split_storages": ["groundwstor", "soilmoist"], "weight_current": 0.5,
     "max_factor": 3.0, "min_sigma": 5.0, "remove_bias": true, "bias_memory": 0.05, "seed": 42}
        as additive, but the std is estimated every window and sub-basin from the innovations. The added variance is
        shared among the storages per calendar month and sub-basin in proportion to the open-loop ENSEMBLE VARIANCE of
        each storage (the model's own error estimate from the forcing / parameter perturbations), computed once from
        Res/<case>/Res_OL.h5; "split_storages" (default: all storages of the DA state) lists the storages that may
        receive noise (e.g. river stays in the state but gets no noise, so the members' discharge is not made noisy).
"""
import warnings
import numpy as np


class Inflation:
    """no spread maintenance; base class: prior() acts on the forecast anomalies, posterior() on the analysis"""
    KIND = 'none'

    def __init__(self):
        self.f = None
        self.stats = {}

    def setup(self, f):
        """f: the filter (EnKF_localized), gives access to _DM, _L_state, _owner, _var_idx, bounds, _today"""
        self.f = f

    def prior(self, A, xm, obs, R):
        return A

    def posterior(self, Xa, A, HA):
        return Xa

    def describe(self):
        return 'none'

    def summary(self):
        return dict(self.stats, kind=self.KIND)

    def log_rows(self):
        return []


class NoInflation(Inflation):
    KIND = 'none'


class MultiplicativeInflation(Inflation):
    KIND = 'multiplicative'

    def __init__(self, factor=1.0):
        super().__init__()
        self.factor = float(factor)

    def prior(self, A, xm, obs, R):
        return A * self.factor

    def describe(self):
        return 'multiplicative, factor %.2f' % self.factor

    def summary(self):
        return dict(kind=self.KIND, factor=self.factor)


class RTPS(Inflation):
    KIND = 'rtps'

    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = float(alpha)

    def describe(self):
        return 'RTPS, alpha %.2f' % self.alpha

    def summary(self):
        return dict(kind=self.KIND, alpha=self.alpha)

    def posterior(self, Xa, A, HA):
        """relax the analysis spread towards the forecast spread; the analysis mean is unchanged"""
        f = self.f
        if self.alpha <= 0:
            return Xa
        xa_m = np.mean(Xa, 1)[:, None]
        Aa = Xa - xa_m
        # per-element bound on the inflation factor, so that no member is pushed across a bound
        # member i stays within its (window-aware) bounds if xa_m + f * Aa_i lies in [LB_i, UB_i]
        with np.errstate(invalid='ignore', divide='ignore'):
            r_lo = np.where(Aa < -1e-12, (xa_m - f.bounds.LB) / np.maximum(-Aa, 1e-12), np.inf).min(1)
            r_hi = np.where(Aa > 1e-12, (f.bounds.UB - xa_m) / np.maximum(Aa, 1e-12), np.inf).min(1)
        lim = np.nan_to_num(np.minimum(r_lo, r_hi), nan=1.0, posinf=np.inf)
        lim = np.maximum(lim, 1.0)                                          # never shrink below the plain analysis
        sig_f = np.std(A, axis=1, ddof=1)
        sig_a = np.std(Aa, axis=1, ddof=1)
        factor = np.ones_like(sig_a)
        ok = sig_a > 1e-12
        factor[ok] = self.alpha * sig_f[ok] / sig_a[ok] + (1.0 - self.alpha)
        factor = np.minimum(factor, lim)
        Xa = xa_m + Aa * factor[:, None]
        return Xa


class AdditiveInflation(Inflation):
    KIND = 'additive'
    CV_MAX = 0.5                    # largest relative perturbation (std / storage) of a bounded storage, e.g. snow
    REASSIGN_LOST_VARIANCE = False  # fixed scheme: variance a scarce bounded storage cannot take is dropped

    def __init__(self, sigma=None, seed=None):
        super().__init__()
        self.sigma_cfg = {v: float(sd) for v, sd in (sigma or {}).items() if float(sd) > 0}
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.sigma = {}

    def setup(self, f):
        super().setup(f)
        names = f.bounds.names
        unknown = [v for v in self.sigma_cfg if v not in names]
        if unknown:
            print('%s inflation ignores %s (not in the DA state %s)' % (self.KIND, unknown, names))
        self.sigma = {v: sd for v, sd in self.sigma_cfg.items() if v in names}
        self.stats = dict(n_applied=0, sum_hx_spread_added=np.zeros(f._L_state.shape[1]))

    def describe(self):
        return 'additive %s mm (sub-basin mean std)' % self.sigma

    def prior(self, A, xm, obs, R):
        X = xm + A
        return A + self.perturbation(xm[:, 0], X, A.shape[1], sigma=self.window_sigma(obs, R, X))

    def window_sigma(self, obs, R, X):
        """std per storage and sub-basin for this window; fixed scheme: None (= the configured values)"""
        return None

    def perturbation(self, xm, X, N, sigma=None):
        """
        Zero-mean perturbations E (n_state x N) that give selected storages extra, sub-basin-coherent spread.
        For every storage v in additive_inflation and every sub-basin j one random number per member,
        e ~ N(0, sigma_v), centred over the members, is added to all cells of that storage in j:
          - unbounded storages (groundwater): the same e in every cell, so the sub-basin mean gets std sigma_v and,
            unlike multiplicative inflation of compensating cell anomalies, the spread is visible to GRACE;
          - storages with a lower bound (snow, soil, river): e weighted by the cell's ensemble-mean storage, so
            snow-free cells stay snow-free; the relative perturbation is capped at CV_MAX (50 %) of the storage, so
            a sub-basin with little snow gets less than sigma; members are kept within the window-aware bounds;
          - storages with a lower AND an upper bound (soil: 0..smax, needs soil_upper_bound): e weighted by the room
            to the nearer bound, min(S - lb, ub - S), so a nearly full (winter) or nearly empty soil gets little
            noise and is not clipped; the same CV_MAX cap applies to the room. Snow has no static upper bound: its room
            is the snow amount, or, with the snow envelope cap on ("envelope_factor" of "snow_bounds"),
            min(S, cap - S) with the cap of the window, so a pack at or above its cap (perennial high-Alpine cells)
            gets no noise and does not keep building ensemble spread there.
          Variance a bounded storage cannot take (no snow, full soil) goes to the unbounded storages (groundwater,
          lakes) in the adaptive scheme.
        The perturbations stay in the members after the update (they are part of Xa), so they act as model error
        that is propagated by WaterGAP, as in the GRACE DA of Girotto et al. (2016) / Kumar et al. (2016).
        """
        f = self.f
        E = np.zeros_like(X)
        n_obs = f._L_state.shape[1]
        if sigma is None:                                   # fixed mode: the same std in every sub-basin
            sigma = {v: np.full(n_obs, sd) for v, sd in self.sigma.items()}
        unbounded = [v for v in sigma if not f.bounds.is_bounded(v)]
        lost = np.zeros(n_obs)                              # variance a scarce bounded storage could not take
        adaptive = self.REASSIGN_LOST_VARIANCE
        order = list(sigma)
        if adaptive:                                        # bounded storages first, their lost variance -> unbounded
            order = [v for v in order if v not in unbounded] + [v for v in order if v in unbounded]
        for v in order:
            rows_v = f._var_idx == f.bounds.names.index(v)
            bounded = v not in unbounded
            for j in range(n_obs):
                rows = np.where(rows_v & (f._owner == j))[0]
                if len(rows) == 0:
                    continue
                sd = float(sigma[v][j])
                if adaptive and not bounded:
                    sd = np.sqrt(sd ** 2 + lost[j] / max(len(unbounded), 1))
                if sd <= 0:
                    continue
                e = self.rng.normal(0.0, sd, N)
                e -= e.mean()
                if bounded:
                    w = self._room(v, xm[rows], rows)
                    if w.mean() <= 1e-9:
                        lost[j] += sd ** 2                                      # e.g. no snow in this sub-basin
                        continue
                    # cell perturbation = storage * relative error, the relative error capped at CV_MAX, so that
                    # a sub-basin with little snow (summer, a few alpine cells) gets a small perturbation and not
                    # sigma concentrated on those few cells
                    fac = min(1.0, self.CV_MAX * w.mean() / sd)
                    lost[j] += (1.0 - fac ** 2) * sd ** 2
                    w = w / w.mean() * fac
                else:
                    w = np.ones(len(rows))
                E[rows] += w[:, None] * e[None, :]
        # keep members within the physical bounds (small shift of the mean only where a bound is hit)
        E = np.maximum(E, f.bounds.LB - X)
        E = np.minimum(E, f.bounds.UB - X)
        HE = f._DM(states=E)
        self.stats['n_applied'] += 1
        self.stats['sum_hx_spread_added'] += np.std(HE, axis=1, ddof=1)
        return E

    def _room(self, v, x, rows):
        """how much noise a bounded storage can take in each cell (rows: its state elements): the distance to the
        nearer static bound, min(S - lb, ub - S); for snow (no static upper bound) the snow amount, limited by the
        distance to the snow cap of the window when the envelope cap is on (EnKF_localized._snow_cap_now)"""
        f = self.f
        w = np.maximum(x - f.bounds.lb[rows], 0.0)
        '''upper bound: the ensemble mean of the members' own static bounds (soil: smax x land fraction x the member's
        smax_multiplier, StateBounds.set_window), so it matches the ensemble-mean storage x; the plain static bound
        (smax per land area, factor 1) where no member has a value or no window has been set'''
        ub = f.bounds.ub[rows]
        ub_m = getattr(f.bounds, 'ub_member', None)
        if ub_m is not None and ub_m.shape[0] == f.bounds.ub.shape[0]:
            with np.errstate(invalid='ignore'), warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)                 # all-NaN cell: static bound
                ub_mean = np.nanmean(ub_m[rows], axis=1)
            ub = np.where(np.isnan(ub_mean), ub, ub_mean)
        fin = np.isfinite(ub)
        if fin.any():
            w = np.where(fin, np.minimum(w, np.maximum(ub - x, 0.0)), w)
        cap = getattr(f, '_snow_cap_now', None)
        if v == 'swe' and cap is not None:
            c = np.asarray(cap['swe'], dtype=float)[rows // len(f.bounds.names)]      # state element -> cell
            fin = np.isfinite(c)
            w = np.where(fin, np.minimum(w, np.maximum(c - x, 0.0)), w)
        return w

    def summary(self):
        n = max(self.stats.get('n_applied', 0), 1)
        return dict(kind=self.KIND, sigma_mm=self.sigma, seed=self.seed, n_applied=self.stats.get('n_applied', 0),
                    mean_obs_space_spread_added_mm=list(np.round(self.stats['sum_hx_spread_added'] / n, 2)))


class AdaptiveAdditiveInflation(AdditiveInflation):
    KIND = 'adaptive_additive'
    REASSIGN_LOST_VARIANCE = True   # variance snow cannot take (no / little snow) goes to the unbounded storages

    def __init__(self, split_storages=None, weight_current=0.5, max_factor=3.0, min_sigma=5.0, remove_bias=True,
                 bias_memory=0.05, seed=None):
        super().__init__(sigma=None, seed=seed)
        self.split_storages = list(split_storages) if split_storages is not None else None
        self.split_source = None
        self._table = None              # (12, n_sub, n_storage) variance shares from the open loop
        self._table_vars = []
        self.weight_current = float(weight_current)
        self.max_factor = float(max_factor)
        self.min_sigma = float(min_sigma)
        self.remove_bias = bool(remove_bias)
        self.bias_memory = float(bias_memory)
        self._s2 = None                 # smoothed innovation variance per sub-basin
        self._bias = None               # slowly varying mean innovation per sub-basin
        self.log = []

    def setup(self, f):
        Inflation.setup(self, f)
        names = f.bounds.names
        n_obs = f._L_state.shape[1]
        from pathlib import Path
        b = f.DA_setting.basic
        src = Path(b.res_permanent) / b.case / 'Res_OL.h5'
        storages = list(names)
        if self.split_storages is not None:
            unknown = [v for v in self.split_storages if v not in names]
            if unknown:
                print('%s inflation: split_storages %s not in the DA state %s - ignored' % (self.KIND, unknown, names))
            storages = [v for v in names if v in self.split_storages]
            if not storages:
                raise ValueError('adaptive_additive: none of split_storages %s is in the DA state %s'
                                 % (self.split_storages, names))
        self._table, self._table_vars = split_table(src, storages, n_obs)
        self.split_source = str(src)
        used = self._table.max(axis=(0, 1)) > 0
        self.split = {v: float(self._table[:, :, i].mean()) for i, v in enumerate(self._table_vars) if used[i]}
        if 'soilmoist' in self.split and not np.isfinite(f.bounds.ub[f._var_idx == names.index('soilmoist')]).any():
            print('%s inflation: soil is perturbed but has no upper bound in the filter (soil_upper_bound off) - '
                  'noise in nearly full soil will be clipped by the threshold' % self.KIND)
        self.sigma = dict(self.split)   # storages that are perturbed (values: (mean) variance shares)
        self._bias = np.zeros(n_obs)
        self.stats = dict(n_applied=0, sum_hx_spread_added=np.zeros(n_obs))

    def describe(self):
        split = 'open-loop ensemble variance from %s (mean shares %s)' % (
            self.split_source, {v: round(s, 2) for v, s in self.split.items()})
        return ('adaptive additive, variance split %s, weight of current window %.2f, sigma in [%.1f mm, %.1f x obs '
                'error], bias removal %s (memory %.2f)'
                % (split, self.weight_current, self.min_sigma, self.max_factor, self.remove_bias, self.bias_memory))

    def window_sigma(self, obs, R, X):
        """
        Std of the additive perturbation per storage and sub-basin, estimated from the innovations
        (Desroziers et al. 2005; Li, Kalnay & Miyoshi 2009; current window included as in Anderson 2007/2009).
        Per sub-basin j:
            d_j    = mean(obs_j) - mean(HX_j)                     innovation of the ensemble mean
            d'_j   = d_j - b_j                                      minus the slowly varying mean innovation (bias)
            s2_j   = (1-w) s2_j(previous) + w d'_j^2               smoothed innovation variance, w = weight_current
            sa2_j  = s2_j - var(HX_j) - R_jj                        missing (model-error) variance
            sigma_j = clip(sqrt(sa2_j), min_sigma, max_factor * sqrt(R_jj))
        The bias is updated after use: b_j <- b_j + bias_memory (d_j - b_j). sigma_j is split over the storages
        with the variance shares of "split": sigma_vj = sqrt(share_v) sigma_j.
        """
        f = self.f
        HX = f._DM(states=X)
        d = np.mean(obs, 1) - np.mean(HX, 1)
        sf2 = np.var(HX, axis=1, ddof=1)
        so2 = np.diag(R).copy()
        dp = d - self._bias if self.remove_bias else d
        w = self.weight_current
        s2_prev = self._s2 if self._s2 is not None else sf2 + so2       # neutral start
        s2 = (1.0 - w) * s2_prev + w * dp ** 2
        self._s2 = s2
        if self.remove_bias:
            self._bias = self._bias + self.bias_memory * (d - self._bias)
        sa2 = s2 - sf2 - so2
        sigma = np.sqrt(np.maximum(sa2, 0.0))
        sigma = np.clip(sigma, self.min_sigma, self.max_factor * np.sqrt(so2))
        shares = self.shares_now(len(d))
        for j in range(len(d)):
            row = dict(date=str(getattr(f, '_today', '')), sub_basin=j + 1,
                       innovation_mm=float(d[j]), bias_mm=float(self._bias[j]),
                       spread_forecast_mm=float(np.sqrt(sf2[j])), obs_error_mm=float(np.sqrt(so2[j])),
                       smoothed_innov_std_mm=float(np.sqrt(s2[j])), sigma_added_mm=float(sigma[j]))
            row.update({'share_%s' % v: float(sh[j]) for v, sh in shares.items()})
            self.log.append(row)
        return {v: np.sqrt(sh) * sigma for v, sh in shares.items()}

    def shares_now(self, n_obs):
        """variance shares {storage: (n_obs,)} for the month of the current window"""
        month = int(str(getattr(self.f, '_today', '0000-01'))[5:7] or 1)
        T = self._table[month - 1]                                      # (n_sub, n_storage)
        return {v: T[:, i] for i, v in enumerate(self._table_vars) if v in self.split}

    def summary(self):
        out = super().summary()
        out.update(kind=self.KIND, split=self.split, weight_current=self.weight_current, max_factor=self.max_factor,
                   min_sigma=self.min_sigma, remove_bias=self.remove_bias, bias_memory=self.bias_memory,
                   split_mode='ol_spread')
        if self._table is not None:
            out.update(split_source=self.split_source, split_storages=self._table_vars,
                       split_table={v: [[round(float(x), 3) for x in self._table[m, :, i]] for m in range(12)]
                                    for i, v in enumerate(self._table_vars)},
                       split_table_layout='[month 1..12][sub-basin 1..n]')
        out.pop('sigma_mm', None)
        return out

    def log_rows(self):
        return self.log


def split_table(path, storages, n_sub):
    """
    Variance shares of the storages per calendar month and sub-basin from the collected open loop (Res_OL.h5:
    <sub_basin_k>/<storage>/<member>, daily; member '0' left out as in the evaluation): the ensemble variance across
    the members, averaged over the days of each calendar month. Storages of the DA state that are not in the file get
    share 0. Returns (table (12, n_sub, n_storage), storages)
    """
    import h5py, datetime as dt
    from pathlib import Path
    if not Path(path).exists():
        raise FileNotFoundError('adaptive_additive inflation needs the open-loop basin series %s - collect the open '
                                'loop first (RDA.post_processing(stages=[Stage.OL]))' % path)
    storages = list(storages)
    T = np.zeros((12, n_sub, len(storages)))
    with h5py.File(path, 'r') as h:
        if 'sub_basin_%d' % n_sub not in h or 'sub_basin_%d' % (n_sub + 1) in h:
            raise ValueError('split_table: %s does not have the %d sub-basins of the DA' % (path, n_sub))
        t = h['time'][:]

        def month_of(x):
            y = int(x)
            n = (dt.date(y + 1, 1, 1) - dt.date(y, 1, 1)).days
            d = dt.date(y, 1, 1) + dt.timedelta(days=min(int(round((x - y) * n)), n - 1))
            return d.year * 100 + d.month
        mon = np.array([month_of(x) for x in t]) % 100
        for j in range(n_sub):
            for i, v in enumerate(storages):
                key = 'sub_basin_%d/%s' % (j + 1, v)
                if key not in h:
                    continue
                X = np.array([h[key][m][:] for m in h[key] if m != '0'])
                var = X.var(axis=0, ddof=1) if X.shape[0] > 1 else np.zeros(X.shape[1])
                T[:, j, i] = [var[mon == m].mean() if (mon == m).any() else 0.0 for m in range(1, 13)]
    T = np.maximum(np.nan_to_num(T), 0.0)
    tot = T.sum(2)
    S = T / np.where(tot > 0, tot, 1.0)[..., None]
    empty = tot <= 0                                   # no variance at all: everything to groundwater
    if empty.any():
        k = storages.index('groundwstor') if 'groundwstor' in storages else 0
        S[empty] = 0.0
        S[empty, k] = 1.0
    return S, storages


INFLATIONS = {c.KIND: c for c in (NoInflation, MultiplicativeInflation, RTPS, AdditiveInflation,
                                  AdaptiveAdditiveInflation)}


def make_inflation(cfg=None):
    """Inflation object from the "inflation" block, e.g. {"kind": "rtps", "alpha": 0.7}"""
    cfg = dict(cfg or {'kind': 'none'})
    kind = str(cfg.pop('kind', 'none')).lower()
    if kind not in INFLATIONS:
        raise ValueError('inflation kind must be one of %s, got %s' % (list(INFLATIONS), kind))
    try:
        return INFLATIONS[kind](**cfg)
    except TypeError as err:
        raise ValueError('inflation "%s": unknown or missing setting (%s)' % (kind, err))
