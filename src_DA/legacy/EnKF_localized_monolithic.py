"""
Single-class version of EnKF_localized as used for the Danube runs 1-8 (until 3 Oct 2026), kept for reference.
The current code (src_DA/EnKF_localized.py + localization.py, inflation.py, partition.py, bounds.py) gives identical
results for every configuration (tested); use that.
"""
"""
EnKF with state-observation localization (and hooks for spread relaxation and observation-error inflation).

Why: with a small ensemble the sample cross-covariance between the state of one sub-basin and the observation of
another is dominated by sampling noise. In the Amazon run this let the large innovations of the estuary unit
remove river storage along the whole mainstem. The variants in src_DA.EnKF localize only in observation space
(tapering R or the observation-space covariance), which does not stop this: the leakage happens in the gain,
through  Pxy = A HA^T / (N-1).  Here Pxy is tapered element-wise (Schur product) with a matrix L (n_state x n_obs)
that is 1 for cells inside the observed sub-basin and decays with distance to it, and Pyy = HA HA^T / (N-1) is
tapered with the corresponding sub-basin to sub-basin taper, so that

        K = (L o Pxy) (L_obs o Pyy + R)^-1 .

Settings (DA_setting.json, block "method"):
    "fusion_method": "EnKF_localized",
    "localization": {"kind": "block" | "gaussian", "length_km": 300, "cutoff": 2.0}
        block    : each observation updates only the cells of its own sub-basin (L = membership matrix)
        gaussian : L = exp(-0.5 (d/length)^2) with d the distance [km] from a cell to the nearest cell of the
                   observed sub-basin (0 inside it), set to 0 beyond cutoff*length; Pyy is tapered with the
                   same function of the mean distance between the cells of two sub-basins
    "inflation": 1.0            multiplicative inflation of the forecast anomalies before the update (1 = off)
    "rtps_alpha": 0.7           relaxation-to-prior-spread (Whitaker & Hamill 2012): the analysis spread of every
                                state element is set to alpha*sigma_forecast + (1-alpha)*sigma_analysis; 0 = off,
                                1 = analysis keeps the full forecast spread. Counters the spread collapse of small
                                ensembles without touching the analysis mean.
    "rtps_space": "state" | "obs"  where the relaxation acts: per state element (classic) or per sub-basin on the
                                spread of the observation equivalents (restores the observable spread; use with
                                integrated observations such as basin-mean TWS)
    "obs_error_inflation": {}   per-sub-basin factor on the observation error variance, e.g. {"18": 9} triples
                                sigma of sub-basin 18 (1-based index); R' = D R D keeps the error correlations
    "increment_partition": "enkf" | "non_negative"
        enkf         : increments as given by the (localized) Kalman gain
        non_negative : the sub-basin TWS increment of the Kalman update is kept, but it is distributed over the
                       cells and storages of the sub-basin with non-negative shares proportional to
                       variance x max(correlation with the sub-basin TWS, 0). No storage is emptied to fill another
                       one, and no element can receive more than the sub-basin increment times its share. Meant for
                       bound-aware: elements are not pushed across their bounds (snow/soil/... >= 0, snow <= 1000),
                       the share is redistributed inside the sub-basin. Meant for
                       small ensembles, whose vertical partition of the increment is dominated by sampling noise and
                       by the parameter perturbation (member with more TWS = member with more groundwater and less
                       river water), which the Amazon 4-member run turned into a filter-model tug of war.
The state vector ordering is that of EnsStates / DM_basin_average: cell-major, vertical_dim storages per cell.
"""
import numpy as np
from src_DA.EnKF import EnKF
from src_DA.configure_DA import config_DA
from src_DA.observations import GRACE_obs
from src_DA.ExtractStates import EnsStates
from src_GHM.Interface.DailyStepRun import DailyModelRun as model_run_daily


class EnKF_localized(EnKF):

    def __init__(self, DA_setting: config_DA, model: model_run_daily, obs: GRACE_obs, sv: EnsStates,
                 sv_excluded: EnsStates, localization: dict = None):
        super().__init__(DA_setting, model, obs, sv, sv_excluded)
        loc = dict(kind='block', length_km=300.0, cutoff=2.0)
        if localization:
            loc.update(localization)
        self.loc = loc
        self._L_state, self._L_obs = self._build_taper(sv, kind=loc['kind'], length_km=float(loc['length_km']),
                                                       cutoff=float(loc['cutoff']))
        n_state, n_obs = self._L_state.shape

        # spread maintenance and observation-error inflation, all read from the "method" block with defaults
        m = DA_setting.method
        self._inflation = float(getattr(m, 'inflation', 1.0))
        self._rtps_alpha = float(getattr(m, 'rtps_alpha', 0.7))
        self._rtps_space = str(getattr(m, 'rtps_space', 'state')).lower()
        assert self._rtps_space in ('state', 'obs'), 'rtps_space must be "state" or "obs"'
        oei = getattr(m, 'obs_error_inflation', None) or {}
        d = np.ones(n_obs)
        for k, f in oei.items():
            d[int(k) - 1] = np.sqrt(float(f))
        self._R_scale = np.outer(d, d)                      # R' = D R D keeps the error correlations

        self._partition = str(getattr(m, 'increment_partition', 'enkf')).lower()
        if self._partition not in ('enkf', 'non_negative'):
            raise ValueError('increment_partition must be "enkf" or "non_negative", got %s' % self._partition)
        self._partition_stats = dict(n_updates=0, n_obs_fallback=0, n_obs_unresolved=0,
                                     max_abs_inc_enkf=0.0, max_abs_inc_new=0.0, n_bound_capped=0, water_redistributed_mm=0.0)
        # bounds of the state elements for the bound-aware partition (cell-major, DM.statesnn order per cell):
        # storages that cannot go negative get 0 (river: its floor), snow 1000 mm on top; groundwater and lakes none
        lb_var = {'swe': 0.0, 'soilmoist': 0.0, 'canopystor': 0.0, 'localwetlandstor': 0.0, 'globalwetlandstor': 0.0,
                  'reservoirstor': 0.0, 'riverstor': 1e-3}
        ub_var = {'swe': 1000.0}
        names = list(sv.DM.statesnn)
        n_cell = n_state // len(names)
        self._lb = np.tile(np.array([lb_var.get(v, -np.inf) for v in names]), n_cell)
        self._ub = np.tile(np.array([ub_var.get(v, np.inf) for v in names]), n_cell)

        # additive inflation ("model error"): sub-basin-coherent random perturbations of selected storages, added to
        # the forecast anomalies before the gain is computed. Config: {"groundwstor": 15, "swe": 5, "months": [..]}
        # (std in mm of the sub-basin mean, optional calendar months of the window in which it is applied)
        add = dict(getattr(m, 'additive_inflation', None) or {})
        self._add_months = set(int(x) for x in add.pop('months', range(1, 13)))
        self._add_seed = add.pop('seed', None)
        self._add_mode = str(add.pop('mode', 'fixed')).lower()
        assert self._add_mode in ('fixed', 'adaptive'), 'additive_inflation mode must be "fixed" or "adaptive"'
        if self._add_mode == 'adaptive':
            # adaptive: the std of the added sub-basin TWS spread is estimated every window from the innovations
            # (see _adaptive_sigma); "split" gives the share of the added VARIANCE per storage
            split = dict(add.pop('split', {'groundwstor': 0.9, 'swe': 0.1}))
            self._add_weight_current = float(add.pop('weight_current', 0.5))
            self._add_max_factor = float(add.pop('max_factor', 3.0))
            self._add_min_sigma = float(add.pop('min_sigma', 5.0))
            self._add_remove_bias = bool(add.pop('remove_bias', True))
            self._add_bias_memory = float(add.pop('bias_memory', 0.05))
            tot = sum(float(w) for v, w in split.items() if v in names)
            self._add_split = {v: float(w) / tot for v, w in split.items() if v in names and float(w) > 0}
            self._add_sigma = dict(self._add_split)          # non-empty -> perturbation switched on
            unknown = [v for v in split if v not in names] + [k for k in add]
            self._add_s2 = None                             # smoothed innovation variance per sub-basin
            self._add_bias = np.zeros(n_obs)                # slowly varying mean innovation per sub-basin
            self._add_log = []
        else:
            self._add_sigma = {v: float(sd) for v, sd in add.items() if v in names and float(sd) > 0}
            unknown = [v for v in add if v not in names]
        if unknown:
            print('EnKF_localized: additive_inflation ignores %s (not in the DA state %s)' % (unknown, names))
        self._names = names
        self._var_idx = np.tile(np.arange(len(names)), n_cell)
        self._owner = np.argmax(self._L_state, axis=1)
        self._owner[self._L_state.max(1) <= 0] = -1
        self._rng = np.random.default_rng(self._add_seed)
        self._add_stats = dict(n_applied=0, n_skipped_month=0, sum_hx_spread_added=np.zeros(n_obs))

        print('EnKF_localized: kind=%s, length=%.0f km, cutoff=%.1f; taper %d states x %d observations, '
              'mean weight %.3f (block would be %.3f); inflation %.2f, RTPS alpha %.2f (%s space), obs-error inflation %s, '
              'increment partition %s'
              % (loc['kind'], loc['length_km'], loc['cutoff'], n_state, n_obs, self._L_state.mean(), 1.0 / n_obs,
                 self._inflation, self._rtps_alpha, self._rtps_space, oei if oei else 'none', self._partition))
        if self._add_sigma and self._add_mode == 'fixed':
            print('EnKF_localized: additive inflation %s mm (sub-basin mean std), months %s'
                  % (self._add_sigma, sorted(self._add_months)))
        elif self._add_sigma:
            print('EnKF_localized: ADAPTIVE additive inflation, variance split %s, weight of current window %.2f, '
                  'sigma in [%.1f mm, %.1f x obs error], bias removal %s (memory %.2f), months %s'
                  % (self._add_split, self._add_weight_current, self._add_min_sigma, self._add_max_factor,
                     self._add_remove_bias, self._add_bias_memory, sorted(self._add_months)))

    # ------------------------------------------------------------------ taper construction
    @staticmethod
    def _cell_coordinates(local_mask):
        """latitude / longitude [deg] of the basin cells in the order of the state vector"""
        res = 0.5
        lat_c = np.arange(90 - res / 2, -90, -res)
        lon_c = np.arange(-180 + res / 2, 180, res)
        lon_mesh, lat_mesh = np.meshgrid(lon_c, lat_c)
        g = local_mask['global_2d'].astype(bool)
        return lat_mesh[g], lon_mesh[g]

    @staticmethod
    def _haversine_matrix(lat, lon):
        """pairwise great-circle distances [km] between cells (n x n)"""
        R = 6371.0
        la, lo = np.deg2rad(lat)[:, None], np.deg2rad(lon)[:, None]
        dlat, dlon = la - la.T, lo - lo.T
        a = np.sin(dlat / 2) ** 2 + np.cos(la) * np.cos(la.T) * np.sin(dlon / 2) ** 2
        return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    def _build_taper(self, sv: EnsStates, kind='block', length_km=300.0, cutoff=2.0):
        lm = sv.DM.local_mask
        nsub = lm['basin_num']
        nvar = sv.DM.vertical_dim
        M = np.column_stack([lm['sub_basin_%d' % k].astype(float) for k in range(1, nsub + 1)])   # n_cell x n_sub
        orphan = M.sum(1) == 0
        if orphan.any():
            print('EnKF_localized: %d basin cells belong to no sub-basin and are never updated' % orphan.sum())

        if kind == 'block':
            L_cell = M
            L_obs = np.eye(nsub)
        elif kind == 'gaussian':
            lat, lon = self._cell_coordinates(lm)
            D = self._haversine_matrix(lat, lon)                                  # n_cell x n_cell
            d_cell_sub = np.column_stack([D[:, M[:, j] > 0].min(1) for j in range(nsub)])   # n_cell x n_sub
            d_cell_sub[M > 0] = 0.0
            L_cell = np.exp(-0.5 * (d_cell_sub / length_km) ** 2)
            L_cell[d_cell_sub > cutoff * length_km] = 0.0
            # sub-basin to sub-basin distance: mean distance of the cells of j to sub-basin k (symmetrised); the
            # nearest-cell distance would be ~one cell for every adjacent pair and taper nothing
            d_sub_sub = np.array([[d_cell_sub[M[:, j] > 0, k].mean() for k in range(nsub)] for j in range(nsub)])
            d_sub_sub = 0.5 * (d_sub_sub + d_sub_sub.T)
            L_obs = np.exp(-0.5 * (d_sub_sub / length_km) ** 2)
            L_obs[d_sub_sub > cutoff * length_km] = 0.0
        else:
            raise ValueError('localization kind must be "block" or "gaussian", got %s' % kind)

        L_state = np.repeat(L_cell, nvar, axis=0)                                 # cell-major, nvar rows per cell
        return L_state, L_obs

    def run_mpi(self):
        super().run_mpi()
        s = self._partition_stats
        if s['n_updates'] > 0:
            print('Increment partition (%s): %d updates, variance-only fallback for %d sub-basin updates, '
                  '%d unresolved; largest |increment| of one element: EnKF %.1f mm -> applied %.1f mm; '
                  'bound-aware: %d element-members capped, %.3g mm redistributed'
                  % (self._partition, s['n_updates'], s['n_obs_fallback'], s['n_obs_unresolved'],
                     s['max_abs_inc_enkf'], s['max_abs_inc_new'], s['n_bound_capped'], s['water_redistributed_mm']))

    # ------------------------------------------------------------------ analysis step
    def update(self, obs, obs_cov, ens_states):
        """
        obs        : perturbed observations of the members (n_obs x N)
        obs_cov    : R (n_obs x n_obs)
        ens_states : forecast states (n_state x N), window means
        """
        R = obs_cov * self._R_scale                                               # per-sub-basin obs-error inflation
        N = self.DA_setting.basic.ensemble

        xm = np.mean(ens_states, 1)[:, None]
        A = (ens_states - xm) * self._inflation                                   # inflated forecast anomalies
        X = xm + A
        self._set_effective_bounds(ens_states)
        if self._add_sigma:
            sig = self._adaptive_sigma(obs, R, X) if self._add_mode == 'adaptive' else None
            A = A + self._additive_perturbation(xm[:, 0], X, N, sigma=sig)
            X = xm + A

        HX = self._DM(states=X)
        HA = HX - np.mean(HX, 1)[:, None]

        Pyy = self._L_obs * (HA @ HA.T) / (N - 1) + R                             # localized obs-space covariance
        Pxy = self._L_state * (A @ HA.T) / (N - 1)                                # localized cross-covariance

        '''K = Pxy Pyy^-1, solved without forming the inverse'''
        K = np.linalg.solve(Pyy.T, Pxy.T).T

        dX = K @ (obs - HX)                                                        # EnKF increments (n_state x N)
        if self._partition == 'non_negative':
            dX = self._partition_non_negative(dX, A, HA, N, X)
        Xa = X + dX

        '''RTPS (Whitaker & Hamill 2012): relax the analysis spread back towards the forecast spread,
           sigma_a' = alpha * sigma_f + (1 - alpha) * sigma_a; the analysis mean is unchanged.
           rtps_space = 'state': element-wise factors (classic). With an integrated observation this restores the
                        cell-level spread but not the spread of the sub-basin TWS, because the analysis anomalies
                        of the cells of a sub-basin compensate each other (Danube: cell spread kept, sub-basin
                        spread stayed collapsed, gain unchanged).
           rtps_space = 'obs'  : one factor per sub-basin from the spread of the observation equivalents HX
                        (forecast vs analysis), applied to all state elements of that sub-basin (weighted by the
                        taper for the gaussian kind), so that the observable spread is restored by construction.'''
        if self._rtps_alpha > 0:
            xa_m = np.mean(Xa, 1)[:, None]
            Aa = Xa - xa_m
            # per-element bound on the inflation factor, so that no member is pushed across a bound
            # member i stays within its (window-aware) bounds if xa_m + f * Aa_i lies in [LB_i, UB_i]
            with np.errstate(invalid='ignore', divide='ignore'):
                r_lo = np.where(Aa < -1e-12, (xa_m - self._LB) / np.maximum(-Aa, 1e-12), np.inf).min(1)
                r_hi = np.where(Aa > 1e-12, (self._UB - xa_m) / np.maximum(Aa, 1e-12), np.inf).min(1)
            lim = np.nan_to_num(np.minimum(r_lo, r_hi), nan=1.0, posinf=np.inf)
            lim = np.maximum(lim, 1.0)                                          # never shrink below the plain analysis
            if self._rtps_space == 'obs':
                HAa = self._DM(states=Aa)                                       # analysis anomalies in obs space
                sf = np.std(HA, axis=1, ddof=1)
                sa = np.std(HAa, axis=1, ddof=1)
                target = self._rtps_alpha * sf + (1.0 - self._rtps_alpha) * sa # wanted sub-basin spread
                owner = np.argmax(self._L_state, axis=1)                        # sub-basin of every state element
                owner[self._L_state.max(1) <= 0] = -1
                factor = np.ones(Aa.shape[0])
                for j in range(len(target)):
                    rows = np.where(owner == j)[0]
                    if len(rows) == 0 or sa[j] <= 1e-12:
                        continue
                    g = target[j] / sa[j]                                       # uniform factor if nothing is bounded
                    fr = np.full(len(rows), g)
                    bounded = np.zeros(len(rows), dtype=bool)
                    for _ in range(8):
                        newly = (~bounded) & (lim[rows] < fr - 1e-12)
                        if not newly.any():
                            break
                        bounded |= newly
                        fr[bounded] = lim[rows][bounded]                         # bounded elements keep their limit ...
                        free = ~bounded
                        if not free.any():
                            break
                        # ... and the free elements get a common factor g so that the sub-basin spread hits the target:
                        # H_j (Aa_bounded * f_bounded + Aa_free * g) has variance target^2  ->  quadratic in g
                        full = np.zeros_like(Aa); full[rows[bounded]] = Aa[rows[bounded]] * fr[bounded, None]
                        c = self._DM(states=full)[j]
                        full = np.zeros_like(Aa); full[rows[free]] = Aa[rows[free]]
                        d = self._DM(states=full)[j]
                        vd, cd, vc = np.var(d, ddof=1), np.cov(c, d, ddof=1)[0, 1], np.var(c, ddof=1)
                        disc = cd ** 2 - vd * (vc - target[j] ** 2)
                        g = (-cd + np.sqrt(disc)) / vd if (vd > 1e-12 and disc >= 0) else g
                        g = max(float(g), 1.0)
                        fr[free] = g
                    factor[rows] = fr
            else:
                sig_f = np.std(A, axis=1, ddof=1)
                sig_a = np.std(Aa, axis=1, ddof=1)
                factor = np.ones_like(sig_a)
                ok = sig_a > 1e-12
                factor[ok] = self._rtps_alpha * sig_f[ok] / sig_a[ok] + (1.0 - self._rtps_alpha)
                factor = np.minimum(factor, lim)
            Xa = xa_m + Aa * factor[:, None]

        return Xa

    # ------------------------------------------------------------------ window-aware bounds
    def _set_effective_bounds(self, ens_states):
        """
        The analysis increment of a window is added equally to every day of the window, and the threshold is then
        applied day by day. A bound checked against the window mean therefore lets the low (high) days of the
        window cross it. With the daily minimum / maximum of each member over the window (set by EnKF.run_mpi as
        self._ens_min / self._ens_max, n_state x N), the condition "every day stays within [lb, ub]" is, for the
        window-mean analysis Xa,
              Xa >= lb + (x_mean - x_min) = LB        and        Xa <= ub - (x_max - x_mean) = UB
        Without the window extremes LB = lb, UB = ub (old behaviour).
        """
        lb, ub = self._lb[:, None], self._ub[:, None]
        mn, mx = getattr(self, '_ens_min', None), getattr(self, '_ens_max', None)
        if mn is not None and mx is not None and np.shape(mn) == ens_states.shape == np.shape(mx):
            with np.errstate(invalid='ignore'):
                self._LB = lb + (ens_states - mn)
                self._UB = ub - (mx - ens_states)
            self._partition_stats['n_window_bounds'] = self._partition_stats.get('n_window_bounds', 0) + 1
        else:
            self._LB = np.broadcast_to(lb, ens_states.shape).copy()
            self._UB = np.broadcast_to(ub, ens_states.shape).copy()

    # ------------------------------------------------------------------ additive inflation
    CV_MAX = 0.5        # largest relative perturbation (std / storage) of a bounded storage, e.g. snow

    def _adaptive_sigma(self, obs, R, X):
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
        HX = self._DM(states=X)
        d = np.mean(obs, 1) - np.mean(HX, 1)
        sf2 = np.var(HX, axis=1, ddof=1)
        so2 = np.diag(R).copy()
        dp = d - self._add_bias if self._add_remove_bias else d
        w = self._add_weight_current
        s2_prev = self._add_s2 if self._add_s2 is not None else sf2 + so2       # neutral start
        s2 = (1.0 - w) * s2_prev + w * dp ** 2
        self._add_s2 = s2
        if self._add_remove_bias:
            self._add_bias = self._add_bias + self._add_bias_memory * (d - self._add_bias)
        sa2 = s2 - sf2 - so2
        sigma = np.sqrt(np.maximum(sa2, 0.0))
        sigma = np.clip(sigma, self._add_min_sigma, self._add_max_factor * np.sqrt(so2))
        for j in range(len(d)):
            self._add_log.append(dict(date=str(getattr(self, '_today', '')), sub_basin=j + 1,
                                      innovation_mm=float(d[j]), bias_mm=float(self._add_bias[j]),
                                      spread_forecast_mm=float(np.sqrt(sf2[j])), obs_error_mm=float(np.sqrt(so2[j])),
                                      smoothed_innov_std_mm=float(np.sqrt(s2[j])), sigma_added_mm=float(sigma[j])))
        return {v: np.sqrt(share) * sigma for v, share in self._add_split.items()}

    def save_filter_log(self, out_dir):
        """write the filter statistics (partition, additive inflation; adaptive: one row per window and sub-basin)"""
        import json, csv
        from pathlib import Path
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        summ = self.partition_summary()
        json.dump(summ, open(out_dir / 'filter_summary.json', 'w'), indent=1, default=lambda o: np.asarray(o).tolist())
        rows = getattr(self, '_add_log', None)
        if rows:
            with open(out_dir / 'adaptive_inflation_log.csv', 'w', newline='') as f:
                wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)

    def _additive_perturbation(self, xm, X, N, sigma=None):
        """
        Zero-mean perturbations E (n_state x N) that give selected storages extra, sub-basin-coherent spread.
        For every storage v in additive_inflation and every sub-basin j one random number per member,
        e ~ N(0, sigma_v), centred over the members, is added to all cells of that storage in j:
          - unbounded storages (groundwater): the same e in every cell, so the sub-basin mean gets std sigma_v and,
            unlike multiplicative inflation of compensating cell anomalies, the spread is visible to GRACE;
          - storages with a lower bound (snow, soil, river): e weighted by the cell's ensemble-mean storage, so
            snow-free cells stay snow-free; the relative perturbation is capped at CV_MAX (50 %) of the storage, so
            a sub-basin with little snow gets less than sigma; members are kept within the window-aware bounds.
        The perturbations stay in the members after the update (they are part of Xa), so they act as model error
        that is propagated by WaterGAP, as in the GRACE DA of Girotto et al. (2016) / Kumar et al. (2016).
        """
        month = int(str(getattr(self, '_today', '0000-00'))[5:7] or 0)
        E = np.zeros_like(X)
        if month not in self._add_months:
            self._add_stats['n_skipped_month'] += 1
            return E
        n_obs = self._L_state.shape[1]
        if sigma is None:                                   # fixed mode: the same std in every sub-basin
            sigma = {v: np.full(n_obs, sd) for v, sd in self._add_sigma.items()}
        unbounded = [v for v in sigma if not np.isfinite(self._lb[self._var_idx == self._names.index(v)]).any()]
        lost = np.zeros(n_obs)                              # variance a scarce bounded storage could not take
        adaptive = self._add_mode == 'adaptive'
        order = list(sigma)
        if adaptive:                                        # bounded storages first, their lost variance -> unbounded
            order = [v for v in order if v not in unbounded] + [v for v in order if v in unbounded]
        for v in order:
            rows_v = self._var_idx == self._names.index(v)
            bounded = v not in unbounded
            for j in range(n_obs):
                rows = np.where(rows_v & (self._owner == j))[0]
                if len(rows) == 0:
                    continue
                sd = float(sigma[v][j])
                if adaptive and not bounded:
                    sd = np.sqrt(sd ** 2 + lost[j] / max(len(unbounded), 1))
                if sd <= 0:
                    continue
                e = self._rng.normal(0.0, sd, N)
                e -= e.mean()
                if bounded:
                    w = np.maximum(xm[rows], 0.0)
                    if w.mean() <= 1e-9:
                        lost[j] += sd ** 2                                      # e.g. no snow in this sub-basin
                        continue
                    # cell perturbation = storage * relative error, the relative error capped at CV_MAX, so that
                    # a sub-basin with little snow (summer, a few alpine cells) gets a small perturbation and not
                    # sigma concentrated on those few cells
                    f = min(1.0, self.CV_MAX * w.mean() / sd)
                    lost[j] += (1.0 - f ** 2) * sd ** 2
                    w = w / w.mean() * f
                else:
                    w = np.ones(len(rows))
                E[rows] += w[:, None] * e[None, :]
        # keep members within the physical bounds (small shift of the mean only where a bound is hit)
        E = np.maximum(E, self._LB - X)
        E = np.minimum(E, self._UB - X)
        HE = self._DM(states=E)
        self._add_stats['n_applied'] += 1
        self._add_stats['sum_hx_spread_added'] += np.std(HE, axis=1, ddof=1)
        return E

    # ------------------------------------------------------------------ non-negative disaggregation
    def _partition_non_negative(self, dX, A, HA, N, X=None):
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
        delta = self._DM(states=dX)                                                # n_obs x N, sub-basin increments
        var = np.sum(A * A, 1) / (N - 1)                                           # n_state
        sd_x = np.sqrt(var)
        sd_y = np.sqrt(np.sum(HA * HA, 1) / (N - 1))                               # n_obs
        with np.errstate(invalid='ignore', divide='ignore'):
            corr = (A @ HA.T) / (N - 1) / np.outer(sd_x, sd_y)                    # n_state x n_obs
        corr = np.nan_to_num(corr)
        W = self._L_state * var[:, None] * np.maximum(corr, 0.0)                   # n_state x n_obs
        HW = self._DM(states=W)                                                    # n_obs x n_obs, column j -> (H w_.j)
        hw = np.diag(HW).copy()
        bad = hw <= 0
        if bad.any():                                                              # fall back to variance-only shares
            W[:, bad] = self._L_state[:, bad] * var[:, None]
            hw[bad] = np.diag(self._DM(states=W))[bad]
        ok = hw > 0
        scale = np.zeros_like(hw)
        scale[ok] = 1.0 / hw[ok]
        dX_new = (W * scale[None, :]) @ delta                                      # n_state x N

        if X is not None and (np.isfinite(self._lb).any() or np.isfinite(self._ub).any()):
            dX_new = self._apply_bounds(dX_new, W * scale[None, :], delta, X)

        self._partition_stats['n_updates'] += 1
        self._partition_stats['n_obs_fallback'] += int(bad.sum())
        self._partition_stats['n_obs_unresolved'] += int((~ok).sum())
        self._partition_stats['max_abs_inc_enkf'] = max(self._partition_stats['max_abs_inc_enkf'], float(np.abs(dX).max()))
        self._partition_stats['max_abs_inc_new'] = max(self._partition_stats['max_abs_inc_new'], float(np.abs(dX_new).max()))
        return dX_new

    def _apply_bounds(self, dX_new, S, delta, X, n_iter=6):
        """
        S     : normalised shares (n_state x n_obs), H S = I on the block diagonal; dX_new = S @ delta
        delta : sub-basin increments (n_obs x N);  X : forecast states (n_state x N)
        For every sub-basin j: elements whose increment would cross a bound are fixed at the bound, the remaining
        increment of j is redistributed over the free elements with their shares; repeated until no new crossing.
        """
        cap_lo = self._LB - X                                 # most negative allowed increment (window-aware)
        cap_hi = self._UB - X                                 # most positive allowed increment
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
                rem = delta[j] - self._DM(states=full)[j]                           # (N,)
                free_s = np.zeros_like(out); free_s[rows] = np.where(fixed, 0.0, s[:, None])
                denom = self._DM(states=free_s)[j]                                  # (N,)
                ok = denom > 1e-12
                scale = np.where(ok, rem / np.where(ok, denom, 1.0), 0.0)
                dj = np.where(fixed, dj, s[:, None] * scale[None, :])
            n_capped += int(fixed.sum())
            moved += float(np.abs(dj - out[rows, :]).sum())
            out[rows, :] = dj
        self._partition_stats['n_bound_capped'] += n_capped
        self._partition_stats['water_redistributed_mm'] += moved
        return out

    def partition_summary(self):
        out = dict(self._partition_stats, partition=self._partition)
        if self._add_sigma:
            n = max(self._add_stats['n_applied'], 1)
            out['additive_inflation'] = dict(sigma_mm=self._add_sigma, n_applied=self._add_stats['n_applied'],
                                             n_skipped_month=self._add_stats['n_skipped_month'],
                                             mean_obs_space_spread_added_mm=list(np.round(
                                                 self._add_stats['sum_hx_spread_added'] / n, 2)))
        return out
