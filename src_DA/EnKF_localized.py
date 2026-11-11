"""
Localized EnKF for basin-averaged TWS observations, assembled from independent components:

    localization  src_DA.localization  NoLocalization | BlockLocalization | GaussianLocalization
    inflation     src_DA.inflation     NoInflation | MultiplicativeInflation | RTPS | AdditiveInflation |
                                       AdaptiveAdditiveInflation
    partition     src_DA.partition     EnKFPartition | NonNegativePartition
    bounds        src_DA.bounds        StateBounds (physical limits, window-aware)

Analysis step (one window, window-mean states):

    A  = inflation.prior(X_f - mean)                    multiplicative / additive inflation (or nothing)
    K  = (L_state o Pxy) (L_obs o Pyy + R)^-1           localization tapers, R' = D R D (obs_error_inflation)
    dX = partition.split(K (y - HX))                    as given by K, or non-negative and bound-aware
    Xa = inflation.posterior(X + dX)                    RTPS (or nothing)

The components are chosen in DA_setting.json ("method" block) and built by src_DA.filter_factory.build_filter;
see doc/configuration.md. The deterministic variant src_DA.LETKF.LETKF_localized replaces only the raw increment
K (y - HX) (analysis_increment) by the ensemble-transform analysis and keeps everything else. The daily loop, MPI, window means / extremes and the post-update thresholds live in the
base class src_DA.EnKF.EnKF. The state vector is that of EnsStates / DM_basin_average: cell-major, vertical_dim
storages per cell.
"""
import numpy as np
from scipy import sparse
from src_DA.EnKF import EnKF
from src_DA.configure_DA import config_DA
from src_DA.observations import GRACE_obs
from src_DA.ExtractStates import EnsStates
from src_GHM.Interface.DailyStepRun import DailyModelRun as model_run_daily
from src_DA.localization import Localization, BlockLocalization, sub_basin_membership, owner_of_cells
from src_DA.inflation import Inflation, NoInflation
from src_DA.partition import Partition, EnKFPartition
from src_DA.bounds import StateBounds


class EnKF_localized(EnKF):
    METHOD = 'EnKF_localized'
    OBS_PERTURBATION_SEED = 20021            # obs_error_correlation 'diagonal': seed of the member perturbations

    def __init__(self, DA_setting: config_DA, model: model_run_daily, obs: GRACE_obs, sv: EnsStates,
                 sv_excluded: EnsStates, localization: Localization = None, inflation: Inflation = None,
                 partition: Partition = None, obs_error_inflation=None, obs_error_correlation: str = 'full',
                 obs_perturbation_centering: bool = False, soil_upper_bound: bool = True, snow_bounds: dict = None,
                 innovation_bias: dict = None):
        super().__init__(DA_setting, model, obs, sv, sv_excluded)
        self.localization = localization or BlockLocalization()
        self.inflation = inflation or NoInflation()
        self.partition = partition or EnKFPartition()

        '''localization tapers'''
        self._L_state, self._L_obs = self.localization.build(sv.DM.local_mask, sv.DM.vertical_dim)
        n_state, n_obs = self._L_state.shape

        '''state layout: storage of every element, sub-basin of every element (from the membership, not the taper)'''
        names = list(sv.DM.statesnn)
        n_cell = n_state // len(names)
        self._var_idx = np.tile(np.arange(len(names)), n_cell)
        self._owner = np.repeat(owner_of_cells(sub_basin_membership(sv.DM.local_mask)), len(names))
        '''row j of the design matrix restricted to a set of elements (partition / inflation): every state element
        has one non-zero in H (its own sub-basin), so H[j, rows] = h[rows] where owner[rows] == j'''
        H = sv.DM.getDM()
        if (H.getnnz(axis=0) > 1).any():
            raise ValueError('%s: the design matrix has state elements in more than one observation' % self.METHOD)
        self._h_state = np.asarray(H.sum(axis=0)).ravel()
        '''soil capacity: the same smax as the post-update threshold, so that the partition hands the water soil cannot
        hold to the other storages (conserved) instead of the threshold deleting it after the update'''
        upper = {}
        self._soil_upper_bound = bool(soil_upper_bound) and 'soilmoist' in names
        if self._soil_upper_bound:
            mk = np.asarray(sv.DM.local_mask['basin_2d']).astype(bool)
            smax = np.asarray(self._thresholder.smax, dtype=float)
            if smax.shape != mk.shape or int(mk.sum()) != n_cell:
                raise ValueError('soil_upper_bound: smax %s / basin mask %s (%d cells) do not match the state vector '
                                 '(%d cells)' % (smax.shape, mk.shape, int(mk.sum()), n_cell))
            upper['soilmoist'] = smax[mk]
        '''snow limits ("snow_bounds", read by src_DA.configure_DA.snow_bounds; None = defaults):
        relative upper bound per member (default 2 x forecast + 20 mm) and relative lower bound (default 0.5: one
        update removes at most half of a snow pack)'''
        from src_DA.configure_DA import snow_bounds as read_snow_bounds
        sb = read_snow_bounds({}) if snow_bounds is None else snow_bounds
        rel, rel_lo = {}, {}
        if 'swe' in names and sb['upper_factor'] is not None:
            rel['swe'] = (sb['upper_factor'], sb['upper_offset_mm'])
        if 'swe' in names and sb['lower_factor'] is not None:
            rel_lo['swe'] = sb['lower_factor']
        '''lakes in the DA state: WaterGAP's floor -max_storage as the lower bound, no lake increment where the cell has
        no lake (a lake written back below its floor stops the lake balance of WaterGAP)'''
        lower = {}
        self._lake_floor_ctx = None                     # lakes in the state: what refresh_lake_floor needs
        if any(v in names for v in ('locallakestor', 'globallakestor', 'reservoirstor')):   # + reservoirs (review 260)
            from src_DA.bounds import lake_floor_fields
            lw = getattr(self._model, 'lateral_waterbalance', None)
            if lw is None:
                raise ValueError('%s: lakes are in the DA state but the model has no lateral_waterbalance for their '
                                 'maximum storages' % self.METHOD)
            lower, up_l = lake_floor_fields(lw, self._model.unit_converter, sv.DM.local_mask['basin_2d'], names)
            upper.update(up_l)
            from src_DA.bounds import lake_floor_grids
            box = np.shape(sv.DM.local_mask['basin_2d'])
            self._thresholder.set_lake_floor(lake_floor_grids(lw, self._model.unit_converter, box, names))
            print('%s: lake bounds - %s' % (self.METHOD, {v: dict(cells_with_lake=int((up_l[v] > 0).sum()),
                                                                   no_lake_cells=int((up_l[v] == 0).sum()))
                                                          for v in lower}))
            self._lake_floor_ctx = dict(lw=lw, names=list(names), mask=sv.DM.local_mask['basin_2d'], box=box,
                                        n_cell=n_cell, lower={v: np.array(a, dtype=float) for v, a in lower.items()},
                                        upper={v: np.array(a, dtype=float) for v, a in up_l.items()})
            self._lake_floor_refresh = dict(n_calls=0, n_changes=0, cells_changed={})
        self.bounds = StateBounds(names, n_cell, upper=upper, relative_upper=rel, relative_lower=rel_lo, lower=lower)
        '''snow: absolute cap per window from the monthly open-loop maximum, factor x max_OL(cell, month) + offset
        (envelope_factor / envelope_offset_mm of "snow_bounds"; None = off). The same cap as Threshold
        (snow_cap_month), on the state cells; the month of the window is set by EnKF.run_mpi (_window_months, the
        largest cap of the months it spans)'''
        self._snow_env = None
        self._snow_cap_now = None                        # cap of the current window (update), None = off
        if sb['envelope_factor'] is not None and 'swe' in names:
            th = self._thresholder
            if th.snow_cap is None or th.snow_env_month is None:
                print('%s: WARNING the snow envelope cap is on but no monthly open-loop snow maximum was loaded -> no snow cap'
                      % self.METHOD)
            else:
                mk = np.asarray(sv.DM.local_mask['basin_2d']).astype(bool)
                env = np.asarray(th.snow_env_month, dtype=float)
                if env.shape[1:] != mk.shape:
                    raise ValueError('snow_bounds envelope: monthly maximum %s does not match the basin box %s'
                                     % (env.shape[1:], mk.shape))
                self._snow_env = env[:, mk]                                       # (12, n_cell)

        '''observation-error inflation: one factor on the error variance of every sub-basin, R' = factor x R (1 = off)'''
        self._oei = float(1.0 if obs_error_inflation is None else obs_error_inflation)
        d = np.full(n_obs, np.sqrt(self._oei))
        self._R_scale = np.outer(d, d)

        '''observation-error correlation between sub-basins: "full" (as delivered with the GRACE product) or "diagonal"
        (sensitivity test: R without off-diagonal terms; the member observations are re-perturbed consistently)'''
        self._obs_corr = str(obs_error_correlation or 'full').lower()
        if self._obs_corr not in ('full', 'diagonal'):
            raise ValueError('obs_error_correlation must be "full" or "diagonal", got %s' % self._obs_corr)
        self._n_obs_decorrelated = 0

        '''centring of the observation perturbations: the members' perturbations (GRACE ens_k - ens_0) are shifted so
        that they average to zero, i.e. the ensemble-mean observation is exactly the unperturbed GRACE value. With
        4 members their mean is otherwise ~sigma/2 per sub-basin (Danube: 12 mm RMS) and pulls every analysis
        towards a randomly shifted observation. The spread of the members is unchanged.'''
        self._obs_centering = bool(obs_perturbation_centering)
        self._n_obs_centered = 0
        self._obs_mean_shift_rms = []

        # bias-aware reference (optional "innovation_bias"; absent / null = off, bit-identical): the systematic part
        # of the forecast-minus-GRACE misfit of every sub-basin is estimated from past innovations and removed from
        # the member observations before the analysis (Dee & da Silva 1998), see _bias_reference
        self._ib = self._innovation_bias_config(innovation_bias, n_obs)

        self.inflation.setup(self)
        self.partition.setup(self)
        print('%s: %d states x %d observations; localization %s (mean taper weight %.3f); inflation: %s; '
              'increment partition: %s; obs-error inflation: %s; obs-error correlation: %s; obs-perturbation '
              'centring: %s; soil upper bound (smax) in the partition: %s; snow upper bound: %s; snow lower bound: %s'
              % (self.METHOD, n_state, n_obs, self.localization.describe(),
                 self._L_state.sum() / (n_state * n_obs),
                 self.inflation.describe(), self.partition.KIND, self._oei if self._oei != 1.0 else 'none', self._obs_corr,
                 self._obs_centering, self._soil_upper_bound,
                 ('%.1f x forecast + %.0f mm' % rel['swe']) if 'swe' in rel else 'none',
                 ('%.2f x forecast' % rel_lo['swe']) if 'swe' in rel_lo else 'none')
              + '; snow cap (monthly open-loop maximum): %s'
              % (('%.1f x max_OL(month) + %.0f mm' % self._thresholder.snow_cap) if self._snow_env is not None
                 else 'none'))

    # ------------------------------------------------------------------ analysis step
    def refresh_lake_floor(self):
        """
        Lakes in the DA state (DA review 251): rebuild the lake bounds from the model's CURRENT maximum lake storages.
        The bounds are first built in __init__, before WaterGAP's first model day; on 1 January WaterGAP merges
        reservoirs with mean_annual_inflow = 0 into the global lakes (waterbalance_lateral.py, reservoir activation),
        which raises max_glolake_storage in those cells (4 cells worldwide in 2002-2020, e.g. Tharthar). With the
        initial bounds (max = 0 -> lower = upper = 0) the filter and the daily threshold clamped these lakes to 0 at
        every write-back. Called by EnKF.run_mpi at every analysis on every rank (each rank clips its own write-back);
        no-op without lakes in the state, and identical bounds when nothing changed.
        """
        ctx = getattr(self, '_lake_floor_ctx', None)
        if ctx is None:
            return
        from src_DA.bounds import lake_floor_fields, lake_floor_grids, LOWER, UPPER
        lower, up_l = lake_floor_fields(ctx['lw'], self._model.unit_converter, ctx['mask'], ctx['names'])
        st = self._lake_floor_refresh
        st['n_calls'] += 1
        ctx.setdefault('upper', {})
        changed = {v: int(np.sum((lower[v] != ctx['lower'][v]) |
                                 (up_l[v] != ctx['upper'].get(v, up_l[v])))) for v in lower}   # reservoirs: capacity (review 260)
        if not any(changed.values()):
            return
        nv = len(self.bounds.names)
        for v, lo in lower.items():
            i = self.bounds.names.index(v)
            self.bounds.lb[i::nv] = np.maximum(LOWER.get(v, -np.inf), np.where(np.isfinite(lo), lo, -np.inf))
            self.bounds.ub[i::nv] = np.minimum(UPPER.get(v, np.inf), np.where(np.isfinite(up_l[v]), up_l[v], np.inf))
            ctx['lower'][v] = np.array(lo, dtype=float)
            ctx['upper'][v] = np.array(up_l[v], dtype=float)
            st['cells_changed'][v] = st['cells_changed'].get(v, 0) + changed[v]
        self._thresholder.set_lake_floor(lake_floor_grids(ctx['lw'], self._model.unit_converter, ctx['box'],
                                                          ctx['names']))
        st['n_changes'] += 1
        print('%s: lake bounds refreshed from the current maximum lake storages - cells changed %s'
              % (self.METHOD, changed))

    def update(self, obs, obs_cov, ens_states):
        """
        obs        : perturbed observations of the members (n_obs x N)
        obs_cov    : R (n_obs x n_obs)
        ens_states : forecast states (n_state x N), window means
        """
        R = obs_cov * self._R_scale
        N = ens_states.shape[1]
        '''unperturbed observation of this window (deterministic filters, src_DA.LETKF): GRACE ens_0 minus the
        member-mean of the excluded storages, which EnKF.run_mpi subtracted from the raw member observations'''
        raw, ref = getattr(self, '_obs_raw', None), getattr(self, '_obs_unperturbed', None)
        self._obs_deterministic = None if (raw is None or ref is None or np.shape(raw) != np.shape(obs)) \
            else np.asarray(ref, dtype=float) - np.mean(raw - obs, 1)
        if self._obs_corr == 'diagonal':
            obs, R = self._decorrelate(obs, obs_cov, R)
        if self._obs_centering:
            obs = self._center_perturbations(obs)
        if self._ib is not None:
            obs = self._bias_reference(obs, ens_states)
        scale = getattr(self, '_ens_soil_scale', None)
        upper_scale = {'soilmoist': scale} if (self._soil_upper_bound and scale is not None and
                                               np.shape(scale) == (ens_states.shape[0] // len(self.bounds.names),
                                                                   ens_states.shape[1])) else None
        '''snow cap of this window (None = off), also used by the inflation to weight the snow noise'''
        self._snow_cap_now = self._snow_cap_window()
        ens_max = getattr(self, '_ens_max', None)
        ens_min = getattr(self, '_ens_min', None)
        if getattr(self, '_soil_cap', None) is not None and 'soilmoist' in self.bounds.names:
            '''"soil_daily_bounds" (EnKF._soil_cap_config): the soil bounds act on the window mean (x_min = x_max := x_mean
            for soil); the daily soil increments are then shaped so that every day stays within its bounds'''
            r = np.arange(self.bounds.names.index('soilmoist'), ens_states.shape[0], len(self.bounds.names))
            if ens_max is not None and np.shape(ens_max) == np.shape(ens_states):
                ens_max = np.array(ens_max, dtype=float, copy=True)
                ens_max[r] = ens_states[r]
            if ens_min is not None and np.shape(ens_min) == np.shape(ens_states):
                ens_min = np.array(ens_min, dtype=float, copy=True)
                ens_min[r] = ens_states[r]
        self.bounds.set_window(ens_states, ens_min, ens_max,
                               upper_scale=upper_scale, upper_window=self._snow_cap_now)

        xm = np.mean(ens_states, 1)[:, None]
        A = self.inflation.prior(ens_states - xm, xm, obs, R)
        X = xm + A

        HX = self._DM(states=X)
        HA = HX - np.mean(HX, 1)[:, None]
        if self._ib is not None and self._ib['mode'] == 'fast_only':
            self.partition._fast_offset = self._fast_only_offset(HA, R, N)
        dX = self.analysis_increment(X, A, HX, HA, obs, R, N)

        dX = self.partition.split(dX, A, HA, N, X)
        return self.inflation.posterior(X + dX, A, HA)

    def analysis_increment(self, X, A, HX, HA, obs, R, N):
        """
        Raw increments of the members (n_state x N) before the partition: the stochastic EnKF update with the
        perturbed observations. The filter variants (src_DA.LETKF) override only this method; the preparation
        (observation-error model, bounds, prior inflation) and the post-processing (partition, posterior inflation)
        in update() are shared.
        X, A   : forecast states and anomalies after the prior inflation (n_state x N)
        HX, HA : their observation equivalents and anomalies (n_obs x N)
        obs    : perturbed member observations (n_obs x N);  R : observation-error covariance (n_obs x n_obs)
        """
        Pyy = self._L_obs * (HA @ HA.T) / (N - 1) + R                             # localized obs-space covariance
        Pxy = self.tapered_cross_cov(A, HA, N)                                    # localized cross-covariance (sparse)
        '''K (obs - HX) with K = Pxy Pyy^-1, without forming the n_state x n_obs gain'''
        return Pxy @ np.linalg.solve(Pyy, obs - HX)

    # ------------------------------------------------------------------ bias-aware reference
    @staticmethod
    def _innovation_bias_config(cfg, n_obs):
        """
        "innovation_bias": {"tau_months": 36, "fraction": 1.0, "init": null, "save": null}
          tau_months  memory of the running mean of the ensemble-mean innovation per sub-basin (in analyses = months):
                      the plain mean of all past innovations until tau analyses are seen, then an exponential mean
                      with weight 1/tau (no ramp-up from zero)
          fraction    share of the estimated bias removed (0 = estimate and log only)
          mode        "observation": the member observations are shifted by -fraction*b (the whole increment changes);
                      "fast_only": the observations are kept; the change of the sub-basin increment that the shift
                      would cause, o = -(Pyy - R) Pyy^-1 fraction*b, is placed in the fast storages only (needs the
                      seasonal split of the non-negative partition), so groundwater increments are not touched
          init/save   .npz with "bias" (n_obs,) and "count" (scalar): start state / state written after every update
          freeze      true: b is taken from "init" and never updated (a fixed per-unit bias, e.g. the mean innovation
                      of an earlier run); a constant b shifts the level only and leaves trends and seasonal cycles
        None, {} or false: off.
        """
        if not cfg:
            return None
        cfg = dict(cfg)
        tau = float(cfg.get('tau_months', 36))
        mode = str(cfg.get('mode', 'observation')).lower()
        if mode not in ('observation', 'fast_only'):
            raise ValueError('innovation_bias: mode must be "observation" or "fast_only", got %s' % mode)
        frac = float(cfg.get('fraction', 1.0))
        if tau < 1 or not (0.0 <= frac <= 1.0):
            raise ValueError('innovation_bias: tau_months >= 1 and 0 <= fraction <= 1 required, got %s / %s' % (tau, frac))
        freeze = bool(cfg.get('freeze', False))
        if freeze and not cfg.get('init'):
            raise ValueError('innovation_bias: freeze needs "init" (a .npz with the fixed bias per sub-basin)')
        st = dict(tau=tau, fraction=frac, mode=mode, freeze=freeze, shift=None, save=cfg.get('save'), init=cfg.get('init'),
                  bias=np.zeros(n_obs), count=0, n_updates=0, log=[])
        if st['init']:
            z = np.load(st['init'])
            if np.shape(z['bias']) != (n_obs,):
                raise ValueError('innovation_bias init: %s has %s sub-basins, the filter %d'
                                 % (st['init'], np.shape(z['bias']), n_obs))
            st['bias'] = np.array(z['bias'], dtype=float)
            st['count'] = int(z['count'])
            if freeze:
                st['count'] = max(st['count'], 1)          # a frozen bias is always applied
        return st

    def _bias_reference(self, obs, ens_states):
        """
        obs (n_obs x N) -> obs - fraction * b, with b the running mean of the PAST ensemble-mean innovations
        d = mean_i(obs_i) - H mean_i(x_i) (forecast window means before inflation). b is updated with this window's d
        after it was used, so the correction is causal. Only the mean observation of every sub-basin is shifted: the
        member spread, R and the seasonal cycle of the innovations are unchanged.
        """
        ib = self._ib
        xm = np.mean(ens_states, 1)[:, None]
        d = np.mean(obs, 1) - np.asarray(self._DM(states=xm)).reshape(-1)
        shift = ib['fraction'] * ib['bias'] if ib['count'] > 0 else np.zeros_like(d)
        ib['shift'] = shift
        if ib['mode'] == 'fast_only':
            out = obs                                                    # offset applied in the partition
        else:
            out = obs - shift[:, None]
            if getattr(self, '_obs_deterministic', None) is not None:
                self._obs_deterministic = self._obs_deterministic - shift
        # update with the raw innovation of this window (not when frozen)
        if not ib['freeze']:
            ib['count'] += 1
            w = 1.0 / min(ib['count'], ib['tau'])
            ib['bias'] = ib['bias'] + w * (d - ib['bias'])
        ib['n_updates'] += 1
        ib['log'].append(dict(window=list(getattr(self, '_window_months', []) or []),
                              innovation_mean_mm=float(np.mean(d)),
                              shift_mean_mm=float(np.mean(shift)),
                              shift_abs_mean_mm=float(np.mean(np.abs(shift))),
                              innovation_after_mean_mm=float(np.mean(d - shift))))
        if ib['save']:
            np.savez(ib['save'], bias=ib['bias'], count=ib['count'])
        return out

    def _fast_only_offset(self, HA, R, N):
        """sub-basin increment change caused by shifting the observations by -shift, o = -(Pyy - R) Pyy^-1 shift
        (ensemble-mean, observation space; the gain is linear in the observations)"""
        shift = self._ib['shift']
        if shift is None or not np.any(shift):
            return None
        Pf = self._L_obs * (HA @ HA.T) / (N - 1)
        o = -Pf @ np.linalg.solve(Pf + R, shift)
        self._ib['log'][-1]['fast_offset_mean_mm'] = float(np.mean(o))
        return o

    def _snow_cap_window(self):
        """{'swe': cap per state cell} for the months of the current window, or None"""
        months = getattr(self, '_window_months', None)
        if self._snow_env is None or not months:
            return None
        fac, off = self._thresholder.snow_cap
        mx = np.fmax.reduce(self._snow_env[[int(m) - 1 for m in sorted(set(months))]], axis=0)   # NaN: no cap
        return {'swe': fac * mx + off}

    def tapered_cross_cov(self, A, HA, N, taper=True):
        """
        L_state o (A HA^T) / (N - 1) evaluated only on the non-zeros of the taper: sparse n_state x n_obs with the
        pattern of L_state (block localization: one entry per state element). taper=False: the plain sample
        cross-covariance (A HA^T) / (N - 1) on the same pattern (used by the non-negative partition for the
        correlations, which are tapered afterwards by the weights themselves).
        """
        L = self._L_state
        rows = np.repeat(np.arange(L.shape[0]), np.diff(L.indptr))
        vals = np.einsum('ik,ik->i', A[rows], HA[L.indices]) / (N - 1)
        if taper:
            vals = L.data * vals
        return sparse.csr_matrix((vals, L.indices, L.indptr), shape=L.shape)

    def h_row(self, j, rows):
        """row j of the design matrix on the state elements `rows`: the sub-basin-mean weights of the elements of
        sub-basin j, 0 for elements of other sub-basins"""
        return np.where(self._owner[rows] == j, self._h_state[rows], 0.0)

    def _center_perturbations(self, obs):
        """
        The member observations are obs_i = y_0 + eps_i - x_excl,i (unperturbed GRACE, perturbation, excluded
        storages of the member). eps_i = GRACE ens_k - ens_0, or its decorrelated version when
        obs_error_correlation is "diagonal". Its member mean is removed: obs_i' = obs_i - mean_i(eps_i), so the
        ensemble-mean observation equals y_0 (minus the mean excluded storages) and the member spread is unchanged.
        """
        eps = getattr(self, '_last_eps', None)
        if eps is None:
            raw, ref = getattr(self, '_obs_raw', None), getattr(self, '_obs_unperturbed', None)
            if raw is None or ref is None or np.shape(raw) != np.shape(obs):
                if self._n_obs_centered == 0:
                    print('%s: obs_perturbation_centering: unperturbed observations not available, not applied'
                          % self.METHOD)
                return obs
            eps = raw - np.asarray(ref)[:, None]
        self._last_eps = None
        shift = eps.mean(1)
        self._n_obs_centered += 1
        self._obs_mean_shift_rms.append(float(np.sqrt(np.mean(shift ** 2))))
        return obs - shift[:, None]

    def _decorrelate(self, obs, obs_cov, R):
        """
        Diagonal R and member observations perturbed consistently with it. The perturbations of the members,
        eps_i = y_i - y_0 (GRACE ens_k - ens_0, drawn from N(0, C) with the full covariance C of the product), are
        replaced by independent draws eps_i' ~ N(0, diag(R)) from a generator seeded with OBS_PERTURBATION_SEED
        (one draw per update, the same sequence in every run), so that the spread of the member observations is
        exactly the R used in the gain.
        (Whitening eps_i with a factor of C is exact only for a well-conditioned C; the sample covariance of many
        units is rank-deficient, and the whitened perturbations would then have less spread than sqrt(R_jj).)
        """
        Rd = np.diag(np.diag(R))
        raw, ref = getattr(self, '_obs_raw', None), getattr(self, '_obs_unperturbed', None)
        if raw is None or ref is None or np.shape(raw) != np.shape(obs):
            if self._n_obs_decorrelated == 0:
                print('%s: obs_error_correlation "diagonal": unperturbed observations not available, only R is made '
                      'diagonal (member perturbations keep their correlation)' % self.METHOD)
            return obs, Rd
        eps = raw - np.asarray(ref)[:, None]
        if not hasattr(self, '_obs_rng'):
            self._obs_rng = np.random.default_rng(self.OBS_PERTURBATION_SEED)
        eps_new = np.sqrt(np.diag(R))[:, None] * self._obs_rng.standard_normal(eps.shape)
        self._n_obs_decorrelated += 1
        self._last_eps = eps_new                    # perturbation now contained in obs (used by the centring)
        return obs - eps + eps_new, Rd

    # ------------------------------------------------------------------ reporting
    def run_mpi(self):
        super().run_mpi()
        s = self.partition.stats
        if s.get('n_updates', 0) > 0 and self.partition.KIND == 'non_negative':
            print('%s, increment partition (non_negative): %d updates, variance-only fallback for %d sub-basin '
                  'updates, %d unresolved; largest |increment| of one element: EnKF %.1f mm -> applied %.1f mm (%s); '
                  'bound-aware: %d element-members capped, %.3g mm redistributed'
                  % (self.METHOD, s['n_updates'], s['n_obs_fallback'], s['n_obs_unresolved'], s['max_abs_inc_enkf'],
                     s['max_abs_inc_new'], s.get('max_inc_where'), s['n_bound_capped'], s['water_redistributed_mm']))

    def filter_summary(self):
        return dict(method=self.METHOD, localization=self.localization.describe(),
                    inflation=self.inflation.summary(), partition=self.partition.summary(),
                    bounds=dict(self.bounds.summary(), soil_upper_bound=self._soil_upper_bound,
                                snow_envelope=None if self._snow_env is None else
                                dict(zip(('factor', 'offset_mm'), self._thresholder.snow_cap))),
                    obs_error_inflation=self._oei,
                    obs_error_correlation=self._obs_corr, n_obs_decorrelated=self._n_obs_decorrelated,
                    obs_perturbation_centering=self._obs_centering, n_obs_centered=self._n_obs_centered,
                    innovation_bias=None if self._ib is None else
                    dict(tau_months=self._ib['tau'], fraction=self._ib['fraction'], mode=self._ib['mode'],
                         freeze=self._ib['freeze'],
                         n_updates=self._ib['n_updates'],
                         final_bias_mean_mm=float(np.mean(self._ib['bias'])),
                         final_bias_abs_mean_mm=float(np.mean(np.abs(self._ib['bias']))),
                         per_window=self._ib['log']),
                    mean_perturbation_removed_rms_mm=(float(np.mean(self._obs_mean_shift_rms))
                                                      if self._obs_mean_shift_rms else None))

    def save_filter_log(self, out_dir):
        """filter_summary.json and, for the adaptive inflation, adaptive_inflation_log.csv (one row per window and
        sub-basin) in out_dir"""
        import json, csv
        from pathlib import Path
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        json.dump(self.filter_summary(), open(out_dir / 'filter_summary.json', 'w'), indent=1,
                  default=lambda o: np.asarray(o).tolist())
        rows = self.inflation.log_rows()
        if rows:
            with open(out_dir / 'adaptive_inflation_log.csv', 'w', newline='') as fh:
                wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
