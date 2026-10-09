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
see doc/configuration.md. The daily loop, MPI, window means / extremes and the post-update thresholds live in the
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
                 partition: Partition = None, obs_error_inflation: dict = None, obs_error_correlation: str = 'full',
                 obs_perturbation_centering: bool = False, soil_upper_bound: bool = True, snow_bounds: dict = None):
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
        self.bounds = StateBounds(names, n_cell, upper=upper, relative_upper=rel, relative_lower=rel_lo)
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

        '''observation-error inflation per sub-basin (1-based keys): R' = D R D keeps the error correlations'''
        self._oei = dict(obs_error_inflation or {})
        d = np.ones(n_obs)
        for k, fac in self._oei.items():
            d[int(k) - 1] = np.sqrt(float(fac))
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

        self.inflation.setup(self)
        self.partition.setup(self)
        print('%s: %d states x %d observations; localization %s (mean taper weight %.3f); inflation: %s; '
              'increment partition: %s; obs-error inflation: %s; obs-error correlation: %s; obs-perturbation '
              'centring: %s; soil upper bound (smax) in the partition: %s; snow upper bound: %s; snow lower bound: %s'
              % (self.METHOD, n_state, n_obs, self.localization.describe(),
                 self._L_state.sum() / (n_state * n_obs),
                 self.inflation.describe(), self.partition.KIND, self._oei if self._oei else 'none', self._obs_corr,
                 self._obs_centering, self._soil_upper_bound,
                 ('%.1f x forecast + %.0f mm' % rel['swe']) if 'swe' in rel else 'none',
                 ('%.2f x forecast' % rel_lo['swe']) if 'swe' in rel_lo else 'none')
              + '; snow cap (monthly open-loop maximum): %s'
              % (('%.1f x max_OL(month) + %.0f mm' % self._thresholder.snow_cap) if self._snow_env is not None
                 else 'none'))

    # ------------------------------------------------------------------ analysis step
    def update(self, obs, obs_cov, ens_states):
        """
        obs        : perturbed observations of the members (n_obs x N)
        obs_cov    : R (n_obs x n_obs)
        ens_states : forecast states (n_state x N), window means
        """
        R = obs_cov * self._R_scale
        N = ens_states.shape[1]
        if self._obs_corr == 'diagonal':
            obs, R = self._decorrelate(obs, obs_cov, R)
        if self._obs_centering:
            obs = self._center_perturbations(obs)
        scale = getattr(self, '_ens_soil_scale', None)
        upper_scale = {'soilmoist': scale} if (self._soil_upper_bound and scale is not None and
                                               np.shape(scale) == (ens_states.shape[0] // len(self.bounds.names),
                                                                   ens_states.shape[1])) else None
        '''snow cap of this window (None = off), also used by the inflation to weight the snow noise'''
        self._snow_cap_now = self._snow_cap_window()
        self.bounds.set_window(ens_states, getattr(self, '_ens_min', None), getattr(self, '_ens_max', None),
                               upper_scale=upper_scale, upper_window=self._snow_cap_now)

        xm = np.mean(ens_states, 1)[:, None]
        A = self.inflation.prior(ens_states - xm, xm, obs, R)
        X = xm + A

        HX = self._DM(states=X)
        HA = HX - np.mean(HX, 1)[:, None]
        Pyy = self._L_obs * (HA @ HA.T) / (N - 1) + R                             # localized obs-space covariance
        Pxy = self.tapered_cross_cov(A, HA, N)                                    # localized cross-covariance (sparse)
        '''K (obs - HX) with K = Pxy Pyy^-1, without forming the n_state x n_obs gain'''
        dX = Pxy @ np.linalg.solve(Pyy, obs - HX)

        dX = self.partition.split(dX, A, HA, N, X)
        return self.inflation.posterior(X + dX, A, HA)

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
        Until 7 Oct 2026 eps_i was whitened with a factor of C (eps_i' = diag(sqrt R_jj) L^-1 eps_i, C = L L^T),
        which is exact only for a well-conditioned C: the sample covariance of 772 global units is rank-deficient
        (fewer samples than units) or ill-conditioned, the eigenvalue floor then shrank most components of L^-1 eps,
        and the member observations had a fraction of the spread sqrt(R_jj) (over-confident analyses).
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
