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
    "obs_error_inflation": {}   per-sub-basin factor on the observation error variance, e.g. {"18": 9} triples
                                sigma of sub-basin 18 (1-based index); R' = D R D keeps the error correlations
    "increment_partition": "enkf" | "non_negative"
        enkf         : increments as given by the (localized) Kalman gain
        non_negative : the sub-basin TWS increment of the Kalman update is kept, but it is distributed over the
                       cells and storages of the sub-basin with non-negative shares proportional to
                       variance x max(correlation with the sub-basin TWS, 0). No storage is emptied to fill another
                       one, and no element can receive more than the sub-basin increment times its share. Meant for
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
        oei = getattr(m, 'obs_error_inflation', None) or {}
        d = np.ones(n_obs)
        for k, f in oei.items():
            d[int(k) - 1] = np.sqrt(float(f))
        self._R_scale = np.outer(d, d)                      # R' = D R D keeps the error correlations

        self._partition = str(getattr(m, 'increment_partition', 'enkf')).lower()
        if self._partition not in ('enkf', 'non_negative'):
            raise ValueError('increment_partition must be "enkf" or "non_negative", got %s' % self._partition)
        self._partition_stats = dict(n_updates=0, n_obs_fallback=0, n_obs_unresolved=0,
                                     max_abs_inc_enkf=0.0, max_abs_inc_new=0.0)

        print('EnKF_localized: kind=%s, length=%.0f km, cutoff=%.1f; taper %d states x %d observations, '
              'mean weight %.3f (block would be %.3f); inflation %.2f, RTPS alpha %.2f, obs-error inflation %s, '
              'increment partition %s'
              % (loc['kind'], loc['length_km'], loc['cutoff'], n_state, n_obs, self._L_state.mean(), 1.0 / n_obs,
                 self._inflation, self._rtps_alpha, oei if oei else 'none', self._partition))

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
                  '%d unresolved; largest |increment| of one element: EnKF %.1f mm -> applied %.1f mm'
                  % (self._partition, s['n_updates'], s['n_obs_fallback'], s['n_obs_unresolved'],
                     s['max_abs_inc_enkf'], s['max_abs_inc_new']))

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

        HX = self._DM(states=X)
        HA = HX - np.mean(HX, 1)[:, None]

        Pyy = self._L_obs * (HA @ HA.T) / (N - 1) + R                             # localized obs-space covariance
        Pxy = self._L_state * (A @ HA.T) / (N - 1)                                # localized cross-covariance

        '''K = Pxy Pyy^-1, solved without forming the inverse'''
        K = np.linalg.solve(Pyy.T, Pxy.T).T

        dX = K @ (obs - HX)                                                        # EnKF increments (n_state x N)
        if self._partition == 'non_negative':
            dX = self._partition_non_negative(dX, A, HA, N)
        Xa = X + dX

        '''RTPS (Whitaker & Hamill 2012): relax the analysis spread back towards the forecast spread,
           sigma_a' = alpha * sigma_f + (1 - alpha) * sigma_a, element-wise; the analysis mean is unchanged'''
        if self._rtps_alpha > 0:
            xa_m = np.mean(Xa, 1)[:, None]
            Aa = Xa - xa_m
            sig_f = np.std(A, axis=1, ddof=1)
            sig_a = np.std(Aa, axis=1, ddof=1)
            factor = np.ones_like(sig_a)
            ok = sig_a > 1e-12
            factor[ok] = self._rtps_alpha * sig_f[ok] / sig_a[ok] + (1.0 - self._rtps_alpha)
            Xa = xa_m + Aa * factor[:, None]

        return Xa

    # ------------------------------------------------------------------ non-negative disaggregation
    def _partition_non_negative(self, dX, A, HA, N):
        """
        Keep the observation-space part of the EnKF update (how much water each sub-basin gains or loses) but
        redistribute it over the cells and storages of that sub-basin with non-negative shares, so that no storage
        is emptied to fill another one.

        dX : EnKF increments (n_state x N);  A, HA : forecast anomalies of states / observation equivalents
        For sub-basin j:   delta_j = (H dX)_j  (per member)
                           w_ij  = L_ij * var_i * max(corr(A_i, HA_j), 0)       (0 for anti-correlated elements)
                           dX_i  = sum_j  w_ij / (H w_.j)_j * delta_j            (so that H dX is unchanged)
        If no element of a sub-basin is positively correlated, the shares fall back to L_ij * var_i.
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

        self._partition_stats['n_updates'] += 1
        self._partition_stats['n_obs_fallback'] += int(bad.sum())
        self._partition_stats['n_obs_unresolved'] += int((~ok).sum())
        self._partition_stats['max_abs_inc_enkf'] = max(self._partition_stats['max_abs_inc_enkf'], float(np.abs(dX).max()))
        self._partition_stats['max_abs_inc_new'] = max(self._partition_stats['max_abs_inc_new'], float(np.abs(dX_new).max()))
        return dX_new

    def partition_summary(self):
        return dict(self._partition_stats, partition=self._partition)
