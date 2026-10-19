"""
LETKF_localized - deterministic (ensemble-transform) analysis for the basin-averaged GRACE observations, built from
the same components as src_DA.EnKF_localized (localization, inflation, increment partition, bounds). It replaces
only the raw member increments K (y - HX) of the stochastic EnKF (EnKF_localized.analysis_increment); the
observation-error model, the window-aware bounds, the prior / posterior inflation and the non-negative partition are
inherited unchanged, so the two filters differ in the analysis equations alone.

Hunt, Kostelich & Szunyogh (2007, Physica D 230, 112), for every state cell c with its local observations J (the
non-zeros of the localization taper L_state in the cell's row, taper weight rho_j in (0, 1]; all observations with
rho = 1 under "none", the cell's own unit under "block", the units within cutoff x length under "gaussian"):

    Rl^-1  = D^1/2 R_J^-1 D^1/2,  D = diag(rho_j)       R-localization: a distant observation counts less
    C      = Yb_J^T Rl^-1                                  N x p      (Yb_J = HA_J, observation-space anomalies)
    Pa~    = [(N-1) I + C Yb_J]^-1                         N x N      (symmetric eigendecomposition)
    w      = Pa~ C (y_J - mean(HX)_J)                      mean weights
    W      = [(N-1) Pa~]^1/2                               symmetric square root (analysis anomalies)
    dX_c   = A_c (w 1^T + W - I)                           member increments of the rows of cell c

y is the UNPERTURBED observation (GRACE ens_0 minus the member-mean of the excluded storages, from
EnKF_localized.update); the perturbed member observations are used only by the adaptive inflation (innovation
statistics). Cells with the same local observation set and tapers share one transform (block localization: one
transform per unit, 772 N x N problems; Gaussian taper: one per cell, a few seconds per window).

Compared with EnKF_localized: no sampling noise from the observation perturbations (the analysis spread is exactly
the Kalman spread, the gain is not randomised), and the cross-covariance taper acts through R (R-localization)
instead of element-wise on Pxy. The partition then redistributes the per-member unit increments H dX over the cells
and storages exactly as for the EnKF, so the unit-mean analysis of every member (mean and spread) is the LETKF's.

DA_setting.json, "method" block: "fusion_method": "LETKF", all other keys as for EnKF_localized, and optionally
    "letkf": {"min_taper": 0.0}     observations with a taper weight below min_taper are dropped from a cell's set
                                    (Gaussian localization only; block and none have weights 0 or 1)
"""
import numpy as np
from src_DA.EnKF_localized import EnKF_localized


class LETKF_localized(EnKF_localized):
    METHOD = 'LETKF'

    def __init__(self, *args, letkf=None, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = dict(letkf or {})
        self.min_taper = float(cfg.get('min_taper', 0.0))
        self._groups = None                      # [(cells, obs_idx, rho)] built on the first update
        self._y_fallback_warned = False
        self.letkf_stats = dict(n_updates=0, n_groups=0, n_cells=0, n_cells_without_obs=0, local_obs_mean=0.0,
                                w_mean_rms=[], spread_ratio_obs=[])
        print('%s: deterministic ensemble-transform analysis with R-localization from the %s taper (min_taper %.2f)'
              % (self.METHOD, self.localization.KIND, self.min_taper))

    # ------------------------------------------------------------------ local observation sets
    def _build_groups(self):
        """cells grouped by their local observation set (indices) and taper weights (rounded), from L_state"""
        nvar = len(self.bounds.names)
        L = self._L_state[::nvar]                                        # rows of the first storage of every cell
        L = L.tocsr()
        groups = {}
        for c in range(L.shape[0]):
            a, b = L.indptr[c], L.indptr[c + 1]
            idx, rho = L.indices[a:b], L.data[a:b]
            keep = rho >= max(self.min_taper, 1e-12)
            idx, rho = idx[keep], np.minimum(rho[keep], 1.0)
            order = np.argsort(idx)
            idx, rho = idx[order], rho[order]
            key = (idx.tobytes(), np.round(rho, 6).tobytes())
            if key not in groups:
                groups[key] = (idx, rho, [])
            groups[key][2].append(c)
        self._groups = [(np.asarray(cells), idx, rho) for idx, rho, cells in groups.values()]
        n_without = sum(len(cells) for cells, idx, _ in self._groups if idx.size == 0)
        n_cell = int(L.shape[0])
        self.letkf_stats.update(n_groups=len(self._groups), n_cells=n_cell, n_cells_without_obs=int(n_without),
                                local_obs_mean=float(sum(len(cells) * idx.size for cells, idx, _ in self._groups)
                                                     / max(n_cell, 1)))
        print('%s: %d transforms for %d cells (mean %.1f local observations per cell, %d cells without observation)'
              % (self.METHOD, len(self._groups), L.shape[0], self.letkf_stats['local_obs_mean'], n_without))

    # ------------------------------------------------------------------ analysis step
    def analysis_increment(self, X, A, HX, HA, obs, R, N):
        """ensemble-transform increments (n_state x N) of all cells, see the module docstring"""
        if self._groups is None:
            self._build_groups()
        y = self._obs_deterministic
        if y is None:
            if not self._y_fallback_warned:
                print('%s: unperturbed observations not available, the member mean of the perturbed observations is '
                      'used instead' % self.METHOD)
                self._y_fallback_warned = True
            y = np.mean(obs, 1)
        d = np.asarray(y, dtype=float) - np.mean(HX, 1)                   # innovation of the ensemble mean (n_obs)
        nvar = len(self.bounds.names)
        km1 = N - 1
        r_diag = np.diag(R)
        is_diag = np.allclose(R, np.diag(r_diag))
        I = np.eye(N)
        dX = np.zeros_like(X)
        w_rms, sr = [], []
        for cells, idx, rho in self._groups:
            if idx.size == 0:
                continue
            Yb = HA[idx]                                                      # p x N
            if is_diag:
                C = (Yb * (rho / r_diag[idx])[:, None]).T                     # N x p = Yb^T diag(rho / R_jj)
            else:
                sq = np.sqrt(rho)
                Rl_inv = sq[:, None] * np.linalg.inv(R[np.ix_(idx, idx)]) * sq[None, :]
                C = Yb.T @ Rl_inv
            lam, V = np.linalg.eigh(km1 * I + C @ Yb)
            lam = np.maximum(lam, 1e-12 * lam.max())
            Pa = (V / lam) @ V.T                                              # (N-1) I + C Yb inverted
            W = (V * np.sqrt(km1 / lam)) @ V.T                                # symmetric square root of (N-1) Pa
            w = Pa @ (C @ d[idx])
            T = w[:, None] + W - I
            rows = (cells[:, None] * nvar + np.arange(nvar)[None, :]).ravel()
            dX[rows] = A[rows] @ T
            w_rms.append(float(np.sqrt(np.mean(w ** 2))))
            '''analysis / forecast spread of the group's own observation equivalents (diagnostic)'''
            Ya = Yb @ W
            sr.append(float(np.sqrt(np.sum(Ya ** 2) / max(np.sum(Yb ** 2), 1e-30))))
        self.letkf_stats['n_updates'] += 1
        self.letkf_stats['w_mean_rms'].append(float(np.mean(w_rms)) if w_rms else 0.0)
        self.letkf_stats['spread_ratio_obs'].append(float(np.median(sr)) if sr else 1.0)
        return dX

    # ------------------------------------------------------------------ reporting
    def filter_summary(self):
        out = super().filter_summary()
        s = self.letkf_stats
        out['letkf'] = dict(min_taper=self.min_taper, n_updates=s['n_updates'], n_transforms=s['n_groups'],
                            n_cells=s['n_cells'], n_cells_without_obs=s['n_cells_without_obs'],
                            local_obs_mean=s['local_obs_mean'],
                            w_mean_rms_mean=float(np.mean(s['w_mean_rms'])) if s['w_mean_rms'] else None,
                            spread_ratio_obs_median=float(np.median(s['spread_ratio_obs']))
                            if s['spread_ratio_obs'] else None)
        return out
