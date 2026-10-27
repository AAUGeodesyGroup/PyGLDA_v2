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
transform per unit, 772 N x N problems; Gaussian taper: about one per cell). Groups with the same number of local
observations are solved in batched NumPy calls; the N x N eigendecomposition is replaced by the exact rank-p form
(p = number of local observations < N, a p x p eigendecomposition), cells are updated in batched chunks.

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
    GROUP_CHUNK = 4096       # groups solved per batched call
    CELL_CHUNK = 8192        # cells per batched application of the transforms

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
        '''groups with the same number of local observations p are solved together (batched): per p the observation
        indices (G x p), taper weights (G x p) and the cells of every group'''
        by_p = {}
        for cells, idx, rho in self._groups:
            if idx.size:
                by_p.setdefault(idx.size, []).append((cells, idx, rho))
        self._buckets = [(np.array([g[1] for g in gr]), np.array([g[2] for g in gr]), [g[0] for g in gr])
                         for p, gr in sorted(by_p.items())]
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
        r_diag = np.diag(R)
        is_diag = np.allclose(R, np.diag(r_diag))
        dX = np.zeros_like(X)
        w_rms, sr = [], []
        for IDX, RHO, cells_list in self._buckets:
            for g0 in range(0, IDX.shape[0], self.GROUP_CHUNK):
                sl = slice(g0, g0 + self.GROUP_CHUNK)
                T, wr, s = self._transforms(IDX[sl], RHO[sl], HA, d, R, r_diag, is_diag, N)
                w_rms.append(wr)
                sr.append(s)
                self._apply(dX, A, T, cells_list[sl], nvar, N)
        w_rms = np.concatenate(w_rms) if w_rms else np.zeros(0)
        sr = np.concatenate(sr) if sr else np.zeros(0)
        self.letkf_stats['n_updates'] += 1
        self.letkf_stats['w_mean_rms'].append(float(np.mean(w_rms)) if w_rms.size else 0.0)
        self.letkf_stats['spread_ratio_obs'].append(float(np.median(sr)) if sr.size else 1.0)
        return dX

    def _transforms(self, IDX, RHO, HA, d, R, r_diag, is_diag, N):
        """
        Ensemble transforms of G groups with p local observations each, in batched NumPy calls.
        With U = Yb^T Rl^-1/2 (N x p) the local matrix is (N-1) I + U U^T: only p of its N eigenvalues differ from
        N-1, so with U = V diag(sqrt mu) Q^T (V: N x p orthonormal, mu: eigenvalues of the p x p matrix U^T U)
            Pa = [(N-1) I + U U^T]^-1 = (I - V diag(mu / (N-1+mu)) V^T) / (N-1)
            W  = [(N-1) Pa]^1/2       = I + V diag(sqrt((N-1)/(N-1+mu)) - 1) V^T
        exactly (Hunt et al. 2007 eqs. 20-24 without the N x N eigendecomposition). For p >= N the full N x N
        eigendecomposition is used. Returns w (G x N), V (G x N x k), s1 = diag of W - I in that basis (G x k), so
        that T = w 1^T + V diag(s1) V^T, and two diagnostics per group (RMS of w; analysis / forecast spread ratio of
        the group's observation equivalents).
        IDX, RHO: G x p observation indices and taper weights.
        """
        km1 = N - 1
        Yb = HA[IDX]                                                      # G x p x N
        dl = d[IDX]                                                       # G x p
        if is_diag:
            sw = np.sqrt(RHO / r_diag[IDX])                               # G x p: Rl^-1/2 = diag(sqrt(rho / R_jj))
            U = (Yb * sw[:, :, None]).transpose(0, 2, 1)                  # G x N x p
            e = sw * dl                                                   # Rl^-1/2 d
        else:
            sq = np.sqrt(RHO)
            Rl_inv = sq[:, :, None] * np.linalg.inv(R[IDX[:, :, None], IDX[:, None, :]]) * sq[:, None, :]
            Lc = np.linalg.cholesky(Rl_inv)                               # Rl^-1 = Lc Lc^T
            U = Yb.transpose(0, 2, 1) @ Lc                                # G x N x p
            e = np.einsum('gqp,gq->gp', Lc, dl)                           # Lc^T d
        Cd = np.einsum('gnp,gp->gn', U, e)                                # Yb^T Rl^-1 d
        if U.shape[2] < N:
            mu, Q = np.linalg.eigh(U.transpose(0, 2, 1) @ U)              # G x p, G x p x p
            ok = mu > 1e-12 * np.maximum(mu.max(axis=1, keepdims=True), 1e-300)
            V = (U @ Q) / np.sqrt(np.where(ok, mu, 1.0))[:, None, :]
            V = V * ok[:, None, :]                                        # directions without information: dropped
            mu = np.where(ok, mu, 0.0)
            w = (Cd - np.einsum('gnk,gk->gn', V, mu / (km1 + mu) * np.einsum('gnk,gn->gk', V, Cd))) / km1
            s1 = np.sqrt(km1 / (km1 + mu)) - 1.0
        else:
            lam, V = np.linalg.eigh(km1 * np.eye(N) + U @ U.transpose(0, 2, 1))
            lam = np.maximum(lam, 1e-12 * lam.max(axis=1, keepdims=True))
            w = np.einsum('gnk,gk->gn', V, np.einsum('gnk,gn->gk', V, Cd) / lam)
            s1 = np.sqrt(km1 / lam) - 1.0
        YV = Yb @ V                                                       # G x p x k
        Ya = Yb + (YV * s1[:, None, :]) @ V.transpose(0, 2, 1)            # Yb W
        ratio = np.sqrt(np.sum(Ya ** 2, axis=(1, 2)) / np.maximum(np.sum(Yb ** 2, axis=(1, 2)), 1e-30))
        return (w, V, s1), np.sqrt(np.mean(w ** 2, axis=1)), ratio

    def _apply(self, dX, A, T, cells_list, nvar, N):
        """dX of the cells of G groups, dX_c = A_c (w 1^T + V diag(s1) V^T) with the transform of the cell's group,
        for all storages of a cell, in batched chunks of CELL_CHUNK cells (the N x N transforms are never formed)"""
        w, V, s1 = T
        cells = np.concatenate(cells_list)
        gid = np.repeat(np.arange(len(cells_list)), [len(c) for c in cells_list])
        for c0 in range(0, cells.size, self.CELL_CHUNK):
            cc, g = cells[c0:c0 + self.CELL_CHUNK], gid[c0:c0 + self.CELL_CHUNK]
            rows = (cc[:, None] * nvar + np.arange(nvar)[None, :]).ravel()
            Ac = A[rows].reshape(cc.size, nvar, N)
            mean = np.einsum('cvn,cn->cv', Ac, w[g])                       # A_c w: shift of the mean
            AV = (Ac @ V[g]) * s1[g][:, None, :]                            # A_c V diag(s1)
            dX[rows] = (mean[:, :, None] + AV @ V[g].transpose(0, 2, 1)).reshape(-1, N)

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
