"""
LETKF - Local Ensemble Transform Kalman Filter for PyGLDA
=========================================================

Reference: Hunt, Kostelich & Szunyogh (2007), "Efficient data assimilation for
spatiotemporal chaos: A local ensemble transform Kalman filter", Physica D 230.

Why this filter for the global GRACE (3-degree) assimilation
------------------------------------------------------------
* Deterministic (square-root) update: the analysis spread is set by the symmetric
  square root of the k x k analysis covariance in ensemble space. No perturbed
  observations are needed, which removes the sampling noise a stochastic EnKF pays
  on every cycle - important at small ensemble size.
* Analysis in ensemble space: every local analysis is a k x k problem (k = ensemble
  size) whatever the state dimension. Cost per grid cell is O(k^3) after one
  eigendecomposition; the whole globe at 0.5 degree is a few seconds per month.
* Localisation in observation space (R-localisation): each 0.5-degree state cell is
  updated only by GRACE observations whose footprint is within a cut-off distance,
  with the observation-error variance inflated by 1/rho(d), rho = Gaspari-Cohn taper.
  This removes spurious long-range covariances of a small ensemble and, unlike a
  covariance-localised global EnKF, never needs an n x n matrix.

How it fits the existing PyGLDA machinery
-----------------------------------------
* Subclass of `EnKF`: the monthly cycle in `EnKF.run_mpi` is reused unchanged -
  states are averaged over the observation window, the analysis increment of the
  monthly mean is added to every day of the window and to the current prediction,
  the OL rank (rank 0) is never updated.
* Observation operator: `DM_basin_average` is reused as-is. For the regional case
  its rows are sub-basin area-weighted means; for the global case build the mask
  file with one `sub_basin_k` per 3-degree GRACE cell and the same class produces
  one row per GRACE cell. The LETKF reads each row's footprint (cells with non-zero
  weight) and its area-weighted centroid to build the localisation.
* State vector layout (from `EnsStates.get_states_by_transfer_single`): cell-major,
  index = cell * V + variable, V = `DM.vertical_dim`. The local analysis for cell c
  therefore updates rows c*V ... (c+1)*V - 1 with one common weight vector, which
  keeps the vertical partition of the TWS increment consistent with the ensemble.
* Observations: the LETKF must be fed the UNPERTURBED observation vector. In the
  MPI cycle that is the vector gathered from the OL rank (rank 0). `EnKF.run_mpi`
  currently discards it; see `_obs_for_update` below for the 4-line hook. Without
  the hook the filter falls back to the column mean of the perturbed ensemble
  observations and prints a warning once.

Algorithm (per state cell, Hunt et al. 2007, eqs. 20-24)
-------------------------------------------------------
    Xb  : n x k background perturbations,  xb_mean : n
    Yb  : p x k perturbations in obs space, yb_mean : p
    for a cell with local obs set L (|L| = p_loc) and taper rho_j, j in L:
        Rl^-1  = D^1/2 R_L^-1 D^1/2,   D = diag(rho_j)           (R-localisation)
        C      = Yb_L^T Rl^-1                                      k x p_loc
        Pa~    = [ (k-1) I / infl  +  C Yb_L ]^-1                  k x k   (eigh)
        w_mean = Pa~ C (yo_L - yb_mean_L)                          k
        W      = [ (k-1) Pa~ ]^1/2                                 k x k   (symmetric)
        xa_i   = xb_mean + Xb (w_mean + W[:, i])                   rows of this cell
Optionally RTPS (Whitaker & Hamill 2012) relaxes the analysis spread toward the
prior spread: sigma_a <- (1 - alpha) sigma_a + alpha sigma_b.
"""

import warnings
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from src_DA.EnKF import EnKF
from src_DA.configure_DA import config_DA
from src_DA.observations import GRACE_obs
from src_DA.ExtractStates import EnsStates
from src_DA.ObsDesignMatrix import DM_basin_average
from src_GHM.Interface.DailyStepRun import DailyModelRun as model_run_daily

EARTH_RADIUS_KM = 6371.0


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def gaspari_cohn(r):
    """Gaspari & Cohn (1999) 5th-order piecewise rational taper.
    r = distance / localisation radius. Equals 1 at r=0, 0 for r >= 2."""
    r = np.abs(np.asarray(r, dtype=float))
    g = np.zeros_like(r)
    m = r < 1.0
    x = r[m]
    g[m] = 1.0 - 5.0 / 3.0 * x ** 2 + 5.0 / 8.0 * x ** 3 + 0.5 * x ** 4 - 0.25 * x ** 5
    m = (r >= 1.0) & (r < 2.0)
    x = r[m]
    g[m] = (4.0 - 5.0 * x + 5.0 / 3.0 * x ** 2 + 5.0 / 8.0 * x ** 3
            - 0.5 * x ** 4 - 1.0 / 12.0 * x ** 5 - 2.0 / (3.0 * x))
    return g


def latlon_to_unit(lat_deg, lon_deg):
    """(lat, lon) in degrees -> unit vectors (N x 3)."""
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def unit_to_latlon(u):
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    return np.rad2deg(np.arcsin(u[..., 2])), np.rad2deg(np.arctan2(u[..., 1], u[..., 0]))


def global_cell_latlon(global_2d_mask, res=0.5):
    """lat/lon of every True cell of a global (lat, lon) mask, in the same order that
    boolean indexing produces - i.e. the order of the PyGLDA state vector cells."""
    err = res / 10
    lat_coords = np.arange(90 - res / 2, -90 + res / 2 - err, -res)
    lon_coords = np.arange(-180 + res / 2, 180 - res / 2 + err, res)
    lon_mesh, lat_mesh = np.meshgrid(lon_coords, lat_coords)
    m = global_2d_mask.astype(bool)
    return lat_mesh[m], lon_mesh[m]


# --------------------------------------------------------------------------------------
# localisation geometry: built once from the design matrix
# --------------------------------------------------------------------------------------
class ObsLocalization:
    """
    From the design matrix H (p x n, n = n_cells * V) derive, for every observation,
    its footprint (cells with non-zero weight) and area-weighted centroid, then build
    a sparse taper matrix T (n_cells x p): T[c, j] = Gaspari-Cohn( d(c, j) / radius ),
    forced to 1 when cell c lies inside the footprint of observation j.
    Observations farther than 2 * radius from a cell are not used for that cell.
    """

    def __init__(self, H, cell_lat, cell_lon, vertical_dim, radius_km=800.0,
                 footprint_full_weight=True):
        p, n = H.shape
        V = int(vertical_dim)
        n_cells = n // V
        assert n_cells * V == n, "state length is not a multiple of vertical_dim"

        # per-cell weight of each observation: rows are repeated over V, take column c*V.
        # H may be dense (current DM_basin_average) or scipy.sparse (global case).
        Hc = sparse.csr_matrix(H)[:, ::V]                # p x n_cells, sparse
        Hc.eliminate_zeros()
        self.footprint = (Hc > 0).T.tocsr()              # n_cells x p membership

        cell_u = latlon_to_unit(cell_lat, cell_lon)      # n_cells x 3
        row_sum = np.asarray(Hc.sum(axis=1)).ravel()
        w = sparse.diags(1.0 / np.maximum(row_sum, 1e-300)) @ Hc
        obs_u = np.asarray(w @ cell_u)                   # weighted centroid (not yet unit)
        self.obs_lat, self.obs_lon = unit_to_latlon(obs_u)
        obs_u = obs_u / np.linalg.norm(obs_u, axis=1, keepdims=True)

        # neighbour search: chord distance on the unit sphere <-> great-circle distance
        cutoff_rad = 2.0 * radius_km / EARTH_RADIUS_KM   # GC support ends at 2 * radius
        chord_cut = 2.0 * np.sin(min(cutoff_rad, np.pi) / 2.0) if cutoff_rad < np.pi else 2.0

        # all (cell, obs) pairs closer than the cut-off, as a sparse matrix of chord
        # distances - computed entirely in C by the KD-trees
        D = cKDTree(cell_u).sparse_distance_matrix(cKDTree(obs_u), chord_cut,
                                                   output_type='coo_matrix')
        d_km = 2.0 * np.arcsin(np.clip(D.data / 2.0, 0.0, 1.0)) * EARTH_RADIUS_KM
        T = sparse.csr_matrix((gaspari_cohn(d_km / radius_km), (D.row, D.col)), shape=(n_cells, p))
        # (a pair at exactly zero distance is dropped by the sparse format; it is
        #  restored below because such a cell is necessarily inside that footprint)

        if footprint_full_weight:
            # a cell inside an observation's footprint always sees that observation at full weight
            T = T.maximum(self.footprint.astype(float))

        T.eliminate_zeros()
        self.taper = T.tocsr()
        self.n_cells, self.p, self.V = n_cells, p, V
        self.radius_km = radius_km

    def local(self, cell):
        """indices and taper values of the observations used for one cell."""
        row = self.taper.getrow(cell)
        return row.indices, row.data

    def summary(self):
        nnz_per_cell = np.diff(self.taper.indptr)
        return (f"LETKF localisation: {self.p} obs, {self.n_cells} cells, radius {self.radius_km:.0f} km, "
                f"obs per cell min/median/max = {nnz_per_cell.min()}/{int(np.median(nnz_per_cell))}/"
                f"{nnz_per_cell.max()}, cells with no obs = {(nnz_per_cell == 0).sum()}")


# --------------------------------------------------------------------------------------
# the filter
# --------------------------------------------------------------------------------------
class LETKF(EnKF):
    """
    Local Ensemble Transform Kalman Filter on the monthly-mean state.

    Parameters (keyword, all optional; can also be given under "method" in DA_setting.json)
    ----------
    loc_radius_km : float   Gaspari-Cohn localisation radius (taper reaches 0 at 2x). 800 km
                            is a sensible start for 3-degree GRACE cells (~330 km).
                            np.inf disables localisation -> global ETKF.
    inflation     : float   multiplicative prior inflation rho >= 1 (enters (k-1)I/rho). 1.0
    rtps          : float   relaxation-to-prior-spread alpha in [0, 1]. 0 disables.  0.0
    use_full_R    : bool    if False, only diag(R) is used (fully vectorised fast path).
                            If True and R has off-diagonal terms, the local block of R is
                            inverted per cell (correlated GRACE errors).            auto
    min_obs       : int     cells with fewer local obs than this are left unchanged.  1
    """

    def __init__(self, DA_setting: config_DA, model: model_run_daily, obs: GRACE_obs, sv: EnsStates,
                 sv_excluded: EnsStates, **kw):
        super().__init__(DA_setting, model, obs, sv, sv_excluded)

        method = getattr(DA_setting, 'method', None)
        cfg = getattr(method, '__dict__', {}) if method is not None else {}

        def opt(name, default):
            return kw.get(name, cfg.get(name, default))

        self.loc_radius_km = float(opt('loc_radius_km', 800.0))
        self._inflation = float(opt('inflation', 1.0))          # rho, overrides EnKF default 1.1
        self.rtps = float(opt('rtps', 0.0))
        self.use_full_R = opt('use_full_R', None)               # None -> decide from R
        self.min_obs = int(opt('min_obs', 1))
        self.batch_cells = int(opt('batch_cells', 4096))

        self._H = None
        self._loc = None
        self._warned_perturbed_obs = False

    # ------------------------------------------------------------------ configuration
    def configure_design_matrix(self, DM: DM_basin_average):
        """keep the matrix itself (footprints/centroids) in addition to the operator."""
        super().configure_design_matrix(DM)
        self._H = DM.getDM()
        cell_lat, cell_lon = global_cell_latlon(DM.local_mask['global_2d'])
        assert cell_lat.size * DM.vertical_dim == self._H.shape[1], \
            "design matrix width does not match n_cells * vertical_dim"
        self._loc = ObsLocalization(self._H, cell_lat, cell_lon, DM.vertical_dim,
                                    radius_km=self.loc_radius_km)
        print(self._loc.summary())
        return self

    def configure_localization(self, radius_km):
        self.loc_radius_km = float(radius_km)
        if self._H is not None:
            cell_lat, cell_lon = global_cell_latlon(self._sv.DM.local_mask['global_2d'])
            self._loc = ObsLocalization(self._H, cell_lat, cell_lon, self._sv.DM.vertical_dim,
                                        radius_km=self.loc_radius_km)
            print(self._loc.summary())
        return self

    def configure_rtps(self, alpha):
        self.rtps = float(alpha)
        return self

    # ------------------------------------------------------------------ hook for run_mpi
    def _obs_for_update(self, ens_obs_perturbed, obs_unperturbed):
        """LETKF is deterministic: use the unperturbed observation (gathered from the OL rank)."""
        return obs_unperturbed

    # ------------------------------------------------------------------ the analysis
    def update(self, obs, obs_cov, ens_states):
        """
        obs        : (p,) unperturbed observation vector   [(p, k) tolerated: column mean is used]
        obs_cov    : (p, p) observation error covariance R
        ens_states : (n, k) background ensemble of the monthly-mean state
        returns    : (n, k) analysis ensemble
        """
        assert self._loc is not None, "call configure_design_matrix() first"
        obs = np.asarray(obs, dtype=float)
        if obs.ndim == 2:
            if not self._warned_perturbed_obs:
                warnings.warn("LETKF received an ensemble of perturbed observations; using their mean. "
                              "Add the _obs_for_update hook to EnKF.run_mpi to pass the unperturbed vector.")
                self._warned_perturbed_obs = True
            obs = obs.mean(axis=1)
        R = np.atleast_2d(np.asarray(obs_cov, dtype=float))
        if R.shape == (1, 1) and obs.size > 1:
            R = np.eye(obs.size) * R[0, 0]

        X = np.asarray(ens_states, dtype=float)
        n, k = X.shape
        V = self._loc.V
        n_cells = self._loc.n_cells
        assert n == n_cells * V

        xb_mean = X.mean(axis=1)
        Xb = X - xb_mean[:, None]                                # n x k

        HX = self._DM(states=X)                                  # p x k
        yb_mean = HX.mean(axis=1)
        Yb = HX - yb_mean[:, None]                               # p x k
        d = obs - yb_mean                                        # innovation, p

        full_R = self.use_full_R
        if full_R is None:
            off = R - np.diag(np.diag(R))
            full_R = np.abs(off).max() > 1e-12 * max(np.abs(np.diag(R)).max(), 1e-300)

        Xa = X.copy()                                            # cells without obs keep background
        km1 = float(k - 1)

        if not full_R:
            self._analysis_diag_R(Xa, xb_mean, Xb, Yb, d, np.diag(R), km1, k, V, n_cells)
        else:
            self._analysis_full_R(Xa, xb_mean, Xb, Yb, d, R, km1, k, V, n_cells)

        if self.rtps > 0.0:
            sb = Xb.std(axis=1, ddof=1)
            Xa_mean = Xa.mean(axis=1)
            Pa = Xa - Xa_mean[:, None]
            sa = Pa.std(axis=1, ddof=1)
            scale = np.where(sa > 0, (1.0 - self.rtps) + self.rtps * sb / np.where(sa > 0, sa, 1.0), 1.0)
            Xa = Xa_mean[:, None] + Pa * scale[:, None]

        return Xa

    # ---- fast path: diagonal R, all cells of a batch solved with one stacked eigh
    def _analysis_diag_R(self, Xa, xb_mean, Xb, Yb, d, r_diag, km1, k, V, n_cells):
        T = self._loc.taper                                      # n_cells x p (sparse)
        n_obs_per_cell = np.diff(T.indptr)
        # localised R^-1 for every (cell, obs) pair: taper * 1/r  -> still sparse
        Tr = (T @ sparse.diags(1.0 / r_diag)).tocsr()
        # per-observation building blocks, computed once:
        #   M[p]  = Yb[p] Yb[p]^T flattened (p x k^2)   so that  C Yb = sum_p Rinv[c,p] M[p]
        #   Yd[p] = Yb[p] d[p]                (p x k)   so that  C d  = sum_p Rinv[c,p] Yd[p]
        M = (Yb[:, :, None] * Yb[:, None, :]).reshape(Yb.shape[0], k * k)
        Yd = Yb * d[:, None]
        for s in range(0, n_cells, self.batch_cells):
            e = min(s + self.batch_cells, n_cells)
            cells = np.arange(s, e)
            ok = n_obs_per_cell[cells] >= self.min_obs
            if not ok.any():
                continue
            cells = cells[ok]
            Trb = Tr[cells]                                      # b x p sparse
            CY = np.asarray(Trb @ M).reshape(cells.size, k, k)   # b x k x k
            Cd = np.asarray(Trb @ Yd)                            # b x k
            self._apply_weights(Xa, xb_mean, Xb, CY, Cd, cells, km1, k, V)

    # ---- general path: correlated R, local block inverted per cell
    def _analysis_full_R(self, Xa, xb_mean, Xb, Yb, d, R, km1, k, V, n_cells):
        T = self._loc.taper
        for c in range(n_cells):
            js, rho = self._loc.local(c)
            if js.size < self.min_obs:
                continue
            RL = R[np.ix_(js, js)]
            sq = np.sqrt(rho)
            # Rl^-1 = D^1/2 R_L^-1 D^1/2
            RLinv = np.linalg.pinv(RL) if js.size > 1 else 1.0 / RL
            RLinv = (sq[:, None] * RLinv * sq[None, :])
            YL = Yb[js]                                          # p_loc x k
            C = YL.T @ RLinv                                     # k x p_loc
            CY = (C @ YL)[None]                                  # 1 x k x k
            Cd = (C @ d[js])[None]                               # 1 x k
            self._apply_weights(Xa, xb_mean, Xb, CY, Cd, np.array([c]), km1, k, V)

    # ---- ensemble-space solve for a batch of cells and write the analysis rows
    def _apply_weights(self, Xa, xb_mean, Xb, CY, Cd, cells, km1, k, V):
        b = cells.size
        A = CY + (km1 / self._inflation) * np.eye(k)[None]       # b x k x k, SPD
        lam, Q = np.linalg.eigh(A)                               # stacked eigendecomposition
        lam = np.maximum(lam, 1e-12)
        # Pa~ = Q diag(1/lam) Q^T ;  W = Q diag(sqrt(km1/lam)) Q^T  (symmetric square root)
        Pa = np.einsum('bik,bk,bjk->bij', Q, 1.0 / lam, Q)
        W = np.einsum('bik,bk,bjk->bij', Q, np.sqrt(km1 / lam), Q)
        w_mean = np.einsum('bij,bj->bi', Pa, Cd)                 # b x k
        Wfull = W + w_mean[:, :, None]                           # b x k x k : column i = weights of member i

        # rows of these cells: cell*V + v
        rows = (cells[:, None] * V + np.arange(V)[None, :])      # b x V
        Xb_rows = Xb[rows]                                       # b x V x k
        Xa[rows] = xb_mean[rows][:, :, None] + np.einsum('bvk,bkl->bvl', Xb_rows, Wfull)


# --------------------------------------------------------------------------------------
# self-test (no MPI, no model): LETKF with infinite radius == analytic Kalman update
# --------------------------------------------------------------------------------------
def _selftest():
    rng = np.random.default_rng(0)
    n_cells, V, k, p = 60, 3, 20, 4
    n = n_cells * V
    # a fake design matrix: p "basins", each averaging a contiguous block of cells, repeated over V
    H = np.zeros((p, n))
    for j in range(p):
        cells = np.arange(j * (n_cells // p), (j + 1) * (n_cells // p))
        for c in cells:
            H[j, c * V:(c + 1) * V] = 1.0 / cells.size
    lat = np.linspace(40, 50, n_cells); lon = np.linspace(5, 30, n_cells)

    X = rng.normal(size=(n, k)) * 10 + 100
    R = np.diag(rng.uniform(1, 4, size=p))
    truth = X.mean(axis=1) + rng.normal(size=n) * 5
    yo = H @ truth + rng.multivariate_normal(np.zeros(p), R)

    class _Stub(LETKF):                                            # bypass model/obs plumbing
        def __init__(self, radius):
            self.loc_radius_km = radius; self._inflation = 1.0; self.rtps = 0.0
            self.use_full_R = None; self.min_obs = 1; self.batch_cells = 16
            self._warned_perturbed_obs = False
            self._DM = lambda states: H @ states
            self._H = H
            self._loc = ObsLocalization(H, lat, lon, V, radius_km=radius)

    # 1) no localisation: analysis mean must equal the ensemble-covariance Kalman mean,
    #    analysis covariance must equal (I-KH) Pb in the ensemble subspace
    f = _Stub(np.inf)
    Xa = f.update(yo, R, X)
    Xb = X - X.mean(1, keepdims=True); Pb = Xb @ Xb.T / (k - 1)
    K = Pb @ H.T @ np.linalg.inv(H @ Pb @ H.T + R)
    xa_kf = X.mean(1) + K @ (yo - H @ X.mean(1))
    Pa_kf = (np.eye(n) - K @ H) @ Pb
    Pa_ens = (Xa - Xa.mean(1, keepdims=True)) @ (Xa - Xa.mean(1, keepdims=True)).T / (k - 1)
    e_mean = np.abs(Xa.mean(1) - xa_kf).max() / np.abs(xa_kf).max()
    e_cov = np.abs(Pa_ens - Pa_kf).max() / np.abs(Pa_kf).max()
    print(f"[selftest] global ETKF vs KF:   mean rel.err {e_mean:.2e}   cov rel.err {e_cov:.2e}")
    assert e_mean < 1e-10 and e_cov < 1e-8

    # 2) full-R path must agree with the diagonal path when R is diagonal
    f.use_full_R = True
    Xa2 = f.update(yo, R, X)
    print(f"[selftest] full-R path vs diag path: max abs diff {np.abs(Xa2 - Xa).max():.2e}")
    assert np.abs(Xa2 - Xa).max() < 1e-8

    # 3) localisation: with a small radius only nearby observations act on a cell
    g = _Stub(150.0)
    print("[selftest]", g.summary() if hasattr(g, 'summary') else g._loc.summary())
    Xa3 = g.update(yo, R, X)
    assert np.isfinite(Xa3).all()
    print("[selftest] OK")


if __name__ == '__main__':
    _selftest()
