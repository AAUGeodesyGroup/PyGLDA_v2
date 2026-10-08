"""
Localization of the Kalman gain for basin-averaged observations: an independent category, combined with any filter
variant of src_DA.EnKF_localized / src_DA.EnKF_variants.

With a small ensemble the sample cross-covariance between the state of one sub-basin and the observation of another
is dominated by sampling noise (Amazon, 4 members: the estuary innovation removed river storage along the whole
mainstem). A localization returns the tapers
        L_state (n_state x n_obs)   applied element-wise to Pxy = A HA^T / (N-1)
        L_obs   (n_obs x n_obs)     applied element-wise to Pyy = HA HA^T / (N-1)
so that K = (L_state o Pxy) (L_obs o Pyy + R)^-1.

DA_setting.json, "method" block:
    "localization": {"kind": "none"}                                      no localization (L = 1)
    "localization": {"kind": "block"}                                     each observation updates its own sub-basin
    "localization": {"kind": "gaussian", "length_km": 300, "cutoff": 2.0} distance taper exp(-0.5 (d/length)^2),
                                                                           0 beyond cutoff * length
"""
import numpy as np
from scipy import sparse


def sub_basin_membership(local_mask):
    """n_cell x n_sub sparse (CSR) membership, 1 where a basin cell belongs to sub-basin k (state-vector cell order)"""
    nsub = local_mask['basin_num']
    rows, cols = [], []
    for k in range(1, nsub + 1):
        idx = np.flatnonzero(local_mask['sub_basin_%d' % k])
        rows.append(idx)
        cols.append(np.full(idx.size, k - 1))
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    return sparse.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(len(local_mask['lat']), nsub))


def owner_of_cells(M):
    """sub-basin index (0-based) of every cell from the membership M, -1 for a cell in no sub-basin"""
    owner = np.full(M.shape[0], -1)
    r, c = M.nonzero()
    owner[r] = c
    return owner


class Localization:
    KIND = None

    def __init__(self, **params):
        self.params = params

    def build(self, local_mask, nvar):
        """
        returns L_state (n_state x n_obs, cell-major: nvar rows per cell) as a scipy.sparse CSR matrix holding only
        the non-zero taper weights, and L_obs (n_obs x n_obs, dense). The dense L_state of the global case
        (166 383 x 772, 1 GB, 99.9 % zeros for block localization) was held on every MPI rank until 8 Oct 2026;
        the filter, the partition and the inflation evaluate their products only on the pattern of L_state.
        """
        M = sub_basin_membership(local_mask)
        L_cell, L_obs = self._tapers(M, local_mask)
        L_cell = sparse.csr_matrix(L_cell)
        L_cell.eliminate_zeros()
        if L_cell.nnz > 1e8:
            print('%s: the localization taper has %.2g non-zeros (%d cells x %d observations): this is a dense '
                  'problem and needs %.1f GB' % (self.__class__.__name__, L_cell.nnz, *L_cell.shape,
                                                  L_cell.nnz * nvar * 8 / 1e9))
        L_state = sparse.kron(L_cell, np.ones((nvar, 1)), format='csr')        # = np.repeat(L_cell, nvar, axis=0)
        L_state.sort_indices()
        return L_state, np.asarray(L_obs, dtype=float)

    def _tapers(self, M, local_mask):
        raise NotImplementedError

    def describe(self):
        return dict(kind=self.KIND, **self.params)


class NoLocalization(Localization):
    """every observation updates every cell (the plain sample covariances are used)"""
    KIND = 'none'

    def _tapers(self, M, local_mask):
        return np.ones(M.shape), np.ones((M.shape[1], M.shape[1]))


class BlockLocalization(Localization):
    """each observation updates only the cells of its own sub-basin; observations are treated as independent in Pyy"""
    KIND = 'block'

    def _tapers(self, M, local_mask):
        orphan = M.getnnz(axis=1) == 0
        if orphan.any():
            print('BlockLocalization: %d basin cells belong to no sub-basin and are never updated' % orphan.sum())
        return M.copy(), np.eye(M.shape[1])


class GaussianLocalization(Localization):
    """distance taper: L = exp(-0.5 (d/length)^2), d = distance [km] from a cell to the nearest cell of the observed
    sub-basin (0 inside), 0 beyond cutoff*length; Pyy tapered with the mean cell distance between two sub-basins"""
    KIND = 'gaussian'

    def __init__(self, length_km=300.0, cutoff=2.0):
        super().__init__(length_km=float(length_km), cutoff=float(cutoff))

    @staticmethod
    def cell_coordinates(local_mask, res=0.5):
        """latitude / longitude [deg] of the basin cells in the order of the state vector"""
        lat_c = np.arange(90 - res / 2, -90, -res)
        lon_c = np.arange(-180 + res / 2, 180, res)
        lon_mesh, lat_mesh = np.meshgrid(lon_c, lat_c)
        g = local_mask['global_2d'].astype(bool)
        return lat_mesh[g], lon_mesh[g]

    @staticmethod
    def haversine_matrix(lat, lon):
        """pairwise great-circle distances [km] between cells (n x n); only for small regions (n^2 memory)"""
        R = 6371.0
        la, lo = np.deg2rad(lat)[:, None], np.deg2rad(lon)[:, None]
        dlat, dlon = la - la.T, lo - lo.T
        a = np.sin(dlat / 2) ** 2 + np.cos(la) * np.cos(la.T) * np.sin(dlon / 2) ** 2
        return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    def _tapers(self, M, local_mask):
        """
        distance of every cell to the nearest cell of every sub-basin with one KD-tree per sub-basin (the full
        cell-to-cell matrix was n_cell^2: 25 GB for the global case); L_cell keeps only the pairs within the cutoff
        """
        from scipy.spatial import cKDTree
        R = 6371.0
        length, cutoff = self.params['length_km'], self.params['cutoff']
        nsub = M.shape[1]
        lat, lon = self.cell_coordinates(local_mask)
        la, lo = np.deg2rad(lat), np.deg2rad(lon)
        xyz = np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
        owner = owner_of_cells(M)
        cols, rows, vals = [], [], []
        d_sub_sub = np.zeros((nsub, nsub))
        for k in range(nsub):
            cells_k = np.flatnonzero(np.asarray(M[:, k].todense()).ravel() > 0)
            chord, _ = cKDTree(xyz[cells_k]).query(xyz)                                   # nearest cell of k
            d = 2 * R * np.arcsin(np.clip(chord / 2, 0, 1))                               # great-circle [km]
            d[cells_k] = 0.0
            '''sub-basin to sub-basin distance: mean distance of the cells of j to sub-basin k'''
            inside = owner >= 0
            d_sub_sub[:, k] = np.bincount(owner[inside], weights=d[inside], minlength=nsub) / \
                np.maximum(np.bincount(owner[inside], minlength=nsub), 1)
            keep = np.flatnonzero(d <= cutoff * length)
            rows.append(keep)
            cols.append(np.full(keep.size, k))
            vals.append(np.exp(-0.5 * (d[keep] / length) ** 2))
        L_cell = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                   shape=(M.shape[0], nsub))
        # (symmetrised; the nearest-cell distance would be ~one cell for every adjacent pair and taper nothing)
        d_sub_sub = 0.5 * (d_sub_sub + d_sub_sub.T)
        L_obs = np.exp(-0.5 * (d_sub_sub / length) ** 2)
        L_obs[d_sub_sub > cutoff * length] = 0.0
        return L_cell, L_obs


LOCALIZATIONS = {c.KIND: c for c in (NoLocalization, BlockLocalization, GaussianLocalization)}


def make_localization(cfg=None):
    """Localization object from the "localization" block (default: block)"""
    cfg = dict(cfg or {'kind': 'block'})
    kind = str(cfg.pop('kind', 'block')).lower()
    if kind not in LOCALIZATIONS:
        raise ValueError('localization kind must be one of %s, got %s' % (list(LOCALIZATIONS), kind))
    if kind == 'gaussian':
        return GaussianLocalization(**{k: cfg[k] for k in ('length_km', 'cutoff') if k in cfg})
    return LOCALIZATIONS[kind]()          # length_km / cutoff are ignored by none and block
