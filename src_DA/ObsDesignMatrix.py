from src_DA.EnumDA import WaterGap_storage_variables
from src_auxiliary.shp2mask import load_mask
import numpy as np
from scipy import sparse
from scipy.linalg import block_diag
import h5py
from pathlib import Path
import xarray as xr


class DM_basin_average:

    """
    The states must be flattened to one-dimension vector, which aligns with the kalman filter.
    """

    def __init__(self, layer: dict = None, is_residual=False, **kwargs):
        """
        when is_residual is set to True, the design matrix is obtained for those that don't participate in DA.
        Parameters
        ----------
        layer
        is_residual
        kwargs
        """
        self.local_mask = None
        self.box_crop = None
        self._A = None
        self.is_residual = is_residual

        '''to decide the dimension of vertical layers to be involved in the computation'''
        if layer is None:
            layer = {
                WaterGap_storage_variables.groundwstor.name: True,
                WaterGap_storage_variables.soilmoist.name: True,
                WaterGap_storage_variables.swe.name: True,
                WaterGap_storage_variables.locallakestor.name: True,
                WaterGap_storage_variables.localwetlandstor.name: True,
                WaterGap_storage_variables.globallakestor.name: True,
                WaterGap_storage_variables.globalwetlandstor.name: True,
                WaterGap_storage_variables.riverstor.name: True,
                WaterGap_storage_variables.reservoirstor.name: True,
                WaterGap_storage_variables.canopystor.name: True
            }

        states_nn = []
        vertical_dim = 0
        for k, v in layer.items():
            if (v and not is_residual) or (not v and is_residual):
                states_nn.append(k)
                vertical_dim += 1

        self.statesnn = sorted(states_nn, key=lambda x: WaterGap_storage_variables[x].value)
        self.vertical_dim = vertical_dim

        '''load DM from a pre-saved matrix: <dir>/DM.hdf5 with the CSR components data / indices / indptr / shape'''
        if 'LoadfromDisk' in kwargs:
            if kwargs['LoadfromDisk'] is True:
                with h5py.File(Path(kwargs['dir']) / 'DM.hdf5', 'r') as f:
                    if 'indptr' in f:
                        self._A = sparse.csr_matrix((f['data'][:], f['indices'][:], f['indptr'][:]),
                                                    shape=tuple(f['shape'][:]))
                    else:
                        self._A = sparse.csr_matrix(f['data'][:])            # file holding a dense matrix
                return

    def vertical_aggregation(self):

        self._A = np.ones((1,self.vertical_dim))

        """this is indeed not right, however, it is right when combined with the horizontal aggregation """

        return self

    def horizontal_aggregation(self):
        """
        design matrix H [n_sub_basin x n_cell * vertical_dim] of the sub-basin means: row i holds, for every cell of
        sub-basin i and every vertical layer, the cos(lat) weight of the cell normalised over the sub-basin (the
        state vector is cell-major: element c * vertical_dim + k is layer k of cell c, see ExtractStates). H has one
        non-zero per state element of the sub-basins and is stored as scipy.sparse CSR: a dense matrix of the
        global case (772 x 166 383 for 3 layers, 772 x 388 227 for the 7 excluded layers) would take 1 + 2.4 GB on
        every MPI rank; the sparse one is a few MB.
        """
        sub_basin_num = self.local_mask['basin_num']
        N = self.vertical_dim
        n_cell = len(self.local_mask['lat'])
        lat_cos = np.cos(np.deg2rad(self.local_mask['lat']))

        rows, cols, vals = [], [], []
        for i in range(1, sub_basin_num + 1):
            '''cells of the sub-basin and their normalised area weights'''
            idx = np.flatnonzero(self.local_mask['sub_basin_%s' % i])
            w = lat_cos[idx] / np.sum(lat_cos[idx])
            '''the same weight for every vertical layer of the cell'''
            rows.append(np.full(idx.size * N, i - 1))
            cols.append((idx[:, None] * N + np.arange(N)[None, :]).ravel())
            vals.append(np.repeat(w, N))
        rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
        self._A = sparse.csr_matrix((vals, (rows, cols)), shape=(sub_basin_num, n_cell * N))
        return self

    def configure_mask(self, mask_path:str):

        self.box_crop, self.local_mask  = load_mask(mask_path=mask_path)

        return self

    def getDM(self):
        """the design matrix as scipy.sparse CSR (no copy; use .toarray() for a dense array)"""
        return self._A

    def operator(self, states):
        """H @ states: sub-basin means of the states [n_state] or [n_state x N]"""
        return self._A @ states
