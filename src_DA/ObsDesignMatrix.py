from src_DA.EnumDA import WaterGap_storage_variables
from src_auxiliary.shp2mask import load_mask
import numpy as np
from scipy.linalg import block_diag
import h5py
from pathlib import Path
from datetime import datetime
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

        # self.layer = layer
        states_nn = []
        vertical_dim = 0
        for k, v in layer.items():
            if (v and not is_residual) or (not v and is_residual):
                states_nn.append(k)
                vertical_dim += 1

        self.statesnn = sorted(states_nn, key=lambda x: WaterGap_storage_variables[x].value)
        self.vertical_dim = vertical_dim

        '''load DM from pre-saved matrix'''
        if 'LoadfromDisk' in kwargs:
            if kwargs['LoadfromDisk'] is True:
                self._A = h5py.File(Path(kwargs['dir']) / 'DM.hdf5', 'r')['data'][:]
                return

    def vertical_aggregation(self):

        self._A = np.ones((1,self.vertical_dim))

        """this is indeed not right, however, it is right when combined with the horizontal aggregation """

        return self

    def horizontal_aggregation(self):

        '''to decide the amount of grid cells to be involved in the computation'''
        sub_basin_num = self.local_mask['basin_num']

        B = []

        lat_cos = np.cos(np.deg2rad(self.local_mask['lat']))

        for i in range(1, sub_basin_num + 1):
            '''area computation for each cell'''
            bm = self.local_mask['sub_basin_%s' % i]
            lat_basin = lat_cos[bm.astype(bool)]
            A = np.sum(lat_basin)

            '''basin-average for each basin'''
            c= np.zeros(np.shape(lat_cos))
            c[bm.astype(bool)] = lat_basin
            x = bm * c /A

            B.append(x)

            pass

        B = np.array(B)

        '''re-organize the matrix to speed up the computation'''
        N = self.vertical_dim
        y0 = np.repeat(B, N, axis=1)
        self._A = y0
        return self

    def configure_mask(self, mask_path:str):

        self.box_crop, self.local_mask  = load_mask(mask_path=mask_path)

        return self


    def upscaling(self):
        return self

    def filtering(self):
        return self

    def getDM(self):
        return self._A.copy()

    def operator(self, states):

        return self.getDM() @ states

    def operator_reduce_running_memory(self, states):
        """
        it is found that for a large study region the design matrix can be huge. Holding this design matrix for each
        ensemble member can produce the error of running out memory. Therefore, this operator is developed to avoid
        the occupation of memory by this design matrix.

        Not tested yet!!
        """
        self._A = h5py.File(Path('../temp') / 'DM.hdf5', 'r')['data'][:]
        B = self.getDM()
        self._A = None  # release the memory
        return B @ states

    def saveDM(self, out_path: str):

        fn = h5py.File(Path(out_path) / 'DM.hdf5', 'w')
        fn.create_dataset('data', data=self._A)
        fn.close()
        print('Finished and save the design matrix: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        pass
def demo2():
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
    dm = DM_basin_average(layer=layer, is_residual=False)
    dm.configure_mask(mask_path='/media/user/My Book/Fan/WaterGap/Basin/mask/Brahmaputra/Brahmaputra_res_0.5.h5')
    dm.vertical_aggregation().horizontal_aggregation()

    dm2 = DM_basin_average(layer=layer, is_residual=True)
    dm2.configure_mask(mask_path='/media/user/My Book/Fan/WaterGap/Basin/mask/Brahmaputra/Brahmaputra_res_0.5.h5')
    dm2.vertical_aggregation().horizontal_aggregation()
    pass


if __name__ == '__main__':
    # demo1()
    demo2()
