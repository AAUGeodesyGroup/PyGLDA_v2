
import numpy as np
import h5py
from src_auxiliary.GeoMathKit import GeoMathKit
import os
from pathlib import Path
from datetime import datetime
from src_OBS.obs_auxiliary import obs_auxiliary


class GRACE_perturbed_obs:
    SEED = 20021            # seed of the observation perturbations: a rerun of DA_run gives the same observation ensemble

    def __init__(self, ens=30, basin_name='MDB'):
        self.ens = ens
        self.basin_name = basin_name
        self.TWS = None
        self.obs_aux = None
        self._rng = np.random.default_rng(self.SEED)
        pass

    def configure_dir(self, input_dir=None, obs_dir=None):
        """input_dir: pre-processed GRACE (<basin>_signal.hdf5, <basin>_cov.hdf5); obs_dir: output of save()"""
        if input_dir is None or obs_dir is None:
            raise ValueError('GRACE_perturbed_obs.configure_dir: input_dir and obs_dir must be given')
        self._input_dir = Path(input_dir)
        self._obs_dir = Path(obs_dir)
        return self

    def configure_obs_aux(self, obs_aux: obs_auxiliary):
        '''time reference (epochs and windows) of the observations'''
        self.obs_aux = obs_aux
        return self

    def perturb_TWS(self):

        in_dir = self._input_dir

        '''signal'''
        signal_fn = in_dir / ('%s_signal.hdf5' % self.basin_name)
        S_h5fn = h5py.File(signal_fn, 'r')

        '''cov'''
        cov_fn = in_dir / ('%s_cov.hdf5' % self.basin_name)
        C_h5fn = h5py.File(cov_fn, 'r')

        basin_num = np.shape(S_h5fn['sub_basin_area'][:])[0]

        self.TWS = {}

        time_epochs_1 = list(S_h5fn['time_epoch'][:].astype('str'))
        time_epochs_2 = list(C_h5fn['time_epoch'][:].astype('str'))

        TWS = []
        un_TWS = []
        COV = []
        new_time = []
        time_duration = []
        print('')
        print('Start to perturb GRACE to obtain appropriate observations...')

        Time_list = self.obs_aux.getTimeReference()['time_epoch']
        Time_duration_list = self.obs_aux.getTimeReference()['duration']

        '''get covariance matrix: either static (2-D) or one per epoch (3-D)'''
        if len(np.shape(C_h5fn['data'])) == 2:
            '''this indicates a static cov'''
            cov = C_h5fn['data']
            isStaticCov= True
        else:
            isStaticCov = False

        for count, time in enumerate(Time_list):
            '''check if this time exists'''
            if time not in time_epochs_1:
                continue
            '''confirm data consistency'''
            index = time_epochs_1.index(time)
            assert time_epochs_1[index] == time_epochs_2[index] == time

            '''obtain the TWS of each basin'''
            a = []
            for id in range(1, basin_num + 1):
                a.append(S_h5fn['sub_basin_%d' % id][index])
            a = np.array(a)

            '''obtain the COV'''
            if isStaticCov:
                mcov = cov
            else:
                mcov = C_h5fn['data'][index]

            '''perturb the signal'''
            if basin_num == 1:
                '''in case there is only one subbasin'''
                perturbed_TWS = self._rng.normal(a, np.sqrt(mcov), self.ens)[:, None]
            else:
                perturbed_TWS = self._rng.multivariate_normal(a, mcov, self.ens)

            new_time.append(time)
            COV.append(mcov)
            un_TWS.append(a)
            TWS.append(perturbed_TWS)
            time_duration.append(Time_duration_list[count])
            pass

        self.TWS['time'] = new_time
        self.TWS['cov'] = COV
        self.TWS['unperturbation'] = np.array(un_TWS)
        self.TWS['ens'] = np.array(TWS)
        self.TWS['duration'] = time_duration

        print('Finished: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        return self

    def remove_temporal_mean(self):

        meanTWs = np.mean(self.TWS['unperturbation'], 0)

        self.TWS['ens'] -= meanTWs
        self.TWS['unperturbation'] -= meanTWs
        return self

    def add_temporal_mean(self, fn: str):

        hh = h5py.File(Path(fn), 'r')
        mean = hh['mean_ensemble'][:]

        self.TWS['ens'] += mean
        self.TWS['unperturbation'] += mean
        return self

    def save(self):
        """
        Save the data
        """

        fn = self._obs_dir / ('%s_obs_GRACE.hdf5' % self.basin_name)

        obs = h5py.File(fn, 'w')

        obs.create_dataset(name='cov', data=self.TWS['cov'])

        dt = h5py.special_dtype(vlen=str)
        obs.create_dataset('time_epoch', data=self.TWS['time'], dtype=dt)

        if 'duration' in self.TWS.keys():
            obs.create_dataset('duration', data=self.TWS['duration'], dtype=dt)

        '''record sub-basin area information'''
        signal_fn = self._input_dir / ('%s_signal.hdf5' % self.basin_name)
        S_h5fn = h5py.File(signal_fn, 'r')
        obs.create_dataset('sub_basin_area', data=S_h5fn['sub_basin_area'][:])
        S_h5fn.close()

        for i in np.arange(self.ens + 1):
            if i == 0:
                obs.create_dataset(name='ens_%s' % i, data=self.TWS['unperturbation'])
            else:
                obs.create_dataset(name='ens_%s' % i, data=self.TWS['ens'][:, i - 1, :])
                pass

        obs.close()
        pass

