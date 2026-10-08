import numpy as np
from datetime import datetime
from pathlib import Path
import h5py
from src_OBS.obs_auxiliary import obs_auxiliary


class GRACE_obs:

    def __init__(self, basin='MDB', dir_obs='/home/user/test/obs/', ens_id=0):
        self.__time_list = None

        '''load the pre-stored observations as well as the covariance matrix'''
        obs_fn = Path(dir_obs) / ('%s_obs_GRACE.hdf5' % basin)
        obs_h5 = h5py.File(obs_fn, 'r')

        self.__time_list = list(obs_h5['time_epoch'][:].astype('str'))

        if 'duration' in obs_h5.keys():
            self.__duration_list = list(obs_h5['duration'][:].astype('str'))

        self.__obs = obs_h5['ens_%s' % ens_id][:]

        '''the covariance of a month is read from the file when asked for (get_cov): the whole history
        (n_month x n_sub x n_sub, 1 GB for 772 sub-basins and 216 months) was loaded on every MPI rank until
        8 Oct 2026, although only the main rank uses one month at a time'''
        self.__obs_fn = obs_fn
        self.__cov_scalar = np.ndim(obs_h5['cov']) < 3                  # one sub-basin: cov stored as [n_month]

        obs_h5.close()
        pass

    def get_time_list(self):
        return self.__time_list.copy()

    def set_date(self, date='2002-01-01'):
        if date in self.__time_list:
            self.__date_index = self.__time_list.index(date)
        else:
            self.__date_index = None

        return self

    def set_month(self, month='2002-01'):

        for date in self.__time_list:
            if len(date.split('-')) == 2:
                thismonth = date
            else:
                thismonth = datetime.strptime(date, '%Y-%m-%d').strftime('%Y-%m')

            if thismonth == month:
                self.__date_index = self.__time_list.index(date)
                break
            else:
                self.__date_index = None

        return self

    def get_obs(self):

        if self.__date_index is None:
            return None
        else:
            return self.__obs[self.__date_index]

    def get_cov(self):
        """covariance matrix [n_sub x n_sub] of the current month (set_date / set_month), read from the file"""
        if self.__date_index is None:
            return None
        with h5py.File(self.__obs_fn, 'r') as f:
            cov = f['cov'][self.__date_index]
        if self.__cov_scalar:
            cov = np.asarray(cov)[None, None]
        return cov

    def get_obs_aux(self):
        obs_aux = {
            'time_epoch': self.__time_list.copy(),
            'duration': self.__duration_list.copy()
        }
        return obs_aux

def demo1():
    gr = GRACE_obs()
    hh = gr.get_time_list()
    pass


if __name__ == '__main__':
    demo1()
