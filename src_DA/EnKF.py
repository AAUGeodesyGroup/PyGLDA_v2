# from src_GHM.hotrun import model_run_daily
from src_GHM.Interface.DailyStepRun import DailyModelRun as model_run_daily
from src_DA.configure_DA import config_DA
from src_auxiliary.GeoMathKit import GeoMathKit
from src_DA.observations import GRACE_obs
from src_DA.ExtractStates import EnsStates
from src_DA.ObsDesignMatrix import DM_basin_average
import numpy as np
from mpi4py import MPI
from datetime import datetime

# from src_DA.CovRegularization import CovRegu, DomainLocalization

from src_DA.Threshold import model_state_threshold
from src_DA.EnumDA import WaterGap_storage_variables


class EnKF:
    """
    Monthly mean update: each day has the same update;
    This is a version to allow one to set observation at any time interval.
    """

    def __init__(self, DA_setting: config_DA, model: model_run_daily, obs: GRACE_obs, sv: EnsStates,
                 sv_excluded: EnsStates):

        self._states_predict = None

        self._DM = None
        self._model = model
        self.DA_setting = DA_setting
        self._obs = obs
        self._sv = sv
        self._sv_excluded = sv_excluded
        self._today = '2000-01-01'
        self.threshold = model_state_threshold(configDA=DA_setting).threshold

        '''obtain info'''
        self._obs_helper = self.helper_resolve_time(obs.get_obs_aux())

        '''inflation factor'''
        self._inflation = 1.1

        '''model re-configuration'''
        self._model.unit_converter= self._model.unit_converter.from_basin_mask(local_mask=self._sv.DM.local_mask,
                                                   cell_area_global=self._model.unit_converter.cell_area_global,
                                                   contfrac_global=self._model.unit_converter.contfrac_global)

        pass

    def _replace_state(self, original_state_a: dict, updated_internal_c: dict) -> None:
        """
        Safely write the values from processed internal state C back into the original state A.
        Supports partial updates, ensuring that variables in A not present in C are preserved.

        Parameters:
        -----------
        original_state_a : dict
            The original model state dictionary A (internal names). Updated in-place.
        updated_internal_c : dict
            The state dictionary C containing updated/assimilated values mapped back to internal names.
        """
        for internal_key, updated_array in updated_internal_c.items():
            if internal_key in original_state_a:
                # In-place assignment if the array supports it to maintain object reference integrity,
                # otherwise fall back to direct key replacement.
                if hasattr(original_state_a[internal_key], '__setitem__'):
                    original_state_a[internal_key][:] = updated_array
                else:
                    original_state_a[internal_key] = updated_array
            else:
                print(f"Warning: Key '{internal_key}' from C is not found in original state A.")

        pass

    def configure_design_matrix(self, DM: DM_basin_average):
        """
        design matrix for the observation equations. It should be constant over time.
        Here we pass the function (operator) instead of design matrix to gain flexibility in the future
        """
        self._DM = DM.operator
        return self

    def configure_inflation_factor(self, inflation=1.0):
        self._inflation = inflation
        return self

    def predict(self, states, today='2002-01-01', is_first_day=False, issave=True):
        """
        prediction is for one ensemble
        """
        today_standard = np.datetime64(today, 'D')
        self._states_predict = self._model.update(is_first_day=is_first_day, previous_states=states, day=today_standard,
                                                  issave=issave)
        self._today = today

        return self

    def update(self, obs, obs_cov, ens_states):
        """
        update is for all ensembles
        reference: WIKI
        """
        R = obs_cov
        # R = np.diag(np.diag(R)) #todo test

        '''calculate the deviation of ens_states'''
        A = ens_states - np.mean(ens_states, 1)[:, None]

        '''Inflation to increase the model perturbation'''
        A = A * self._inflation
        ens_states_inf = np.mean(ens_states, 1)[:, None] + A

        '''propagate it into obs-equivalent variable'''
        HX = self._DM(states=ens_states_inf)

        '''to calculate the deviation'''
        HA = HX - np.mean(HX, 1)[:, None]

        '''calculate matrix P'''
        P = np.cov(HA) + R

        '''calculate the gain factor K'''
        N = self.DA_setting.basic.ensemble
        '''method-1: straight-forward'''
        # K = 1 / (N - 1) * A @ HA.T @ np.linalg.inv(P)
        '''method-2: via linear solver'''
        Bt = np.linalg.lstsq(P.T, HA, rcond=None)[0]
        K = 1 / (N - 1) * A @ Bt.T

        '''update the states'''
        states_update = ens_states_inf + K @ (obs - HX)
        print(obs-HX)

        return states_update

    def run_mpi(self):
        """
        parallelized running with MPI4py
        """
        from datetime import datetime, timedelta

        '''main process'''
        main_thread = 1

        '''OL process: no perturbation'''
        OL_thread = 0

        '''preparation'''
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        '''make sure that each ensemble is given a process'''
        assert size == self.DA_setting.basic.ensemble + 1

        '''configure setting'''
        daylist = GeoMathKit.dayListByDay(begin=self.DA_setting.basic.fromdate,
                                          end=self.DA_setting.basic.todate)

        '''assign job to each ensemble'''
        firstday = True

        historic_states = []
        historic_mean_states = None
        obs = None
        obs_cov = None
        rr = -1
        previous_month = -1

        '''this is to collect the states that will not participate in the data assimilation'''
        historic_states_excluded = []
        historic_mean_states_excluded = None

        print('=================== Data assimilation =======================')
        for count, day in enumerate(daylist):
            "print information"
            if day.month != previous_month:
                previous_month = day.month
                print('\nDoing year/month: %04d/%02d' % (day.year, day.month))
            today = day.strftime('%Y-%m-%d')
            print('.', end='')
            '''==================kalman filter: prediction step=================================='''
            self.predict(today=today, states=self._states_predict, is_first_day=firstday, issave=True)
            local_predict_of_today = self._sv.load_state_dict(date=today)  # load state from hard disk

            local_predict_of_today_excluded = self._sv_excluded.load_state_dict(date=today)  # load state from hard disk

            firstday = False

            newRecord = self._obs_helper['NewRecord'][count]
            keepRecord = self._obs_helper['KeepRecord'][count]
            assimilationRecord = self._obs_helper['AssimilationRecord'][count]
            # print(newRecord, keepRecord, assimilationRecord)
            if newRecord:
                """create a new vector to prepare for assimilation of next time"""
                historic_mean_states = None
                historic_mean_states_excluded = None

            if keepRecord:
                '''record the previous states over areas of interest'''
                sv = self._sv.get_states_by_transfer_single(states=local_predict_of_today)
                sv_excluded = self._sv_excluded.get_states_by_transfer_single(states=local_predict_of_today_excluded)
                if historic_mean_states is None:
                    historic_mean_states = sv
                    historic_mean_states_excluded = sv_excluded
                else:
                    historic_mean_states += sv
                    historic_mean_states_excluded += sv_excluded
            else:
                historic_mean_states = None
                historic_mean_states_excluded = None

            if not assimilationRecord:
                continue

            '''====================start assimilation================================'''
            # print('====> GRACE data has been assimilated.')
            print('|', end='')
            rr += 1
            '''get obs and cov'''
            info = self._obs_helper['AssimilationInfo'][rr]
            self._obs.set_date(date=info[0])
            obs = self._obs.get_obs()

            '''if this is subspace data assimilation, the specified water compartments must be removed beforehand '''
            if self._sv_excluded.DM.vertical_dim == 0:
                '''nothing happens. All will participate in the data assimilation'''
                pass
            else:
                historic_mean_states_excluded = historic_mean_states_excluded / len(info[1])
                obs = obs - self._sv_excluded.DM.getDM()@historic_mean_states_excluded
                historic_mean_states_excluded = None # free memory

            obs_cov = self._obs.get_cov()

            '''EnKF'''
            historic_mean_states = historic_mean_states / len(info[1])

            '''synchronization to prepare for the data assimilation'''
            this_day = comm.gather(today, root=main_thread)
            if rank == main_thread:
                for iday in this_day:
                    '''confirm the synchronization again'''
                    assert iday == today

            '''collect obs from each ensemble'''
            ens_obs = comm.gather(root=main_thread, sendobj=obs)

            '''collect states from each ensemble'''
            ens_states = comm.gather(root=main_thread, sendobj=historic_mean_states)

            '''free memory'''
            historic_mean_states = None
            sv = None

            if rank != main_thread:
                delta_state = None
                pass
            else:
                '''delete the OL process: it does not participate in data assimilation'''
                del ens_obs[OL_thread]
                del ens_states[OL_thread]

                '''load ensemble states'''
                ens_obs = np.array(ens_obs).T
                ens_states = np.array(ens_states).T

                '''kalman filter: update step'''
                states_update = self.update(obs=ens_obs, obs_cov=obs_cov, ens_states=ens_states)
                # print(np.shape(states_update.T), '=============================')

                '''delta for monthly mean'''
                delta_state = states_update - ens_states
                delta_state = list(delta_state.T)

                '''add an arbitrary matrix to states to be able to use comm.scatter: 
                by default, the process id of OL_process must be 0'''
                delta_state = [np.zeros(np.shape(delta_state[0]))] + delta_state

            '''every thread should wait until the main thread finishes its job, and redistribute the updated states'''
            states_ens_update_delta = comm.scatter(sendobj=delta_state, root=main_thread)
            delta_state = None  # free the memory
            '''update the states for each ensemble: equal increment for each day'''
            for his_day_datetime in info[1]:
                his_day = his_day_datetime.strftime("%Y-%m-%d")
                states_old = self._sv.load_state_dict(date=his_day)

                if rank != OL_thread:
                    '''do not update the OL thread'''
                    new_state = self._sv.restore_states(old_states=states_old,
                                                        new_states=states_ens_update_delta.flatten(), isdelta=True)
                    '''possible negative value exists in the updated states. replace the negative value with zero'''
                    new_state = self.threshold(state=new_state)

                    '''save the new states, and overwrite the old one'''
                    self._sv.save_to_nc(state=new_state, date=his_day)

            if rank != OL_thread:
                # Retrieve the per_contfrac that the VB module used when it
                # wrote soilmoist/swe/canopystor to disk for the last day.
                # (update_landareafrac is called *after* calculate() in
                #  DailyStepRun.update, so we cannot read current_landareafrac
                #  from land_water_frac here — it has already been updated.)
                _per_contfrac = self._model.vertical_waterbalance._per_contfrac

                '''to assign the last day to the current prediction vector to enable another round of kalman filter'''
                new_state_internal= self._model.unit_converter.state_dict_to_internal(state=new_state, per_contfrac=_per_contfrac)  # convert to internal units
                self._sv.regional2Dstate_transfer_global2Dstate(rs=new_state_internal, gs=self._states_predict)
                pass

            '''free memory'''
            states_old = None
            vv = None
            states_ens_update_delta = None

        pass

    def helper_resolve_time(self, obs_aux: dict):
        """
        It is assumed that the date increases at daily basis.
        This tool helps to decide when and how the update takes place
        """

        '''configure time period'''
        daylist = GeoMathKit.dayListByDay(begin=self.DA_setting.basic.fromdate,
                                          end=self.DA_setting.basic.todate)

        '''configure observation reference'''
        duration = obs_aux['duration']
        data_first = []
        data_end = []
        for data in duration:
            a, b = data.split('_')
            data_first.append(datetime.strptime(a, "%Y-%m-%d"))
            data_end.append(datetime.strptime(b, "%Y-%m-%d"))
            pass

        ''''''
        NewRecord = []
        KeepRecord = []
        AssimilationRecord = []
        AssimilationInfo = []

        newRecord = False
        keepRecord = False
        assimilation = False

        nod = 0
        dates_in_list = []

        '''find the match of the first data set'''
        data_index = None
        for day in daylist:
            if day in data_first:
                data_index = data_first.index(day)
                break

        assert data_index is not None
        current_index = data_index

        Nlen = len(data_first)

        '''start loop'''
        for day in daylist:

            if day > data_end[-1]:
                '''this means that no observation is available'''
                NewRecord.append(False)
                KeepRecord.append(False)
                AssimilationRecord.append(False)
                continue

            if data_index == Nlen:
                '''this is to avoid the mistake of the last observation'''
                data_first.append(-1)

            if day == data_first[data_index]:
                newRecord = True
                keepRecord = True

                current_index = data_index
                data_index += 1
                nod = 1
                dates_in_list = [day]

            else:
                newRecord = False

                if data_end[current_index] >= day > data_first[current_index]:
                    keepRecord = True
                    nod += 1
                    dates_in_list.append(day)
                else:
                    keepRecord = False

            if day == data_end[current_index]:
                assimilation = True
                time_epoch = obs_aux['time_epoch'][current_index]
                number_of_days = nod
                AssimilationInfo.append([time_epoch, dates_in_list, current_index])
            else:
                assimilation = False

            NewRecord.append(newRecord)
            KeepRecord.append(keepRecord)
            AssimilationRecord.append(assimilation)

            pass

        return {'NewRecord': NewRecord,
                'KeepRecord': KeepRecord,
                'AssimilationRecord': AssimilationRecord,
                'AssimilationInfo': AssimilationInfo}


class EnKF_localization_v1(EnKF):
    """
    Model covariance localization is added. The obs and obs_cov are not changeable.
    """

    def __init__(self, DA_setting: config_DA, model: model_run_daily, obs: GRACE_obs, sv: EnsStates):
        super().__init__(DA_setting, model, obs, sv)

    def update(self, obs, obs_cov, ens_states):
        """
        update is for all ensembles
        reference: WIKI
        """
        R = obs_cov.copy()

        '''calculate the deviation of ens_states'''
        A = ens_states - np.mean(ens_states, 1)[:, None]

        '''Inflation to increase the model perturbation'''
        A = A * self._inflation
        ens_states_inf = np.mean(ens_states, 1)[:, None] + A

        '''propagate it into obs-equivalent variable'''
        HX = self._DM(states=ens_states_inf)

        '''to calculate the deviation'''
        HA = HX - np.mean(HX, 1)[:, None]

        '''calculate the model COV'''
        CA = np.cov(HA)

        '''regularization: solve the stability problem by adding a tiny diagonal matrix'''
        cc = CovRegu().set_COV(cov=CA)
        cc.method_shrinkage(alpha=0.10)  # covariance in obs space is better not too strong to prevent from sharpness
        CA = cc.get_COV()

        cc.set_COV(cov=R)
        cc.method_shrinkage(alpha=0.50)
        R = cc.get_COV()

        '''calculate matrix P'''
        P = CA + R

        '''calculate the gain factor K'''
        N = self.DA_setting.basic.ensemble
        '''method-1: straight-forward'''
        # K = 1 / (N - 1) * A @ HA.T @ np.linalg.inv(P)
        '''method-2: via linear solver'''
        Bt = np.linalg.lstsq(P.T, HA, rcond=None)[0]
        K = 1 / (N - 1) * A @ Bt.T

        '''update the states'''
        states_update = ens_states_inf + K @ (obs - HX)

        return states_update


class EnKF_domain_localization(EnKF):

    def __init__(self, DA_setting: config_DA, model: model_run_daily, obs: GRACE_obs, sv: EnsStates):
        super().__init__(DA_setting, model, obs, sv)
        self._dl_Pmatrix = DomainLocalization(shapefile=DA_setting.basic.basin_shp, radius=3).Pmatrix()

    def update(self, obs, obs_cov, ens_states):
        """
        update is for all ensembles
        reference: WIKI
        """
        R = obs_cov.copy()

        '''calculate the deviation of ens_states'''
        A = ens_states - np.mean(ens_states, 1)[:, None]

        '''Inflation to increase the model perturbation'''
        A = A * self._inflation
        ens_states_inf = np.mean(ens_states, 1)[:, None] + A

        '''propagate it into obs-equivalent variable'''
        HX = self._DM(states=ens_states_inf)

        '''to calculate the deviation'''
        HA = HX - np.mean(HX, 1)[:, None]

        '''calculate the model COV'''
        CA = np.cov(HA)

        '''regularization: solve the stability problem by adding a tiny diagonal matrix'''
        cc = CovRegu().set_COV(cov=CA)
        cc.method_shrinkage(
            alpha=0.10)  # model covariance in obs space is better not too strong to prevent from sharpness
        CA = cc.get_COV()

        '''domain localization applied to the observation'''
        R = R * self._dl_Pmatrix
        cc.set_COV(cov=R)
        cc.method_shrinkage(alpha=0.20)
        R = cc.get_COV()

        '''calculate matrix P'''
        P = CA + R

        '''calculate the gain factor K'''
        N = self.DA_setting.basic.ensemble
        '''method-1: straight-forward'''
        # K = 1 / (N - 1) * A @ HA.T @ np.linalg.inv(P)
        '''method-2: via linear solver'''
        Bt = np.linalg.lstsq(P.T, HA, rcond=None)[0]
        K = 1 / (N - 1) * A @ Bt.T

        '''update the states'''
        states_update = ens_states_inf + K @ (obs - HX)

        return states_update


class EnKF_localization_v2(EnKF):
    """
    compared to v1, this version allows for a flexible adjustment of R
    """

    def __init__(self, DA_setting: config_DA, model: model_run_daily, obs: GRACE_obs, sv: EnsStates):
        super().__init__(DA_setting, model, obs, sv)

    def update(self, obs, obs_cov, ens_states):
        """
        update is for all ensembles
        reference: WIKI
        """
        R = obs_cov / 5  # test what happens if reducing the obs_cov

        '''calculate the deviation of ens_states'''
        A = ens_states - np.mean(ens_states, 1)[:, None]

        '''Inflation to increase the model perturbation'''
        A = A * self._inflation
        ens_states_inf = np.mean(ens_states, 1)[:, None] + A

        '''propagate it into obs-equivalent variable'''
        HX = self._DM(states=ens_states_inf)

        '''to calculate the deviation'''
        HA = HX - np.mean(HX, 1)[:, None]

        '''calculate the model COV'''
        CA = np.cov(HA)

        '''regularization: solve the stability problem by adding a tiny diagonal matrix'''
        cc = CovRegu().set_COV(cov=CA)
        cc.method_shrinkage(alpha=0.5)
        CA = cc.get_COV()
        '''Most regularization/localization method does not work for EnSQRA since the regularization would 
        lead to inconsistency in computing the posterior covariance.'''

        '''calculate matrix P'''
        P = CA + R

        '''calculate the gain factor K'''
        N = self.DA_setting.basic.ensemble
        '''method-1: straight-forward'''
        # K = 1 / (N - 1) * A @ HA.T @ np.linalg.inv(P)
        '''method-2: via linear solver'''
        Bt = np.linalg.lstsq(P.T, HA, rcond=None)[0]
        K = 1 / (N - 1) * A @ Bt.T

        '''regenerate the observation ensemble according to R'''
        if obs.size == 1:
            new_obs = np.random.normal(obs, np.sqrt(R), size=N)[:, None]
        else:
            new_obs = np.random.multivariate_normal(mean=obs, cov=R, size=N)

        new_obs = new_obs.T

        '''update the states'''
        states_update = ens_states_inf + K @ (new_obs - HX)

        return states_update

    def run_mpi(self):
        """
        parallelized running with MPI4py
        """
        from datetime import datetime, timedelta

        '''main process'''
        main_thread = 1

        '''OL process: no perturbation'''
        OL_thread = 0

        '''preparation'''
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        '''make sure that each ensemble is given a process'''
        assert size == self.DA_setting.basic.ensemble + 1

        '''configure setting'''
        daylist = GeoMathKit.dayListByDay(begin=self.DA_setting.basic.fromdate,
                                          end=self.DA_setting.basic.todate)

        '''assign job to each ensemble'''
        firstday = True

        historic_states = []
        historic_mean_states = None
        obs = None
        obs_cov = None
        rr = -1
        previous_month = -1

        print('=====================Data assimilation=========================')
        for count, day in enumerate(daylist):
            "print information"
            if day.month != previous_month:
                previous_month = day.month
                print('\nDoing year/month: %04d/%02d' % (day.year, day.month))
            today = day.strftime('%Y-%m-%d')
            print('.', end='')
            '''==================kalman filter: prediction step=================================='''
            self.predict(today=today, states=self._states_predict, is_first_day=firstday, issave=True)
            firstday = False

            newRecord = self._obs_helper['NewRecord'][count]
            keepRecord = self._obs_helper['KeepRecord'][count]
            assimilationRecord = self._obs_helper['AssimilationRecord'][count]
            # print(newRecord, keepRecord, assimilationRecord)
            if newRecord:
                """create a new vector to prepare for assimilation of next time"""
                historic_mean_states = None

            if keepRecord:
                '''record the previous states over areas of interest'''
                sv = self._sv.get_states_by_transfer_single(states=self._states_predict)
                if historic_mean_states is None:
                    historic_mean_states = sv
                else:
                    historic_mean_states += sv
            else:
                historic_mean_states = None

            if not assimilationRecord:
                continue

            '''====================start assimilation================================'''
            # print('====> GRACE data has been assimilated.')
            print('|', end='')
            rr += 1
            '''get obs and cov'''
            info = self._obs_helper['AssimilationInfo'][rr]
            self._obs.set_date(date=info[0])
            obs = self._obs.get_obs()
            obs_cov = self._obs.get_cov()
            '''EnKF'''
            historic_mean_states = historic_mean_states / len(info[1])

            '''synchronization to prepare for the data assimilation'''
            this_day = comm.gather(today, root=main_thread)
            if rank == main_thread:
                for iday in this_day:
                    '''confirm the synchronization again'''
                    assert iday == today

            '''collect obs from each ensemble'''
            ens_obs = comm.gather(root=main_thread, sendobj=obs)

            '''collect states from each ensemble'''
            ens_states = comm.gather(root=main_thread, sendobj=historic_mean_states)

            if rank != main_thread:
                delta_state = None
                pass
            else:
                '''delete the OL process: it does not participate in data assimilation'''
                obs_unperturbation = ens_obs[OL_thread].copy()
                del ens_obs[OL_thread]
                del ens_states[OL_thread]

                '''load ensemble states'''
                ens_obs = np.array(ens_obs).T
                ens_states = np.array(ens_states).T

                '''kalman filter: update step'''
                states_update = self.update(obs=obs_unperturbation, obs_cov=obs_cov, ens_states=ens_states)
                # print(np.shape(states_update.T), '=============================')

                '''delta for monthly mean'''
                delta_state = states_update - ens_states
                delta_state = list(delta_state.T)

                '''add an arbitrary matrix to states to be able to use comm.scatter: 
                by default, the process id of OL_process must be 0'''
                delta_state = [np.zeros(np.shape(delta_state[0]))] + delta_state

            '''every thread should wait until the main thread finishes its job, and redistribute the updated states'''
            states_ens_update_delta = comm.scatter(sendobj=delta_state, root=main_thread)
            delta_state = None  # free the memory
            '''update the states for each ensemble: equal increment for each day'''
            for his_day_datetime in info[1]:
                his_day = his_day_datetime.strftime("%Y-%m-%d")
                states_old = self._sv.load_state_dict(date=his_day)

                if rank != OL_thread:
                    '''do not update the OL thread'''
                    new_state = self._sv.restore_states(old_states=states_old,
                                                        new_states=states_ens_update_delta.flatten(), isdelta=True)

                    '''possible negative value exists in the updated states. replace the negative value with zero'''
                    for key, vv in new_state.items():
                        vv[vv < 0] = 0.0001

                    '''save the new states, and overwrite the old one'''
                    self._model.save(states=new_state, day=his_day)

            if rank != OL_thread:
                '''to assign the last day to the current prediction vector to enable another round of kalman filter'''
                self._states_predict = new_state

            '''free memory'''
            states_old = None
            vv = None
            states_ens_update_delta = None

        pass


class EnKF_adaptive(EnKF):
    """
    In this approach, we wish to increase the weight of obs when the deviation between model and obs becomes big.
    In another word, we trust obs more than model when they have big deviation. In this case, covariance of obs must be
    diagonal.# todo: not finished yet
    """

    def __init__(self, DA_setting: config_DA, model: model_run_daily, obs: GRACE_obs, sv: EnsStates):
        super().__init__(DA_setting, model, obs, sv)

    def update(self, obs, obs_cov, ens_states):
        """
        update is for all ensembles
        reference: WIKI
        """
        R = obs_cov

        '''calculate the deviation of ens_states'''
        A = ens_states - np.mean(ens_states, 1)[:, None]

        '''Inflation to increase the model perturbation'''
        A = A * self._inflation
        ens_states_inf = np.mean(ens_states, 1)[:, None] + A

        '''propagate it into obs-equivalent variable'''
        HX = self._DM(states=ens_states_inf)

        '''to calculate the deviation'''
        HA = HX - np.mean(HX, 1)[:, None]

        '''calculate the increment'''
        increment = np.mean(obs - HX, 1)

        '''adjust the cov according to the increment'''
        cr = np.diag(np.diag(R) * np.abs(increment) / HX)

        '''calculate matrix P'''
        P = np.cov(HA) + R

        '''calculate the gain factor K'''
        N = self.DA_setting.basic.ensemble
        '''method-1: straight-forward'''
        # K = 1 / (N - 1) * A @ HA.T @ np.linalg.inv(P)
        '''method-2: via linear solver'''
        Bt = np.linalg.lstsq(P.T, HA, rcond=None)[0]
        K = 1 / (N - 1) * A @ Bt.T

        '''update the states'''
        states_update = ens_states_inf + K @ (obs - HX)

        return states_update
