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

    Driver of the data assimilation (daily loop, MPI, window means and daily extremes, post-update thresholds) with
    the plain EnKF analysis in update(). The localized filter with its components (localization, inflation,
    increment partition, bounds) is src_DA.EnKF_localized, built by src_DA.filter_factory.build_filter.
    Earlier experimental variants are in src_DA/legacy/EnKF_legacy.py.
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
        self._thresholder = model_state_threshold(configDA=DA_setting)
        self.threshold = self._thresholder.threshold

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

    def _land_ratio_box(self):
        """land / continental fraction of the model's current day on the regional box (the factor WaterGAP applies
        to soil, snow and canopy when writing the daily files); 1 where undefined"""
        pc = np.asarray(self._model.vertical_waterbalance._per_contfrac, dtype=float)
        uc = self._model.unit_converter
        if hasattr(uc, '_r0') and pc.shape != np.shape(self._sv.DM.local_mask['basin_2d']):
            pc = pc[uc._r0:uc._r1, uc._c0:uc._c1]
        return np.where(np.isfinite(pc), np.clip(pc, 0.0, 1.0), 1.0)

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
        # print(obs-HX)

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
        historic_min_states = None          # daily minimum / maximum over the window (window-aware bounds)
        historic_max_states = None
        historic_land_ratio = None          # largest land / continental fraction of the window (soil capacity)
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
                historic_min_states = None
                historic_max_states = None
                historic_land_ratio = None

            if keepRecord:
                '''record the previous states over areas of interest'''
                sv = self._sv.get_states_by_transfer_single(states=local_predict_of_today)
                sv_excluded = self._sv_excluded.get_states_by_transfer_single(states=local_predict_of_today_excluded)
                if historic_mean_states is None:
                    historic_min_states = np.array(sv, dtype=float, copy=True)
                    historic_max_states = np.array(sv, dtype=float, copy=True)
                    historic_mean_states = sv
                    historic_mean_states_excluded = sv_excluded
                else:
                    historic_min_states = np.minimum(historic_min_states, sv)
                    historic_max_states = np.maximum(historic_max_states, sv)
                    historic_mean_states += sv
                    historic_mean_states_excluded += sv_excluded
                lr = self._land_ratio_box()
                historic_land_ratio = lr if historic_land_ratio is None else np.fmax(historic_land_ratio, lr)
            else:
                historic_mean_states = None
                historic_mean_states_excluded = None
                historic_min_states = None
                historic_max_states = None
                historic_land_ratio = None

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
            obs_raw = np.array(obs, dtype=float, copy=True)     # perturbed GRACE before removing excluded storages

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
            ens_obs_raw = comm.gather(root=main_thread, sendobj=obs_raw)

            '''collect states from each ensemble'''
            ens_states = comm.gather(root=main_thread, sendobj=historic_mean_states)
            ens_min = comm.gather(root=main_thread, sendobj=historic_min_states)
            ens_max = comm.gather(root=main_thread, sendobj=historic_max_states)
            land_ratio_cells = None if historic_land_ratio is None else \
                historic_land_ratio[np.asarray(self._sv.DM.local_mask['basin_2d']).astype(bool)]
            ens_land_ratio = comm.gather(root=main_thread, sendobj=land_ratio_cells)
            historic_min_states = None
            historic_max_states = None

            '''free memory'''
            historic_mean_states = None
            sv = None

            if rank != main_thread:
                delta_state = None
                pass
            else:
                '''delete the OL process: it does not participate in data assimilation'''
                '''unperturbed GRACE (OL rank reads ens_0) and the perturbed values of the members: their difference is
                the observation perturbation, used by filters that change the observation-error model'''
                self._obs_unperturbed = np.array(ens_obs_raw[OL_thread])
                del ens_obs_raw[OL_thread]
                self._obs_raw = np.array(ens_obs_raw).T
                ens_obs_raw = None
                del ens_obs[OL_thread]
                del ens_states[OL_thread]
                del ens_min[OL_thread]
                del ens_max[OL_thread]
                del ens_land_ratio[OL_thread]
                '''soil capacity of the members in the units of the states (cells x members), see Threshold.set_soil_scale'''
                self._ens_soil_scale = None if any(x is None for x in ens_land_ratio) else np.array(ens_land_ratio).T
                ens_land_ratio = None

                '''load ensemble states'''
                ens_obs = np.array(ens_obs).T
                ens_states = np.array(ens_states).T
                '''daily extremes over the window, used by filters with window-aware bounds (EnKF_localized)'''
                self._ens_min = np.array(ens_min).T
                self._ens_max = np.array(ens_max).T
                ens_min = ens_max = None

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
            self._thresholder.set_soil_scale(historic_land_ratio)        # soil capacity of this member and window
            for his_day_datetime in info[1]:
                his_day = his_day_datetime.strftime("%Y-%m-%d")
                states_old = self._sv.load_state_dict(date=his_day)

                if rank != OL_thread:
                    '''do not update the OL thread'''
                    new_state = self._sv.restore_states(old_states=states_old,
                                                        new_states=states_ens_update_delta.flatten(), isdelta=True)
                    '''possible negative value exists in the updated states. replace the negative value with zero'''
                    new_state = self.threshold(state=new_state, date=his_day)

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

        '''clipping statistics of this member (printed to the rank log, saved next to its daily output)'''
        if rank != OL_thread:
            self._thresholder.print_summary()
            try:
                from pathlib import Path
                out_dir = Path(self.DA_setting.basic.DA_output_temp_dir) / self.DA_setting.basic.case / ('Ens_%d' % rank)
                self._thresholder.save_log(out_dir / 'threshold_log.json')
                self._thresholder.save_monthly_csv(out_dir / 'threshold_monthly.csv')
            except Exception as err:            # logging must never stop the run
                print('threshold log not written: %s' % err)

        '''filter statistics (partition, additive inflation, adaptive-inflation log): main thread only'''
        if rank == main_thread and hasattr(self, 'save_filter_log'):
            try:
                from pathlib import Path
                self.save_filter_log(Path(self.DA_setting.basic.DA_output_temp_dir) / self.DA_setting.basic.case)
            except Exception as err:
                print('filter log not written: %s' % err)

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
