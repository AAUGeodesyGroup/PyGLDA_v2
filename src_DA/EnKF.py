from src_GHM.Interface.DailyStepRun import DailyModelRun as model_run_daily
from src_DA.configure_DA import config_DA
from src_auxiliary.GeoMathKit import GeoMathKit
from src_DA.observations import GRACE_obs
from src_DA.ExtractStates import EnsStates
from src_DA.ObsDesignMatrix import DM_basin_average
import numpy as np
from mpi4py import MPI
from datetime import datetime

from src_DA.Threshold import model_state_threshold
from src_DA.EnumDA import WaterGap_storage_variables


class EnKF:
    """
    Driver of the data assimilation: the daily model loop over MPI ranks (rank 0 = unperturbed open loop, ranks 1..N =
    members), the window means and daily extremes of the states, the analysis at the end of each observation window
    (any window length), the daily write-back of the increments and the post-update thresholds.
    The analysis itself (update) is implemented by src_DA.EnKF_localized.EnKF_localized and src_DA.LETKF, built from
    the settings by src_DA.filter_factory.build_filter.
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

        '''time-released increment (method block "increment_release", default off): see _release_config'''
        self._release = self._release_config()
        self._release_pending = None        # per-day share of the withheld increment (state vector of this member)
        self._release_left = 0              # days still to release
        self._release_windows = 0

        '''daily soil increments within the soil bounds (method key "soil_daily_bounds", default off): see
        _soil_cap_config'''
        self._soil_cap = self._soil_cap_config()
        self._soil_cap_stats = dict(n_windows=0, n_cells_shifted=0, excess_mm=0.0, max_excess_mm=0.0)

        '''soil capacity multiplier of this member (ensemble perturbation "smax_multiplier", see src_DA.perturbation_par and
        ReWaterGAP model/parameters.py; 1 = standard WaterGAP): the soil bound of the filter and of the threshold is
        smax.nc x land fraction x this factor, applied with the land fraction in run_mpi'''
        self._smax_multiplier = self._member_smax_multiplier()

        '''obtain info'''
        self._obs_helper = self.helper_resolve_time(obs.get_obs_aux())

        '''model re-configuration'''
        self._model.unit_converter= self._model.unit_converter.from_basin_mask(local_mask=self._sv.DM.local_mask,
                                                   cell_area_global=self._model.unit_converter.cell_area_global,
                                                   contfrac_global=self._model.unit_converter.contfrac_global)

    def _member_smax_multiplier(self):
        """smax_multiplier of the member this rank runs (one value per member), 1.0 if the parameter file has none"""
        try:
            m = np.asarray(self._model.parameters.global_params['smax_multiplier'].values, dtype=float)
        except (AttributeError, KeyError):
            return 1.0
        m = m[np.isfinite(m)]
        if m.size == 0:
            return 1.0
        if np.ptp(m) > 0:
            raise ValueError('smax_multiplier varies in space; the DA soil bound supports one value per member')
        return float(m[0])

    def _release_config(self):
        """
        "increment_release": {"storages": ["riverstor"], "days": 5} in the method block of DA_setting.json (absent or
        null = off, the default). The analysis increment of the listed storages is NOT put into the state the model
        restarts from at the end of a window; it is added in equal parts on the first `days` days of the next window
        (a simple incremental analysis update, IAU, for fast storages). Without it the whole river increment enters the
        model at once and is routed out on the next day, which gives a discharge pulse at the window start.
        The stored daily analysis of the window itself keeps the full increment of every storage (the product is
        unchanged); only the trajectory that continues into the next window receives the listed storages' share
        gradually. A share still pending when the next window is analysed is added to that restart at once.
        """
        cfg = getattr(self.DA_setting.method, 'increment_release', None)
        if not cfg:
            return None
        names = list(self._sv.DM.statesnn)
        storages = [v for v in cfg.get('storages', ['riverstor']) if v in names]
        days = int(cfg.get('days', 5))
        if not storages or days < 2:
            print('increment_release %s: no storage of the DA state %s or days < 2 - off' % (cfg, names))
            return None
        return dict(storages=storages, days=days, var_idx=[names.index(v) for v in storages], nvar=len(names))

    def _soil_cap_config(self):
        """
        "soil_daily_bounds": true in the method block (default false). The window-mean soil increment delta is bounded
        by 0 <= x_mean + delta <= cap (the filter's bounds act on the window mean for soil), and the soil increment of
        every day is proportional to what that day can take, so that every day of the window stays within [0, cap]:
            delta > 0:  d_t = delta * r_t / mean(r),  r_t = max(cap - x_t, 0)   (room to capacity)
            delta < 0:  d_t = delta * a_t / mean(a),  a_t = max(x_t, 0)          (water available)
        mean_t(d_t) = delta exactly. Without it every day receives the same increment, and a soil that touches its
        capacity (or zero) on one day of the window blocks the increment of the whole window.
        """
        if not getattr(self.DA_setting.method, 'soil_daily_bounds', False):
            return None
        names = list(self._sv.DM.statesnn)
        if 'soilmoist' not in names:
            print('soil_daily_bounds: soilmoist is not in the DA state %s - off' % names)
            return None
        return dict(nvar=len(names), i_soil=names.index('soilmoist'))

    def _soil_proportional(self, delta, soil_days, cap):
        """
        delta: window-mean increment (state vector); soil_days: (n_days x n_cell) forecast
        soil of this member on the days of the window; cap: (n_cell,) capacity. Returns the soil increments
        (n_days x n_cell): delta > 0 shared in proportion to the room to capacity, delta < 0 in proportion to the water
        available, so that no day leaves [0, cap]. A cell without any room / water where delta needs it falls back to
        the equal increment (the threshold clips).
        """
        sc = self._soil_cap
        d_s = delta[sc['i_soil']::sc['nvar']]
        x = np.asarray(soil_days, dtype=float)
        capb = np.broadcast_to(np.asarray(cap, dtype=float)[None, :], x.shape)
        room = np.where(np.isfinite(capb), np.maximum(capb - x, 0.0), 1.0)          # no capacity: equal weights
        avail = np.maximum(x, 0.0)
        w = np.where(d_s[None, :] > 0, room, avail)
        wm = w.mean(0)
        ok = wm > 1e-12
        out = np.where(ok[None, :], d_s[None, :] * w / np.where(ok, wm, 1.0)[None, :], d_s[None, :])
        # never more than a day's own room / water (rounding, or delta beyond the window-mean bound)
        out = np.where(ok[None, :] & np.isfinite(capb), np.minimum(out, room), out)
        out = np.where(ok[None, :], np.maximum(out, -avail), out)
        st = self._soil_cap_stats
        st['n_windows'] += 1
        st['n_cells_shifted'] += int((~ok & (d_s != 0)).sum())                 # here: cells on the equal fallback
        lost = np.abs(out.mean(0) - d_s)
        st['excess_mm'] += float(lost.sum())                                   # here: mean change lost to the limits
        if lost.size:
            st['max_excess_mm'] = max(st['max_excess_mm'], float(lost.max()))
        return out

    def _release_mask(self, n_state):
        """True for the elements of the released storages (state vector cell-major, nvar storages per cell)"""
        return np.isin(np.arange(n_state) % self._release['nvar'], self._release['var_idx'])

    def _released_part(self, state: dict, values: dict):
        """state with the released storages replaced by `values` (dict storage -> 2-D field), the others unchanged"""
        out = dict(state)
        for v in self._release['storages']:
            if v in values:
                out[v] = values[v]
        return out

    def _apply_release(self, today):
        """add the day's share of the withheld increment to today's state (saved file and the model's restart
        vector); only the released storages are re-checked by the threshold"""
        st = self._sv.load_state_dict(date=today)
        st = self._sv.restore_states(old_states=st, new_states=self._release_pending, isdelta=True)
        sub = self.threshold(state={v: st[v] for v in self._release['storages'] if v in st}, date=today)
        st = self._released_part(st, sub)
        self._sv.save_to_nc(state=st, date=today)
        _per_contfrac = self._model.vertical_waterbalance._per_contfrac
        internal = self._model.unit_converter.state_dict_to_internal(state=st, per_contfrac=_per_contfrac)
        self._sv.regional2Dstate_transfer_global2Dstate(rs=internal, gs=self._states_predict)
        self._release_left -= 1

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
        """analysis of one window for all members (implemented by the filters, see src_DA.filter_factory)"""
        raise NotImplementedError('EnKF is the driver; the analysis is EnKF_localized.update or LETKF_localized')

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
        historic_soil_days = None           # forecast soil of every day of the window (soil_daily_bounds)
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
            if self._release_left > 0 and rank != OL_thread:
                self._apply_release(today)          # share of the previous window's withheld increment
            local_predict_of_today = self._sv.load_state_dict(date=today)  # load state from hard disk

            local_predict_of_today_excluded = self._sv_excluded.load_state_dict(date=today)  # load state from hard disk

            firstday = False

            newRecord = self._obs_helper['NewRecord'][count]
            keepRecord = self._obs_helper['KeepRecord'][count]
            assimilationRecord = self._obs_helper['AssimilationRecord'][count]
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
                if self._soil_cap is not None:
                    if historic_mean_states is None:
                        historic_soil_days = []
                    historic_soil_days.append(np.array(sv[self._soil_cap['i_soil']::self._soil_cap['nvar']],
                                                       dtype=float, copy=True))
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
                lr = self._land_ratio_box() * self._smax_multiplier     # member's soil capacity factor
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
                obs = obs - self._sv_excluded.DM.getDM() @ historic_mean_states_excluded
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

                '''calendar months of the window (monthly snow cap of EnKF_localized)'''
                self._window_months = sorted({d.month for d in info[1]})

                '''kalman filter: update step'''
                states_update = self.update(obs=ens_obs, obs_cov=obs_cov, ens_states=ens_states)

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
            release_now = self._release is not None and rank != OL_thread
            if release_now:
                '''time-released increment: the listed storages restart from the forecast (plus a share still pending
                from the previous window); their increment is added over the first days of the next window'''
                delta_full = states_ens_update_delta.flatten()
                mask = self._release_mask(delta_full.size)
                carry = self._release_pending * self._release_left if self._release_left > 0 else None
                self._release_pending = np.where(mask, delta_full, 0.0) / self._release['days']
                self._release_left = self._release['days']
                self._release_windows += 1
            soil_cap_now = self._soil_cap is not None and rank != OL_thread
            if soil_cap_now:
                '''soil capacity of this member and window (the threshold's bound) on the mask cells'''
                _cap2d = self._thresholder._bounds('soilmoist', None)[1]
                _mk = np.asarray(self._sv.DM.local_mask['basin_2d']).astype(bool)
                _cap = np.broadcast_to(np.asarray(_cap2d, dtype=float), _mk.shape)[_mk]
                if historic_soil_days is None or len(historic_soil_days) != len(info[1]):
                    print('soil_daily_bounds: %s recorded days for a window of %d days - equal increments'
                          % (None if historic_soil_days is None else len(historic_soil_days), len(info[1])))
                    soil_cap_now = False
                else:
                    _dsoil = self._soil_proportional(states_ens_update_delta.flatten(),
                                                     np.vstack(historic_soil_days), _cap)
                    _rs = slice(self._soil_cap['i_soil'], None, self._soil_cap['nvar'])
            historic_soil_days = None
            for k_day, his_day_datetime in enumerate(info[1]):
                his_day = his_day_datetime.strftime("%Y-%m-%d")
                states_old = self._sv.load_state_dict(date=his_day)

                if rank != OL_thread:
                    '''do not update the OL thread'''
                    last_day = his_day_datetime == info[1][-1]
                    if release_now and last_day:
                        restart_forecast = {v: states_old[v].copy() for v in self._release['storages'] if v in states_old}
                    if self._thresholder.snow_cap is not None and 'swe' in states_old:
                        self._thresholder.set_snow_forecast(states_old['swe'].copy())   # cap never removes it
                    if soil_cap_now:
                        d_day = np.array(states_ens_update_delta.flatten(), dtype=float, copy=True)
                        d_day[_rs] = _dsoil[k_day]
                        new_state = self._sv.restore_states(old_states=states_old, new_states=d_day, isdelta=True)
                    else:
                        new_state = self._sv.restore_states(old_states=states_old,
                                                            new_states=states_ens_update_delta.flatten(), isdelta=True)
                    '''physical limits after the update (negative storages, soil capacity, snow cap, ...)'''
                    new_state = self.threshold(state=new_state, date=his_day)

                    '''save the new states, and overwrite the old one'''
                    self._sv.save_to_nc(state=new_state, date=his_day)
                    if release_now and last_day:
                        if carry is not None:
                            restart_forecast = self._sv.restore_states(
                                old_states=self._released_part(new_state, restart_forecast), new_states=carry,
                                isdelta=True)
                            restart_forecast = self.threshold(
                                state={v: restart_forecast[v] for v in self._release['storages']}, date=his_day)
                        restart_state = self._released_part(new_state, restart_forecast)

            if rank != OL_thread:
                # Retrieve the per_contfrac that the VB module used when it
                # wrote soilmoist/swe/canopystor to disk for the last day.
                # (update_landareafrac is called *after* calculate() in
                #  DailyStepRun.update, so we cannot read current_landareafrac
                #  from land_water_frac here — it has already been updated.)
                _per_contfrac = self._model.vertical_waterbalance._per_contfrac

                '''to assign the last day to the current prediction vector to enable another round of kalman filter'''
                new_state_internal= self._model.unit_converter.state_dict_to_internal(
                    state=restart_state if release_now else new_state, per_contfrac=_per_contfrac)  # convert to internal units
                self._sv.regional2Dstate_transfer_global2Dstate(rs=new_state_internal, gs=self._states_predict)
                pass

            '''free memory'''
            states_old = None
            vv = None
            states_ens_update_delta = None

        if self._soil_cap is not None and rank != OL_thread:
            st = self._soil_cap_stats
            print('soil_daily_bounds: %d windows, %d cell-windows on the equal fallback, %.1f mm of window-mean soil '
                  'increment not placed because of the daily limits (largest %.1f mm)'
                  % (st['n_windows'], st['n_cells_shifted'], st['excess_mm'], st['max_excess_mm']))
        if self._release is not None and rank == main_thread:
            print('increment_release: %s released over %d days after each of %d windows'
                  % (self._release['storages'], self._release['days'], self._release_windows))

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
