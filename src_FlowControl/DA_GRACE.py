import os
from datetime import datetime, timedelta
from pathlib import Path
# from OpenLoop import OpenLoop
# from src_OBS.prepare_GRACE import GRACE_preparation
from src_DA.configure_DA import config_DA
# from src_OBS.GRACE_perturbation_old import GRACE_perturbed_obs
from src_DA.ObsDesignMatrix import DM_basin_average
import h5py
import numpy as np
import sys
import json
import xarray as xr
from src_FlowControl.OpenLoop import OpenLoop
from src_GHM.Interface.DailyStepRun import DailyModelRun
from src_auxiliary.shp2mask import load_mask
from src_DA.ExtractStates import EnsStates
from src_auxiliary.banner import print_pyglda_banner


class ensemble_model_daily_step(OpenLoop):
    """
    This can only be used for model resume mode
    """

    def __init__(self, configDA: config_DA, setting_dir='/media/user/My Book/Fan/WaterGap/Extensions/',
                 ensemble_id: int = 3):
        case = configDA.basic.case
        super().__init__(setting_dir=setting_dir, case=case, ensemble_id=ensemble_id)

        self.configDA = configDA

        self.__internal_configuration()

        box_crop, _ = load_mask(mask_path=configDA.basic.basin_mask)

        self.__model_instance = DailyModelRun(is_crop_save=True,
                                              **box_crop)  # this is produced based on the CM configuration.
        pass

    def __internal_configuration(self):
        """This is very important! All the configuration has been done here so that the others do not need to
        configure the CM again"""
        configDA = self.configDA
        self.configure_time(begin_time=configDA.basic.fromdate, end_time=configDA.basic.todate)

        pp = Path(configDA.basic.DA_output_temp_dir) / configDA.basic.case
        pp.mkdir(parents=False, exist_ok=True)
        self.configure_Ens_output(output_dir=pp)
        self.configure_Ens_input(input_dir=configDA.basic.Ensemble_input_dir)
        self.configure_ini_for_resume(save_init_dir=configDA.basic.Ensemble_ini_dir,
                                      read_init_dir=configDA.basic.Ensemble_ini_dir)

        pass

    def get_model_instance(self):
        return self.__model_instance


class DA_GRACE:

    def __init__(self, case='test', setting_dir='../settings/DA_local', ens_size=3):
        self.ens_size = ens_size
        self.case = case
        self.setting_dir = Path(setting_dir)
        dp_dir = self.setting_dir / 'DA_setting.json'
        self.configDA = config_DA.loadjson(dp_dir).process()
        pass

    def configure_setting(self, ens_size, case_name, basin_name, basin_dir, basin_shp=None):
        """
        basin_shp : path of the basin shapefile. If None, the shapefile is looked for in
                    <basin_dir>/shp/<basin_name>/ (whatever *.shp is there - the file is not
                    necessarily called <basin_name>.shp, e.g. Danube3subbasins_subbasins.shp);
                    only if nothing is found does it fall back to <basin_name>.shp.
        The mask always follows the convention <basin_dir>/mask/<basin_name>/<basin_name>_res_0.5.h5
        (it is written that way by RDA.config_basin_mask).
        """
        self.configDA.basic.ensemble = ens_size
        self.configDA.basic.case = case_name
        self.configDA.basic.basin = basin_name
        shp_dir = Path(basin_dir) / 'shp' / basin_name
        if basin_shp is None:
            found = sorted(shp_dir.glob('*.shp')) if shp_dir.is_dir() else []
            if len(found) > 1:
                print(f"[configure_setting] {len(found)} shapefiles in {shp_dir}, using {found[0].name}; "
                      f"pass basin_shp explicitly to choose")
            basin_shp = found[0] if found else shp_dir / ('%s.shp' % basin_name)
        if not Path(basin_shp).exists():
            print(f"[configure_setting] WARNING: basin shapefile not found: {basin_shp}")
        self.configDA.basic.basin_shp = str(basin_shp)
        self.configDA.basic.basin_mask = str(Path(basin_dir) / 'mask' / basin_name / ('%s_res_0.5.h5' % basin_name))
        return self

    def configure_date(self, begin_date="2002-01-01", end_date='2005-04-30'):
        self.configDA.basic.fromdate = begin_date
        self.configDA.basic.todate = end_date
        return self

    def save_configuration(self):
        self.configDA.save_json(save_path=Path(self.setting_dir) / 'DA_setting.json')
        pass

    def reload_setting(self):
        dp_dir = self.setting_dir / 'DA_setting.json'
        self.configDA = config_DA.loadjson(dp_dir).process()
        pass

    def gather_OLmean(self):
        """
        Open-loop TWS mean per sub-basin, added to the GRACE anomalies to give observations in model TWS
        (GRACE_perturbed_obs.remove_temporal_mean / add_temporal_mean): the GRACE mean is taken over the GRACE epochs
        present in the assimilation period, so the open-loop mean is taken over exactly the same epochs, each one
        averaged over its observation window (the mean of the daily OL TWS over the days of the GRACE month, as the
        filter compares the window mean of the model with the observation). Until 7 Oct 2026 the open loop was
        averaged over every day of fromdate..todate, i.e. also over months without GRACE (gaps, 2002-01..03), which
        left a constant offset per sub-basin between the two means.
        Output: <OL_mean>/<case>_<basin>.hdf5 with mean_unperturbed, mean_ensemble (over members 1..N), time_epoch
        and duration of the epochs used.
        """
        import pandas as pd
        configDA = self.configDA
        days = pd.date_range(configDA.basic.fromdate, configDA.basic.todate)

        '''step-1: GRACE epochs of the period that exist in the signal file, with their observation windows'''
        obs_aux = self._obs_aux()
        ref = obs_aux.getTimeReference()
        signal_fn = Path(configDA.obs.GRACE['preprocess_res']) / ('%s_signal.hdf5' % configDA.basic.basin)
        with h5py.File(signal_fn, 'r') as f:
            available = set(f['time_epoch'][:].astype(str))
        epochs = [(t, d) for t, d in zip(ref['time_epoch'], ref['duration']) if t in available]
        if not epochs:
            raise ValueError('gather_OLmean: no GRACE epoch of %s..%s in %s' %
                             (configDA.basic.fromdate, configDA.basic.todate, signal_fn))
        windows = []
        for t, d in epochs:
            d1, d2 = d.split('_')
            w = (days >= pd.Timestamp(d1)) & (days <= pd.Timestamp(d2))
            if not w.any():
                raise ValueError('gather_OLmean: observation window %s of epoch %s is outside the OL period' % (d, t))
            windows.append(w)

        '''step-2: open-loop mean of every member over these windows'''
        ens_TWS = []
        for ens_id in range(configDA.basic.ensemble + 1):
            ens_dir = Path(configDA.basic.res_permanent) / configDA.basic.case / 'OL' / ('Ens_%s' % ens_id)
            with h5py.File(str(ens_dir / 'basin_ts_OL.h5'), 'r') as hf:
                mm = hf['tws']
                subbasin_num = len(list(mm.keys())) - 1
                basin_tws = []
                for subbasin in range(1, subbasin_num + 1):
                    nn = mm['sub_basin_%s' % subbasin][:]
                    if len(nn) != len(days):
                        raise ValueError('gather_OLmean: basin_ts_OL.h5 of Ens_%d has %d days, the period %s..%s has '
                                         '%d' % (ens_id, len(nn), configDA.basic.fromdate, configDA.basic.todate,
                                                 len(days)))
                    basin_tws.append(np.mean([nn[w].mean() for w in windows]))
            ens_TWS.append(basin_tws)
        ens_TWS = np.array(ens_TWS)

        mean_0 = ens_TWS[0]
        mean_1 = np.mean(ens_TWS[1:], axis=0)

        fn = Path(configDA.obs.GRACE['OL_mean']) / ('%s_%s.hdf5' % (configDA.basic.case, configDA.basic.basin))
        dt = h5py.special_dtype(vlen=str)
        with h5py.File(fn, 'w') as ww:
            ww.create_dataset(data=mean_0, name='mean_unperturbed')
            ww.create_dataset(data=mean_1, name='mean_ensemble')
            ww.create_dataset(data=[t for t, _ in epochs], name='time_epoch', dtype=dt)
            ww.create_dataset(data=[d for _, d in epochs], name='duration', dtype=dt)
        print('OL mean over %d GRACE epochs (%s .. %s) -> %s' % (len(epochs), epochs[0][0], epochs[-1][0], fn))
        pass

    def _obs_aux(self):
        """time reference (epochs and observation windows) of the observation kind in DA_setting.json for the
        assimilation period; used by gather_OLmean and generate_perturbed_GRACE_obs"""
        from src_OBS.obs_auxiliary import aux_ESAsing_5daily, aux_GRACE_SH_monthly, aux_GRACE_mascon_monthly, \
            aux_ESM3_5daily, aux_TUD_5daily

        configDA = self.configDA
        begin_day = configDA.basic.fromdate
        end_day = configDA.basic.todate
        kind = configDA.obs.GRACE['kind']
        if kind == 'SH_monthly':
            t1 = datetime.strptime(begin_day, '%Y-%m-%d').strftime('%Y-%m')
            t2 = datetime.strptime(end_day, '%Y-%m-%d').strftime('%Y-%m')
            return aux_GRACE_SH_monthly().setTimeReference(month_begin=t1, month_end=t2,
                                                           dir_in=configDA.obs.GRACE['aux_for_time_epochs'])
        if kind == 'ESA_SING':
            return aux_ESAsing_5daily().setTimeReference(day_begin=begin_day, day_end=end_day,
                                                         dir_in=configDA.obs.GRACE['aux_for_time_epochs'])
        if kind == 'ESA_SING_ESM3':
            return aux_ESM3_5daily().setTimeReference(day_begin=begin_day, day_end=end_day,
                                                      dir_in=configDA.obs.GRACE['aux_for_time_epochs'])
        if kind == 'Mascon_monthly':
            t1 = datetime.strptime(begin_day, '%Y-%m-%d').strftime('%Y-%m')
            t2 = datetime.strptime(end_day, '%Y-%m-%d').strftime('%Y-%m')
            return aux_GRACE_mascon_monthly().setTimeReference(month_begin=t1, month_end=t2,
                                                               dir_in=configDA.obs.GRACE['aux_for_time_epochs'])
        if kind == 'TUD_5daily':
            '''TU Delft 5-daily/weekly hybrid product: aux_for_time_epochs = folder holding the netCDF files'''
            return aux_TUD_5daily().setTimeReference(day_begin=begin_day, day_end=end_day,
                                                     dir_in=configDA.obs.GRACE['aux_for_time_epochs'],
                                                     filename=configDA.obs.GRACE.get(
                                                         'ewh_file', 'TUD-L3-5dayEWH-GRACEv2-Hybrid-2002_2017-0.5x0.5.nc'))
        raise ValueError('unknown observation kind: %s' % kind)

    def make_state_envelope(self, variables=None, force=False, comm=None):
        """
        Per-cell min/max of selected storages over the collected open loop (Res/<case>/OL/Ens_1..N), written to
        <Auxiliary_dir>/state_envelope_<basin>.nc. src_DA.Threshold uses it to keep river storage (and, only with
        gw_lower / gw_upper, groundwater) in the analysis within an envelope of what the model produced. Like
        gather_OLmean it needs the OL to be collected; it is skipped when the file exists and holds all variables,
        unless force=True (recompute after a new open loop).
        variables=None: only the storages the threshold of this run actually bounds with the envelope, i.e.
        riverstor when it is part of the DA state; nothing to do (and no file read) otherwise. Until 8 Oct 2026
        the envelope of river storage and groundwater was rebuilt at every DA start, reading two full-grid
        variables of every member and year on rank 0 while the other ranks waited, even when no storage used it.
        With the snow envelope cap on ("envelope_factor" in "snow_bounds") and swe in the state the monthly snow maximum swe_max_month is
        added in the same pass (snow cap, src_DA.Threshold / EnKF_localized).
        comm: all ranks of the communicator call it and each reads its own member (src_DA.state_envelope);
        the file is reused while the open-loop files are unchanged (fingerprint), force=True rebuilds it.
        """
        from src_DA.state_envelope import make_state_envelope
        cfg = self.configDA
        if variables is None:
            variables = ['riverstor'] if cfg.model.layer.get('riverstor', False) else []
        '''monthly snow maximum for the snow cap ("envelope_factor" in "snow_bounds", swe in the DA state)'''
        from src_DA.configure_DA import snow_bounds
        cap_on = snow_bounds(cfg.method)['envelope_factor'] is not None
        monthly = ['swe'] if (cap_on and cfg.model.layer.get('swe', False)) else []
        if not variables and not monthly:
            print('state envelope: not needed (no state variable is bounded by the open-loop envelope)')
            return None
        return make_state_envelope(res_dir=cfg.basic.res_permanent, case=cfg.basic.case, ens=cfg.basic.ensemble,
                                   basin=cfg.basic.basin, out_dir=cfg.basic.Auxiliary_dir, variables=variables,
                                   force=force, comm=comm, monthly=monthly)

    def prepare_design_matrix(self):
        dm = DM_basin_average(layer=self.configDA.model.layer, is_residual=False)
        dm.configure_mask(mask_path=self.configDA.basic.basin_mask)
        dm.vertical_aggregation().horizontal_aggregation()
        # dm.saveDM(out_path=)
        self.dm_included = dm

        dm2 = DM_basin_average(layer=self.configDA.model.layer, is_residual=True)

        dm2.configure_mask(mask_path=self.configDA.basic.basin_mask)
        dm2.vertical_aggregation().horizontal_aggregation()
        # dm.saveDM(out_path=)
        self.dm_excluded = dm2
        pass

    def generate_perturbed_GRACE_obs(self):
        from src_OBS.GRACE_perturbation import GRACE_perturbed_obs

        configDA = self.configDA
        ob = GRACE_perturbed_obs(ens=configDA.basic.ensemble, basin_name=configDA.basic.basin)
        ob.configure_dir(input_dir=configDA.obs.GRACE['preprocess_res'], obs_dir=configDA.obs.dir)
        ob.configure_obs_aux(obs_aux=self._obs_aux())
        ob.perturb_TWS().remove_temporal_mean()
        fn = Path(configDA.obs.GRACE['OL_mean']) / ('%s_%s.hdf5' % (configDA.basic.case, configDA.basic.basin))
        ob.add_temporal_mean(fn=str(fn))
        ob.save()

        pass

    def run_DA(self, rank: int):
        """The main entrance to the data assimilation experiment"""
        import json
        from src_DA.observations import GRACE_obs
        from src_DA.EnKF import EnKF
        # earlier variants (EnKF_localization_v1/v2, EnKF_domain_localization): src_DA.legacy.EnKF_legacy
        # from src_DA.EnSQRA import EnSQRA, EnSQRA_V2

        if rank != 1:
            log_dir = str(Path(__file__).resolve().parent.parent / 'parallel_logs' / 'DA')
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"rank_{rank}.log")
            sys.stdout = open(log_path, 'w', encoding='utf-8', buffering=1)
            # sys.stdout = TqdmLogFilter(log_path)
            sys.stderr = sys.stdout

        # banner AFTER the redirect: rank 1 -> terminal (colour), every other rank -> top of its own log (plain)
        print_pyglda_banner()
        from misc.time_checker_and_ascii_image import check_time

        # print('\n================ Configure DA experiment ===================')

        '''configure DA'''
        configDA = self.configDA

        '''configure model'''
        model_instance = ensemble_model_daily_step(configDA=self.configDA, setting_dir=str(self.setting_dir),
                                                   ensemble_id=rank).get_model_instance()

        '''obtain the GRACE observation'''
        gr = GRACE_obs(basin=self.configDA.basic.basin, dir_obs=configDA.obs.dir, ens_id=rank)

        '''states extract operator'''
        sv_included = EnsStates(DM=self.dm_included, configDA=configDA, ensemble_id=rank)
        sv_excluded = EnsStates(DM=self.dm_excluded, configDA=configDA, ensemble_id=rank)

        '''DA experiment'''
        # # da = DataAssimilation(DA_setting=configDA, model=model_instance, obs=gr, sv=sv)
        # # da = DataAssimilation_monthly(DA_setting=configDA, model=model_instance, obs=gr, sv=sv)
        '''choice of the filter: configDA.method.fusion_method (EnKF_v0 = plain EnKF; EnKF_localized = EnKF assembled
        from localization, inflation and increment-partition components, see src_DA.filter_factory)'''
        from src_DA.EnumDA import FusionMethod
        if configDA.method.fusion_method == FusionMethod.EnKF_localized:
            from src_DA.filter_factory import build_filter
            da = build_filter(configDA, model=model_instance, obs=gr, sv=sv_included, sv_excluded=sv_excluded)
        else:
            da = EnKF(DA_setting=configDA, model=model_instance, obs=gr, sv=sv_included, sv_excluded=sv_excluded)
        # # da = EnSQRA(DA_setting=configDA, model=model_instance, obs=gr, sv=sv)
        # # da = EnSQRA_V2(DA_setting=configDA, model=model_instance, obs=gr, sv=sv)
        # # da = EnKF_localization_v1(DA_setting=configDA, model=model_instance, obs=gr, sv=sv)
        # # da = EnKF_domain_localization(DA_setting=configDA, model=model_instance, obs=gr, sv=sv)
        # # da = EnKF_localization_v2(DA_setting=configDA, model=model_instance, obs=gr, sv=sv)
        # # da = DataAssimilation_monthlymean_dailyupdate(DA_setting=configDA, model=model_instance, obs=gr, sv=sv)
        da.configure_design_matrix(DM=self.dm_included)
        #
        '''running with MPI parallelization'''
        print('User case: %s' % self.case)
        print()
        da.run_mpi()
        print('\nDA finished (rank %s)!\n' % rank)

        pass


def demo1():
    da = DA_GRACE(setting_dir='/media/user/My Book/Fan/WaterGap/Extensions/DA_settings')
    da.configure_setting(ens_size=4, case_name='test', basin_name='Brahmaputra',
                         basin_dir='/media/user/My Book/Fan/WaterGap/Basin')
    da.configure_date(begin_date='2002-01-01', end_date='2005-04-30')
    da.save_configuration()
    da.reload_setting()

    da.prepare_design_matrix()

    da.gather_OLmean()

    da.generate_perturbed_GRACE_obs()
    pass


def demo2():
    from mpi4py import MPI
    """Parallel execution using MPI. Each rank will have its own log file."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # if rank != 0:
    #     log_dir = "./parallel_logs"
    #     os.makedirs(log_dir, exist_ok=True)
    #
    #     log_path = os.path.join(log_dir, f"rank_{rank}.log")
    #
    #     sys.stdout = open(log_path, 'w', encoding='utf-8')
    #     # sys.stdout = TqdmLogFilter(log_path)
    #     sys.stderr = sys.stdout

    if rank == 0:
        print_pyglda_banner()
        from misc.time_checker_and_ascii_image import check_time
        pass
    # rank=1

    da = DA_GRACE(setting_dir='/media/user/My Book/Fan/WaterGap/Extensions/DA_settings')
    da.configure_setting(ens_size=4, case_name='test', basin_name='Brahmaputra',
                         basin_dir='/media/user/My Book/Fan/WaterGap/Basin')
    da.configure_date(begin_date='2002-01-01', end_date='2005-04-30')
    # da.configure_date(begin_date='2002-01-01', end_date='2002-03-31')
    if rank == 0:
        da.save_configuration()
    comm.barrier()

    da.reload_setting()

    if rank == 0:
        da.gather_OLmean()
    comm.barrier()

    da.prepare_design_matrix()

    da.run_DA(rank=rank)


if __name__ == '__main__':
    demo2()
