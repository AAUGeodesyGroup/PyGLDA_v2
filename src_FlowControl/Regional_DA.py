from src_auxiliary.banner import print_pyglda_banner
from src_DA.EnumDA import Stage
from pathlib import Path
from datetime import datetime, timedelta


# ── controller alias: bare-path ↔ full-path ────────────────────────────────
# WaterGAP's internals import controller.* via bare path; our Interface files
# import via src_GHM.ReWaterGAP.controller.*. Without this block Python creates
# two separate module objects so configure_time's updates never reach WaterGAP.
# Fix: load everything via bare path first, then register under the full path
# so both spellings resolve to the same object in sys.modules.
# import sys, importlib, pkgutil
# from pathlib import Path as _Path
#
# _rewatergap = str(_Path(__file__).resolve().parent.parent / 'src_GHM' / 'ReWaterGAP')
# if _rewatergap not in sys.path:
#     sys.path.insert(0, _rewatergap)
#
# import controller as _ctrl
# sys.modules.setdefault('src_GHM.ReWaterGAP.controller', _ctrl)
# for _mi in pkgutil.iter_modules(_ctrl.__path__):
#     _bare_mod = importlib.import_module(f'controller.{_mi.name}')
#     sys.modules[f'src_GHM.ReWaterGAP.controller.{_mi.name}'] = _bare_mod
# # ───────────────────────────────────────────────────────────────────────────


def _abort_with_report(comm, rank, stage):
    """
    Call from an `except` block inside an MPI stage.
    Writes the full traceback to this rank's current stdout (its log file, or the terminal
    for the rank that owns it), then - if this rank's output has been redirected to a log -
    echoes a one-line summary to the real terminal (sys.__stderr__) so a failure on any
    rank is visible without opening the log files. Finally aborts the whole MPI job.
    """
    import sys, traceback
    exc_type, exc, _ = sys.exc_info()
    print(f"[ERROR] Rank {rank} encountered an exception: {exc}")
    traceback.print_exc()
    sys.stdout.flush()
    if sys.stdout is not sys.__stdout__:          # output redirected -> terminal has seen nothing yet
        print(f"[ERROR] rank {rank} ({stage}): {exc_type.__name__}: {exc}  "
              f"-- full traceback in parallel_logs/{stage}/rank_{rank}.log",
              file=sys.__stderr__, flush=True)
    comm.Abort(1)


class RDA:
    setting_dir = '/media/user/My Book/Fan/PyGLDA_v2/settings/demo_1'
    ens = 4

    external_data_path = '/media/user/My Book/Fan/PyGLDA_v2_external_data'
    case = 'demo_1'
    basin = 'Brahmaputra'
    shp_path = '/media/user/My Book/Fan/PyGLDA_v2_external_data/Basin/shp/Brahmaputra/Brahmaputra.shp'

    '''for spin-up'''
    spin_up_start = '2000-01-01'
    spin_up_end = '2001-12-31'

    spin_up_years = 5

    '''for open-loop and data assimilation'''
    sim_begin_time = '2002-05-01'
    sim_end_time = '2005-04-30'

    @staticmethod
    def config_external_data():
        import json

        def reformulate(st: str):
            new = None
            for keyword in ['/Input_data', '/Ensemble_input', '/output_data', '/Initialization', '/ReWaterGAP', '/Basin', '/Res',
                            '/OL_output', '/DA_output', '/Ensemble_Initialization', '/Auxiliary',
                            '/GRACE']:
                if keyword in st:
                    # print(keyword)
                    # print(st)
                    new = RDA.external_data_path + keyword + st.split(keyword)[1]
                    # print(new)
                    break

            return new

        def func(data):
            if isinstance(data, dict):
                for key, vv in data.items():
                    if isinstance(vv, str):
                        new = reformulate(vv)
                        if new is not None:
                            data[key] = new
                    elif isinstance(vv, (dict, list)):
                        func(vv)

            elif isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, str):
                        new = reformulate(item)
                        if new is not None:
                            data[i] = new
                    elif isinstance(item, (dict, list)):
                        func(item)

        for file_name in ['DA_setting.json', 'perturbation.json', 'Config_ReWaterGAP.json']:
            print(file_name)
            config = json.load(open(Path(RDA.setting_dir) / file_name, 'r'))
            # print(Path(RDA.setting_dir) / file_name)
            func(config)
            with open(Path(RDA.setting_dir) / file_name, 'w') as f:
                json.dump(config, f, indent=4)

        pass

    @staticmethod
    def read_config_and_save():
        from src_DA.configure_DA import config_DA

        configDA = config_DA.loadjson(Path(RDA.setting_dir) / 'DA_setting.json').process()

        configDA.basic.case = RDA.case
        configDA.basic.basin = RDA.basin
        configDA.basic.basin_shp = RDA.shp_path
        configDA.basic.ensemble = RDA.ens
        configDA.basic.basin_mask = str(Path(RDA.external_data_path) / 'Basin/mask' / RDA.basin / ('%s_res_0.5.h5' % RDA.basin))

        configDA.save_json(save_path=Path(RDA.setting_dir) / 'DA_setting.json')

        pass

    @staticmethod
    def config_basin_mask():
        from src_auxiliary.shp2mask import basin_shp_process

        '''for WaterGAP'''
        basin_shp = basin_shp_process(save_dir=Path(RDA.external_data_path) / 'Basin/mask',
                                      basin_name=RDA.basin, res=0.5)
        basin_shp.shp_to_mask(shp_path=RDA.shp_path, issave=True)

        '''for GRACE'''
        basin_shp = basin_shp_process(save_dir=Path(RDA.external_data_path) / 'Basin/mask',
                                      basin_name=RDA.basin, res=1)
        basin_shp.shp_to_mask(shp_path=RDA.shp_path, issave=True)

        pass

    @staticmethod
    def get_GRACE_obs(**kwargs):

        """preparation for GRACE and forcing fields for global tiles"""
        from src_OBS.prepare_GRACE_mascon import GRACE_CSR_mascon
        from src_DA.configure_DA import config_DA
        GR = GRACE_CSR_mascon(basin_name=RDA.basin, shp_path=RDA.shp_path)

        GR.configure_global_land_ocean_mask(
            fn=Path(RDA.external_data_path) / 'GRACE/global_mask/GlobalLandMaskForGRACE.hdf5')
        GR.generate_mask(save_dir=Path(RDA.external_data_path) / 'Basin/mask')

        configDA = config_DA.loadjson(Path(RDA.setting_dir) / 'DA_setting.json').process()
        t1 = datetime.strptime(RDA.sim_begin_time, '%Y-%m-%d').strftime('%Y-%m')
        t2 = datetime.strptime(RDA.sim_end_time, '%Y-%m-%d').strftime('%Y-%m')
        dir_in = configDA.obs.GRACE['EWH_grid_dir']
        cov_dir_in = configDA.obs.GRACE['cov_dir']
        dir_out = configDA.obs.GRACE['preprocess_res']

        kind = configDA.obs.GRACE.get('kind', 'Mascon_monthly')
        if kind == 'TUD_5daily':
            '''TU Delft 5-daily/weekly hybrid product (gridded EWH + per-cell uncertainty). Optional keys in
            DA_setting.json -> obs.GRACE: ewh_file, unc_file, unit ('cm'), corr_length_km (300), add_gia (false).
            EWH_grid_dir must hold both netCDF files; cov_dir is not used.'''
            from src_OBS.prepare_GRACE_TUD import GRACE_TUD_5daily
            g = configDA.obs.GRACE
            GR = GRACE_TUD_5daily(basin_name=RDA.basin, shp_path=RDA.shp_path)
            GR.configure_global_land_ocean_mask(
                fn=Path(RDA.external_data_path) / 'GRACE/global_mask/GlobalLandMaskForGRACE.hdf5')
            GR.generate_mask(save_dir=Path(RDA.external_data_path) / 'Basin/mask')
            GR.set_extra_info(dir_in=dir_in,
                              ewh_file=g.get('ewh_file', 'TUD-L3-5dayEWH-GRACEv2-Hybrid-2002_2017-0.5x0.5.nc'),
                              unc_file=g.get('unc_file', 'TUD-L3Uncorr-5dayEWH_UNC-GRACEv2-Hybrid-2002_2017-0.5x0.5.nc'),
                              unit=g.get('unit', 'cm'), corr_length_km=g.get('corr_length_km', 300.0),
                              add_gia=g.get('add_gia', False))
            GR.inspect(day_begin=RDA.sim_begin_time, day_end=RDA.sim_end_time)
            GR.basin_TWS(day_begin=RDA.sim_begin_time, day_end=RDA.sim_end_time, dir_out=dir_out)
            GR.grid_TWS(day_begin=RDA.sim_begin_time, day_end=RDA.sim_end_time, dir_out=dir_out)
            GR.basin_COV(day_begin=RDA.sim_begin_time, day_end=RDA.sim_end_time, dir_out=dir_out,
                         is_diagonal=kwargs['is_diagonal'])
            return

        GR.basin_TWS(month_begin=t1, month_end=t2, dir_in=dir_in, dir_out=dir_out)
        GR.grid_TWS(month_begin=t1, month_end=t2, dir_in=dir_in, dir_out=dir_out)
        GR.basin_COV(month_begin=t1, month_end=t2, dir_in=cov_dir_in, dir_out=dir_out,
                     is_diagonal=kwargs['is_diagonal'])

        # which spends much time.

        pass

    @staticmethod
    def model_perturbation():
        """
        Global perturbation of parameters and forcing -> Ensemble_input/Ens_k (k = 0..ens, Ens_0 unperturbed).
        MPI stage (mpiexec -n ens+1): each rank writes its own member; rank 0 draws the random numbers and
        broadcasts them. Rank 1 prints the progress, the other ranks log to parallel_logs/perturbation/.
        """
        from mpi4py import MPI
        import sys, os
        from src_DA.Perturbation import perturbation

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        assert size == (RDA.ens + 1), 'Not enough threads for parallelization! Required threads are %s' % (RDA.ens + 1)

        if rank != 1:
            log_dir = str(Path(__file__).resolve().parent.parent / 'parallel_logs' / 'perturbation')
            os.makedirs(log_dir, exist_ok=True)
            sys.stdout = open(os.path.join(log_dir, f"rank_{rank}.log"), 'w', encoding='utf-8', buffering=1)
            sys.stderr = sys.stdout

        dp = Path(RDA.setting_dir) / 'perturbation.json'

        '''the begin time should cover the spin-up period, and the end time should cover the simulation period'''
        t1 = datetime.strptime(RDA.spin_up_start, '%Y-%m-%d').strftime('%Y-%m')
        t2 = datetime.strptime(RDA.sim_end_time, '%Y-%m-%d').strftime('%Y-%m')

        try:
            if rank == 1:
                print('\n=================== Model perturbation ===================')
                print('%d members (+ unperturbed) | %s to %s | %d ranks' % (RDA.ens, t1, t2, size), flush=True)
            pp = perturbation(dp=dp, ens_size=RDA.ens).setDate(month_begin=t1, month_end=t2)
            pp.perturb_par(comm=comm)
            pp.perturb_forcing(comm=comm)
        except Exception:
            _abort_with_report(comm, rank, 'perturbation')

        comm.barrier()
        if rank == 1:
            print('Model perturbation finished: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'), flush=True)
        pass

    @staticmethod
    def spin_up():
        from mpi4py import MPI
        import sys, os
        from src_FlowControl.OpenLoop import OpenLoop
        from src_DA.configure_DA import config_DA

        """Parallel execution using MPI. Each rank will have its own log file."""
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        assert size == (RDA.ens + 1), 'Not enough threads for parallelization! Required threads are %s' % (RDA.ens + 1)

        if rank != 0:
            log_dir = str(Path(__file__).resolve().parent.parent / 'parallel_logs' / 'spin_up')
            os.makedirs(log_dir, exist_ok=True)

            log_path = os.path.join(log_dir, f"rank_{rank}.log")

            # buffering=1 enables line-buffering so every print is flushed
            # immediately — errors are captured even if comm.Abort(1) fires
            sys.stdout = open(log_path, 'w', encoding='utf-8', buffering=1)
            sys.stderr = sys.stdout

        try:
            from misc.time_checker_and_ascii_image import check_time

            configDA = config_DA.loadjson(Path(RDA.setting_dir) / 'DA_setting.json').process()

            OL = OpenLoop(ensemble_id=rank, setting_dir=RDA.setting_dir)
            OL.configure_time(begin_time=RDA.spin_up_start, end_time=RDA.spin_up_end)
            OL.configure_Ens_output(output_dir=configDA.basic.OL_output_temp_dir)
            OL.configure_Ens_input(input_dir=configDA.basic.Ensemble_input_dir)
            OL.configure_ini_for_resume(save_init_dir=configDA.basic.Ensemble_ini_dir,
                                        read_init_dir=configDA.basic.Ensemble_ini_dir)

            OL.model_spinup(spinup_years=RDA.spin_up_years)

        except Exception:
            _abort_with_report(comm, rank, 'spin_up')

        pass

    @staticmethod
    def OL_run():
        from mpi4py import MPI
        import sys, os
        from src_FlowControl.OpenLoop import OpenLoop
        from src_DA.configure_DA import config_DA

        """Parallel execution using MPI. Each rank will have its own log file."""
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        assert size == (RDA.ens + 1), 'Not enough threads for parallelization! Required threads are %s' % (RDA.ens + 1)

        if rank != 0:
            log_dir = str(Path(__file__).resolve().parent.parent / 'parallel_logs' / 'OL')
            os.makedirs(log_dir, exist_ok=True)

            log_path = os.path.join(log_dir, f"rank_{rank}.log")

            # buffering=1 enables line-buffering so every print is flushed
            # immediately — errors are captured even if comm.Abort(1) fires
            sys.stdout = open(log_path, 'w', encoding='utf-8', buffering=1)
            sys.stderr = sys.stdout

        try:
            configDA = config_DA.loadjson(Path(RDA.setting_dir) / 'DA_setting.json').process()

            OL = OpenLoop(ensemble_id=rank, setting_dir=RDA.setting_dir)
            OL.configure_time(begin_time=RDA.sim_begin_time, end_time=RDA.sim_end_time)
            OL.configure_Ens_output(output_dir=configDA.basic.OL_output_temp_dir)
            OL.configure_Ens_input(input_dir=configDA.basic.Ensemble_input_dir)
            OL.configure_ini_for_resume(save_init_dir=configDA.basic.Ensemble_ini_dir,
                                        read_init_dir=configDA.basic.Ensemble_ini_dir)

            OL.model_resume()

        except Exception:
            _abort_with_report(comm, rank, 'OL')

        pass

    @staticmethod
    def collect_and_statistics(stage, skip_collect=False, post_process=True):
        """
        MPI stage after OL_run / DA_run (mpiexec -n ens+1):
          step 1  daily output -> yearly files Res/<case>/<stage>/Ens_k/daily_output_<year>.nc (skip_collect=True:
                  use the existing yearly files)
          step 2  basin / sub-basin series -> Res/<case>/<stage>/Ens_k/basin_ts_<stage>.h5
          step 3  (post_process=True) post-processing of this stage:
                  3a  harmonic maps Harmonic_<stage>.nc on all ranks (members spread over the ranks, rank 1 writes;
                      every member is fitted, see statistical_analysis.HarmonicMapAnalysis)
                  3b  RDA.post_processing(stages=[stage], harmonic=False) on rank 1: Res_<stage>.h5 and, for the DA,
                      the GRACE files; the other ranks wait.
                  A failure in step 3 is reported but does not abort the job (the collected files are complete);
                  rerun it serially with RDA.post_processing(stages=[stage]).
        """
        from mpi4py import MPI
        import sys, os
        from src_postprocessing.merge_standardize import yearly_merge, Stage
        from src_DA.configure_DA import config_DA
        from src_postprocessing.statistical_analysis import BasinAverageAnalysis

        """Parallel execution using MPI. Each rank will have its own log file."""
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        assert size == (RDA.ens + 1), 'Not enough threads for parallelization! Required threads are %s' % (RDA.ens + 1)

        if rank != 1:
            log_dir = str(Path(__file__).resolve().parent.parent / 'parallel_logs' / 'collect')
            os.makedirs(log_dir, exist_ok=True)

            log_path = os.path.join(log_dir, f"rank_{rank}.log")

            # buffering=1 enables line-buffering so every print is flushed
            # immediately - otherwise the log is empty if the process is killed
            sys.stdout = open(log_path, 'w', encoding='utf-8', buffering=1)
            # sys.stdout = TqdmLogFilter(log_path)
            sys.stderr = sys.stdout

        configDA = config_DA.loadjson(Path(RDA.setting_dir) / 'DA_setting.json').process()

        '''the first stage: reorganize the data over specific area'''
        state_dir = configDA.basic.OL_output_temp_dir if stage == Stage.OL else configDA.basic.DA_output_temp_dir

        if rank == 1:
            print('\n=================== Collection and statistics: %s ===================' % stage.name)
            print('case %s | basin %s | %s to %s | %d members (+ unperturbed)' %
                  (RDA.case, RDA.basin, RDA.sim_begin_time, RDA.sim_end_time, RDA.ens))
            print('step 1/2: %s daily output from %s into yearly files under %s' %
                  ('skipped (skip_collect=True), using existing yearly files' if skip_collect else 'merging',
                   state_dir, Path(configDA.basic.res_permanent) / RDA.case / stage.name))

        ps = yearly_merge(basin_mask_dir=Path(RDA.external_data_path) / 'Basin/mask', basin_name=RDA.basin,
                          state_dir=state_dir,
                          output_dir=configDA.basic.res_permanent, stage=stage, case_name=RDA.case)

        ps.configure_ensID(ens_id=rank).configure_date(begin_date=RDA.sim_begin_time, end_date=RDA.sim_end_time)

        try:
            if not skip_collect:
                if stage == Stage.OL:
                    ps.merge_and_crop_by_year_with_mask()
                else:
                    ps.merge_by_year()
                pass

        except Exception:
            _abort_with_report(comm, rank, 'collect')

        comm.barrier()

        '''the second stage: do statistical analysis and save it'''
        if rank == 1:
            print('step 2/2: basin-average time series of all storages -> Res/%s/%s/Ens_k/basin_ts_%s.h5' %
                  (RDA.case, stage.name, stage.name))
        analysis = BasinAverageAnalysis(
            basin_mask_path=configDA.basic.basin_mask,
            state_dir=configDA.basic.res_permanent, stage=stage, ens_id=rank, case_name=RDA.case)

        try:
            analysis.load_nc_files(start_date=RDA.sim_begin_time, end_date=RDA.sim_end_time).\
                select_variables().get_basin_average()
            pass
        except Exception:
            _abort_with_report(comm, rank, 'collect')

        comm.barrier()
        if rank == 1:
            print('Collection and statistics of %s finished: %s' % (stage.name, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

        '''the third stage: post-processing of this stage'''
        if post_process:
            if rank == 1:
                print('step 3/3: post-processing %s -> Res/%s/Harmonic_%s.nc (all ranks), Res_%s.h5%s (rank 1)' %
                      (stage.name, RDA.case, stage.name, stage.name, ', GRACE files' if stage == Stage.DA else ''))
            '''3a: harmonic maps, members spread over all ranks (an error is raised on every rank, so no hang)'''
            try:
                RDA.harmonic_fit(stage, comm=comm, root=1)
            except Exception:
                import traceback
                traceback.print_exc()
                if rank == 1:
                    print('[WARNING] harmonic maps of %s failed; rerun serially: RDA.harmonic_fit(Stage.%s)' %
                          (stage.name, stage.name), flush=True)
            comm.barrier()
            '''3b: basin series and GRACE files on rank 1'''
            if rank == 1:
                try:
                    RDA.post_processing(stages=[stage], harmonic=False)
                    print('Post-processing of %s finished: %s' % (stage.name, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                except Exception:
                    import traceback
                    traceback.print_exc()
                    print('[WARNING] post-processing of %s failed; the collected files are complete. Rerun serially: '
                          'RDA.post_processing(stages=[Stage.%s])' % (stage.name, stage.name), flush=True)
            comm.barrier()

        pass

    @staticmethod
    def clean_temp_output(stage, dry_run=True, force=False):
        """
        Remove the temporary daily output (daily_output_YYYY-MM-DD.nc) of a finished stage.
        The folders are taken from DA_setting.json of the current case:
            Stage.OL -> configDA.basic.OL_output_temp_dir / Ens_k
            Stage.DA -> configDA.basic.DA_output_temp_dir / <case> / Ens_k
        The stage must already be collected (Res/<case>/<stage>/Ens_k/basin_ts_<stage>.h5 present for
        every member), otherwise nothing is deleted unless force=True.
        dry_run=True (default) only reports the files and their size. Run it as a single process
        (not under mpiexec), e.g. from demo1() after collect_and_statistics() has finished.
        """
        from src_DA.configure_DA import config_DA
        from src_auxiliary.clean_temp_output import clean_daily_output

        if stage not in (Stage.OL, Stage.DA):
            raise ValueError('clean_temp_output: stage must be Stage.OL or Stage.DA, got %r' % (stage,))
        configDA = config_DA.loadjson(Path(RDA.setting_dir) / 'DA_setting.json').process()
        temp_dir = configDA.basic.OL_output_temp_dir if stage == Stage.OL else configDA.basic.DA_output_temp_dir
        res_dir = Path(configDA.basic.res_permanent) / RDA.case

        print('Cleaning temporary %s output of case %s' % (stage.name, RDA.case))
        return clean_daily_output(temp_dir=temp_dir, res_dir=res_dir, stage=stage, case=RDA.case,
                                  dry_run=dry_run, force=force)

    @staticmethod
    def DA_run():
        from mpi4py import MPI
        from src_FlowControl.DA_GRACE import DA_GRACE

        """Parallel execution using MPI. Each rank will have its own log file."""
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # banner is printed inside da.run_DA(), after each rank's stdout redirect,
        # so it reaches the terminal (rank 1) and the top of every rank log

        da = DA_GRACE(setting_dir=Path(RDA.setting_dir), case=RDA.case)
        da.configure_setting(ens_size=RDA.ens, case_name=RDA.case, basin_name=RDA.basin,
                             basin_dir=Path(RDA.external_data_path) / 'Basin', basin_shp=RDA.shp_path)
        da.configure_date(begin_date=RDA.sim_begin_time, end_date=RDA.sim_end_time)
        # da.configure_date(begin_date='2002-01-01', end_date='2002-03-31')
        if rank == 0:
            da.save_configuration()
        comm.barrier()

        da.reload_setting()

        if rank == 0:
            da.gather_OLmean()

            da.generate_perturbed_GRACE_obs()

            '''open-loop envelope for river storage and groundwater (see src_DA.Threshold): rebuilt from the collected
            open loop at every DA start, so that it always matches the current case, basin and period'''
            da.make_state_envelope(force=True)

        comm.barrier()

        da.prepare_design_matrix()

        try:
            da.run_DA(rank=rank)
            pass
        except Exception:
            _abort_with_report(comm, rank, 'DA')

        pass

    @staticmethod
    def harmonic_fit(stage, comm=None, root=0):
        """
        Per-grid-cell harmonic maps of one stage -> Res/<case>/Harmonic_<stage>.nc (HarmonicMapAnalysis).
        Every member is fitted (per-member maps, ensemble spread) plus the ensemble mean; the options
        (per_member, n_workers) stay in statistical_analysis.HarmonicMapAnalysis.
        comm : MPI communicator (all its ranks call this; members are spread over the ranks, `root` writes)
        """
        from src_DA.configure_DA import config_DA
        from src_postprocessing.statistical_analysis import HarmonicMapAnalysis
        if not isinstance(stage, Stage) or stage not in (Stage.OL, Stage.DA):
            raise ValueError('harmonic_fit: stage must be Stage.OL or Stage.DA, got %r' % (stage,))
        configDA = config_DA.loadjson(Path(RDA.setting_dir) / 'DA_setting.json').process()
        hm = HarmonicMapAnalysis(res_dir=configDA.basic.res_permanent, case=RDA.case, ens=RDA.ens,
                                 date_begin=RDA.sim_begin_time, date_end=RDA.sim_end_time)
        return hm.run(stage, comm=comm, root=root)

    @staticmethod
    def post_processing(stages=(Stage.OL, Stage.DA), grace=None, harmonic=True):
        """
        Post-processing of one or both stages, from their collected results (Res/<case>/<stage>/Ens_k/):
            Res/<case>/Res_<stage>.h5         basin / sub-basin daily series of all storages, all members
            Res/<case>/Harmonic_<stage>.nc    per-cell trend, annual and semi-annual cycle maps
        and, with grace=True, GRACE_<basin>.h5 and Harmonic_GRACE.nc (GRACE basin series and maps).
        It runs automatically at the end of collect_and_statistics(stage) (post_process=True, rank 1), so in
        the normal workflow it is not called by hand. Call it directly only to rerun it serially, e.g. after a
        failure or a code change:  RDA.post_processing(stages=[Stage.OL])  /  RDA.post_processing(stages=[Stage.DA])
            Res_OL.h5 is needed by the DA when the inflation split is "ol_spread" or "ol_variability";
            the DA stage includes GRACE, whose observation file is written at the start of the DA.
        stages   : Stage.OL, Stage.DA or both (a single Stage or a list); grace: None = with the DA stage
        harmonic : False skips Harmonic_<stage>.nc (collect_and_statistics makes it in parallel, RDA.harmonic_fit);
        """
        from src_DA.configure_DA import config_DA
        from src_postprocessing.statistical_analysis import BasinAverageAnalysis_post, HarmonicMapAnalysis

        if isinstance(stages, Stage):
            stages = [stages]
        stages = list(stages)
        bad = [s for s in stages if s not in (Stage.OL, Stage.DA)]
        if bad:
            raise ValueError('post_processing: stages must be Stage.OL and/or Stage.DA, got %s' % bad)
        if grace is None:
            grace = Stage.DA in stages

        configDA = config_DA.loadjson(Path(RDA.setting_dir) / 'DA_setting.json').process()
        res_case = Path(configDA.basic.res_permanent) / RDA.case

        bp = BasinAverageAnalysis_post(ens=RDA.ens, case=RDA.case, basin=RDA.basin, date_begin=RDA.sim_begin_time,
                                       date_end=RDA.sim_end_time)
        hm = HarmonicMapAnalysis(res_dir=configDA.basic.res_permanent, case=RDA.case, ens=RDA.ens,
                                 date_begin=RDA.sim_begin_time, date_end=RDA.sim_end_time)

        for stage in stages:
            '''basin / sub-basin time series -> Res/<case>/Res_<stage>.h5'''
            bp.get_states(dir=configDA.basic.res_permanent, stage=stage)
            bp.save_states(save_dir=res_case, prefix=stage.name)
            '''per-grid-cell harmonic analysis of the yearly files -> Res/<case>/Harmonic_<stage>.nc'''
            if harmonic:
                hm.run(stage)
            print('post-processing %s done: Res_%s.h5%s' % (stage.name, stage.name,
                                                            ', Harmonic_%s.nc' % stage.name if harmonic else ''))

        if grace:
            '''GRACE basin series -> GRACE_<basin>.h5, gridded GRACE TWS (from get_GRACE_obs) -> Harmonic_GRACE.nc'''
            bp.get_GRACE(obs_dir=configDA.obs.dir)
            bp.save_GRACE(prefix=RDA.basin, save_dir=res_case)
            hm.run_GRACE(grace_dir=configDA.obs.GRACE['preprocess_res'], basin=RDA.basin,
                         basin_mask_path=configDA.basic.basin_mask,
                         land_mask_path=Path(RDA.external_data_path) / 'GRACE/global_mask/GlobalLandMaskForGRACE.hdf5')
            print('post-processing GRACE done: GRACE_%s.h5, Harmonic_GRACE.nc' % RDA.basin)


    @staticmethod
    def visualization():
        from src_postprocessing.Visualization import visualization
        from src_DA.configure_DA import config_DA
        from src_DA.EnumDA import WaterGap_storage_variables, Stage

        configDA = config_DA.loadjson(Path(RDA.setting_dir) / 'DA_setting.json').process()
        vv = visualization(configDA=configDA)

        fig_path = Path(configDA.basic.res_permanent) / RDA.case
        '''ensemble time series of the storage compartments, one figure per stage (Components_<stage>.png);
        panels with a range below min_range [mm] (e.g. constant global lakes) are skipped, ncol sets the layout'''
        for stage in [Stage.OL, Stage.DA]:
            vv.basin_ensemble(allow_pop_up=False, fig_path=fig_path, stage=stage, ncol=2, min_range=1.0)
        vv.GRACE_OL_DA(allow_pop_up=False, fig_path=fig_path, signal=WaterGap_storage_variables.tws.name)
        '''evaluation on the observation windows: OL vs DA vs GRACE, innovation/residual with sigma, spread;
        basin_id=0 is the whole basin, k a sub-basin; zoom adds a daily close-up (figures/DA_eval_*.png,
        statistics for all sub-basins in figures/DA_eval_stats.json)'''
        vv.DA_evaluation(fig_path=fig_path, basin_id=0, zoom=None)
        '''2-D maps of trend / annual amplitude / annual peak day: OL | DA | GRACE'''
        vv.harmonic_maps(allow_pop_up=False, fig_path=Path(configDA.basic.res_permanent)/RDA.case,
                         variable=WaterGap_storage_variables.tws.name, style='pixel')

    @staticmethod
    def increment_diagnosis(tag=None, event_threshold_mm=40.0):
        """
        Diagnosis of the assimilation increments from the collected basin time series (needs post_processing):
        A. mean increment per window vs drift inside the windows, per storage (filter-model tug of war)
        B. windows with a basin-mean increment above event_threshold_mm in one storage, sub-basin breakdown
        C. cross-member correlation of every storage with TWS in OL and DA
        D. yearly maxima / minima of the window-mean TWS anomaly (OL, DA, GRACE)
        Written to Res/<case>/figures/da_increment_diagnosis_<tag>.{txt,json} and da_increments_<tag>.png.
        """
        from src_DA.configure_DA import config_DA
        from src_postprocessing.da_increment_diagnosis import run_diagnosis

        configDA = config_DA.loadjson(Path(RDA.setting_dir) / 'DA_setting.json').process()
        res_dir = Path(configDA.basic.res_permanent) / RDA.case
        return run_diagnosis(res_dir=res_dir, obs_file=Path(configDA.obs.dir) / ('%s_obs_GRACE.hdf5' % RDA.basin),
                             out_dir=res_dir / 'figures', tag=tag or RDA.case, nens=RDA.ens,
                             start=datetime.strptime(RDA.sim_begin_time, '%Y-%m-%d').date(),
                             event_threshold_mm=event_threshold_mm)

    @staticmethod
    def da_evaluation(tag=None, reference='previous', extra_files=None, thresholds=None):
        """
        Standard evaluation report of the run (needs post_processing; best after increment_diagnosis, whose output
        is then archived with it). Written to Res/<case>/evaluation/<tag>/:
          summary.txt / summary.json   fit to GRACE (overall, by season, per sub-basin, key months), trends,
                                       gain, forecast spread, chi^2, comparison with the reference run, warnings
          fig1..fig5                   basin series, seasonal correction per storage, groundwater drift,
                                       sub-basins, filter health
          DA_setting.json, logs/       copies of the configuration and of the filter / clipping / adaptive-inflation
                                       logs and the DA rank logs (parallel_logs/DA/rank_k.log); DA_output/<case> is
                                       temporary and overwritten by the next run
        and one row in Res/<case>/evaluation/runs_index.csv.
        tag       : name of the run, e.g. 'run10' (None: date and time)
        reference : tag of an earlier report, 'previous' (last row of runs_index.csv) or None
        """
        from src_DA.configure_DA import config_DA
        from src_postprocessing.da_evaluation import evaluate_run

        configDA = config_DA.loadjson(Path(RDA.setting_dir) / 'DA_setting.json').process()
        res_dir = Path(configDA.basic.res_permanent) / RDA.case
        return evaluate_run(res_dir=res_dir, obs_file=Path(configDA.obs.dir) / ('%s_obs_GRACE.hdf5' % RDA.basin),
                            out_root=res_dir / 'evaluation', tag=tag, basin=RDA.basin,
                            da_output_dir=Path(configDA.basic.DA_output_temp_dir) / RDA.case,
                            setting_file=Path(RDA.setting_dir) / 'DA_setting.json', reference=reference,
                            start=datetime.strptime(RDA.sim_begin_time, '%Y-%m-%d').date(),
                            thresholds=thresholds, figures_dir=res_dir / 'figures',
                            extra_files=list(extra_files or []) +
                            sorted((Path(__file__).resolve().parents[1] / 'parallel_logs' / 'DA').glob('*.log')))

    @staticmethod
    def export_product(version='v1.0'):
        """
        Data product of the case for distribution (called by hand, only when the data are shared; needs
        collect_and_statistics of OL and DA). Written to Res/<case>/product/PyGLDA-v2_<case>_<version>/:
          gridded/     daily ensemble mean and spread of every variable (OL, DA), one file per year, and the
                       gridded GRACE TWS (monthly)
          timeseries/  basin and sub-basin daily series (OL, DA) and the GRACE TWS assimilated
          ancillary/   cell area, basin mask, sub-basin id, per-cell TWS trend, cell_flag
          README.md
        Further options (output folder, anomaly baseline): src_postprocessing.export_product.product_export
        """
        from src_postprocessing.export_product import product_export

        return product_export(setting_dir=RDA.setting_dir, version=version).run()




def demo1():

    '''Before open loop'''
    # RDA.config_external_data()
    # RDA.read_config_and_save()
    RDA.model_perturbation()

    '''before data assimilation'''
    # RDA.config_basin_mask()  # pygmt
    # RDA.get_GRACE_obs(is_diagonal=False) # pygmt

    '''after data assimilation'''
    # RDA.post_processing()
    # RDA.visualization()

    pass


def demo2():
    from src_DA.EnumDA import Stage

    RDA.spin_up()
    #
    # RDA.OL_run()

    # RDA.collect_and_statistics(Stage.OL, skip_collect=False)

    # try:
    #     RDA.DA_run()  # or whatever your entry point is
    # except Exception:
    #     import traceback
    #     traceback.print_exc()
    #     MPI.COMM_WORLD.Abort(1)
    #
    # RDA.collect_and_statistics(Stage.DA, skip_collect=False)

    pass


if __name__ == '__main__':
    demo1()
    # demo2()
