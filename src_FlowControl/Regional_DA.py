from src_auxiliary.banner import print_pyglda_banner
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
            for keyword in ['/Input_data', '/Ensemble_input', '/output_data', '/Initialization', '/ReWaterGAP', '/Basin'
                                                                                                                '/Res',
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

        GR.basin_TWS(month_begin=t1, month_end=t2, dir_in=dir_in, dir_out=dir_out)
        GR.grid_TWS(month_begin=t1, month_end=t2, dir_in=dir_in, dir_out=dir_out)
        GR.basin_COV(month_begin=t1, month_end=t2, dir_in=cov_dir_in, dir_out=dir_out,
                     is_diagonal=kwargs['is_diagonal'])

        # which spends much time.

        pass

    @staticmethod
    def model_perturbation():
        """This is global-wise perturbation"""

        from src_DA.Perturbation import perturbation
        dp = Path(RDA.setting_dir) / 'perturbation.json'

        '''the begin time should cover the spin-up period, and the end time should cover the simulation period'''
        t1 = datetime.strptime(RDA.spin_up_start, '%Y-%m-%d').strftime('%Y-%m')
        t2 = datetime.strptime(RDA.sim_end_time, '%Y-%m-%d').strftime('%Y-%m')

        pp = perturbation(dp=dp, ens_size=RDA.ens).setDate(month_begin=t1, month_end=t2)
        pp.perturb_par()
        pp.perturb_forcing()

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

            OL.model_spinup()

        except Exception as e:
            import traceback
            print(f"[ERROR] Rank {rank} encountered an exception: {e}")
            traceback.print_exc()
            sys.stdout.flush()
            comm.Abort(1)

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

        except Exception as e:
            import traceback
            print(f"[ERROR] Rank {rank} encountered an exception: {e}")
            traceback.print_exc()
            sys.stdout.flush()
            comm.Abort(1)

        pass

    @staticmethod
    def collect_and_statistics(stage, skip_collect= False):
        from mpi4py import MPI
        import sys, os
        from src_DA.merge_standardize import yearly_merge, Stage
        from src_DA.configure_DA import config_DA
        from src_DA.statistical_analysis import BasinAverageAnalysis

        """Parallel execution using MPI. Each rank will have its own log file."""
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        assert size == (RDA.ens + 1), 'Not enough threads for parallelization! Required threads are %s' % (RDA.ens + 1)

        if rank != 1:
            log_dir = str(Path(__file__).resolve().parent.parent / 'parallel_logs' / 'collect')
            os.makedirs(log_dir, exist_ok=True)

            log_path = os.path.join(log_dir, f"rank_{rank}.log")

            sys.stdout = open(log_path, 'w', encoding='utf-8')
            # sys.stdout = TqdmLogFilter(log_path)
            sys.stderr = sys.stdout

        configDA = config_DA.loadjson(Path(RDA.setting_dir) / 'DA_setting.json').process()

        '''the first stage: reorganize the data over specific area'''
        state_dir = configDA.basic.OL_output_temp_dir if stage == Stage.OL else configDA.basic.DA_output_temp_dir

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

        except Exception as e:
            print(f"[ERROR] Rank {rank} encountered an exception: {e}")
            import traceback
            traceback.print_exc()

            # force exist= 1
            comm.Abort(1)

        '''the second stage: do statistical analysis and save it'''
        analysis = BasinAverageAnalysis(
            basin_mask_path=configDA.basic.basin_mask,
            state_dir=configDA.basic.res_permanent, stage=stage, ens_id=rank, case_name=RDA.case)

        try:
            analysis.load_nc_files(start_date=RDA.sim_begin_time, end_date=RDA.sim_end_time).\
                select_variables().get_basin_average()
            pass
        except Exception as e:
            print(f"[ERROR] Rank {rank} encountered an exception: {e}")
            import traceback
            traceback.print_exc()

            # force exist= 1
            comm.Abort(1)

        pass

    @staticmethod
    def DA_run():
        from mpi4py import MPI
        from src_FlowControl.DA_GRACE import DA_GRACE

        """Parallel execution using MPI. Each rank will have its own log file."""
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 1:
            print_pyglda_banner()
            pass

        da = DA_GRACE(setting_dir=Path(RDA.setting_dir))
        da.configure_setting(ens_size=RDA.ens, case_name=RDA.case, basin_name=RDA.basin,
                             basin_dir=Path(RDA.external_data_path) / 'Basin')
        da.configure_date(begin_date=RDA.sim_begin_time, end_date=RDA.sim_end_time)
        # da.configure_date(begin_date='2002-01-01', end_date='2002-03-31')
        if rank == 0:
            da.save_configuration()
        comm.barrier()

        da.reload_setting()

        if rank == 0:
            da.gather_OLmean()
        comm.barrier()

        da.prepare_design_matrix()

        try:
            da.run_DA(rank=rank)
            pass
        except Exception as e:
            print(f"[ERROR] Rank {rank} encountered an exception: {e}")
            import traceback
            traceback.print_exc()

            # force exist= 1
            comm.Abort(1)

        pass

    @staticmethod
    def post_processing():
        """
        post-processing of the data assimilation and open loop results
        Returns
        -------

        """

        from src_DA.configure_DA import config_DA
        from src_DA.statistical_analysis import BasinAverageAnalysis_post, Stage

        configDA = config_DA.loadjson(Path(RDA.setting_dir) / 'DA_setting.json').process()

        bp = BasinAverageAnalysis_post(ens=RDA.ens, case=RDA.case, basin=RDA.basin, date_begin=RDA.sim_begin_time,
                                       date_end=RDA.sim_end_time)

        for stage in [Stage.OL, Stage.DA]:
            bp.get_states(dir=configDA.basic.res_permanent, stage=stage)
            bp.save_states(save_dir=Path(configDA.basic.res_permanent)/RDA.case, prefix=stage.name)

        '''collect GRACE basin average results'''

        bp.get_GRACE(obs_dir=configDA.obs.dir)
        bp.save_GRACE(prefix=RDA.basin, save_dir=Path(configDA.basic.res_permanent)/RDA.case)

        pass


    @staticmethod
    def visualization():
        from src_postprocessing.Visualization import visualization
        from src_DA.configure_DA import config_DA
        from src_DA.EnumDA import WaterGap_storage_variables

        configDA = config_DA.loadjson(Path(RDA.setting_dir) / 'DA_setting.json').process()
        vv = visualization(configDA=configDA)

        vv.basin_ensemble(allow_pop_up=True, fig_path=Path(configDA.basic.res_permanent)/RDA.case)
        vv.GRACE_OL_DA(allow_pop_up=True, fig_path=Path(configDA.basic.res_permanent)/RDA.case,
                       signal=WaterGap_storage_variables.tws.name)




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


def demo3():
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

    print_pyglda_banner()

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

    # da.run_DA(rank=rank)


if __name__ == '__main__':
    demo1()
    # demo2()
