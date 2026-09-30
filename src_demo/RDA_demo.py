import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src_FlowControl.Regional_DA import RDA

RDA.setting_dir = '/media/user/My Book/Fan/PyGLDA_v2/settings/demo_3'
RDA.ens = 4

RDA.external_data_path = '/media/user/My Book/Fan/PyGLDA_v2_external_data'
RDA.case = 'demo_3'
RDA.basin = 'Amazon'
RDA.shp_path = '/media/user/My Book/Fan/PyGLDA_v2_external_data/Basin/shp/Amazon/Amazon.shp'

'''for spin-up'''
RDA.spin_up_start = '2000-01-01'
RDA.spin_up_end = '2001-12-31'

RDA.spin_up_years = 5

'''for open-loop and data assimilation'''
RDA.sim_begin_time = '2002-01-01'
# RDA.sim_end_time = '2005-04-30'
RDA.sim_end_time = '2016-04-30'


def demo1():

    '''Before open loop'''
    # RDA.config_external_data()
    # RDA.read_config_and_save()
    # RDA.model_perturbation()

    '''before data assimilation'''
    # RDA.config_basin_mask()
    # RDA.get_GRACE_obs(is_diagonal=False)

    '''after data assimilation'''
    RDA.post_processing()
    RDA.visualization()
    RDA.increment_diagnosis()

    '''house-keeping: remove the temporary daily output of a collected stage (dry run first, then dry_run=False)'''
    # from src_DA.EnumDA import Stage
    # RDA.clean_temp_output(Stage.OL, dry_run=True)
    # RDA.clean_temp_output(Stage.DA, dry_run=True)

    pass


def demo2():
    from src_DA.EnumDA import Stage
    from mpi4py import MPI

    # RDA.spin_up()
    # #
    # RDA.OL_run()
    #
    # RDA.collect_and_statistics(Stage.OL, skip_collect=False)

    '''DA of 2026-09-29 (TUD 5-daily, 4 members): run and collection done'''
    try:
        RDA.DA_run()  # or whatever your entry point is
    except Exception:
        import traceback
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
    RDA.collect_and_statistics(Stage.DA, skip_collect=False)

    pass


if __name__ == '__main__':
    # demo1()     # post_processing / visualization: run after the OL statistics below
    demo2()      # mpiexec -n 5 python -u RDA_demo.py Config_ReWaterGAP.json