"""PyGLDA v2 driver - local workstation. Remote twin: RDA_Ucloud.py"""
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent          # the PyGLDA_v2 folder
sys.path.insert(0, str(CODE_ROOT))

from src_FlowControl.Regional_DA import RDA
from src_DA.EnumDA import Stage

DATA_ROOT = '/media/user/My Book/Fan/PyGLDA_v2_external_data'

RDA.case = 'demo_2'                                          # local 4-member Danube case (runs 10 - 11a2)
RDA.basin = 'Danube'
RDA.ens = 4
RDA.setting_dir = str(CODE_ROOT / 'settings' / RDA.case)
RDA.external_data_path = DATA_ROOT
RDA.shp_path = DATA_ROOT + '/Basin/shp/Danube/Danube.shp'

'''for spin-up'''
RDA.spin_up_start = '2000-01-01'
RDA.spin_up_end = '2001-12-31'
RDA.spin_up_years = 5

'''for open-loop and data assimilation'''
RDA.sim_begin_time = '2002-01-01'
RDA.sim_end_time = '2016-04-30'

def demo1():
    '''serial: python RDA_local.py Config_ReWaterGAP.json'''

    '''before the open loop'''
    # RDA.config_external_data()
    # RDA.read_config_and_save()

    '''before the data assimilation'''
    RDA.config_basin_mask()
    RDA.get_GRACE_obs(is_diagonal=False)

    '''after the data assimilation'''
    # RDA.visualization()
    # RDA.increment_diagnosis()
    # RDA.da_evaluation(tag='run11a2')      # set a new tag for every run

    '''house-keeping: remove the temporary daily output of a collected stage (dry run first)'''
    # from src_DA.EnumDA import Stage
    # RDA.clean_temp_output(Stage.OL, dry_run=True)
    # RDA.clean_temp_output(Stage.DA, dry_run=True)
    pass


def demo2():
    '''MPI: mpiexec -n <ens+1> python -u RDA_local.py Config_ReWaterGAP.json'''
    from src_DA.EnumDA import Stage
    from mpi4py import MPI

    '''settings are synced between machines: rank 0 first writes the paths of this machine into the JSON files'''
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        RDA.config_external_data()
    comm.barrier()

    RDA.model_perturbation()
    # RDA.spin_up()
    # RDA.OL_run()
    # RDA.collect_and_statistics(Stage.OL, skip_collect=False)

    # try:
    #     RDA.DA_run()
    # except Exception:
    #     import traceback
    #     traceback.print_exc()
    #     MPI.COMM_WORLD.Abort(1)
    # RDA.collect_and_statistics(Stage.DA, skip_collect=False)
    pass


if __name__ == '__main__':
    # demo1()
    demo2()
