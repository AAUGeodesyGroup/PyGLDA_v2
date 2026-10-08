"""PyGLDA v2 driver - global data assimilation, local workstation (src_FlowControl.Global_DA.GDA)"""
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent          # the PyGLDA_v2 folder
sys.path.insert(0, str(CODE_ROOT))

from src_FlowControl.Global_DA import GDA
from src_DA.EnumDA import Stage

DATA_ROOT = '/media/user/My Book/Fan/PyGLDA_v2_external_data'

GDA.case = 'demo_Global'                                     # global units GlobalBasins v1.0 (772 units)
GDA.basin = 'GlobalBasins_v1.0'
GDA.ens = 4
GDA.setting_dir = str(CODE_ROOT / 'settings' / GDA.case)
GDA.external_data_path = DATA_ROOT
GDA.shp_path = DATA_ROOT + '/Basin/shp/Global_v1.0/GlobalBasins_v1.0.shp'

'''for spin-up'''
GDA.spin_up_start = '2000-01-01'
GDA.spin_up_end = '2001-12-31'
GDA.spin_up_years = 5

'''for open-loop and data assimilation'''
GDA.sim_begin_time = '2002-01-01'
GDA.sim_end_time = '2003-12-31'

def demo1():
    '''serial: python demo_global.py Config_ReWaterGAP.json'''

    '''before the data assimilation'''
    # GDA.config_external_data()
    # GDA.read_config_and_save()                 # case, basin, shp, mask and ensemble size into DA_setting.json
    # GDA.check_units()                          # frozen unit set GlobalBasins v1.0
    # GDA.get_GRACE_obs()                        # unit TWS + 772 x 772 COV -> GRACE/output/GlobalBasins_v1.0_*.hdf5

    '''after the data assimilation: global evaluation (to come)'''
    pass


def demo2():
    '''MPI: mpiexec -n <ens+1> python -u demo_global.py Config_ReWaterGAP.json'''
    from src_DA.EnumDA import Stage
    from mpi4py import MPI

    '''settings are synced between machines: rank 0 first writes the paths of this machine into the JSON files'''
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        GDA.config_external_data()
    comm.barrier()

    # GDA.collect_and_statistics(Stage.OL)     # unit series, envelope and Res_OL.h5 from the global open loop

    '''the filter still needs the sparse path for 772 units (see Global_DA.py)'''
    # try:
    #     GDA.DA_run()
    # except Exception:
    #     import traceback
    #     traceback.print_exc()
    #     MPI.COMM_WORLD.Abort(1)
    # GDA.collect_and_statistics(Stage.DA)
    pass


if __name__ == '__main__':
    demo1()
    # demo2()
